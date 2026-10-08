"""API (TestClient), backtesting (look-ahead, costs, execution) and alerts."""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.backtesting.engine import CostModel, run_backtest
from app.backtesting.strategies import Signal, Strategy, StrategySpec
from app.data_providers.synthetic import make_ohlcv, random_walk


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.sqlite3'}")
    from app.config import settings
    from app.data_providers import manager
    from app.database import repository
    settings.get_settings.cache_clear()
    manager.get_provider.cache_clear()
    repository.get_engine.cache_clear()
    from app.main import app
    with TestClient(app) as c:
        yield c
    settings.get_settings.cache_clear()
    manager.get_provider.cache_clear()
    repository.get_engine.cache_clear()


# ------------------------------------------------------------------ API
def test_health_and_symbols(client):
    assert client.get("/health").json()["status"] == "ok"
    syms = {s["symbol"]: s for s in client.get("/api/symbols").json()}
    assert syms["SYN-DBOT"]["has_data"] and syms["SYN-DBOT"]["synthetic"]
    assert syms["OGDC"]["has_data"] is False  # real symbol: no fabricated data


def test_real_symbol_without_data_404(client):
    r = client.get("/api/analysis/OGDC")
    assert r.status_code == 404 and "Upload" in r.json()["detail"]


def test_upload_then_analyze_and_report(client):
    df = random_walk(300, seed=4).reset_index()
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    r = client.post("/api/data/upload", files={"file": ("ogdc.csv", df.to_csv(index=False).encode(), "text/csv")},
                    data={"symbol": "OGDC", "timeframe": "1d", "attribution": "test fixture (not real prices)"})
    assert r.status_code == 200, r.text
    rep = client.get("/api/analysis/OGDC").json()["report"]
    assert rep["overview"]["company_name"].startswith("Oil & Gas")
    for k in ("trend", "indicators", "patterns", "key_levels", "long_setup", "short_setup", "score", "commentary"):
        assert k in rep
    o = client.get("/api/ohlcv/OGDC?indicators=sma_50,rsi,vwap").json()
    assert len(o["candles"]) == 300 and "sma_50" in o["indicators"] and "vwap" in o["indicators_unavailable"]
    assert client.get("/api/ohlcv/OGDC?timeframe=1h").status_code == 404  # never derived from daily
    html = client.get("/api/report/OGDC").text
    assert "F. Trading Report Card" in html and "data:image/png;base64" in html
    png = client.get("/api/analysis/OGDC/chart.png")
    assert png.headers["content-type"] == "image/png"


def test_reserved_synthetic_prefix(client):
    r = client.post("/api/data/upload", files={"file": ("x.csv", b"date,open,high,low,close\n", "text/csv")},
                    data={"symbol": "SYN-FAKE", "timeframe": "1d"})
    assert r.status_code == 400


def test_screenshot_endpoint(client):
    import cv2
    from app.screenshot_analysis.render import RenderSpec, render_chart
    from tests.pattern_fixtures import DOUBLE_TOP, mirror
    img, _ = render_chart(make_ohlcv(mirror(DOUBLE_TOP["valid"])), RenderSpec())
    data = cv2.imencode(".png", img)[1].tobytes()
    r = client.post("/api/screenshot", files=[("files", ("a.png", data, "image/png"))])
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["tier"] == "visual_only" and res["preliminary"]
    bad = client.post("/api/screenshot", files=[("files", ("a.gif", b"GIF89a", "image/gif"))])
    assert bad.status_code == 400


def test_scan_watchlist_presets(client):
    assert client.post("/api/watchlists", json={"name": "demo", "symbols": ["SYN-DBOT", "SYN-HS"]}).status_code == 200
    r = client.post("/api/scan", json={"watchlist": "demo", "with_mtf": False}).json()
    assert r["scanned"] == 2 and {row["symbol"] for row in r["rows"]} <= {"SYN-DBOT", "SYN-HS"}
    f = client.post("/api/scan", json={"symbols": ["SYN-DBOT", "SYN-HS", "SYN-FLAG"], "with_mtf": False,
                                       "rules": {"directions": ["bullish"]}}).json()
    assert all(row["direction"] == "bullish" for row in f["rows"])
    assert client.post("/api/scan/presets", json={"name": "bull", "rules": {"min_score": 60}}).status_code == 200
    assert client.get("/api/scan/presets").json()[0]["name"] == "bull"


def test_backtest_endpoint(client):
    r = client.post("/api/backtest", json={"symbols": ["SYN-DBOT", "OGDC"], "strategy": {"type": "sma_cross"},
                                           "walk_forward": True, "folds": 2}).json()
    assert r["results"][0]["result"]["metrics"]["trades"] >= 0 and "OGDC" in r["errors"]
    assert "walk_forward" in r["results"][0]


def test_alert_lifecycle_and_dedup(client):
    assert client.post("/api/alerts", json={"symbol": "SYN-DBOT", "condition": "nope"}).status_code == 400
    a = client.post("/api/alerts", json={"symbol": "SYN-DBOT", "condition": "price_above", "params": {"level": 1}}).json()
    assert a["id"]
    ev = client.post("/api/alerts/evaluate").json()
    st = ev["status"]
    assert st["monitoring_active"] is False and st["notes"]
    assert client.get("/api/alerts").json()[0]["last_status"]
    assert client.get("/api/audit").json()  # mutating calls are audited
    b = client.post("/api/alerts", json={"symbol": "SYN-DBOT", "condition": "unusual_volume", "params": {"mult": 0.0},
                                         "channels": [{"type": "webhook", "url": "http://127.0.0.1:9/unreachable"}]}).json()
    first = [x for x in client.post("/api/alerts/evaluate").json()["results"] if x["rule_id"] == b["id"]][0]
    assert first["status"] == "triggered" and first["deliveries"][0]["ok"] is False  # failure recorded, not raised
    hit = first["hit"]
    assert hit["symbol"] == "SYN-DBOT" and hit["bar_timestamp"] and hit["data_as_of"] and hit["link"]
    second = [x for x in client.post("/api/alerts/evaluate").json()["results"] if x["rule_id"] == b["id"]][0]
    assert second["status"] == "duplicate_suppressed"
    assert len(client.get("/api/alerts/events").json()) == 1


def test_api_key_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_API_KEYS", "secret-key-1")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'k.sqlite3'}")
    from app.config import settings
    from app.database import repository
    settings.get_settings.cache_clear()
    repository.get_engine.cache_clear()
    from app.main import app
    with TestClient(app) as c:
        assert c.get("/api/symbols").status_code == 401
        assert c.get("/api/symbols", headers={"X-API-Key": "secret-key-1"}).status_code == 200
        cfg = c.get("/api/config", headers={"X-API-Key": "secret-key-1"}).json()
        assert "secret" not in str(cfg)
    settings.get_settings.cache_clear()
    repository.get_engine.cache_clear()


# ------------------------------------------------------------------ backtesting
def test_backtest_no_lookahead_future_garbage():
    """Trades entered before the cutoff are identical when the future is replaced by garbage."""
    df = random_walk(500, seed=31, vol=0.02)
    spec = StrategySpec(type="sma_cross", max_hold=15)
    base = run_backtest(df, spec)
    alt = df.copy()
    cut = 350
    rng = np.random.default_rng(0)
    alt.iloc[cut:, :4] = alt.iloc[cut:, :4].to_numpy() * rng.uniform(0.5, 1.5, (len(df) - cut, 1))
    alt["high"] = alt[["open", "high", "low", "close"]].max(axis=1)
    alt["low"] = alt[["open", "high", "low", "close"]].min(axis=1)
    other = run_backtest(alt, spec)
    done = lambda r: [(t.entry_time, t.exit_time, t.pnl) for t in r.trades if t.exit_time < df.index[cut].isoformat()]  # noqa: E731
    assert done(base) == done(other) and done(base)


def test_pattern_strategy_signals_only_from_past():
    class Spy(Strategy):
        def __init__(self):
            super().__init__(StrategySpec())
            self.lengths = []

        def on_bar(self, hist, t):
            self.lengths.append((len(hist), t))
            return []

    spy = Spy()
    run_backtest(random_walk(120, seed=2), StrategySpec(), strategy=spy)
    assert all(n == t + 1 for n, t in spy.lengths)


class OneShot(Strategy):
    def __init__(self, at, stop_off=-5.0, target_off=5.0):
        super().__init__(StrategySpec())
        self.at, self.so, self.to = at, stop_off, target_off

    def on_bar(self, hist, t):
        if t == self.at:
            c = float(hist["close"].iloc[-1])
            return [Signal("long", c + self.so, c + self.to, "test", 30)]
        return []


def flat_df(n=60, price=100.0):
    idx = random_walk(n).index
    return pd.DataFrame({"open": price, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": 1e6}, index=idx)


def test_costs_reduce_pnl_and_time_exit():
    df = flat_df()
    free = CostModel(commission_pct=0, commission_min_per_share=0, levies_pct=0, slippage_bps=0, spread_ticks=0)
    r0 = run_backtest(df, StrategySpec(), strategy=OneShot(10), costs=free)
    r1 = run_backtest(df, StrategySpec(), strategy=OneShot(10))
    assert r0.trades[0].exit_reason == "time" and r0.trades[0].pnl == pytest.approx(0, abs=1e-6)
    assert r1.trades[0].pnl < 0 and r1.trades[0].fees > 0


def test_gap_through_stop_skips_entry():
    df = flat_df()
    df.iloc[11, df.columns.get_loc("open")] = 90.0
    df.iloc[11, df.columns.get_loc("low")] = 89.5
    r = run_backtest(df, StrategySpec(), strategy=OneShot(10))
    assert not r.trades and r.metrics.skipped_entries.get("gap_through_stop") == 1


def test_limit_locked_bar_skips_entry():
    df = flat_df()
    df.iloc[11, :4] = 107.5  # locked limit-up: o=h=l=c at +7.5%
    r = run_backtest(df, StrategySpec(), strategy=OneShot(10, stop_off=-5, target_off=20))
    assert r.metrics.skipped_entries.get("price_limit_locked") == 1


def test_stop_assumed_before_target_same_bar():
    df = flat_df()
    df.iloc[13, df.columns.get_loc("high")] = 110
    df.iloc[13, df.columns.get_loc("low")] = 90
    r = run_backtest(df, StrategySpec(), strategy=OneShot(10))
    assert r.trades[0].exit_reason == "stop"


def test_liquidity_cap():
    df = flat_df()
    df["volume"] = 100.0
    r = run_backtest(df, StrategySpec(), strategy=OneShot(10))
    assert r.trades[0].shares <= 10


def test_sharpe_withheld_when_not_meaningful():
    r = run_backtest(flat_df(), StrategySpec(), strategy=OneShot(10))
    assert r.metrics.sharpe is None and "not statistically meaningful" in r.metrics.sharpe_note
