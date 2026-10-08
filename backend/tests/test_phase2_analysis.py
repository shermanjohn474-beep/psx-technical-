"""Candlesticks, divergences, structure, fibonacci, harmonics, MTF, setups, scoring, service."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.analysis.service import analyze_frame
from app.candlestick_patterns.engine import detect_candles
from app.chart_patterns.base import build_context
from app.chart_patterns.registry import detect_patterns
from app.data_providers.resample import resample_ohlcv
from app.data_providers.synthetic import make_ohlcv, random_walk, trading_index
from app.data_validation.calendar import PKT
from app.data_validation.validator import validate_ohlcv
from app.divergences.divergence import DivergenceConfig, detect_divergences
from app.harmonic_patterns.fibonacci import fibonacci
from app.indicators.momentum import rsi
from app.market_structure.structure import analyze_structure
from app.multi_timeframe.mtf import analyze_mtf
from app.risk_management.setups import build_setup, reward_risk
from app.schemas.market import DataSourceInfo, Timeframe
from app.schemas.patterns import Direction, PatternStage
from app.support_resistance.levels import detect_levels
from app.swing_detection.swings import zigzag_atr

AS_OF = datetime(2030, 1, 1, tzinfo=PKT)


def candles(rows, trend="down", n_prior=15):
    """Build a frame: a prior trend then explicit (o, h, l, c) candles."""
    prior = []
    p = 120.0 if trend == "down" else 80.0
    step = -1.0 if trend == "down" else 1.0
    if trend == "flat":
        step = 0.0
    for k in range(n_prior):
        o = p
        c = p + step + (0.3 if k % 2 else -0.3) * (trend == "flat")
        prior.append((o, max(o, c) + 0.4, min(o, c) - 0.4, c))
        p = c
    allrows = prior + [tuple(x) for x in rows]
    d = pd.DataFrame(allrows, columns=["open", "high", "low", "close"], index=trading_index(len(allrows)))
    d["volume"] = 1e6
    return d


# ------------------------------------------------------------------ candlesticks
def ids(sig):
    return {s.pattern_id for s in sig}


def test_bullish_engulfing_after_downtrend_high_score():
    d = candles([(105.5, 105.8, 104.4, 104.6), (104.3, 106.4, 104.1, 106.2)])
    sig = [s for s in detect_candles(d, all_bars=True) if s.pattern_id == "bullish_engulfing" and s.index == len(d) - 1]
    assert sig and sig[0].components["trend_context"] > 0.5


def test_bullish_engulfing_in_uptrend_is_low_significance():
    d = candles([(95.5, 95.8, 94.4, 94.6), (94.3, 96.4, 94.1, 96.2)], trend="up")
    up = [s for s in detect_candles(d, all_bars=True) if s.pattern_id == "bullish_engulfing" and s.index == len(d) - 1]
    dn = [s for s in detect_candles(candles([(105.5, 105.8, 104.4, 104.6), (104.3, 106.4, 104.1, 106.2)]), all_bars=True)
          if s.pattern_id == "bullish_engulfing"]
    assert up and dn and up[0].score < dn[-1].score
    assert any("low reversal significance" in e for e in up[0].evidence)


@pytest.mark.parametrize("rows,trend,pid", [
    ([(105.0, 105.1, 102.5, 104.4)], "down", "hammer"),
    ([(95.0, 95.1, 92.5, 94.4)], "up", "hanging_man"),
    ([(95.0, 97.5, 94.95, 95.6)], "up", "shooting_star"),
    ([(100.0, 100.3, 99.9, 100.01)], "flat", "doji"),
    ([(100.0, 101.0, 99.0, 100.02)], "flat", "long_legged_doji"),
    ([(100.0, 100.05, 97.0, 100.02)], "down", "dragonfly_doji"),
    ([(100.0, 103.0, 99.98, 100.01)], "up", "gravestone_doji"),
    ([(104.0, 104.05, 100.95, 101.0)], "down", "bearish_marubozu"),
    ([(106.0, 106.2, 103.0, 103.2), (102.9, 104.8, 102.8, 104.7)], "down", "piercing"),
    ([(106.0, 106.2, 102.0, 102.2), (102.0, 102.6, 101.8, 102.3), (102.5, 105.5, 102.4, 105.3)], "down", "morning_star"),
    ([(106.0, 106.2, 102.0, 102.2), (102.0, 102.4, 101.8, 102.02), (102.5, 105.5, 102.4, 105.3)], "down", "morning_doji_star"),
    ([(105.0, 106.6, 104.9, 106.5), (106.1, 107.8, 106.0, 107.6), (107.2, 109.0, 107.1, 108.8)], "down", "three_white_soldiers"),
    ([(106.0, 106.2, 102.0, 102.2), (102.8, 104.0, 102.6, 103.8), (103.9, 106.8, 103.8, 106.6)], "down", "three_inside_up"),
])
def test_candlestick_shapes(rows, trend, pid):
    d = candles(rows, trend=trend)
    got = {s.pattern_id for s in detect_candles(d, all_bars=True) if s.index == len(d) - 1}
    assert pid in got, got


def test_incomplete_candle_not_classified():
    d = candles([(105.5, 105.8, 104.4, 104.6), (104.3, 106.4, 104.1, 106.2)])
    assert not [s for s in detect_candles(d, all_bars=True, last_candle_complete=False) if s.index == len(d) - 1]


def test_talib_crosscheck_present():
    pytest.importorskip("talib")
    d = candles([(105.5, 105.8, 104.4, 104.6), (104.3, 106.4, 104.1, 106.2)])
    s = [x for x in detect_candles(d, all_bars=True) if x.pattern_id == "bullish_engulfing"][-1]
    assert s.talib_agrees is True


# ------------------------------------------------------------------ divergences
def test_regular_bullish_divergence_rsi():
    # sharp first low, slower lower second low -> RSI higher low
    wp = [(0, 130), (20, 100), (32, 115), (52, 97), (65, 112)]
    d = make_ohlcv(wp)
    sw = zigzag_atr(d)
    div = detect_divergences(d, sw, {"rsi": rsi(d["close"])})
    assert any(x.type == "regular_bullish" for x in div), div
    for x in div:
        assert x.signal_index >= x.pivot2_index  # signal only once second pivot is confirmed


def test_divergence_is_causal():
    d = random_walk(400, seed=8)
    cfg = DivergenceConfig(recent_bars=10_000)
    full = detect_divergences(d, zigzag_atr(d), {"rsi": rsi(d["close"])}, cfg)
    t = 300
    part = detect_divergences(d.iloc[: t + 1], zigzag_atr(d.iloc[: t + 1]), {"rsi": rsi(d["close"].iloc[: t + 1])}, cfg)
    assert [(x.type, x.pivot2_index) for x in full if x.signal_index <= t] == [(x.type, x.pivot2_index) for x in part]


# ------------------------------------------------------------------ structure / fib / harmonics
def test_structure_labels_and_bos():
    d = make_ohlcv([(0, 100), (15, 115), (25, 108), (40, 125), (50, 117), (65, 135)])
    st = analyze_structure(d, zigzag_atr(d))
    labs = [x["label"] for x in st.labels if x["label"]]
    assert "HH" in labs and "HL" in labs
    assert st.trend == "uptrend"
    assert any(e.type == "BOS" and e.direction == "bullish" for e in st.events)


def test_choch_on_reversal():
    d = make_ohlcv([(0, 100), (15, 115), (25, 108), (40, 125), (50, 117), (60, 122), (75, 100)])
    st = analyze_structure(d, zigzag_atr(d))
    assert any(e.type == "CHoCH" and e.direction == "bearish" for e in st.events)


def test_fvg_detection():
    d = make_ohlcv([(0, 100), (30, 100), (31, 108), (60, 109)], noise=0.05, wick=0.05)
    st = analyze_structure(d, zigzag_atr(d))
    assert any(z.direction == "bullish" and z.status != "filled" for z in st.fvgs)
    assert all(z.low < z.high for z in st.fvgs)


def test_fibonacci_levels():
    d = make_ohlcv([(0, 100), (40, 150), (55, 130)])
    f = fibonacci(d, zigzag_atr(d))
    assert f.direction == "up-leg"
    r618 = next(lv for lv in f.levels if lv.ratio == 0.618 and lv.kind == "retracement")
    assert r618.price == pytest.approx(f.anchor_end_price - 0.618 * (f.anchor_end_price - f.anchor_start_price), abs=1e-3)


def test_gartley_detected_and_experimental():
    # X=100, A=130 (XA=30), B=130-0.618*30=111.46, C=111.46+0.6*18.54=122.58, D=130-... AD=0.786*XA -> D=106.42
    wp = [(0, 115), (10, 100), (30, 130), (45, 111.46), (55, 122.58), (75, 106.42), (90, 116)]
    res = [r for r in detect_patterns(make_ohlcv(wp, noise=0.1, wick=0.15)) if r.pattern_id == "gartley_bullish"]
    assert res and all(r.experimental for r in res)


# ------------------------------------------------------------------ MTF
def test_mtf_alignment_bullish():
    d = make_ohlcv([(0, 50), (700, 150)], noise=0.3)
    frames = {Timeframe.D1: d, Timeframe.W1: resample_ohlcv(d, Timeframe.D1, Timeframe.W1),
              Timeframe.MN1: resample_ohlcv(d, Timeframe.D1, Timeframe.MN1)}
    m = analyze_mtf(frames)
    assert m.alignment in ("strong_bullish", "moderate_bullish")
    hourly = next(t for t in m.timeframes if t.timeframe == "1h")
    assert hourly.trend == "unavailable"  # never derived from daily
    assert any("Not assessed" in c for c in m.commentary)


def test_mtf_countertrend_commentary():
    daily = make_ohlcv([(0, 150), (400, 60)], noise=0.3)
    idx = pd.date_range("2024-01-01 09:30", periods=300, freq="60min", tz=PKT)
    hourly = make_ohlcv([(0, 60), (299, 75)], noise=0.05, index=idx)
    frames = {Timeframe.D1: daily, Timeframe.W1: resample_ohlcv(daily, Timeframe.D1, Timeframe.W1), Timeframe.H1: hourly}
    m = analyze_mtf(frames)
    assert any("countertrend" in c for c in m.commentary)


# ------------------------------------------------------------------ setups / R:R
def test_reward_risk_formulas():
    assert reward_risk("long", 100, 95, 110) == 2.0
    assert reward_risk("short", 100, 105, 90) == 2.0
    assert reward_risk("long", 100, 101, 110) is None  # invalid stop
    assert reward_risk("short", 100, 105, 99) == 0.2


def _report(wp, **kw):
    d = make_ohlcv(wp, **kw)
    frame, q = validate_ohlcv(d.reset_index(), "SYN-T", Timeframe.D1, as_of=AS_OF)
    src = DataSourceInfo(provider="test", attribution="synthetic", is_synthetic=True, adjusted=True)
    return analyze_frame(frame, symbol="SYN-T", timeframe=Timeframe.D1, source=src, quality=q, as_of=AS_OF)


def test_setup_stops_are_not_moved_for_rr():
    from tests.pattern_fixtures import DOUBLE_TOP
    wp = DOUBLE_TOP["incomplete"]
    rep = _report(wp)
    short = rep.short_setup
    pat = next(p for p in rep.patterns if p.pattern_id == short.pattern_id)
    # stop sits at pattern invalidation + fixed buffer regardless of resulting R:R
    assert short.stop_loss == pytest.approx(pat.invalidation_level + 0.2 * rep.indicators["atr"]["value"], rel=1e-2)
    for t in short.targets:
        assert t.rr == reward_risk("short", short.entry_price, short.stop_loss, t.price)
    assert "ANALYTICAL ONLY" in short.execution


def test_low_rr_flagged_not_hidden():
    from tests.pattern_fixtures import DOUBLE_TOP
    rep = _report(DOUBLE_TOP["incomplete"])
    s = rep.short_setup
    if s.primary_rr is not None and s.primary_rr < 2:
        assert not s.meets_min_rr and any("lower-quality" in n for n in s.notes)


def test_score_reports_missing_volume_coverage():
    from tests.pattern_fixtures import ASC_TRIANGLE
    d = make_ohlcv(ASC_TRIANGLE["incomplete"])
    d["volume"] = np.nan
    frame, q = validate_ohlcv(d.reset_index(), "SYN-T", Timeframe.D1, as_of=AS_OF)
    rep = analyze_frame(frame, symbol="SYN-T", timeframe=Timeframe.D1,
                        source=DataSourceInfo(provider="t", attribution="t", is_synthetic=True), quality=q, as_of=AS_OF)
    vol = next(c for c in rep.score.components if c.name == "volume")
    assert vol.value is None and rep.score.coverage < 1.0
    assert "not a probability" in rep.score.disclaimer


def test_full_report_sections_present():
    from tests.pattern_fixtures import ASC_TRIANGLE
    rep = _report(ASC_TRIANGLE["incomplete"] + [(95, 129), (100, 129.5)])
    assert rep.overview.symbol == "SYN-T"
    assert rep.trend.timeframes and rep.indicators["rsi"] is not None
    assert rep.key_levels is not None and rep.long_setup and rep.short_setup
    assert rep.commentary.engine == "deterministic-template" and rep.commentary.conclusion
    assert any("SYNTHETIC" in w for w in rep.warnings)


def test_neutral_conclusion_on_noise():
    rep = _report([(0, 100), (200, 100)], noise=0.3)
    assert rep.commentary.stay_neutral
    assert "NEUTRAL" in rep.commentary.conclusion


def test_pipeline_no_lookahead():
    """Confirmed-pattern events at bar t are identical whether or not future bars exist."""
    d = random_walk(450, seed=21, vol=0.02)
    full = detect_patterns(d)
    for t in (250, 320, 400):
        part = detect_patterns(d.iloc[: t + 1])
        conf_part = {(p.pattern_id, p.confirmation_index) for p in part if p.confirmation_index is not None}
        # every confirmation that the truncated run reports must reference bars <= t
        assert all(ci <= t for _, ci in conf_part)
        # and each full-run confirmation at bar <= t must have been visible at time t (same trigger bar)
        ctx = build_context(d.iloc[: t + 1])
        visible = {(p.pattern_id, p.confirmation_index) for p in detect_patterns(ctx=ctx)}
        for p in full:
            if p.confirmation_index is not None and p.confirmation_index == t:
                assert (p.pattern_id, p.confirmation_index) in visible
