"""Screenshot analysis (three tiers) and AI analyst validation, with fake providers."""
import base64
from datetime import datetime

import cv2
import numpy as np
import pytest

from app.ai_analyst.analyst import AICommentary, ai_commentary, build_evidence, validate_against_evidence
from app.ai_analyst.providers import AIResult, LLMProvider, NullProvider
from app.analysis.service import analyze_frame
from app.data_providers.file_provider import FileProvider
from app.data_providers.synthetic import make_ohlcv
from app.data_validation.calendar import PKT
from app.data_validation.validator import validate_ohlcv
from app.schemas.market import DataSourceInfo, Timeframe
from app.screenshot_analysis.extract import AxisCalibration, extract_candles
from app.screenshot_analysis.render import RenderSpec, render_chart
from app.screenshot_analysis.service import analyze_screenshot, match_to_data, media_type
from tests.pattern_fixtures import DOUBLE_TOP, HEAD_SHOULDERS, mirror

DB_WP = mirror(DOUBLE_TOP["valid"])


def png(img) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def chart(wp=DB_WP, **spec):
    d = make_ohlcv(wp)
    img, g = render_chart(d, RenderSpec(**spec))
    return d, img, g


class FakeProvider(LLMProvider):
    name = "fake"
    model = "fake-1"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete_json(self, system, text, schema, images=None, max_tokens=16000):
        self.calls += 1
        data = self.payload(text) if callable(self.payload) else self.payload
        return AIResult(data=data, provider=self.name, model=self.model, latency_s=0.0)


def vision_payload(g, symbol="SYN-DBOT", tf="1D", patterns=None):
    return {"symbol": symbol, "symbol_confidence": 0.9, "timeframe": tf, "timeframe_confidence": 0.9,
            "platform": "TradingView", "chart_type": "candlestick", "log_scale": "no",
            "axis_ticks": [{"y_frac": g.y(g.pmax) / 700, "price": g.pmax}, {"y_frac": g.y(g.pmin) / 700, "price": g.pmin}],
            "indicators_visible": [], "visible_patterns": patterns or [], "market_structure": "", "bullish_scenario": "",
            "bearish_scenario": "", "image_quality": "good", "notes": []}


# --------------------------------------------------------------- extraction
@pytest.mark.parametrize("spec", [{}, {"labels_overlay": True, "title": "SYN 1D"}, {"log_scale": True},
                                  {"width": 700, "height": 420}, {"volume_panel": False}, {"dark": True}])
def test_extraction_recovers_candles(spec):
    d, img, g = chart(**spec)
    px, res = extract_candles(img)
    assert res.ok and res.candles == len(d)
    c = px["close"].to_numpy()
    ref = np.log(d["close"].to_numpy()) if spec.get("log_scale") else d["close"].to_numpy()
    assert np.corrcoef(c, ref)[0, 1] > 0.999


def test_multi_panel_detected():
    _, img, _ = chart(volume_panel=True)
    _, res = extract_candles(img)
    assert sum(p.role == "sub_panel" for p in res.panels) >= 1


def test_grayscale_fails_gracefully():
    _, img, _ = chart()
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = analyze_screenshot(png(gray))
    assert out.tier == "visual_only" and not out.extraction.ok and out.confidence == 0.0
    assert not out.patterns


def test_media_type_validation():
    _, img, _ = chart()
    assert media_type(png(img)) == "image/png"
    assert media_type(cv2.imencode(".jpg", img)[1].tobytes()) == "image/jpeg"
    assert media_type(cv2.imencode(".webp", img)[1].tobytes()) == "image/webp"
    with pytest.raises(ValueError):
        media_type(b"GIF89a....")


# --------------------------------------------------------------- tiers
def test_visual_only_tier_detects_pattern_without_prices():
    _, img, _ = chart()
    out = analyze_screenshot(png(img), filename="db.png")
    assert out.tier == "visual_only" and out.units == "pixels" and out.preliminary
    assert {"symbol", "timeframe"} <= set(out.needs_confirmation)
    assert any(p.pattern_id == "double_bottom" for p in out.patterns)
    assert out.report is None and out.annotated_png_b64
    cv2.imdecode(np.frombuffer(base64.b64decode(out.annotated_png_b64), np.uint8), cv2.IMREAD_COLOR).shape


def test_estimated_tier_with_user_calibration():
    d, img, g = chart()
    cal = AxisCalibration(y1=g.y(g.pmax), price1=g.pmax, y2=g.y(g.pmin), price2=g.pmin)
    out = analyze_screenshot(png(img), calibration=cal)
    assert out.tier == "estimated" and out.price_uncertainty is not None
    db = next(p for p in out.patterns if p.pattern_id == "double_bottom")
    real = [p for p in __import__("app.chart_patterns.registry", fromlist=["x"]).detect_patterns(d) if p.pattern_id == "double_bottom"][0]
    assert abs(db.breakout_level - real.breakout_level) < 1.0  # estimated neckline close to true neckline


def test_log_scale_calibration():
    d, img, g = chart(log_scale=True)
    cal = AxisCalibration(y1=g.y(g.pmax), price1=g.pmax, y2=g.y(g.pmin), price2=g.pmin, log_scale=True)
    out = analyze_screenshot(png(img), calibration=cal, log_scale=True)
    px, _ = extract_candles(img)
    from app.screenshot_analysis.extract import to_price_frame
    pf, _ = to_price_frame(px, cal)
    assert np.abs(pf["close"].to_numpy() - d["close"].to_numpy()).max() < 0.5
    assert out.tier == "estimated"


def test_vision_calibration_and_claim_crosscheck():
    _, img, g = chart()
    claims = [{"name": "Double Bottom", "direction": "bullish", "stage": "confirmed", "description": "two lows"},
              {"name": "Cup and Handle", "direction": "bullish", "stage": "forming", "description": "u shape"}]
    prov = FakeProvider(vision_payload(g, symbol="", tf="", patterns=claims))
    out = analyze_screenshot(png(img), vision_provider=prov)
    assert out.vision_status.startswith("ok:fake")
    assert out.tier == "estimated" and out.calibration.source == "vision_model"
    sup = {c.name: c.geometry_supported for c in out.pattern_claims}
    assert sup["Double Bottom"] is True and sup["Cup and Handle"] is False


def test_data_verified_tier(tmp_path):
    d, img, g = chart()
    full = make_ohlcv(DB_WP)  # same seed => identical series
    pre = make_ohlcv([(0, 150), (60, 120)], seed=3)
    pre.index = pre.index - (pre.index[-1] - full.index[0]) - __import__("pandas").Timedelta(days=1)
    data = __import__("pandas").concat([pre, full])
    fp = FileProvider(tmp_path)
    fp.save_upload("SYN-DBOT", Timeframe.D1, data.reset_index().to_csv(index=False).encode(), "x.csv")
    out = analyze_screenshot(png(img), symbol="SYN-DBOT", timeframe=Timeframe.D1, data_provider=fp)
    assert out.tier == "data_verified" and not out.preliminary
    assert out.data_match.matched and out.data_match.price_correlation > 0.99
    assert out.report is not None and out.units == "price"


def test_wrong_symbol_does_not_verify(tmp_path):
    _, img, _ = chart()
    other = make_ohlcv(HEAD_SHOULDERS["valid"], seed=11)
    fp = FileProvider(tmp_path)
    fp.save_upload("SYN-OTHER", Timeframe.D1, other.reset_index().to_csv(index=False).encode(), "x.csv")
    out = analyze_screenshot(png(img), symbol="SYN-OTHER", timeframe=Timeframe.D1, data_provider=fp)
    assert out.tier != "data_verified" and out.data_match and not out.data_match.matched


def test_match_requires_return_correlation():
    d = make_ohlcv(DB_WP)
    px = d[["open", "high", "low", "close"]].copy()
    trend_only = make_ohlcv([(0, d["close"].iloc[0]), (100, d["close"].iloc[-1])], seed=5)
    dm, _ = match_to_data(px, trend_only)
    assert not dm.matched


# --------------------------------------------------------------- AI analyst
def _report():
    d = make_ohlcv(DOUBLE_TOP["incomplete"])
    frame, q = validate_ohlcv(d.reset_index(), "SYN-T", Timeframe.D1, as_of=datetime(2030, 1, 1, tzinfo=PKT))
    return analyze_frame(frame, symbol="SYN-T", timeframe=Timeframe.D1, quality=q,
                         source=DataSourceInfo(provider="t", attribution="t", is_synthetic=True))


def _commentary(rep, text_price=None):
    ev = build_evidence(rep)
    price = ev["last_price"] if text_price is None else text_price
    return {"bias": rep.score.rating, "summary": f"Last price {price:.2f}.", "chart_reading": "Two highs.",
            "pattern_assessment": "Double top forming, not confirmed.", "bullish_confirmation": "Close above invalidation.",
            "bearish_confirmation": "Close below neckline.", "key_levels": "See evidence.", "trade_plan": "Wait.",
            "strongest_evidence": ["Momentum"], "conflicting_signals": [], "what_changes_view": [],
            "stay_neutral": True, "conclusion": "NEUTRAL - NO TRADE - AWAITING CONFIRMATION"}


def test_ai_commentary_accepted_when_grounded(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.config import settings as s
    s.get_settings.cache_clear()
    rep = _report()
    c, rec = ai_commentary(rep, FakeProvider(_commentary(rep)))
    assert rec["status"] == "accepted" and c.engine == "fake:fake-1"
    assert rec["evidence_sha256"] and (tmp_path / "audit" / "ai_audit.jsonl").exists()
    s.get_settings.cache_clear()


def test_ai_commentary_rejected_when_inventing_numbers(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.config import settings as s
    s.get_settings.cache_clear()
    rep = _report()
    c, rec = ai_commentary(rep, FakeProvider(_commentary(rep, text_price=987.65)))
    assert rec["status"] == "rejected" and 987.65 in rec["validation"]["unsupported_numbers"]
    assert c.engine == "deterministic-template"
    s.get_settings.cache_clear()


def test_ai_rejects_invented_pattern():
    rep = _report()
    payload = _commentary(rep)
    payload["pattern_assessment"] = "A bullish pennant and a cup and handle are visible."
    val = validate_against_evidence(AICommentary.model_validate(payload), build_evidence(rep))
    assert not val["passed"] and val["unknown_patterns"]


def test_null_provider_keeps_deterministic():
    rep = _report()
    c, rec = ai_commentary(rep, NullProvider())
    assert rec["status"] == "skipped" and c.engine == "deterministic-template"


def test_schema_violation_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.config import settings as s
    s.get_settings.cache_clear()
    rep = _report()
    c, rec = ai_commentary(rep, FakeProvider({"bias": "Bullish"}))
    assert rec["status"] == "error" and c.engine == "deterministic-template"
    s.get_settings.cache_clear()
