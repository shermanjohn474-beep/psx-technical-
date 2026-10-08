"""Chart screenshot analysis with a three-tier confidence model.

* ``visual_only``  - geometry from OpenCV in pixel space (+ optional vision-model
  observations). No prices, dates or indicator values are reported.
* ``estimated``    - an axis calibration (user-supplied or read by the vision model)
  maps pixels to prices; values carry an explicit +/- uncertainty.
* ``data_verified`` - symbol and timeframe are known and the extracted candle path
  matches licensed/user-supplied OHLCV (price *and* return correlation thresholds);
  the full numerical engine then runs on the real data.

The vision model is used for what geometry cannot do (reading the symbol,
timeframe, axis labels and indicator legends). Its pattern claims are
cross-checked against the geometric detections and labelled accordingly.
"""
from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from app.ai_analyst.providers import AIProviderError, ImageInput, LLMProvider, NullProvider
from app.analysis.service import AnalysisOptions, analyze_frame, default_pattern_cfg
from app.chart_patterns.base import PatternConfig, build_context
from app.chart_patterns.registry import detect_patterns
from app.data_providers.base import DataUnavailable, MarketDataProvider
from app.data_providers.symbols import resolve_symbol
from app.data_providers.synthetic import trading_index
from app.schemas.market import Timeframe
from app.schemas.patterns import PatternResult
from app.schemas.report import AnalysisReport
from app.screenshot_analysis.extract import AxisCalibration, ExtractionResult, decode_image, extract_candles, to_price_frame
from app.support_resistance.levels import detect_levels

VISION_SYSTEM = """You are a chart-reading assistant for a technical analysis platform. Describe ONLY what is visible in the screenshot.
- If the ticker symbol, timeframe or a value is not clearly legible, return an empty string / empty list. Never guess.
- axis_ticks: up to 4 clearly legible price-axis labels, each with its vertical position as a fraction of image height (0=top, 1=bottom).
- Pattern descriptions are visual impressions only; they will be verified geometrically.
- Do not state exact indicator values unless the number is printed on the chart."""

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "symbol_confidence": {"type": "number"},
        "timeframe": {"type": "string"},
        "timeframe_confidence": {"type": "number"},
        "platform": {"type": "string"},
        "chart_type": {"type": "string", "enum": ["candlestick", "ohlc_bar", "line", "heikin_ashi", "other"]},
        "log_scale": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "axis_ticks": {"type": "array", "items": {"type": "object", "properties": {
            "y_frac": {"type": "number"}, "price": {"type": "number"}}, "required": ["y_frac", "price"],
            "additionalProperties": False}},
        "indicators_visible": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "reading": {"type": "string"}}, "required": ["name", "reading"],
            "additionalProperties": False}},
        "visible_patterns": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "stage": {"type": "string", "enum": ["forming", "confirmed", "unclear"]}, "description": {"type": "string"}},
            "required": ["name", "direction", "stage", "description"], "additionalProperties": False}},
        "market_structure": {"type": "string"},
        "bullish_scenario": {"type": "string"},
        "bearish_scenario": {"type": "string"},
        "image_quality": {"type": "string", "enum": ["good", "fair", "poor"]},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["symbol", "symbol_confidence", "timeframe", "timeframe_confidence", "platform", "chart_type", "log_scale",
                 "axis_ticks", "indicators_visible", "visible_patterns", "market_structure", "bullish_scenario",
                 "bearish_scenario", "image_quality", "notes"],
    "additionalProperties": False,
}


class VisionObservation(BaseModel):
    symbol: str = ""
    symbol_confidence: float = 0.0
    timeframe: str = ""
    timeframe_confidence: float = 0.0
    platform: str = ""
    chart_type: str = "other"
    log_scale: str = "unknown"
    axis_ticks: list[dict] = Field(default_factory=list)
    indicators_visible: list[dict] = Field(default_factory=list)
    visible_patterns: list[dict] = Field(default_factory=list)
    market_structure: str = ""
    bullish_scenario: str = ""
    bearish_scenario: str = ""
    image_quality: str = "fair"
    notes: list[str] = Field(default_factory=list)


class PatternClaim(BaseModel):
    name: str
    direction: str
    stage: str
    description: str
    geometry_supported: bool
    matched_detection: str | None = None


class DataMatch(BaseModel):
    matched: bool
    price_correlation: float | None = None
    return_correlation: float | None = None
    end_timestamp: str | None = None
    bars: int | None = None
    note: str | None = None


class ScreenshotAnalysis(BaseModel):
    analysis_id: str
    filename: str | None
    tier: str  # visual_only | estimated | data_verified
    preliminary: bool
    needs_confirmation: list[str] = Field(default_factory=list)
    symbol: str | None = None
    timeframe: str | None = None
    extraction: ExtractionResult
    calibration: AxisCalibration | None = None
    price_uncertainty: float | None = None
    vision: VisionObservation | None = None
    vision_status: str
    pattern_claims: list[PatternClaim] = Field(default_factory=list)
    patterns: list[PatternResult] = Field(default_factory=list)
    units: str  # "pixels" | "price (estimated)" | "price"
    dates_are_placeholders: bool = True
    levels: list[dict] = Field(default_factory=list)
    data_match: DataMatch | None = None
    report: AnalysisReport | None = None
    annotated_png_b64: str | None = None
    confidence: float = 0.0
    notes: list[str] = Field(default_factory=list)


def media_type(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unsupported image format (PNG, JPG, JPEG, WEBP only).")


def _vision(provider: LLMProvider, data: bytes, mt: str) -> tuple[VisionObservation | None, str]:
    if isinstance(provider, NullProvider):
        return None, "not_configured"
    try:
        res = provider.complete_json(VISION_SYSTEM, "Describe this chart screenshot.", VISION_SCHEMA,
                                     images=[ImageInput(data, mt)], max_tokens=8000)
        return VisionObservation.model_validate(res.data), f"ok:{res.provider}:{res.model}"
    except (AIProviderError, ValidationError) as exc:
        return None, f"error:{str(exc)[:120]}"


def calibration_from_ticks(ticks: list[dict], height: int, log_scale: bool) -> AxisCalibration | None:
    pts = sorted({(round(float(t["y_frac"]) * height, 1), float(t["price"])) for t in ticks
                  if 0 <= float(t.get("y_frac", -1)) <= 1 and float(t.get("price", 0)) > 0})
    if len(pts) < 2:
        return None
    (y1, p1), (y2, p2) = pts[0], pts[-1]
    if not (y2 - y1 > 0.1 * height and p1 > p2):  # higher on screen must be higher price
        return None
    return AxisCalibration(y1=y1, price1=p1, y2=y2, price2=p2, log_scale=log_scale, source="vision_model")


def match_to_data(pixel: pd.DataFrame, actual: pd.DataFrame, log_scale: bool = False,
                  min_price_corr: float = 0.95, min_ret_corr: float = 0.8) -> tuple[DataMatch, int | None]:
    n = len(pixel)
    c = actual["close"].to_numpy(dtype=float)
    if len(c) < n:
        return DataMatch(matched=False, note=f"Only {len(c)} data bars; screenshot has {n} candles."), None
    p = pixel["close"].to_numpy(dtype=float)
    if log_scale:
        c = np.log(c)  # pixel heights are linear in log price on a log chart
    pr = np.diff(p)
    best = (-2.0, -2.0, None)
    for e in range(n - 1, len(c)):
        w = c[e - n + 1:e + 1]
        if np.std(w) == 0:
            continue
        pc = float(np.corrcoef(p, w)[0, 1])
        if pc < min_price_corr - 0.05:
            continue
        rc = float(np.corrcoef(pr, np.diff(w))[0, 1])
        if pc + rc > best[0] + best[1] + 1e-9:
            best = (pc, rc, e)
    pc, rc, e = best
    if e is None or pc < min_price_corr or rc < min_ret_corr:
        return DataMatch(matched=False, price_correlation=None if e is None else round(pc, 4),
                         return_correlation=None if e is None else round(rc, 4),
                         note="Extracted candle path does not match the available data (wrong symbol/timeframe, "
                              "different adjustment, or data does not cover the screenshot period)."), None
    return DataMatch(matched=True, price_correlation=round(pc, 4), return_correlation=round(rc, 4),
                     end_timestamp=actual.index[e].isoformat(), bars=n), e


def _fit_calibration(pixel: pd.DataFrame, prices: np.ndarray, log_scale: bool) -> AxisCalibration:
    y = pixel["y_close"].to_numpy(dtype=float)
    v = np.log(prices) if log_scale else prices
    m, b = np.polyfit(y, v, 1)
    y1, y2 = float(y.min()), float(y.max())
    f = (lambda yy: float(np.exp(m * yy + b))) if log_scale else (lambda yy: float(m * yy + b))
    return AxisCalibration(y1=y1, price1=f(y1), y2=y2, price2=f(y2), log_scale=log_scale, source="data_fit")


def _annotate(img: np.ndarray, x_centers: list[float], y_of, patterns: list[PatternResult], offset: int = 0,
              zones=None, max_patterns: int = 4) -> np.ndarray:
    out = img.copy()
    n = len(x_centers)

    def X(i):
        k = int(i) - offset
        return int(x_centers[min(max(k, 0), n - 1)]) if n else 0

    if zones:
        ov = out.copy()
        for z in zones[:6]:
            ya, yb = int(y_of(z["high"])), int(y_of(z["low"]))
            cv2.rectangle(ov, (int(x_centers[0]), ya), (int(x_centers[-1]), max(yb, ya + 2)),
                          (120, 200, 120) if z["kind"] == "support" else (120, 120, 230), -1)
        out = cv2.addWeighted(ov, 0.2, out, 0.8, 0)
    palette = [(223, 146, 0), (0, 140, 255), (160, 60, 160), (60, 150, 60)]
    shown = [p for p in patterns if p.stage.value not in ("invalidated",)][:max_patterns]
    for k_, p in enumerate(shown):
        col = palette[k_ % len(palette)]
        for ln in p.lines:
            cv2.line(out, (X(ln.start.index), int(y_of(ln.start.price))), (X(ln.end.index), int(y_of(ln.end.price))),
                     col, 2, cv2.LINE_AA)
        for kp in p.key_points:
            pt = (X(kp.index), int(y_of(kp.price)))
            cv2.circle(out, pt, 5, col, 2, cv2.LINE_AA)
        # legend (top-left, one row per pattern) avoids overlapping labels
        cv2.rectangle(out, (8, 8 + 22 * k_), (22, 22 + 22 * k_), col, -1)
        cv2.putText(out, f"{p.name} [{p.stage.value}]", (28, 21 + 22 * k_), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (40, 30, 20), 1, cv2.LINE_AA)
    return out


def _claims(vision: VisionObservation | None, patterns: list[PatternResult]) -> list[PatternClaim]:
    out = []
    for vp in (vision.visible_patterns if vision else []):
        name = vp.get("name", "")
        toks = {t for t in name.lower().replace("&", "and").replace("-", " ").split() if len(t) > 2}
        match = None
        for p in patterns:
            ptoks = set(p.name.lower().replace("&", "and").replace("-", " ").split())
            if toks and len(toks & ptoks) >= max(1, len(toks) - 1):
                match = p.name
                break
        out.append(PatternClaim(name=name, direction=vp.get("direction", "neutral"), stage=vp.get("stage", "unclear"),
                                description=vp.get("description", ""), geometry_supported=match is not None,
                                matched_detection=match))
    return out


def analyze_screenshot(data: bytes, *, filename: str | None = None, symbol: str | None = None,
                       timeframe: Timeframe | None = None, calibration: AxisCalibration | None = None,
                       vision_provider: LLMProvider | None = None, data_provider: MarketDataProvider | None = None,
                       log_scale: bool | None = None) -> ScreenshotAnalysis:
    mt = media_type(data)
    img = decode_image(data)
    h = img.shape[0]
    pixel, ext = extract_candles(img)
    vision, vstatus = _vision(vision_provider or NullProvider(), data, mt)
    notes: list[str] = []
    needs: list[str] = []
    if vstatus == "not_configured":
        notes.append("Vision model not configured: symbol, timeframe, axis labels and indicator legends cannot be read "
                     "automatically; supply them manually for higher-confidence analysis.")
    # resolve symbol / timeframe
    sym = symbol
    if not sym and vision and vision.symbol and vision.symbol_confidence >= 0.6:
        sym = resolve_symbol(vision.symbol).symbol
        notes.append(f"Symbol '{sym}' read by the vision model (confidence {vision.symbol_confidence:.2f}); please confirm.")
        needs.append("symbol")
    tf = timeframe
    if tf is None and vision and vision.timeframe and vision.timeframe_confidence >= 0.6:
        tf = _parse_tf(vision.timeframe)
        if tf:
            notes.append(f"Timeframe '{tf.value}' read by the vision model; please confirm.")
            needs.append("timeframe")
    if not sym:
        needs.append("symbol")
    if tf is None:
        needs.append("timeframe")
    is_log = log_scale if log_scale is not None else (vision is not None and vision.log_scale == "yes")
    cal = calibration
    if cal is None and vision and vision.axis_ticks:
        cal = calibration_from_ticks(vision.axis_ticks, h, is_log)
        if cal:
            notes.append("Price axis calibrated from vision-model tick readings (estimated).")
    if vision and vision.chart_type not in ("candlestick", "other"):
        notes.append(f"Chart type '{vision.chart_type}': candle extraction assumes standard candlesticks.")

    result = ScreenshotAnalysis(analysis_id=uuid.uuid4().hex[:12], filename=filename, tier="visual_only",
                                preliminary=True, extraction=ext, vision=vision, vision_status=vstatus, units="pixels",
                                symbol=sym, timeframe=tf.value if tf else None, notes=notes)
    if not ext.ok or pixel is None:
        result.notes += ext.issues + ["Geometric extraction failed: result limited to visual observations."]
        result.pattern_claims = _claims(vision, [])
        result.confidence = 0.15 if vision else 0.0
        result.needs_confirmation = sorted(set(needs))
        return result

    # ---------- tier 3: data-verified
    if sym and tf and data_provider is not None:
        try:
            actual = data_provider.get_ohlcv(sym, tf)
            dm, end = match_to_data(pixel, actual.frame, is_log)
            result.data_match = dm
            if dm.matched:
                frame = actual.frame.iloc[: end + 1]
                rep = analyze_frame(frame, symbol=sym, timeframe=tf, source=actual.source, quality=actual.quality,
                                    options=AnalysisOptions(with_mtf=True))
                offset = end - len(pixel) + 1
                fitcal = _fit_calibration(pixel, frame["close"].to_numpy()[offset:], is_log)
                inv = _inverse(fitcal)
                vis = [p for p in rep.patterns if p.start_index >= offset]
                ann = _annotate(img, ext.x_centers, inv, vis, offset,
                                [z.model_dump() for z in rep.levels.zones if z.last_index >= offset])
                result.tier, result.preliminary, result.units = "data_verified", False, "price"
                result.dates_are_placeholders = False
                result.report, result.patterns, result.calibration = rep, rep.patterns, fitcal
                result.annotated_png_b64 = base64.b64encode(cv2.imencode(".png", ann)[1].tobytes()).decode()
                result.pattern_claims = _claims(vision, rep.patterns)
                result.confidence = round(0.6 + 0.4 * min(dm.price_correlation, dm.return_correlation), 2)
                result.needs_confirmation = []
                if any(n_ == "symbol" for n_ in needs):
                    result.notes.append("Symbol/timeframe were inferred but are corroborated by the data match.")
                return result
            result.notes.append(dm.note or "Data match failed.")
        except DataUnavailable as exc:
            result.notes.append(f"Data-verified tier unavailable: {exc}")

    # ---------- tier 2 (estimated) or tier 1 (visual-only) on extracted geometry
    idx = trading_index(len(pixel), "2000-01-03")
    if cal is not None:
        frame, unc = to_price_frame(pixel, cal)
        frame.index = idx
        result.tier, result.units, result.calibration, result.price_uncertainty = "estimated", "price (estimated)", cal, round(unc, 4)
        pcfg = default_pattern_cfg(tf or Timeframe.D1)
        y_of = _inverse(cal)
        result.notes.append(f"Prices estimated from pixel positions (+/- {unc:.4g}); not exact quotes. No volume or dates.")
    else:
        frame = pixel[["open", "high", "low", "close", "volume"]].copy()
        frame.index = idx
        pcfg = PatternConfig(min_height_pct=0.0)
        y_of = (lambda price, H=float(h): H - price)
        result.notes.append("Visual-only tier: levels are in pixel units; no prices, dates or indicator values are inferred.")
    ctx = build_context(frame, pcfg, None, True)
    pats = [p for p in detect_patterns(ctx=ctx, include_experimental=False) if p.end_index >= ctx.last - 60]
    pats.sort(key=lambda p: (-int(p.is_active), -p.quality_score))
    lv = detect_levels(frame, ctx.swings)
    result.patterns = pats[:8]
    result.levels = [{"low": z.low, "high": z.high, "kind": z.kind, "touches": z.touches, "strength": z.strength}
                     for z in lv.zones]
    ann = _annotate(img, ext.x_centers, y_of, result.patterns, 0, result.levels)
    result.annotated_png_b64 = base64.b64encode(cv2.imencode(".png", ann)[1].tobytes()).decode()
    result.pattern_claims = _claims(vision, result.patterns)
    base = 0.45 if result.tier == "estimated" else 0.3
    result.confidence = round(base * ext.extraction_confidence + (0.1 if vision else 0.0), 2)
    result.needs_confirmation = sorted(set(needs))
    if result.needs_confirmation:
        result.notes.append("PRELIMINARY: confirm " + " and ".join(result.needs_confirmation) +
                            " (and upload matching OHLCV) to upgrade to data-verified analysis.")
    return result


def _inverse(cal: AxisCalibration):
    def y_of(price: float) -> float:
        if cal.log_scale:
            t = (np.log(price) - np.log(cal.price1)) / (np.log(cal.price2) - np.log(cal.price1))
        else:
            t = (price - cal.price1) / (cal.price2 - cal.price1)
        return cal.y1 + t * (cal.y2 - cal.y1)
    return y_of


def _parse_tf(s: str) -> Timeframe | None:
    k = s.strip().lower().replace(" ", "")
    table = {"5": "5m", "5m": "5m", "5min": "5m", "15": "15m", "15m": "15m", "15min": "15m", "1h": "1h", "60": "1h",
             "60m": "1h", "h1": "1h", "1d": "1d", "d": "1d", "daily": "1d", "1day": "1d", "1w": "1w", "w": "1w",
             "weekly": "1w", "1m": None, "m": "1M", "monthly": "1M", "1mo": "1M"}
    v = table.get(k)
    return Timeframe(v) if v else None


def analyze_screenshots(items: list[tuple[bytes, str]], **kw) -> dict:
    """Multiple screenshots (e.g. the same stock on several timeframes)."""
    results = [analyze_screenshot(d, filename=f, **kw) for d, f in items]
    syms = {r.symbol for r in results if r.symbol}
    notes = []
    if len(syms) > 1:
        notes.append(f"Screenshots appear to show different symbols ({', '.join(sorted(syms))}); analysed independently.")
    tfs = [r.timeframe for r in results if r.timeframe]
    if len(set(tfs)) > 1:
        notes.append("Multiple timeframes supplied: higher timeframes set bias, lower timeframes refine timing.")
    return {"results": results, "notes": notes, "generated_at": datetime.now(timezone.utc).isoformat()}
