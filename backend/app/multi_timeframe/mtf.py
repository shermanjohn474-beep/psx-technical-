"""Multi-timeframe trend assessment and alignment.

Framework: Monthly -> Weekly -> Daily -> Hourly -> 15 minute. Higher timeframes
set the directional bias; lower timeframes refine timing. Only timeframes that
the data source actually provides (or can be aggregated to on real calendar
boundaries) are assessed; missing ones are reported, never fabricated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.indicators.momentum import macd, rsi
from app.indicators.trend import adx_dmi
from app.market_structure.structure import classify_trend, label_swings
from app.schemas.market import Timeframe
from app.swing_detection.swings import SwingConfig, zigzag_atr

ORDER = [Timeframe.MN1, Timeframe.W1, Timeframe.D1, Timeframe.H1, Timeframe.M15]
WEIGHTS = {Timeframe.MN1: 1.5, Timeframe.W1: 2.0, Timeframe.D1: 2.0, Timeframe.H1: 1.0, Timeframe.M15: 0.5, Timeframe.M5: 0.25}
LABEL = {Timeframe.MN1: "Monthly", Timeframe.W1: "Weekly", Timeframe.D1: "Daily", Timeframe.H1: "Hourly",
         Timeframe.M15: "15 Minute", Timeframe.M5: "5 Minute"}


class TimeframeTrend(BaseModel):
    timeframe: str
    label: str
    available: bool = True
    bars: int = 0
    trend: str = "unavailable"  # bullish | bearish | neutral | unavailable
    score: float = 0.0  # -100..100
    strength: str = "n/a"  # strong | moderate | weak
    structure: str = "undetermined"
    ma_structure: str | None = None
    momentum: str | None = None
    components: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    last_bar_complete: bool | None = None


class MTFResult(BaseModel):
    timeframes: list[TimeframeTrend]
    alignment: str  # strong_bullish | moderate_bullish | mixed | moderate_bearish | strong_bearish | insufficient
    alignment_score: float
    commentary: list[str] = Field(default_factory=list)


def assess_timeframe(df: pd.DataFrame, tf: Timeframe, last_complete: bool | None = None) -> TimeframeTrend:
    n = len(df)
    out = TimeframeTrend(timeframe=tf.value, label=LABEL[tf], bars=n, last_bar_complete=last_complete)
    if n < 20:
        out.available, out.trend = n > 0, "unavailable"
        out.notes.append(f"Only {n} bars: insufficient for trend assessment.")
        return out
    c = df["close"]
    fast_p, slow_p = (50, 200) if n >= 220 else (20, 50) if n >= 60 else (10, 20)
    if (fast_p, slow_p) != (50, 200):
        out.notes.append(f"Using SMA{fast_p}/SMA{slow_p} (insufficient bars for SMA50/200).")
    fast, slow = c.rolling(fast_p).mean(), c.rolling(slow_p).mean()
    last = float(c.iloc[-1])
    comps: dict[str, float] = {}
    if not np.isnan(slow.iloc[-1]):
        comps["price_vs_slow_ma"] = 1.0 if last > slow.iloc[-1] else -1.0
        comps["ma_stack"] = 1.0 if fast.iloc[-1] > slow.iloc[-1] else -1.0
        slope = (slow.iloc[-1] - slow.iloc[-6]) / slow.iloc[-6] if len(slow.dropna()) > 6 else 0.0
        comps["slow_ma_slope"] = float(np.clip(slope * 100, -1, 1))
        out.ma_structure = (f"price {'above' if last > slow.iloc[-1] else 'below'} SMA{slow_p}; "
                            f"SMA{fast_p} {'above' if fast.iloc[-1] > slow.iloc[-1] else 'below'} SMA{slow_p}")
    sw = zigzag_atr(df, SwingConfig(atr_mult=2.0))
    labels = label_swings([s for s in sw.swings])
    struct = classify_trend(labels)
    out.structure = struct
    comps["structure"] = {"uptrend": 1.0, "downtrend": -1.0, "range": 0.0}.get(struct, 0.0)
    m = macd(c)
    if m["macd"].notna().any():
        comps["macd"] = 0.5 * (1 if m["macd"].iloc[-1] > m["macd_signal"].iloc[-1] else -1) + 0.5 * (1 if m["macd"].iloc[-1] > 0 else -1)
    r = rsi(c)
    if r.notna().any():
        comps["rsi"] = float(np.clip((r.iloc[-1] - 50) / 20, -1, 1))
        out.momentum = f"RSI {r.iloc[-1]:.1f}, MACD {'above' if comps.get('macd', 0) > 0 else 'below'} signal"
    a = adx_dmi(df)
    adx_v = a["adx"].iloc[-1]
    w = {"price_vs_slow_ma": 2, "ma_stack": 2, "slow_ma_slope": 1.5, "structure": 2, "macd": 1, "rsi": 1}
    raw = sum(comps[k] * w[k] for k in comps) / sum(w[k] for k in comps)
    if not np.isnan(adx_v):
        comps["adx"] = float(adx_v)
        raw *= 0.7 + 0.3 * min(1.0, adx_v / 25)  # weak ADX dampens conviction
    out.components = {k: round(v, 3) for k, v in comps.items()}
    out.score = round(100 * raw, 1)
    out.trend = "bullish" if out.score >= 30 else "bearish" if out.score <= -30 else "neutral"
    out.strength = "strong" if abs(out.score) >= 65 else "moderate" if abs(out.score) >= 30 else "weak"
    if last_complete is False:
        out.notes.append("Latest bar of this timeframe is still forming.")
    return out


def combine(trends: list[TimeframeTrend]) -> MTFResult:
    avail = [t for t in trends if t.trend != "unavailable"]
    if not avail:
        return MTFResult(timeframes=trends, alignment="insufficient", alignment_score=0.0,
                         commentary=["No timeframe had sufficient data."])
    tf = {Timeframe(t.timeframe): t for t in avail}
    tot = sum(WEIGHTS[k] for k in tf)
    score = sum(WEIGHTS[k] * tf[k].score for k in tf) / tot
    htf = [tf[k] for k in (Timeframe.MN1, Timeframe.W1, Timeframe.D1) if k in tf]
    htf_dirs = {t.trend for t in htf}
    if score >= 55 and htf_dirs == {"bullish"}:
        align = "strong_bullish"
    elif score >= 20 and "bearish" not in htf_dirs:
        align = "moderate_bullish"
    elif score <= -55 and htf_dirs == {"bearish"}:
        align = "strong_bearish"
    elif score <= -20 and "bullish" not in htf_dirs:
        align = "moderate_bearish"
    else:
        align = "mixed"
    notes = []
    missing = [LABEL[k] for k in ORDER if k not in tf]
    if missing:
        notes.append("Not assessed (no source data): " + ", ".join(missing) + ".")
    w, d, h = tf.get(Timeframe.W1), tf.get(Timeframe.D1), tf.get(Timeframe.H1) or tf.get(Timeframe.M15)
    if w and d and h:
        if w.trend == d.trend == "bearish" and h.trend == "bullish":
            notes.append("Weekly and daily are bearish while the lower timeframe turns bullish: treat as a countertrend "
                         "rally unless the daily structure breaks (e.g. a higher high above the last daily swing high).")
        if w.trend == d.trend == "bullish" and h.trend == "bearish":
            notes.append("Weekly and daily are bullish while the lower timeframe is bearish: likely a pullback within the "
                         "uptrend; a broader reversal needs daily confirmation (lower low below the last daily swing low).")
    if w and d and w.trend != d.trend and "neutral" not in (w.trend, d.trend):
        notes.append(f"Weekly ({w.trend}) and daily ({d.trend}) disagree: higher-timeframe bias is unresolved.")
    notes.append("Lower-timeframe signals do not override higher-timeframe evidence without confirmation.")
    return MTFResult(timeframes=trends, alignment=align, alignment_score=round(score, 1), commentary=notes)


def analyze_mtf(frames: dict[Timeframe, pd.DataFrame], completeness: dict[Timeframe, bool] | None = None) -> MTFResult:
    completeness = completeness or {}
    trends = []
    for tf in ORDER:
        df = frames.get(tf)
        if df is None or df.empty:
            trends.append(TimeframeTrend(timeframe=tf.value, label=LABEL[tf], available=False,
                                         notes=["No data available for this timeframe."]))
        else:
            trends.append(assess_timeframe(df, tf, completeness.get(tf)))
    return combine(trends)
