"""Japanese candlestick recognition with contextual filtering.

Shape rules are expressed with body/shadow/range ratios and the average body of the
previous ``avg_period`` candles. A shape alone is not a signal: every match is
scored on *context* — preceding trend, location relative to support/resistance,
candle size vs ATR, and relative volume — and checked for next-bar confirmation.
A bullish engulfing inside an established uptrend therefore scores low and is
labelled as lacking reversal context.

TA-Lib (if installed) is used as an independent cross-check, reported in evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.indicators.volatility import causal_atr
from app.indicators.volume import has_volume, relative_volume

try:  # optional reference implementation
    import talib  # type: ignore
except Exception:  # pragma: no cover
    talib = None


class CandleSignal(BaseModel):
    pattern_id: str
    name: str
    index: int
    timestamp: str
    direction: str  # bullish | bearish | neutral
    kind: str  # reversal | continuation | indecision
    candles: int
    score: float  # 0..100 context-adjusted quality (not a probability)
    components: dict[str, float] = Field(default_factory=dict)
    confirmation: str  # confirmed | not_confirmed | pending | n/a
    evidence: list[str] = Field(default_factory=list)
    talib_agrees: bool | None = None


@dataclass
class CandleConfig:
    avg_period: int = 10
    trend_lookback: int = 10
    trend_atr: float = 2.0
    doji_body_frac: float = 0.1
    near_level_atr: float = 0.6
    scan_bars: int = 60


NAMES = {
    "doji": "Doji", "dragonfly_doji": "Dragonfly Doji", "gravestone_doji": "Gravestone Doji",
    "long_legged_doji": "Long-legged Doji", "hammer": "Hammer", "inverted_hammer": "Inverted Hammer",
    "hanging_man": "Hanging Man", "shooting_star": "Shooting Star", "spinning_top": "Spinning Top",
    "bullish_marubozu": "Bullish Marubozu", "bearish_marubozu": "Bearish Marubozu",
    "bullish_engulfing": "Bullish Engulfing", "bearish_engulfing": "Bearish Engulfing",
    "bullish_harami": "Bullish Harami", "bearish_harami": "Bearish Harami", "piercing": "Piercing Pattern",
    "dark_cloud_cover": "Dark Cloud Cover", "tweezer_top": "Tweezer Top", "tweezer_bottom": "Tweezer Bottom",
    "matching_low": "Matching Low", "bullish_kicking": "Bullish Kicking", "bearish_kicking": "Bearish Kicking",
    "morning_star": "Morning Star", "evening_star": "Evening Star", "morning_doji_star": "Morning Doji Star",
    "evening_doji_star": "Evening Doji Star", "three_white_soldiers": "Three White Soldiers",
    "three_black_crows": "Three Black Crows", "three_inside_up": "Three Inside Up",
    "three_inside_down": "Three Inside Down", "three_outside_up": "Three Outside Up",
    "three_outside_down": "Three Outside Down", "rising_three_methods": "Rising Three Methods",
    "falling_three_methods": "Falling Three Methods",
}
# pattern -> (direction, kind, required prior trend: -1 down, +1 up, 0 none)
META = {
    "doji": ("neutral", "indecision", 0), "dragonfly_doji": ("bullish", "reversal", -1),
    "gravestone_doji": ("bearish", "reversal", 1), "long_legged_doji": ("neutral", "indecision", 0),
    "hammer": ("bullish", "reversal", -1), "inverted_hammer": ("bullish", "reversal", -1),
    "hanging_man": ("bearish", "reversal", 1), "shooting_star": ("bearish", "reversal", 1),
    "spinning_top": ("neutral", "indecision", 0), "bullish_marubozu": ("bullish", "continuation", 0),
    "bearish_marubozu": ("bearish", "continuation", 0), "bullish_engulfing": ("bullish", "reversal", -1),
    "bearish_engulfing": ("bearish", "reversal", 1), "bullish_harami": ("bullish", "reversal", -1),
    "bearish_harami": ("bearish", "reversal", 1), "piercing": ("bullish", "reversal", -1),
    "dark_cloud_cover": ("bearish", "reversal", 1), "tweezer_top": ("bearish", "reversal", 1),
    "tweezer_bottom": ("bullish", "reversal", -1), "matching_low": ("bullish", "reversal", -1),
    "bullish_kicking": ("bullish", "reversal", 0), "bearish_kicking": ("bearish", "reversal", 0),
    "morning_star": ("bullish", "reversal", -1), "evening_star": ("bearish", "reversal", 1),
    "morning_doji_star": ("bullish", "reversal", -1), "evening_doji_star": ("bearish", "reversal", 1),
    "three_white_soldiers": ("bullish", "reversal", -1), "three_black_crows": ("bearish", "reversal", 1),
    "three_inside_up": ("bullish", "reversal", -1), "three_inside_down": ("bearish", "reversal", 1),
    "three_outside_up": ("bullish", "reversal", -1), "three_outside_down": ("bearish", "reversal", 1),
    "rising_three_methods": ("bullish", "continuation", 1), "falling_three_methods": ("bearish", "continuation", -1),
}
TALIB_MAP = {
    "doji": "CDLDOJI", "dragonfly_doji": "CDLDRAGONFLYDOJI", "gravestone_doji": "CDLGRAVESTONEDOJI",
    "long_legged_doji": "CDLLONGLEGGEDDOJI", "hammer": "CDLHAMMER", "inverted_hammer": "CDLINVERTEDHAMMER",
    "hanging_man": "CDLHANGINGMAN", "shooting_star": "CDLSHOOTINGSTAR", "spinning_top": "CDLSPINNINGTOP",
    "bullish_marubozu": "CDLMARUBOZU", "bearish_marubozu": "CDLMARUBOZU", "bullish_engulfing": "CDLENGULFING",
    "bearish_engulfing": "CDLENGULFING", "bullish_harami": "CDLHARAMI", "bearish_harami": "CDLHARAMI",
    "piercing": "CDLPIERCING", "dark_cloud_cover": "CDLDARKCLOUDCOVER", "matching_low": "CDLMATCHINGLOW",
    "bullish_kicking": "CDLKICKING", "bearish_kicking": "CDLKICKING", "morning_star": "CDLMORNINGSTAR",
    "evening_star": "CDLEVENINGSTAR", "morning_doji_star": "CDLMORNINGDOJISTAR", "evening_doji_star": "CDLEVENINGDOJISTAR",
    "three_white_soldiers": "CDL3WHITESOLDIERS", "three_black_crows": "CDL3BLACKCROWS",
    "three_inside_up": "CDL3INSIDE", "three_inside_down": "CDL3INSIDE", "three_outside_up": "CDL3OUTSIDE",
    "three_outside_down": "CDL3OUTSIDE", "rising_three_methods": "CDLRISEFALL3METHODS",
    "falling_three_methods": "CDLRISEFALL3METHODS",
}


def _shapes(o, h, l, c, i, avg_body, cfg: CandleConfig) -> dict[str, int]:
    """Return {pattern_id: n_candles} for shapes completing at bar i (context-free)."""
    out: dict[str, int] = {}
    rng = h - l
    body = np.abs(c - o)
    up = h - np.maximum(o, c)
    lo = np.minimum(o, c) - l
    bull = c > o
    bear = c < o
    ab = avg_body[i] if np.isfinite(avg_body[i]) and avg_body[i] > 0 else body[max(0, i - 10):i + 1].mean() or 1e-9

    def doji(k):
        return rng[k] > 0 and body[k] <= cfg.doji_body_frac * rng[k]

    def long_body(k):
        return body[k] >= ab and rng[k] > 0 and body[k] >= 0.5 * rng[k]

    def small(k):
        return body[k] <= 0.5 * ab

    r = rng[i]
    if r <= 0:
        return out
    if doji(i):
        if up[i] <= 0.1 * r and lo[i] >= 0.6 * r:
            out["dragonfly_doji"] = 1
        elif lo[i] <= 0.1 * r and up[i] >= 0.6 * r:
            out["gravestone_doji"] = 1
        elif up[i] >= 0.3 * r and lo[i] >= 0.3 * r and r >= 1.5 * ab:
            out["long_legged_doji"] = 1
        else:
            out["doji"] = 1
    else:
        if lo[i] >= 2 * body[i] and up[i] <= 0.1 * r:
            out["hammer_shape"] = 1
        if up[i] >= 2 * body[i] and lo[i] <= 0.1 * r:
            out["inverted_hammer_shape"] = 1
        if body[i] <= 0.3 * r and up[i] >= body[i] and lo[i] >= body[i]:
            out["spinning_top"] = 1
        if body[i] >= 0.9 * r and body[i] >= 1.2 * ab:
            out["bullish_marubozu" if bull[i] else "bearish_marubozu"] = 1
    if i >= 1:
        p = i - 1
        if bear[p] and bull[i] and o[i] <= c[p] and c[i] >= o[p] and body[i] > body[p]:
            out["bullish_engulfing"] = 2
        if bull[p] and bear[i] and o[i] >= c[p] and c[i] <= o[p] and body[i] > body[p]:
            out["bearish_engulfing"] = 2
        if long_body(p) and bear[p] and max(o[i], c[i]) <= o[p] and min(o[i], c[i]) >= c[p] and body[i] < 0.6 * body[p]:
            out["bullish_harami"] = 2
        if long_body(p) and bull[p] and max(o[i], c[i]) <= c[p] and min(o[i], c[i]) >= o[p] and body[i] < 0.6 * body[p]:
            out["bearish_harami"] = 2
        mid_p = (o[p] + c[p]) / 2
        if long_body(p) and bear[p] and bull[i] and o[i] < c[p] and mid_p < c[i] < o[p]:
            out["piercing"] = 2
        if long_body(p) and bull[p] and bear[i] and o[i] > c[p] and o[p] < c[i] < mid_p:
            out["dark_cloud_cover"] = 2
        tol = 0.05 * ab + 0.002 * c[i]
        if abs(l[i] - l[p]) <= tol and bear[p] and bull[i]:
            out["tweezer_bottom"] = 2
        if abs(h[i] - h[p]) <= tol and bull[p] and bear[i]:
            out["tweezer_top"] = 2
        if bear[p] and bear[i] and abs(c[i] - c[p]) <= tol and long_body(p):
            out["matching_low"] = 2
        maru = lambda k: body[k] >= 0.9 * rng[k] and body[k] >= ab  # noqa: E731
        if maru(p) and maru(i) and bear[p] and bull[i] and o[i] > o[p]:
            out["bullish_kicking"] = 2
        if maru(p) and maru(i) and bull[p] and bear[i] and o[i] < o[p]:
            out["bearish_kicking"] = 2
    if i >= 2:
        a, b = i - 2, i - 1
        mid_a = (o[a] + c[a]) / 2
        if long_body(a) and bear[a] and small(b) and max(o[b], c[b]) <= c[a] + 0.1 * ab and bull[i] and c[i] > mid_a:
            out["morning_doji_star" if doji(b) else "morning_star"] = 3
        if long_body(a) and bull[a] and small(b) and min(o[b], c[b]) >= c[a] - 0.1 * ab and bear[i] and c[i] < mid_a:
            out["evening_doji_star" if doji(b) else "evening_star"] = 3
        trio = (a, b, i)
        if all(bull[k] and body[k] >= 0.8 * ab and up[k] <= 0.35 * body[k] for k in trio) and c[a] < c[b] < c[i] \
                and o[a] < o[b] <= c[a] and o[b] < o[i] <= c[b]:
            out["three_white_soldiers"] = 3
        if all(bear[k] and body[k] >= 0.8 * ab and lo[k] <= 0.35 * body[k] for k in trio) and c[a] > c[b] > c[i] \
                and o[a] > o[b] >= c[a] and o[b] > o[i] >= c[b]:
            out["three_black_crows"] = 3
        if long_body(a) and bear[a] and max(o[b], c[b]) <= o[a] and min(o[b], c[b]) >= c[a] and body[b] < 0.6 * body[a] \
                and bull[b] and c[i] > o[a]:
            out["three_inside_up"] = 3
        if long_body(a) and bull[a] and max(o[b], c[b]) <= c[a] and min(o[b], c[b]) >= o[a] and body[b] < 0.6 * body[a] \
                and bear[b] and c[i] < o[a]:
            out["three_inside_down"] = 3
        if bear[a] and bull[b] and o[b] <= c[a] and c[b] >= o[a] and body[b] > body[a] and c[i] > c[b]:
            out["three_outside_up"] = 3
        if bull[a] and bear[b] and o[b] >= c[a] and c[b] <= o[a] and body[b] > body[a] and c[i] < c[b]:
            out["three_outside_down"] = 3
    if i >= 4:
        f = i - 4
        mids = range(f + 1, i)
        if long_body(f) and bull[f] and long_body(i) and bull[i] and c[i] > c[f] and \
                all(small(k) and h[k] <= h[f] and l[k] >= l[f] for k in mids) and c[i - 1] < c[f + 1] + 0.5 * ab:
            out["rising_three_methods"] = 5
        if long_body(f) and bear[f] and long_body(i) and bear[i] and c[i] < c[f] and \
                all(small(k) and h[k] <= h[f] and l[k] >= l[f] for k in mids) and c[i - 1] > c[f + 1] - 0.5 * ab:
            out["falling_three_methods"] = 5
    return out


def detect_candles(df: pd.DataFrame, zones: list | None = None, cfg: CandleConfig | None = None,
                   last_candle_complete: bool = True, all_bars: bool = False) -> list[CandleSignal]:
    cfg = cfg or CandleConfig()
    n = len(df)
    if n < cfg.avg_period + 5:
        return []
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr = causal_atr(df).to_numpy()
    body = np.abs(c - o)
    avg_body = pd.Series(body).shift(1).rolling(cfg.avg_period, min_periods=3).mean().to_numpy()
    rel = relative_volume(df).to_numpy() if has_volume(df) else None
    talib_hits = _talib_hits(df) if talib is not None else {}
    zones = zones or []
    out: list[CandleSignal] = []
    start = cfg.avg_period if all_bars else max(cfg.avg_period, n - cfg.scan_bars)
    last = n - 1
    for i in range(start, n):
        if i == last and not last_candle_complete:
            continue  # never classify an incomplete live candle
        shapes = _shapes(o, h, l, c, i, avg_body, cfg)
        if not shapes:
            continue
        k0 = i - max(shapes.values()) + 1
        prior = (c[max(0, k0 - 1)] - c[max(0, k0 - 1 - cfg.trend_lookback)]) / max(atr[k0 - 1] if k0 > 0 else atr[i], 1e-9)
        trend = 1 if prior >= cfg.trend_atr else -1 if prior <= -cfg.trend_atr else 0
        resolved = {}
        for sid, nc in shapes.items():
            if sid == "hammer_shape":
                if trend <= 0:
                    resolved["hammer"] = nc
                if trend >= 0:
                    resolved["hanging_man"] = nc
            elif sid == "inverted_hammer_shape":
                if trend <= 0:
                    resolved["inverted_hammer"] = nc
                if trend >= 0:
                    resolved["shooting_star"] = nc
            else:
                resolved[sid] = nc
        for pid, nc in resolved.items():
            direction, kind, need = META[pid]
            comps: dict[str, float] = {}
            ev: list[str] = []
            if need == 0:
                comps["trend_context"] = 0.6
            elif need == trend:
                comps["trend_context"] = min(1.0, abs(prior) / (2 * cfg.trend_atr))
                ev.append(f"Preceding {'down' if need < 0 else 'up'}trend ({prior:+.1f} ATR over {cfg.trend_lookback} bars)")
            else:
                comps["trend_context"] = 0.1
                ev.append(f"No preceding {'down' if need < 0 else 'up'}trend ({prior:+.1f} ATR): low "
                          f"{'reversal' if kind == 'reversal' else 'continuation'} significance")
            rng_i = h[k0:i + 1].max() - l[k0:i + 1].min()
            comps["size"] = float(np.clip(rng_i / (1.2 * atr[i]), 0, 1))
            if zones and direction != "neutral":
                near = _near_zone(zones, l[k0:i + 1].min() if direction == "bullish" else h[k0:i + 1].max(),
                                  direction, cfg.near_level_atr * atr[i])
                comps["location"] = 1.0 if near else 0.3
                if near:
                    ev.append(f"Formed at {'support' if direction == 'bullish' else 'resistance'} zone {near}")
            if rel is not None and np.isfinite(rel[i]):
                comps["volume"] = float(np.clip(rel[i] / 1.5, 0, 1))
                ev.append(f"Relative volume {rel[i]:.2f}x")
            conf = "n/a" if direction == "neutral" else "pending"
            if direction != "neutral" and i < last:
                nxt = c[i + 1]
                ok = nxt > c[i] if direction == "bullish" else nxt < c[i]
                conf = "confirmed" if ok else "not_confirmed"
                comps["confirmation"] = 1.0 if ok else 0.0
            w = {"trend_context": 3, "location": 2, "volume": 1, "size": 1, "confirmation": 2}
            score = 100 * sum(comps[k] * w[k] for k in comps) / sum(w[k] for k in comps)
            agree = None
            if pid in TALIB_MAP and TALIB_MAP[pid] in talib_hits:
                v = talib_hits[TALIB_MAP[pid]][i]
                agree = bool(v != 0 and (direction == "neutral" or (v > 0) == (direction == "bullish")))
            out.append(CandleSignal(
                pattern_id=pid, name=NAMES[pid], index=i, timestamp=df.index[i].isoformat(),
                direction=direction, kind=kind, candles=nc, score=round(float(score), 1),
                components={k: round(v, 3) for k, v in comps.items()}, confirmation=conf, evidence=ev, talib_agrees=agree,
            ))
    return out


def _near_zone(zones, price: float, direction: str, tol: float) -> str | None:
    for z in zones:
        lo, hi = (z.low, z.high) if hasattr(z, "low") else (z["low"], z["high"])
        if lo - tol <= price <= hi + tol:
            return f"{lo:.4g}-{hi:.4g}"
    return None


def _talib_hits(df: pd.DataFrame) -> dict[str, np.ndarray]:
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    out = {}
    for fn in set(TALIB_MAP.values()):
        try:
            out[fn] = getattr(talib, fn)(o, h, l, c)
        except Exception:  # pragma: no cover
            continue
    return out
