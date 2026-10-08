"""Price/indicator divergence detection on explicit, aligned pivots.

For two consecutive *confirmed* swing lows (or highs) of price within
``[min_bars, max_bars]`` of each other, the indicator value at each pivot is the
indicator's own extreme within ``+/- align_bars`` of the price pivot, but never
using bars after the pivot's ``confirmed_index`` (no look-ahead). Classification:

=================  ==============  ====================
type               price pivots    indicator pivots
=================  ==============  ====================
regular bullish    lower low       higher low
hidden bullish     higher low      lower low
regular bearish    higher high     lower high
hidden bearish     lower high      higher high
=================  ==============  ====================

Minimum price (``min_price_atr``) and indicator (``min_ind_diff``, relative to the
indicator's rolling std) differences filter out noise. The signal time is the
confirmation bar of the second pivot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.swing_detection.swings import SwingSet


class Divergence(BaseModel):
    indicator: str
    type: str  # regular_bullish | hidden_bullish | regular_bearish | hidden_bearish
    direction: str
    pivot1_index: int
    pivot2_index: int
    pivot1_time: str
    pivot2_time: str
    price1: float
    price2: float
    ind1: float
    ind2: float
    signal_index: int
    signal_time: str
    strength: float  # 0..100 heuristic


@dataclass
class DivergenceConfig:
    min_bars: int = 5
    max_bars: int = 80
    align_bars: int = 3
    min_price_atr: float = 0.25
    min_ind_diff: float = 0.15  # in units of the indicator's 100-bar std
    recent_bars: int = 120


def _aligned(ind: np.ndarray, idx: int, confirmed: int, k: int, low: bool) -> tuple[int, float] | None:
    lo, hi = max(0, idx - k), min(len(ind) - 1, idx + k, confirmed)
    w = ind[lo:hi + 1]
    if len(w) == 0 or np.all(np.isnan(w)):
        return None
    j = int(np.nanargmin(w) if low else np.nanargmax(w))
    return lo + j, float(w[j])


def detect_divergences(df: pd.DataFrame, swings: SwingSet, indicators: dict[str, pd.Series],
                       cfg: DivergenceConfig | None = None) -> list[Divergence]:
    cfg = cfg or DivergenceConfig()
    n = len(df)
    last = n - 1
    atr = swings.atr.to_numpy()
    sw = [s for s in swings.swings if s.confirmed_index <= last]
    out: list[Divergence] = []
    ts = df.index
    for name, series in indicators.items():
        ind = series.to_numpy(dtype=float)
        if np.all(np.isnan(ind)):
            continue
        scale = pd.Series(ind).rolling(100, min_periods=20).std().to_numpy()
        for kind in ("L", "H"):
            piv = [s for s in sw if s.kind == kind]
            for p1, p2 in zip(piv, piv[1:]):
                if not (cfg.min_bars <= p2.index - p1.index <= cfg.max_bars):
                    continue
                if p2.confirmed_index < last - cfg.recent_bars:
                    continue
                a1 = _aligned(ind, p1.index, p1.confirmed_index, cfg.align_bars, kind == "L")
                a2 = _aligned(ind, p2.index, p2.confirmed_index, cfg.align_bars, kind == "L")
                if a1 is None or a2 is None:
                    continue
                (j1, v1), (j2, v2) = a1, a2
                sc = scale[p2.index] if np.isfinite(scale[p2.index]) and scale[p2.index] > 0 \
                    else np.nanstd(ind[:p2.confirmed_index + 1])  # causal fallback
                dp = (p2.price - p1.price) / atr[p2.index]
                di = (v2 - v1) / sc
                if abs(dp) < cfg.min_price_atr or abs(di) < cfg.min_ind_diff:
                    continue
                if kind == "L":
                    typ = "regular_bullish" if dp < 0 < di else "hidden_bullish" if di < 0 < dp else None
                else:
                    typ = "regular_bearish" if di < 0 < dp else "hidden_bearish" if dp < 0 < di else None
                if typ is None:
                    continue
                strength = float(np.clip(50 * min(abs(di), 2) / 2 + 50 * min(abs(dp), 3) / 3, 0, 100))
                out.append(Divergence(
                    indicator=name, type=typ, direction="bullish" if "bullish" in typ else "bearish",
                    pivot1_index=p1.index, pivot2_index=p2.index, pivot1_time=ts[p1.index].isoformat(),
                    pivot2_time=ts[p2.index].isoformat(), price1=round(p1.price, 4), price2=round(p2.price, 4),
                    ind1=round(v1, 4), ind2=round(v2, 4), signal_index=p2.confirmed_index,
                    signal_time=ts[p2.confirmed_index].isoformat(), strength=round(strength, 1),
                ))
    return sorted(out, key=lambda d: d.signal_index)
