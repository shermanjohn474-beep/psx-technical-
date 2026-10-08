"""Swing-point detection.

Three methods:

* ``zigzag_atr`` (default, **causal**): a pivot is confirmed on the first bar where
  price has reversed by ``atr_mult x ATR`` from the running extreme. Each swing
  records ``confirmed_index`` so backtests only see pivots knowable at that time.
* ``fractal``: Bill Williams style N-bar fractals; confirmed ``n`` bars later.
* ``peaks``: ``scipy.signal.find_peaks`` with ATR-scaled prominence. Prominence is
  computed with hindsight, so this method is for visualisation / screenshot pixel
  series only and must not be used for historical signal generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from app.indicators.volatility import causal_atr


@dataclass(frozen=True)
class Swing:
    index: int
    price: float
    kind: str  # "H" or "L"
    confirmed_index: int
    provisional: bool = False


@dataclass
class SwingConfig:
    method: str = "zigzag_atr"
    atr_period: int = 14
    atr_mult: float = 2.0
    min_pct: float = 0.0  # optional minimum reversal in % of price
    fractal_n: int = 3
    peaks_prominence_atr: float = 2.0
    peaks_distance: int = 3


@dataclass
class SwingSet:
    swings: list[Swing]
    provisional: Swing | None
    atr: pd.Series
    config: SwingConfig = field(default_factory=SwingConfig)

    def confirmed_upto(self, t: int) -> list[Swing]:
        return [s for s in self.swings if s.confirmed_index <= t]

    def with_provisional(self) -> list[Swing]:
        return self.swings + ([self.provisional] if self.provisional else [])


def zigzag_atr(df: pd.DataFrame, cfg: SwingConfig | None = None, atr: pd.Series | None = None) -> SwingSet:
    cfg = cfg or SwingConfig()
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    atr = causal_atr(df, cfg.atr_period) if atr is None else atr
    a = atr.to_numpy(dtype=float)
    swings: list[Swing] = []
    if n == 0:
        return SwingSet([], None, atr, cfg)

    def thr(i: int, ref: float) -> float:
        return max(cfg.atr_mult * a[i], cfg.min_pct / 100.0 * ref)

    direction = 0  # +1 looking for a high, -1 looking for a low
    hi_i, hi_p = 0, high[0]
    lo_i, lo_p = 0, low[0]
    for i in range(1, n):
        if direction == 0:
            if high[i] > hi_p:
                hi_i, hi_p = i, high[i]
            if low[i] < lo_p:
                lo_i, lo_p = i, low[i]
            if hi_i > lo_i and hi_p - lo_p >= thr(i, lo_p) and i == hi_i:
                swings.append(Swing(lo_i, lo_p, "L", i))
                direction = 1
            elif lo_i > hi_i and hi_p - lo_p >= thr(i, hi_p) and i == lo_i:
                swings.append(Swing(hi_i, hi_p, "H", i))
                direction = -1
            continue
        if direction == 1:
            if high[i] >= hi_p:
                hi_i, hi_p = i, high[i]
            elif hi_p - low[i] >= thr(i, hi_p):
                swings.append(Swing(hi_i, hi_p, "H", i))
                direction = -1
                lo_i, lo_p = i, low[i]
        else:
            if low[i] <= lo_p:
                lo_i, lo_p = i, low[i]
            elif high[i] - lo_p >= thr(i, lo_p):
                swings.append(Swing(lo_i, lo_p, "L", i))
                direction = 1
                hi_i, hi_p = i, high[i]
    provisional = None
    if direction == 1:
        provisional = Swing(hi_i, hi_p, "H", n - 1, provisional=True)
    elif direction == -1:
        provisional = Swing(lo_i, lo_p, "L", n - 1, provisional=True)
    return SwingSet(swings, provisional, atr, cfg)


def fractal_swings(df: pd.DataFrame, cfg: SwingConfig | None = None) -> SwingSet:
    cfg = cfg or SwingConfig(method="fractal")
    k = cfg.fractal_n
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    raw: list[Swing] = []
    for i in range(k, len(df) - k):
        if high[i] == high[i - k:i + k + 1].max() and (high[i] > high[i - k:i]).all():
            raw.append(Swing(i, float(high[i]), "H", i + k))
        if low[i] == low[i - k:i + k + 1].min() and (low[i] < low[i - k:i]).all():
            raw.append(Swing(i, float(low[i]), "L", i + k))
    return SwingSet(_alternate(raw), None, causal_atr(df, cfg.atr_period), cfg)


def peak_swings(series: np.ndarray, scale: float, prominence_mult: float = 2.0, distance: int = 3) -> list[Swing]:
    """Non-causal swings on an arbitrary 1-D series (e.g. pixel-space close path)."""
    series = np.asarray(series, dtype=float)
    prom = max(prominence_mult * scale, 1e-9)
    hi, _ = find_peaks(series, prominence=prom, distance=distance)
    lo, _ = find_peaks(-series, prominence=prom, distance=distance)
    raw = [Swing(int(i), float(series[i]), "H", int(i)) for i in hi] + [Swing(int(i), float(series[i]), "L", int(i)) for i in lo]
    return _alternate(sorted(raw, key=lambda s: s.index))


def _alternate(raw: list[Swing]) -> list[Swing]:
    """Collapse consecutive same-type swings keeping the more extreme one."""
    out: list[Swing] = []
    for s in sorted(raw, key=lambda s: (s.index, s.kind)):
        if out and out[-1].kind == s.kind:
            prev = out[-1]
            better = s.price > prev.price if s.kind == "H" else s.price < prev.price
            if better:
                out[-1] = Swing(s.index, s.price, s.kind, max(s.confirmed_index, prev.confirmed_index))
            continue
        out.append(s)
    return out


def detect_swings(df: pd.DataFrame, cfg: SwingConfig | None = None) -> SwingSet:
    cfg = cfg or SwingConfig()
    if cfg.method == "fractal":
        return fractal_swings(df, cfg)
    if cfg.method == "peaks":
        atr = causal_atr(df, cfg.atr_period)
        sw = peak_swings(df["close"].to_numpy(), float(atr.median()), cfg.peaks_prominence_atr, cfg.peaks_distance)
        return SwingSet(sw, None, atr, cfg)
    return zigzag_atr(df, cfg)
