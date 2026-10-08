"""Fibonacci retracements and extensions anchored on the dominant recent swing leg."""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from app.swing_detection.swings import SwingSet

RETRACEMENTS = (0.236, 0.382, 0.5, 0.618, 0.786)
EXTENSIONS = (1.272, 1.618, 2.0, 2.618)


class FibLevel(BaseModel):
    ratio: float
    price: float
    kind: str  # retracement | extension


class FibAnalysis(BaseModel):
    anchor_start_index: int
    anchor_end_index: int
    anchor_start_time: str
    anchor_end_time: str
    anchor_start_price: float
    anchor_end_price: float
    direction: str  # up-leg | down-leg
    levels: list[FibLevel] = Field(default_factory=list)
    current_retracement: float | None = None
    nearest_level: FibLevel | None = None


def fibonacci(df: pd.DataFrame, swings: SwingSet, lookback_swings: int = 8) -> FibAnalysis | None:
    last = len(df) - 1
    sw = [s for s in swings.swings if s.confirmed_index <= last][-lookback_swings:]
    if swings.provisional is not None:
        sw = sw + [swings.provisional]
    if len(sw) < 2:
        return None
    # dominant leg = largest absolute move between consecutive swings in the window
    a, b = max(zip(sw, sw[1:]), key=lambda p: abs(p[1].price - p[0].price))
    up = b.price > a.price
    span = b.price - a.price
    levels = [FibLevel(ratio=r, price=round(b.price - r * span, 4), kind="retracement") for r in RETRACEMENTS]
    levels += [FibLevel(ratio=r, price=round(a.price + r * span, 4), kind="extension") for r in EXTENSIONS]
    close = float(df["close"].iloc[-1])
    cur = (b.price - close) / span if span else None
    nearest = min(levels, key=lambda lv: abs(lv.price - close))
    ts = df.index
    return FibAnalysis(
        anchor_start_index=a.index, anchor_end_index=b.index, anchor_start_time=ts[a.index].isoformat(),
        anchor_end_time=ts[b.index].isoformat(), anchor_start_price=round(a.price, 4), anchor_end_price=round(b.price, 4),
        direction="up-leg" if up else "down-leg", levels=levels,
        current_retracement=None if cur is None else round(cur, 3), nearest_level=nearest,
    )
