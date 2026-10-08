"""Timeframe aggregation using real calendar boundaries (Asia/Karachi).

Only *upsampling in period length* is allowed (e.g. daily -> weekly/monthly,
5m -> 15m -> 1h). Intraday candles are never manufactured from daily data.
"""
from __future__ import annotations

import pandas as pd

from app.schemas.market import Timeframe

_RULES = {
    Timeframe.M15: "15min",
    Timeframe.H1: "60min",
    Timeframe.W1: "W-FRI",  # PSX week ends Friday
    Timeframe.MN1: "ME",
}


class ResampleNotAllowed(ValueError):
    pass


def can_resample(src: Timeframe, dst: Timeframe) -> bool:
    if dst == src:
        return True
    if dst.rank < src.rank:
        return False
    if src.is_intraday:
        if dst.is_intraday:
            return (dst.minutes or 0) % (src.minutes or 1) == 0
        return True
    if src == Timeframe.D1:
        return dst in (Timeframe.W1, Timeframe.MN1)
    # Weekly -> monthly is not exact (weeks straddle month boundaries).
    return False


def resample_ohlcv(df: pd.DataFrame, src: Timeframe, dst: Timeframe) -> pd.DataFrame:
    """Aggregate OHLCV. Bars are labelled with the period's *first traded* timestamp."""
    if not can_resample(src, dst):
        raise ResampleNotAllowed(f"Cannot derive {dst.value} candles from {src.value} data.")
    if dst == src or df.empty:
        return df.copy()
    if dst == Timeframe.D1:
        key = df.index.normalize()
        g = df.groupby(key)
    else:
        rule = _RULES[dst]
        if dst.is_intraday:
            g = df.groupby(pd.Grouper(freq=rule, origin="start_day", label="left"))
        else:
            g = df.groupby(pd.Grouper(freq=rule, label="right", closed="right"))
    agg = g.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", lambda v: v.sum(min_count=1)),
        first_ts=("close", lambda s: s.index[0] if len(s) else pd.NaT),
        n=("close", "count"),
    )
    agg = agg[agg["n"] > 0]
    agg.index = pd.DatetimeIndex(agg.pop("first_ts"))
    agg.index.name = "timestamp"
    agg = agg.drop(columns=["n"])
    if dst in (Timeframe.W1, Timeframe.MN1):
        agg.index = agg.index.normalize()
    return agg
