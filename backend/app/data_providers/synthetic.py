"""Deterministic synthetic OHLCV generator.

Used for unit tests and the offline demo. Every frame produced here is synthetic
and must be labelled as such wherever it is displayed.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from app.data_validation.calendar import PKT, load_holidays


def trading_index(n: int, start: str = "2023-01-02") -> pd.DatetimeIndex:
    holidays = load_holidays()
    days = pd.bdate_range(start, periods=int(n * 1.1) + 30)
    days = [d for d in days if d.date() not in holidays][:n]
    return pd.DatetimeIndex(days, name="timestamp").tz_localize(PKT)


def make_ohlcv(
    waypoints: Sequence[tuple[int, float]],
    *,
    noise: float = 0.25,
    wick: float = 0.35,
    seed: int = 7,
    start: str = "2023-01-02",
    volume: float = 1_000_000,
    volume_spikes: dict[int, float] | None = None,
    volume_profile: Sequence[tuple[int, float]] | None = None,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build OHLCV whose closes follow piecewise-linear ``waypoints`` (bar, price).

    ``volume_spikes`` maps bar index -> multiplier. ``volume_profile`` is an optional
    piecewise-linear multiplier path (bar, multiplier) e.g. for volume dry-ups.
    """
    xs = np.array([w[0] for w in waypoints], dtype=float)
    ys = np.array([w[1] for w in waypoints], dtype=float)
    n = int(xs[-1]) + 1
    rng = np.random.default_rng(seed)
    path = np.interp(np.arange(n), xs, ys)
    close = path + rng.normal(0, noise, n)
    open_ = np.r_[path[0], close[:-1]] + rng.normal(0, noise * 0.3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, wick, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, wick, n))
    low = np.maximum(low, 0.01)
    vol = volume * np.clip(1 + 0.15 * rng.standard_normal(n), 0.4, None)
    if volume_profile:
        vx = np.array([p[0] for p in volume_profile], dtype=float)
        vy = np.array([p[1] for p in volume_profile], dtype=float)
        vol *= np.interp(np.arange(n), vx, vy)
    for i, m in (volume_spikes or {}).items():
        if 0 <= i < n:
            vol[i] *= m
    idx = index if index is not None else trading_index(n, start)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": np.round(vol)}, index=idx[:n]
    )


def random_walk(n: int = 500, *, seed: int = 1, start_price: float = 100.0, vol: float = 0.015,
                drift: float = 0.0, start: str = "2022-01-03") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.r_[start_price, close[:-1]] * np.exp(rng.normal(0, vol * 0.2, n))
    spread = np.abs(rng.normal(0, vol * 0.6, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - np.abs(rng.normal(0, vol * 0.6, n)) * close
    volume = np.round(1_000_000 * np.exp(rng.normal(0, 0.35, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                        index=trading_index(n, start))
