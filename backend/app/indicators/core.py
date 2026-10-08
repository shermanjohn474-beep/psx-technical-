"""Low-level, causal moving-average primitives.

All functions are strictly causal: the value at bar ``t`` uses only bars ``<= t``.
Seeding conventions follow TA-Lib (EMA/Wilder seeded with the SMA of the first
``period`` values) so results can be checked against that reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sma(x: pd.Series, period: int) -> pd.Series:
    return x.rolling(period, min_periods=period).mean()


def _seeded_recursive(values: np.ndarray, period: int, alpha: float) -> np.ndarray:
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) == 0:
        return out
    first = valid[0]
    if len(values) - first < period:
        return out
    seed_end = first + period - 1
    window = values[first:seed_end + 1]
    if np.isnan(window).any():
        # fall back to pandas ewm when there are gaps inside the seed window
        return pd.Series(values).ewm(alpha=alpha, adjust=False, ignore_na=True).mean().to_numpy()
    prev = window.mean()
    out[seed_end] = prev
    for i in range(seed_end + 1, len(values)):
        v = values[i]
        if not np.isnan(v):
            prev = prev + alpha * (v - prev)
        out[i] = prev
    return out


def ema(x: pd.Series, period: int) -> pd.Series:
    return pd.Series(_seeded_recursive(_arr(x), period, 2.0 / (period + 1)), index=x.index)


def wilder(x: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA), alpha = 1/period."""
    return pd.Series(_seeded_recursive(_arr(x), period, 1.0 / period), index=x.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan  # TA-Lib convention: first TR undefined (needs previous close)
    return tr


def rolling_linreg(x: pd.Series, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (value_at_end, slope_per_bar, residual_std) of a rolling OLS fit."""
    y = _arr(x)
    n = len(y)
    val = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    rstd = np.full(n, np.nan)
    t = np.arange(period, dtype=float)
    tm = t.mean()
    denom = ((t - tm) ** 2).sum()
    for i in range(period - 1, n):
        w = y[i - period + 1:i + 1]
        if np.isnan(w).any():
            continue
        b = ((t - tm) * (w - w.mean())).sum() / denom
        a = w.mean() - b * tm
        slope[i] = b
        val[i] = a + b * (period - 1)
        rstd[i] = np.std(w - (a + b * t))
    return pd.Series(val, index=x.index), pd.Series(slope, index=x.index), pd.Series(rstd, index=x.index)
