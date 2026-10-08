"""Volatility indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.indicators.core import ema, true_range, wilder


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return wilder(true_range(df), period).rename("atr")


def causal_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR with an expanding-mean warm-up so early bars still have a usable scale.

    Used for normalisation in swing detection/pattern tolerances (not for display).
    """
    tr = true_range(df)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    a = wilder(tr, period)
    warm = tr.expanding().mean()
    return a.fillna(warm).clip(lower=1e-9)


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper, lower = mid + k * std, mid - k * std
    width = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_pct_b": pct_b},
                        index=close.index)


def bollinger_squeeze(close: pd.Series, period: int = 20, k: float = 2.0, lookback: int = 120,
                      pct: float = 0.10) -> pd.Series:
    """True where band width is in the lowest ``pct`` of the trailing ``lookback`` bars."""
    width = bollinger(close, period, k)["bb_width"]
    thresh = width.rolling(lookback, min_periods=max(period, lookback // 2)).quantile(pct)
    return (width <= thresh).rename("bb_squeeze")


def keltner(df: pd.DataFrame, period: int = 20, atr_period: int = 10, k: float = 2.0) -> pd.DataFrame:
    mid = ema(df["close"], period)
    a = wilder(true_range(df), atr_period)
    return pd.DataFrame({"kc_mid": mid, "kc_upper": mid + k * a, "kc_lower": mid - k * a}, index=df.index)


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    up = df["high"].rolling(period).max()
    lo = df["low"].rolling(period).min()
    return pd.DataFrame({"dc_upper": up, "dc_lower": lo, "dc_mid": (up + lo) / 2}, index=df.index)


def historical_volatility(close: pd.Series, period: int = 20, periods_per_year: int = 245) -> pd.Series:
    """Annualised close-to-close volatility (PSX ~245 sessions/year, configurable)."""
    lr = np.log(close / close.shift(1))
    return (lr.rolling(period).std(ddof=1) * np.sqrt(periods_per_year)).rename("hv")
