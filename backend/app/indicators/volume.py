"""Volume indicators. Every function returns NaN when volume is unavailable."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.indicators.core import sma


def has_volume(df: pd.DataFrame) -> bool:
    v = df.get("volume")
    return v is not None and v.notna().any() and float(v.fillna(0).sum()) > 0


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return sma(df["volume"], period).rename("vol_sma")


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume / average of the *previous* ``period`` bars (excludes the current bar)."""
    avg = df["volume"].shift(1).rolling(period, min_periods=period).mean()
    return (df["volume"] / avg.replace(0, np.nan)).rename("rel_volume")


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff()).fillna(0)
    # TA-Lib convention: OBV starts at the first bar's volume
    return (sign * df["volume"]).fillna(0).cumsum().add(df["volume"].iloc[0]).rename("obv")


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    return (clv.fillna(0) * df["volume"]).cumsum().rename("ad")


def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng).fillna(0)
    return ((mfm * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)).rename("cmf")


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    d = tp.diff()
    pos = rmf.where(d > 0, 0.0)
    neg = rmf.where(d < 0, 0.0)
    pos.iloc[0] = np.nan
    neg.iloc[0] = np.nan
    ps = pos.rolling(period).sum()
    ns = neg.rolling(period).sum()
    out = 100 * ps / (ps + ns).replace(0, np.nan)
    return out.rename("mfi")


def volume_dry_up(df: pd.DataFrame, short: int = 5, long: int = 50, ratio: float = 0.6) -> pd.Series:
    """True when short-term average volume is below ``ratio`` x the long-term average."""
    return (sma(df["volume"], short) < ratio * sma(df["volume"], long)).rename("vol_dry_up")


def volume_price_divergence(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """+1: price falling while OBV rising (accumulation); -1: price rising while OBV falling."""
    price_slope = df["close"].diff(period)
    obv_slope = obv(df).diff(period)
    out = pd.Series(0, index=df.index, dtype=int)
    out[(price_slope < 0) & (obv_slope > 0)] = 1
    out[(price_slope > 0) & (obv_slope < 0)] = -1
    return out.rename("vol_price_div")
