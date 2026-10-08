"""Momentum indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.indicators.core import ema, sma, wilder


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = wilder(gain, period)
    avg_loss = wilder(loss, period)
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & avg_gain.notna()] = 100.0
    out[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return out.rename("rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    line[slow_ema.isna()] = np.nan
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig}, index=close.index)


def stochastic(df: pd.DataFrame, k_period: int = 14, k_smooth: int = 3, d_period: int = 3) -> pd.DataFrame:
    ll = df["low"].rolling(k_period).min()
    hh = df["high"].rolling(k_period).max()
    fast_k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    slow_k = sma(fast_k, k_smooth)
    slow_d = sma(slow_k, d_period)
    return pd.DataFrame({"stoch_k": slow_k, "stoch_d": slow_d}, index=df.index)


def stoch_rsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    r = rsi(close, rsi_period)
    lo = r.rolling(stoch_period).min()
    hi = r.rolling(stoch_period).max()
    raw = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    kk = sma(raw, k)
    return pd.DataFrame({"stochrsi_k": kk, "stochrsi_d": sma(kk, d)}, index=close.index)


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return (100 * (close / close.shift(period) - 1)).rename("roc")


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    return (close - close.shift(period)).rename("momentum")


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda w: np.mean(np.abs(w - w.mean())), raw=True)
    return ((tp - ma) / (0.015 * md.replace(0, np.nan))).rename("cci")


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return (-100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)).rename("williams_r")
