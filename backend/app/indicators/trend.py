"""Trend indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.indicators.core import ema, rolling_linreg, sma, true_range, wilder


def moving_averages(close: pd.Series, sma_periods=(10, 20, 50, 100, 200), ema_periods=(9, 20, 50, 100, 200)) -> pd.DataFrame:
    out = {f"sma_{p}": sma(close, p) for p in sma_periods}
    out.update({f"ema_{p}": ema(close, p) for p in ema_periods})
    return pd.DataFrame(out, index=close.index)


def crossovers(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 on the bar fast closes above slow, -1 on cross below, else 0 (causal)."""
    diff = fast - slow
    prev = diff.shift(1)
    out = pd.Series(0, index=fast.index, dtype=int)
    out[(diff > 0) & (prev <= 0)] = 1
    out[(diff < 0) & (prev >= 0)] = -1
    return out


def adx_dmi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low = df["high"], df["low"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    tr = true_range(df)
    # TA-Lib style: Wilder *sums* seeded with the sum of the first `period` values.
    atr_s = wilder(tr, period)
    pdm_s = wilder(plus_dm, period)
    mdm_s = wilder(minus_dm, period)
    plus_di = 100 * pdm_s / atr_s
    minus_di = 100 * mdm_s / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = wilder(dx, period)
    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di}, index=df.index)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    atr = wilder(true_range(df), period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    close = df["close"].to_numpy()
    n = len(df)
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    st = np.full(n, np.nan)
    direction = np.zeros(n)
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        if i == 0 or np.isnan(fu[i - 1]):
            fu[i], fl[i] = upper[i], lower[i]
            direction[i] = 1
            st[i] = fl[i]
            continue
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < fl[i] else 1
        else:
            direction[i] = 1 if close[i] > fu[i] else -1
        st[i] = fl[i] if direction[i] == 1 else fu[i]
    return pd.DataFrame({"supertrend": st, "supertrend_dir": direction}, index=df.index)


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52, displacement: int = 26) -> pd.DataFrame:
    """Ichimoku. Senkou spans are reported *unshifted* (value computed at t, plotted
    ``displacement`` bars ahead) to avoid any confusion with look-ahead; chikou is
    omitted from signal logic because it references the future when plotted."""
    h, l = df["high"], df["low"]
    t = (h.rolling(tenkan).max() + l.rolling(tenkan).min()) / 2
    k = (h.rolling(kijun).max() + l.rolling(kijun).min()) / 2
    a = (t + k) / 2
    b = (h.rolling(senkou_b).max() + l.rolling(senkou_b).min()) / 2
    # cloud currently in force at t was computed `displacement` bars ago
    return pd.DataFrame({
        "ichimoku_tenkan": t, "ichimoku_kijun": k,
        "ichimoku_span_a_raw": a, "ichimoku_span_b_raw": b,
        "ichimoku_cloud_a": a.shift(displacement), "ichimoku_cloud_b": b.shift(displacement),
    }, index=df.index)


def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=df.index)
    # TA-Lib initial direction: compare first two bars' minus-DM vs plus-DM
    up_move, down_move = high[1] - high[0], low[0] - low[1]
    long = not (down_move > 0 and down_move > up_move)
    af = step
    if long:
        ep, s = high[1], low[0]
    else:
        ep, s = low[1], high[0]
    for i in range(1, n):
        if i > 1:
            s = s + af * (ep - s)
        if long:
            s = min(s, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < s:
                long, s, ep, af = False, ep, low[i], step
                s = max(s, high[i], high[i - 1])
            elif high[i] > ep:
                ep, af = high[i], min(af + step, max_step)
        else:
            s = max(s, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > s:
                long, s, ep, af = True, ep, high[i], step
                s = min(s, low[i], low[i - 1])
            elif low[i] < ep:
                ep, af = low[i], min(af + step, max_step)
        sar[i] = s
    return pd.Series(sar, index=df.index, name="psar")


def vwap_intraday(df: pd.DataFrame) -> pd.Series:
    """Session VWAP; resets each Karachi trading day. Only valid for intraday data."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"]
    day = df.index.tz_convert("Asia/Karachi").date
    pv = (tp * vol).groupby(day).cumsum()
    cv = vol.groupby(day).cumsum()
    return (pv / cv.replace(0, np.nan)).rename("vwap")


def linreg_channel(close: pd.Series, period: int = 100, k: float = 2.0) -> pd.DataFrame:
    mid, slope, resid_std = rolling_linreg(close, period)
    return pd.DataFrame({"linreg_mid": mid, "linreg_upper": mid + k * resid_std, "linreg_lower": mid - k * resid_std,
                         "linreg_slope": slope}, index=close.index)
