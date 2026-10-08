"""Indicator engine: computes the full indicator set and a latest-value snapshot.

Only indicators supported by the available data are populated: e.g. VWAP only for
intraday data with volume, SMA200 only with >= 200 bars, volume indicators only
when volume exists. Unavailable values are reported as ``None`` with a reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.indicators import momentum as mom
from app.indicators import trend as tr
from app.indicators import volatility as vola
from app.indicators import volume as vol
from app.schemas.market import Timeframe


@dataclass
class IndicatorConfig:
    sma_periods: tuple[int, ...] = (10, 20, 50, 100, 200)
    ema_periods: tuple[int, ...] = (9, 20, 50, 100, 200)
    rsi_period: int = 14
    macd: tuple[int, int, int] = (12, 26, 9)
    bb_period: int = 20
    bb_k: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    stoch: tuple[int, int, int] = (14, 3, 3)
    volume_period: int = 20
    supertrend: tuple[int, float] = (10, 3.0)
    linreg_period: int = 100


@dataclass
class IndicatorSet:
    frame: pd.DataFrame
    unavailable: dict[str, str] = field(default_factory=dict)

    def latest(self) -> dict[str, float | None]:
        if self.frame.empty:
            return {}
        row = self.frame.iloc[-1]
        return {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}


def compute_indicators(df: pd.DataFrame, timeframe: Timeframe = Timeframe.D1,
                       cfg: IndicatorConfig | None = None) -> IndicatorSet:
    cfg = cfg or IndicatorConfig()
    close = df["close"]
    parts = [
        tr.moving_averages(close, cfg.sma_periods, cfg.ema_periods),
        mom.rsi(close, cfg.rsi_period).to_frame(),
        mom.macd(close, *cfg.macd),
        mom.stochastic(df, *cfg.stoch),
        mom.stoch_rsi(close),
        mom.roc(close).to_frame(),
        mom.momentum(close).to_frame(),
        mom.cci(df).to_frame(),
        mom.williams_r(df).to_frame(),
        vola.atr(df, cfg.atr_period).to_frame(),
        vola.bollinger(close, cfg.bb_period, cfg.bb_k),
        vola.bollinger_squeeze(close, cfg.bb_period, cfg.bb_k).astype(float).to_frame(),
        vola.keltner(df),
        vola.donchian(df),
        vola.historical_volatility(close).to_frame(),
        tr.adx_dmi(df, cfg.adx_period),
        tr.supertrend(df, *cfg.supertrend),
        tr.ichimoku(df),
        tr.parabolic_sar(df).to_frame(),
        tr.linreg_channel(close, cfg.linreg_period),
    ]
    unavailable: dict[str, str] = {}
    if vol.has_volume(df):
        parts += [
            vol.volume_sma(df, cfg.volume_period).to_frame(),
            vol.relative_volume(df, cfg.volume_period).to_frame(),
            vol.obv(df).to_frame(),
            vol.accumulation_distribution(df).to_frame(),
            vol.chaikin_money_flow(df).to_frame(),
            vol.money_flow_index(df).to_frame(),
            vol.volume_dry_up(df).astype(float).to_frame(),
            vol.volume_price_divergence(df).astype(float).to_frame(),
        ]
        if timeframe.is_intraday:
            parts.append(tr.vwap_intraday(df).to_frame())
        else:
            unavailable["vwap"] = "VWAP requires intraday data."
    else:
        for k in ("vol_sma", "rel_volume", "obv", "ad", "cmf", "mfi", "vwap"):
            unavailable[k] = "Volume data unavailable."
    frame = pd.concat(parts, axis=1)
    n = len(df)
    for p in cfg.sma_periods:
        if n < p:
            unavailable[f"sma_{p}"] = f"Needs {p} bars; have {n}."
    for p in cfg.ema_periods:
        if n < p:
            unavailable[f"ema_{p}"] = f"Needs {p} bars; have {n}."
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return IndicatorSet(frame=frame, unavailable=unavailable)


def indicator_snapshot(df: pd.DataFrame, ind: IndicatorSet) -> dict:
    """Human-relevant summary of latest indicator state (Report section C)."""
    if df.empty:
        return {}
    f = ind.frame
    last = ind.latest()
    close = float(df["close"].iloc[-1])

    def val(k):
        return last.get(k)

    def direction(series: str, lookback: int = 3) -> str | None:
        s = f[series].dropna()
        if len(s) <= lookback:
            return None
        d = s.iloc[-1] - s.iloc[-1 - lookback]
        return "rising" if d > 0 else "falling" if d < 0 else "flat"

    snap: dict = {"close": close, "moving_averages": {}}
    for k in ("sma_20", "sma_50", "sma_100", "sma_200", "ema_20", "ema_50", "ema_100", "ema_200"):
        v = val(k)
        snap["moving_averages"][k] = None if v is None else {
            "value": round(v, 4), "price_above": close > v, "distance_pct": round(100 * (close / v - 1), 2)}
    r = val("rsi")
    snap["rsi"] = None if r is None else {
        "value": round(r, 2), "direction": direction("rsi"),
        "zone": "overbought" if r >= 70 else "oversold" if r <= 30 else "neutral"}
    m, s_, h = val("macd"), val("macd_signal"), val("macd_hist")
    if m is not None and s_ is not None:
        hist = f["macd_hist"].dropna()
        cross = None
        if len(hist) >= 2:
            if hist.iloc[-1] > 0 >= hist.iloc[-2]:
                cross = "bullish_cross_this_bar"
            elif hist.iloc[-1] < 0 <= hist.iloc[-2]:
                cross = "bearish_cross_this_bar"
        snap["macd"] = {"macd": round(m, 4), "signal": round(s_, 4), "hist": round(h, 4),
                        "state": "above_signal" if m > s_ else "below_signal",
                        "zero_line": "above" if m > 0 else "below", "cross": cross,
                        "hist_direction": direction("macd_hist", 1)}
    else:
        snap["macd"] = None
    bu, bl, pb = val("bb_upper"), val("bb_lower"), val("bb_pct_b")
    snap["bollinger"] = None if bu is None else {
        "upper": round(bu, 4), "lower": round(bl, 4), "mid": round(val("bb_mid"), 4), "pct_b": round(pb, 3),
        "position": "above_upper" if close > bu else "below_lower" if close < bl else
        ("upper_half" if pb >= 0.5 else "lower_half"), "squeeze": bool(val("bb_squeeze"))}
    a = val("atr")
    snap["atr"] = None if a is None else {"value": round(a, 4), "pct_of_price": round(100 * a / close, 2),
                                          "hv_annualized": None if val("hv") is None else round(val("hv"), 4)}
    rv = val("rel_volume")
    snap["volume"] = None if val("vol_sma") is None else {
        "last": float(df["volume"].iloc[-1]), "avg_20": round(val("vol_sma"), 0),
        "relative": None if rv is None else round(rv, 2),
        "obv_direction": direction("obv", 10) if "obv" in f else None,
        "cmf": None if val("cmf") is None else round(val("cmf"), 3)}
    adx = val("adx")
    snap["adx"] = None if adx is None else {
        "adx": round(adx, 2), "plus_di": round(val("plus_di"), 2), "minus_di": round(val("minus_di"), 2),
        "strength": "strong" if adx >= 25 else "developing" if adx >= 20 else "weak/no trend"}
    snap["unavailable"] = ind.unavailable
    return snap
