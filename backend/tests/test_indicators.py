"""Indicator calculations checked against TA-Lib (trusted reference implementation).

Recursive (EMA/Wilder) indicators are compared after warm-up, where seeding
differences have decayed below tolerance.
"""
import numpy as np
import pandas as pd
import pytest

from app.data_providers.synthetic import random_walk
from app.indicators import momentum as mom
from app.indicators import trend as tr
from app.indicators import volatility as vola
from app.indicators import volume as vol
from app.indicators.core import ema, sma
from app.indicators.engine import compute_indicators, indicator_snapshot

talib = pytest.importorskip("talib")

DF = random_walk(600, seed=42)
H, L, C, V = (DF[k].to_numpy() for k in ("high", "low", "close", "volume"))
TAIL = slice(300, None)


def close_enough(ours, ref, tail=TAIL, rtol=1e-6, atol=1e-6):
    ours = np.asarray(ours, dtype=float)[tail]
    ref = np.asarray(ref, dtype=float)[tail]
    np.testing.assert_allclose(ours, ref, rtol=rtol, atol=atol)


def test_sma_exact():
    close_enough(sma(DF["close"], 20), talib.SMA(C, 20), tail=slice(19, None))


def test_ema_exact_including_seed():
    close_enough(ema(DF["close"], 20), talib.EMA(C, 20), tail=slice(19, None))


def test_rsi():
    close_enough(mom.rsi(DF["close"], 14), talib.RSI(C, 14), tail=slice(14, None), atol=1e-6)


def test_macd_tail():
    m = mom.macd(DF["close"])
    ref_m, ref_s, ref_h = talib.MACD(C, 12, 26, 9)
    close_enough(m["macd"], ref_m)
    close_enough(m["macd_signal"], ref_s)
    close_enough(m["macd_hist"], ref_h)


def test_bollinger_exact():
    b = vola.bollinger(DF["close"], 20, 2.0)
    u, mid, lo = talib.BBANDS(C, 20, 2.0, 2.0)
    close_enough(b["bb_upper"], u, tail=slice(19, None))
    close_enough(b["bb_lower"], lo, tail=slice(19, None))


def test_atr():
    close_enough(vola.atr(DF, 14), talib.ATR(H, L, C, 14), tail=slice(14, None))


def test_adx_dmi_tail():
    a = tr.adx_dmi(DF, 14)
    close_enough(a["adx"], talib.ADX(H, L, C, 14), rtol=1e-4, atol=1e-4)
    close_enough(a["plus_di"], talib.PLUS_DI(H, L, C, 14), rtol=1e-4, atol=1e-4)
    close_enough(a["minus_di"], talib.MINUS_DI(H, L, C, 14), rtol=1e-4, atol=1e-4)


def test_stochastic():
    s = mom.stochastic(DF, 14, 3, 3)
    k, d = talib.STOCH(H, L, C, 14, 3, 0, 3, 0)
    close_enough(s["stoch_k"], k, tail=slice(20, None))
    close_enough(s["stoch_d"], d, tail=slice(20, None))


def test_cci_willr_roc_mom():
    close_enough(mom.cci(DF, 20), talib.CCI(H, L, C, 20), tail=slice(19, None), rtol=1e-6, atol=1e-5)
    close_enough(mom.williams_r(DF, 14), talib.WILLR(H, L, C, 14), tail=slice(13, None))
    close_enough(mom.roc(DF["close"], 10), talib.ROC(C, 10), tail=slice(10, None))
    close_enough(mom.momentum(DF["close"], 10), talib.MOM(C, 10), tail=slice(10, None))


def test_volume_indicators():
    close_enough(vol.obv(DF), talib.OBV(C, V), tail=slice(0, None))
    close_enough(vol.accumulation_distribution(DF), talib.AD(H, L, C, V), tail=slice(0, None), rtol=1e-6)
    close_enough(vol.money_flow_index(DF, 14), talib.MFI(H, L, C, V, 14), tail=slice(14, None), atol=1e-6)


def test_parabolic_sar_tail():
    ours = tr.parabolic_sar(DF).to_numpy()
    ref = talib.SAR(H, L, 0.02, 0.2)
    # SAR is path dependent; require near-identical values on >= 97% of the tail
    o, r = ours[50:], ref[50:]
    agree = np.isclose(o, r, rtol=1e-6, atol=1e-6).mean()
    assert agree >= 0.97, agree


def test_indicators_are_causal():
    """Values at bar t must not change when future bars are appended."""
    full = compute_indicators(DF).frame
    part = compute_indicators(DF.iloc[:400]).frame
    cols = [c for c in part.columns if c not in ("ichimoku_cloud_a", "ichimoku_cloud_b")]
    pd.testing.assert_frame_equal(full[cols].iloc[:400], part[cols], check_exact=False, rtol=1e-9, atol=1e-9)


def test_missing_volume_marks_unavailable():
    df = DF.copy()
    df["volume"] = np.nan
    ind = compute_indicators(df)
    assert "obv" not in ind.frame.columns
    assert "rel_volume" in ind.unavailable
    snap = indicator_snapshot(df, ind)
    assert snap["volume"] is None


def test_short_history_marks_long_mas_unavailable():
    ind = compute_indicators(DF.iloc[:120])
    assert "sma_200" in ind.unavailable
    assert ind.latest()["sma_200"] is None
    assert ind.latest()["sma_100"] is not None


def test_supertrend_flips_with_trend():
    up = DF.copy()
    up[["open", "high", "low", "close"]] = np.linspace(50, 150, len(DF))[:, None] + np.array([0, 0.5, -0.5, 0.1])
    st = tr.supertrend(up)
    assert st["supertrend_dir"].iloc[-1] == 1
