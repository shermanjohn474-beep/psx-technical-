import numpy as np

from app.data_providers.synthetic import make_ohlcv, random_walk
from app.support_resistance.levels import detect_levels
from app.swing_detection.swings import SwingConfig, detect_swings, fractal_swings, zigzag_atr


def test_zigzag_alternates_and_finds_major_turns():
    df = make_ohlcv([(0, 100), (30, 130), (60, 105), (90, 140)])
    sw = zigzag_atr(df).swings
    kinds = [s.kind for s in sw]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))
    highs = [s for s in sw if s.kind == "H"]
    assert any(abs(s.index - 30) <= 2 for s in highs)
    assert any(abs(s.index - 60) <= 2 for s in sw if s.kind == "L")


def test_zigzag_is_causal():
    """Swings confirmed by bar t are identical whether or not later bars exist (no look-ahead)."""
    df = random_walk(500, seed=11)
    full = zigzag_atr(df)
    for t in (100, 200, 333, 450):
        part = zigzag_atr(df.iloc[: t + 1])
        assert part.swings == full.confirmed_upto(t)
        assert all(s.confirmed_index >= s.index for s in part.swings)


def test_fractal_confirmation_lag():
    df = random_walk(200, seed=5)
    sw = fractal_swings(df, SwingConfig(method="fractal", fractal_n=3)).swings
    assert sw and all(s.confirmed_index - s.index >= 3 for s in sw)


def test_peaks_method_runs():
    df = random_walk(200, seed=5)
    assert detect_swings(df, SwingConfig(method="peaks")).swings


def test_levels_cluster_into_zones():
    # repeated tests of ~130 resistance and ~110 support
    wp = [(0, 110), (15, 130), (30, 110), (45, 130), (60, 110), (75, 130), (90, 120)]
    df = make_ohlcv(wp)
    lv = detect_levels(df)
    res = lv.immediate_resistance
    sup = lv.immediate_support
    assert res and res.low <= 131 and res.high >= 129 and res.touches >= 3
    assert sup and sup.low <= 111 and sup.high >= 109 and sup.touches >= 3
    assert res.high > res.low  # zones, not single prices
    assert 0 <= res.strength <= 100


def test_levels_insufficient_data():
    df = make_ohlcv([(0, 100), (10, 101)])
    assert detect_levels(df).zones == []
