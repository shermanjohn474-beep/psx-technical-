"""Remaining classical patterns (Phase 2): valid detection, negative geometry and noise."""
import numpy as np
import pytest

from app.chart_patterns.registry import all_pattern_ids, detect_patterns
from app.schemas.patterns import Direction, PatternStage
from tests.pattern_fixtures import NOISE, TREND_UP, df, mirror

CONF = (PatternStage.CONFIRMED, PatternStage.RETESTING)
_U = [(int(10 + x), 130 - 25 * (1 - ((x - 50) / 50) ** 2)) for x in np.arange(0, 101, 5)]
ROUND = [(0, 100), (10, 132)] + _U + [(125, 140)]
CUP = [(0, 100), (10, 132)] + _U + [(118, 124), (126, 128.5), (135, 142)]
V_SHAPE = [(0, 100), (40, 140), (52, 112), (62, 136), (70, 140)]
BULL_TRAP = [(0, 100), (20, 110), (30, 104), (45, 110), (55, 104), (62, 109), (65, 112.5), (68, 106), (80, 101)]
MEASURED = [(0, 100), (25, 130), (40, 118), (60, 150)]
COMPRESS = [(0, 100), (40, 130), (80, 120), (110, 126)] + [(110 + k, 126 + (0.6 if k % 2 else -0.6)) for k in range(1, 25)] + [(140, 138)]
DIAMOND = [(0, 100), (30, 125), (38, 118), (48, 130), (60, 112), (72, 126), (82, 116), (90, 121), (96, 118), (105, 105)]
VCP = [(0, 80), (60, 120), (70, 108), (85, 121), (92, 114), (102, 121.5), (107, 118), (115, 128)]

VALID = [
    ("rounding_bottom", ROUND, Direction.BULLISH),
    ("rounding_top", mirror(ROUND), Direction.BEARISH),
    ("cup_and_handle", CUP, Direction.BULLISH),
    ("inverted_cup_and_handle", mirror(CUP), Direction.BEARISH),
    ("v_bottom", V_SHAPE, Direction.BULLISH),
    ("inverted_v_top", mirror(V_SHAPE), Direction.BEARISH),
    ("bull_trap", BULL_TRAP, Direction.BEARISH),
    ("bear_trap", mirror(BULL_TRAP), Direction.BULLISH),
    ("measured_move_up", MEASURED, Direction.BULLISH),
    ("measured_move_down", mirror(MEASURED), Direction.BEARISH),
    ("compression_breakout", COMPRESS, Direction.BULLISH),
    ("diamond_top", DIAMOND, Direction.BEARISH),
    ("diamond_bottom", mirror(DIAMOND), Direction.BULLISH),
    ("vcp", VCP, Direction.BULLISH),
]


def find(wp, pid, **kw):
    return [r for r in detect_patterns(df(wp, **kw)) if r.pattern_id == pid]


@pytest.mark.parametrize("pid,wp,direction", VALID, ids=[v[0] for v in VALID])
def test_valid_confirmed(pid, wp, direction):
    res = find(wp, pid)
    assert any(r.stage in CONF and r.direction == direction for r in res), [(r.stage, r.direction) for r in res]
    r = next(r for r in res if r.stage in CONF)
    assert r.breakout_level is not None and r.target is not None
    assert (r.target > r.breakout_level) == (direction == Direction.BULLISH)


@pytest.mark.parametrize("pid,wp,direction", VALID, ids=[v[0] for v in VALID])
def test_incomplete_not_confirmed(pid, wp, direction):
    """Truncating right before the confirmation bar must never yield a confirmed result."""
    full = next(r for r in find(wp, pid) if r.stage in CONF)
    d = df(wp).iloc[: full.confirmation_index]
    res = [r for r in detect_patterns(d) if r.pattern_id == pid and r.start_index == full.start_index]
    assert all(r.confirmation_index is None for r in res)


@pytest.mark.parametrize("pid,wp,direction", VALID, ids=[v[0] for v in VALID])
@pytest.mark.parametrize("seed", [1, 2])
def test_noise_no_confirmed_pattern(pid, wp, direction, seed):
    for base in (NOISE, TREND_UP):
        assert not [r for r in find(base, pid, seed=seed, noise=0.3) if r.stage in CONF]


def test_v_shape_rejects_slow_trend():
    slow = [(0, 100), (60, 160), (75, 130), (90, 155)]
    assert not find(slow, "inverted_v_top")


def test_rounding_rejects_v_shape():
    assert not find([(0, 100), (10, 132), (40, 105), (70, 132), (85, 140)], "rounding_bottom")


def test_cup_without_handle_is_rounding_only():
    assert not find(ROUND, "cup_and_handle")


def test_compression_direction_only_from_breakout():
    down = COMPRESS[:-1] + [(140, 114)]
    r = [x for x in find(down, "compression_breakout") if x.stage in CONF]
    assert r and r[-1].direction == Direction.BEARISH
    pending = [x for x in find(COMPRESS[:-1], "compression_breakout") if x.end_index >= 130]
    assert pending and pending[-1].direction == Direction.NEUTRAL


def test_vcp_and_harmonics_marked_experimental():
    assert all(r.experimental for r in find(VCP, "vcp"))


def test_registry_covers_required_patterns():
    ids = set(all_pattern_ids())
    required = {
        "head_and_shoulders", "inverse_head_and_shoulders", "double_top", "double_bottom", "triple_top", "triple_bottom",
        "rounding_top", "rounding_bottom", "cup_and_handle", "inverted_cup_and_handle", "broadening_top",
        "broadening_bottom", "diamond_top", "diamond_bottom", "v_bottom", "inverted_v_top", "adam_eve_top",
        "adam_eve_bottom", "bull_trap", "bear_trap", "bull_flag", "bear_flag", "bull_pennant", "bear_pennant",
        "rectangle", "ascending_triangle", "descending_triangle", "symmetrical_triangle", "rising_wedge", "falling_wedge",
        "ascending_channel", "descending_channel", "horizontal_channel", "expanding_triangle", "broadening_formation",
        "measured_move_up", "measured_move_down", "compression_breakout", "vcp",
        "gartley_bullish", "bat_bullish", "butterfly_bullish", "crab_bullish", "cypher_bullish", "abcd_bullish",
        "elliott_impulse_up",
    }
    assert required <= ids, required - ids
