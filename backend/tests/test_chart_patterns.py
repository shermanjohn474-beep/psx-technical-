"""Phase-1 chart pattern tests.

For each pattern: valid completed formation, incomplete formation, invalid geometry,
false breakout, failed/invalidated pattern, and noise without a genuine pattern.
"""
import pytest

from app.chart_patterns.base import PatternConfig, build_context
from app.chart_patterns.registry import detect_patterns
from app.schemas.patterns import Direction, PatternStage

from tests.pattern_fixtures import (
    ASC_TRIANGLE, BULL_FLAG, DOUBLE_TOP, HEAD_SHOULDERS, NOISE, SYM_TRIANGLE, TREND_UP, df, mirror,
)

ACTIVE_UNCONFIRMED = {PatternStage.FORMING, PatternStage.APPROACHING_CONFIRMATION}


def find(wp, pid, **kw):
    return [r for r in detect_patterns(df(wp, **kw)) if r.pattern_id == pid]


def best(wp, pid, **kw):
    res = find(wp, pid, **kw)
    assert res, f"{pid} not detected"
    return max(res, key=lambda r: r.quality_score)


# --------------------------------------------------------------------- generic matrix
CASES = [
    # (pattern_id, scenarios, mirror?, expected direction)
    ("double_top", DOUBLE_TOP, False, Direction.BEARISH),
    ("double_bottom", DOUBLE_TOP, True, Direction.BULLISH),
    ("head_and_shoulders", HEAD_SHOULDERS, False, Direction.BEARISH),
    ("inverse_head_and_shoulders", HEAD_SHOULDERS, True, Direction.BULLISH),
    ("ascending_triangle", ASC_TRIANGLE, False, Direction.BULLISH),
    ("descending_triangle", ASC_TRIANGLE, True, Direction.BEARISH),
    ("bull_flag", BULL_FLAG, False, Direction.BULLISH),
    ("bear_flag", BULL_FLAG, True, Direction.BEARISH),
]


def _wp(sc, key, mir):
    return mirror(sc[key]) if mir else sc[key]


# Flags are small structures: test the geometry with lower noise (robustness is
# checked separately across seeds below).
KW = {"bull_flag": {"noise": 0.12, "wick": 0.3}, "bear_flag": {"noise": 0.12, "wick": 0.3}}


@pytest.mark.parametrize("pid,sc,mir,direction", CASES, ids=[c[0] for c in CASES])
def test_valid_completed(pid, sc, mir, direction):
    r = best(_wp(sc, "valid", mir), pid, **KW.get(pid, {}))
    assert r.stage in (PatternStage.CONFIRMED, PatternStage.RETESTING)
    assert r.direction == direction
    assert r.confirmation_index is not None and r.confirmation_time is not None
    assert r.breakout_level is not None and r.target is not None and r.invalidation_level is not None
    # target lies beyond the breakout in the pattern direction; invalidation on the other side
    if direction == Direction.BULLISH:
        assert r.target > r.breakout_level and r.invalidation_level < r.breakout_level
    else:
        assert r.target < r.breakout_level and r.invalidation_level > r.breakout_level
    assert 0 < r.quality_score <= 100
    assert r.key_points and all(k.timestamp for k in r.key_points)


@pytest.mark.parametrize("pid,sc,mir,direction", CASES, ids=[c[0] for c in CASES])
def test_incomplete_is_not_confirmed(pid, sc, mir, direction):
    res = find(_wp(sc, "incomplete", mir), pid, **KW.get(pid, {}))
    assert res, f"{pid} should be detected as forming"
    assert all(r.stage in ACTIVE_UNCONFIRMED for r in res)
    assert all(r.confirmation_index is None for r in res)


@pytest.mark.parametrize("pid,sc,mir,direction", CASES, ids=[c[0] for c in CASES])
def test_invalid_geometry_rejected(pid, sc, mir, direction):
    res = find(_wp(sc, "invalid_geometry", mir), pid, **KW.get(pid, {}))
    assert not [r for r in res if r.stage in (PatternStage.CONFIRMED, PatternStage.RETESTING)], res


@pytest.mark.parametrize("pid,sc,mir,direction", CASES, ids=[c[0] for c in CASES])
def test_false_breakout_marked_failed(pid, sc, mir, direction):
    res = find(_wp(sc, "false_breakout", mir), pid, **KW.get(pid, {}))
    assert res
    assert any(r.stage == PatternStage.FAILED for r in res), [(r.stage, r.start_index) for r in res]
    assert not any(r.stage in (PatternStage.CONFIRMED, PatternStage.RETESTING) for r in res)


@pytest.mark.parametrize("pid,sc,mir,direction", CASES, ids=[c[0] for c in CASES])
def test_failed_pattern_invalidated(pid, sc, mir, direction):
    res = find(_wp(sc, "failed", mir), pid, **KW.get(pid, {}))
    assert res
    assert all(r.stage == PatternStage.INVALIDATED for r in res), [(r.stage, r.evidence) for r in res]


@pytest.mark.parametrize("pid,sc,mir,direction", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("noise", ["flat", "trend"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_noise_has_no_pattern(pid, sc, mir, direction, noise, seed):
    wp = NOISE if noise == "flat" else (mirror(TREND_UP) if mir else TREND_UP)
    assert not find(wp, pid, seed=seed, noise=0.3)


# --------------------------------------------------------------------- specifics
@pytest.mark.parametrize("pid,sc,mir,direction", CASES, ids=[c[0] for c in CASES])
def test_valid_detection_robust_across_seeds(pid, sc, mir, direction):
    """Completed formations should be recognised for most noise realisations."""
    hits = 0
    for seed in range(1, 11):
        res = find(_wp(sc, "valid", mir), pid, seed=seed)
        hits += any(r.stage in (PatternStage.CONFIRMED, PatternStage.RETESTING) and r.direction == direction for r in res)
    assert hits >= 8, f"{pid}: only {hits}/10 seeds"


def test_double_top_measured_move_and_neckline():
    r = best(DOUBLE_TOP["valid"], "double_top")
    trough = next(k for k in r.key_points if k.label == "Neckline")
    peak = max(k.price for k in r.key_points if k.label.startswith("Top"))
    assert r.breakout_level == pytest.approx(trough.price, abs=1e-3)
    assert r.target == pytest.approx(trough.price - (peak - trough.price), abs=1e-3)
    assert any(line.label == "Neckline" for line in r.lines)


def test_head_shoulders_structure_and_target():
    r = best(HEAD_SHOULDERS["valid"], "head_and_shoulders")
    labels = [k.label for k in r.key_points]
    assert labels == ["Left Shoulder", "Neck 1", "Head", "Neck 2", "Right Shoulder"]
    head = r.key_points[2].price
    assert head > r.key_points[0].price and head > r.key_points[4].price
    assert r.target < r.breakout_level
    assert r.invalidation_level == pytest.approx(r.key_points[4].price, abs=1e-3)


def test_head_shoulders_not_confirmed_without_neckline_break():
    for r in find(HEAD_SHOULDERS["incomplete"], "head_and_shoulders"):
        assert r.stage != PatternStage.CONFIRMED


def test_symmetrical_triangle_direction_from_breakout_only():
    up = best(SYM_TRIANGLE["valid_up"], "symmetrical_triangle")
    dn = best(SYM_TRIANGLE["valid_down"], "symmetrical_triangle")
    inc = best(SYM_TRIANGLE["incomplete"], "symmetrical_triangle")
    assert up.stage == PatternStage.CONFIRMED and up.direction == Direction.BULLISH
    assert dn.stage == PatternStage.CONFIRMED and dn.direction == Direction.BEARISH
    assert inc.stage in ACTIVE_UNCONFIRMED and inc.direction == Direction.NEUTRAL
    assert any("Bilateral" in e for e in inc.evidence)


def test_symmetrical_triangle_invalid_and_false_breakout():
    assert not [r for r in find(SYM_TRIANGLE["invalid_geometry"], "symmetrical_triangle")
                if r.stage == PatternStage.CONFIRMED]
    res = find(SYM_TRIANGLE["false_breakout"], "symmetrical_triangle")
    assert any(r.stage == PatternStage.FAILED for r in res)


def test_volume_confirmation_flag():
    wp = DOUBLE_TOP["valid"]
    plain = best(wp, "double_top")
    brk = plain.confirmation_index
    spiked = best(wp, "double_top", volume_spikes={brk: 3.0})
    assert plain.volume_confirmation is False
    assert spiked.volume_confirmation is True
    assert spiked.quality_score > plain.quality_score


def test_missing_volume_gives_none_confirmation():
    d = df(DOUBLE_TOP["valid"])
    d["volume"] = float("nan")
    r = [x for x in detect_patterns(d) if x.pattern_id == "double_top"][0]
    assert r.volume_confirmation is None
    assert "volume" not in r.quality_components


def test_incomplete_last_candle_is_provisional():
    d = df(DOUBLE_TOP["valid"])
    full = best(DOUBLE_TOP["valid"], "double_top")
    cut = d.iloc[: full.confirmation_index + 1]
    ctx = build_context(cut, last_candle_complete=False)
    res = [r for r in detect_patterns(ctx=ctx) if r.pattern_id == "double_top"]
    assert res and res[0].stage == PatternStage.APPROACHING_CONFIRMATION and res[0].provisional
    ctx2 = build_context(cut, last_candle_complete=True)
    res2 = [r for r in detect_patterns(ctx=ctx2) if r.pattern_id == "double_top"]
    assert res2[0].stage == PatternStage.CONFIRMED


def test_configurable_thresholds_change_detection():
    strict = PatternConfig(equal_level_atr=0.05)
    d = df(DOUBLE_TOP["valid"])
    assert not [r for r in detect_patterns(d, strict) if r.pattern_id == "double_top"]


def test_pattern_coordinates_have_timestamps():
    r = best(ASC_TRIANGLE["valid"], "ascending_triangle")
    d = df(ASC_TRIANGLE["valid"])
    for k in r.key_points:
        assert d.index[k.index].to_pydatetime() == k.timestamp
    assert len(r.lines) == 2


def test_empty_and_tiny_inputs():
    d = df(DOUBLE_TOP["valid"])
    assert detect_patterns(d.iloc[:0]) == []
    assert detect_patterns(d.iloc[:10]) == []
