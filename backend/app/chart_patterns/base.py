"""Shared infrastructure for chart-pattern detectors.

Design rules enforced here:

* Detectors only see the frame they are given (``ctx.df``) — the backtester passes
  truncated frames, so nothing beyond ``ctx.last`` is ever visible.
* Only *confirmed* swings (``confirmed_index <= last``) plus the single provisional
  running extreme are available. A pattern built on the provisional swing can never
  be CONFIRMED.
* Confirmation requires a **completed** candle close beyond the trigger by a
  configurable ATR tolerance. A break on an incomplete live candle is reported as
  APPROACHING_CONFIRMATION with ``provisional=True``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from app.indicators.volatility import causal_atr
from app.indicators.volume import has_volume, relative_volume
from app.schemas.patterns import (
    Direction, KeyPoint, LinePoint, PatternCategory, PatternLine, PatternResult, PatternStage,
)
from app.swing_detection.swings import Swing, SwingConfig, SwingSet, zigzag_atr


@dataclass
class PatternConfig:
    """Global tolerances (ATR-normalised). Every detector reads from here."""

    swing: SwingConfig = field(default_factory=SwingConfig)
    breakout_tol_atr: float = 0.25   # close must exceed trigger by this many ATR
    approach_atr: float = 1.0         # within this distance => approaching confirmation
    retest_atr: float = 0.6           # pullback to within this distance of trigger => retest
    fail_lookahead_bars: int = 15     # a close back through the trigger within N bars => failed
    max_wait_bars: int = 60           # forming pattern expires if no resolution in N bars
    max_age_bars: int = 250           # ignore patterns that ended longer ago than this
    volume_confirm_mult: float = 1.5  # breakout relative volume needed for volume confirmation
    volume_period: int = 20
    trend_lookback: int = 40
    min_trend_atr: float = 3.0        # preceding move (in ATR) to qualify as a prior trend
    equal_level_atr: float = 1.0      # "equal" highs/lows tolerance (double/triple tops)
    min_height_pct: float = 3.0       # minimum pattern height as % of price (use ~1% intraday)
    params: dict[str, dict] = field(default_factory=dict)  # per-detector overrides

    def p(self, detector_id: str, key: str, default):
        return self.params.get(detector_id, {}).get(key, default)


@dataclass
class AnalysisContext:
    df: pd.DataFrame
    cfg: PatternConfig
    swings: SwingSet
    timeframe: str | None = None
    last_candle_complete: bool = True

    def __post_init__(self):
        self.n = len(self.df)
        self.last = self.n - 1
        self.o = self.df["open"].to_numpy(dtype=float)
        self.h = self.df["high"].to_numpy(dtype=float)
        self.l = self.df["low"].to_numpy(dtype=float)
        self.c = self.df["close"].to_numpy(dtype=float)
        self.atr = self.swings.atr.to_numpy(dtype=float)
        self.index = self.df.index
        self.volume_available = has_volume(self.df)
        self.rel_vol = relative_volume(self.df, self.cfg.volume_period).to_numpy(dtype=float) if self.volume_available else None

    # ---- time helpers --------------------------------------------------
    def ts(self, i: int) -> datetime:
        return self.index[int(i)].to_pydatetime()

    def kp(self, label: str, i: int, price: float) -> KeyPoint:
        return KeyPoint(label=label, index=int(i), timestamp=self.ts(i), price=float(price))

    def line(self, label: str, i0: int, p0: float, i1: int, p1: float, style: str = "solid") -> PatternLine:
        i1 = min(int(i1), self.last)
        return PatternLine(label=label, start=LinePoint(index=int(i0), timestamp=self.ts(i0), price=float(p0)),
                           end=LinePoint(index=i1, timestamp=self.ts(i1), price=float(p1)), style=style)

    # ---- swing helpers -------------------------------------------------
    def confirmed_swings(self) -> list[Swing]:
        return [s for s in self.swings.swings if s.confirmed_index <= self.last]

    def swings_with_provisional(self) -> list[Swing]:
        sw = self.confirmed_swings()
        p = self.swings.provisional
        if p is not None and p.index <= self.last and (not sw or p.index > sw[-1].index):
            sw = sw + [p]
        return sw

    # ---- context helpers ----------------------------------------------
    def prior_move_atr(self, idx: int, lookback: int | None = None) -> float:
        """Net close change into ``idx`` over ``lookback`` bars, in ATR units (+ = up)."""
        lb = lookback or self.cfg.trend_lookback
        j = max(0, idx - lb)
        if idx <= j:
            return 0.0
        a = self.atr[idx] if self.atr[idx] > 0 else 1.0
        return float((self.c[idx] - self.c[j]) / a)

    def prior_extreme_move_atr(self, idx: int, lookback: int | None = None) -> float:
        """Up-move: high[idx] - min low before; down-move: negative of low[idx] - max high before."""
        lb = lookback or self.cfg.trend_lookback
        j = max(0, idx - lb)
        if idx <= j:
            return 0.0
        a = self.atr[idx] if self.atr[idx] > 0 else 1.0
        up = (self.h[idx] - self.l[j:idx].min()) / a
        down = (self.h[j:idx].max() - self.l[idx]) / a
        return float(up if up >= down else -down)

    def volume_at(self, i: int) -> float | None:
        if self.rel_vol is None or i < 0 or i > self.last:
            return None
        v = self.rel_vol[i]
        return None if np.isnan(v) else float(v)


# ---------------------------------------------------------------------------
# Lifecycle evaluation


@dataclass
class BreakoutEval:
    stage: PatternStage
    confirmation_index: int | None = None
    invalidation_index: int | None = None
    failed_index: int | None = None
    target_reached: bool = False
    provisional: bool = False
    rel_volume: float | None = None
    notes: list[str] = field(default_factory=list)


def evaluate_breakout(
    ctx: AnalysisContext,
    start: int,
    level_fn: Callable[[int], float],
    direction: Direction,
    invalidation: float | Callable[[int], float] | None,
    target: float | None = None,
    *,
    expires: bool = True,
) -> BreakoutEval:
    """Walk forward from ``start`` and classify the lifecycle stage.

    ``direction`` is the expected resolution (BULLISH = close above level,
    BEARISH = close below level). ``invalidation`` is a price (or function) that,
    if closed beyond *before* confirmation, invalidates the setup.
    """
    cfg = ctx.cfg
    bull = direction == Direction.BULLISH
    sign = 1.0 if bull else -1.0
    inval_fn = invalidation if callable(invalidation) else (lambda _j, v=invalidation: v)
    confirmed_at = None
    frozen = None
    retest_seen_at = None
    ev = BreakoutEval(stage=PatternStage.FORMING)
    for j in range(max(start, 0), ctx.n):
        lvl = level_fn(j)
        tol = cfg.breakout_tol_atr * ctx.atr[j]
        close = ctx.c[j]
        complete = j < ctx.last or ctx.last_candle_complete
        if confirmed_at is None:
            inv = inval_fn(j)
            if inv is not None and sign * (inv - close) > tol:
                ev.stage, ev.invalidation_index = PatternStage.INVALIDATED, j
                ev.notes.append(f"Closed beyond invalidation level {inv:.4g} before confirmation.")
                return ev
            if sign * (close - lvl) > tol:
                if not complete:
                    ev.stage, ev.provisional = PatternStage.APPROACHING_CONFIRMATION, True
                    ev.notes.append("Live (incomplete) candle beyond trigger: confirmation pending candle close.")
                    return ev
                confirmed_at = j
                frozen = lvl  # failure is judged against the breakout price itself
                ev.confirmation_index = j
                ev.rel_volume = ctx.volume_at(j)
                continue
            if expires and j - start > cfg.max_wait_bars:
                ev.stage, ev.invalidation_index = PatternStage.INVALIDATED, j
                ev.notes.append(f"Expired: no resolution within {cfg.max_wait_bars} bars.")
                return ev
        else:
            if target is not None and sign * (ctx.h[j] if bull else ctx.l[j]) >= sign * target:
                ev.target_reached = True
            # failure: close back through trigger by tolerance, before target, within window
            if not ev.target_reached and j - confirmed_at <= cfg.fail_lookahead_bars and sign * (frozen - close) > tol and complete:
                ev.stage, ev.failed_index = PatternStage.FAILED, j
                ev.notes.append("Closed back through the breakout level shortly after confirmation (failed breakout).")
                return ev
            near = (ctx.l[j] - lvl) if bull else (lvl - ctx.h[j])
            if j > confirmed_at and near <= cfg.retest_atr * ctx.atr[j]:
                retest_seen_at = j
    if confirmed_at is None:
        lvl = level_fn(ctx.last)
        dist = sign * (lvl - ctx.c[ctx.last])
        ev.stage = PatternStage.APPROACHING_CONFIRMATION if dist <= cfg.approach_atr * ctx.atr[ctx.last] else PatternStage.FORMING
        return ev
    if retest_seen_at is not None and ctx.last - retest_seen_at <= 2 and not ev.target_reached:
        ev.stage = PatternStage.RETESTING
        ev.notes.append("Price is retesting the breakout level.")
    else:
        ev.stage = PatternStage.CONFIRMED
    if retest_seen_at is not None:
        ev.notes.append("Breakout level has been retested at least once.")
    return ev


def volume_confirmation(ctx: AnalysisContext, rel: float | None) -> bool | None:
    if not ctx.volume_available or rel is None:
        return None
    return rel >= ctx.cfg.volume_confirm_mult


def quality(components: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Weighted mean of 0..1 components -> 0..100. Missing (None) components are skipped."""
    comps = {k: float(np.clip(v, 0, 1)) for k, v in components.items() if v is not None}
    if not comps:
        return 0.0
    w = {k: (weights or {}).get(k, 1.0) for k in comps}
    tot = sum(w.values())
    return round(100 * sum(comps[k] * w[k] for k in comps) / tot, 1)


def closeness(diff: float, tol: float) -> float:
    """1 when diff=0, linearly to 0 at diff>=tol."""
    if tol <= 0:
        return 0.0
    return float(max(0.0, 1.0 - abs(diff) / tol))


def stage_bonus(stage: PatternStage) -> float:
    return {PatternStage.CONFIRMED: 1.0, PatternStage.RETESTING: 1.0, PatternStage.APPROACHING_CONFIRMATION: 0.6,
            PatternStage.FORMING: 0.4, PatternStage.FAILED: 0.0, PatternStage.INVALIDATED: 0.0}[stage]


def build_result(
    ctx: AnalysisContext, *, pattern_id: str, name: str, category: PatternCategory, direction: Direction,
    ev: BreakoutEval, start: int, end: int, key_points: list[KeyPoint], lines: list[PatternLine],
    breakout_level: float | None, invalidation_level: float | None, target: float | None,
    components: dict[str, float], weights: dict[str, float] | None = None, evidence: list[str] | None = None,
    experimental: bool = False,
) -> PatternResult:
    vc = volume_confirmation(ctx, ev.rel_volume) if ev.confirmation_index is not None else None
    if ev.confirmation_index is not None:
        components = {**components, "volume": None if vc is None else (1.0 if vc else 0.3)}
    components = {**components, "confirmation": stage_bonus(ev.stage)}
    evidence = list(evidence or [])
    if ev.confirmation_index is not None:
        evidence.append(f"Confirmed by close on {ctx.ts(ev.confirmation_index).date()}"
                        + (f" with relative volume {ev.rel_volume:.2f}x" if ev.rel_volume is not None else " (volume n/a)"))
    evidence += ev.notes
    end_idx = max(end, ev.confirmation_index or end, ev.failed_index or end, ev.invalidation_index or end)
    end_idx = min(end_idx, ctx.last)
    return PatternResult(
        pattern_id=pattern_id, name=name, category=category, direction=direction, stage=ev.stage,
        timeframe=ctx.timeframe, start_index=int(start), end_index=int(end_idx), start_time=ctx.ts(start),
        end_time=ctx.ts(end_idx), key_points=key_points, lines=lines,
        breakout_level=_r(breakout_level), invalidation_level=_r(invalidation_level), target=_r(target),
        target_reached=ev.target_reached, confirmation_index=ev.confirmation_index,
        confirmation_time=ctx.ts(ev.confirmation_index) if ev.confirmation_index is not None else None,
        volume_confirmation=vc, relative_volume=None if ev.rel_volume is None else round(ev.rel_volume, 2),
        quality_score=quality(components, weights),
        quality_components={k: round(float(v), 3) for k, v in components.items() if v is not None},
        evidence=evidence, provisional=ev.provisional, experimental=experimental, detected_at_index=ctx.last,
    )


def _r(x: float | None) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


class PatternDetector(ABC):
    pattern_ids: tuple[str, ...] = ()
    category: PatternCategory = PatternCategory.REVERSAL
    experimental: bool = False

    @abstractmethod
    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        ...


def build_context(df: pd.DataFrame, cfg: PatternConfig | None = None, timeframe: str | None = None,
                  last_candle_complete: bool = True, swings: SwingSet | None = None) -> AnalysisContext:
    cfg = cfg or PatternConfig()
    if swings is None:
        swings = zigzag_atr(df, cfg.swing, causal_atr(df, cfg.swing.atr_period))
    return AnalysisContext(df=df, cfg=cfg, swings=swings, timeframe=timeframe, last_candle_complete=last_candle_complete)


def dedupe(results: list[PatternResult]) -> list[PatternResult]:
    """Drop overlapping duplicates of the same pattern type, keeping the best quality."""
    out: list[PatternResult] = []
    for r in sorted(results, key=lambda r: (-r.quality_score, -r.end_index)):
        clash = False
        for o in out:
            if o.pattern_id != r.pattern_id:
                continue
            ov = min(o.end_index, r.end_index) - max(o.start_index, r.start_index)
            span = min(o.end_index - o.start_index, r.end_index - r.start_index) or 1
            shared = {k.index for k in o.key_points} & {k.index for k in r.key_points}
            if ov / span > 0.6 or len(shared) >= 2:
                clash = True
                break
        if not clash:
            out.append(r)
    return sorted(out, key=lambda r: r.end_index)
