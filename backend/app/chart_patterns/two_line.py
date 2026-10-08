"""Two-boundary formations fitted to alternating swing points.

One geometric engine covers triangles, wedges, channels, rectangles and
broadening structures. For a run of alternating swings we fit an upper line
through swing highs and a lower line through swing lows (least squares), then:

* reject fits whose touch residuals exceed ``touch_tol_atr`` ATR,
* reject formations with too many closes outside the boundaries (containment),
* compute each line's total change across the formation in ATR units
  (``|change| <= flat_atr`` => flat), and the width ratio end/start
  (``< converge_ratio`` converging, ``> diverge_ratio`` diverging, else parallel),
* classify:

==========================  ==========  ===========  ======================
pattern                     upper       lower        width
==========================  ==========  ===========  ======================
ascending_triangle          flat        rising       converging
descending_triangle         falling     flat         converging
symmetrical_triangle        falling     rising       converging
rising_wedge                rising      rising       converging
falling_wedge               falling     falling      converging
ascending_channel           rising      rising       parallel
descending_channel          falling     falling      parallel
rectangle                   flat        flat         parallel (+ prior trend)
horizontal_channel          flat        flat         parallel (no prior trend)
expanding_triangle          rising      falling      diverging, symmetric
broadening_top / _bottom    rising      falling      diverging after up/down trend
broadening_formation        other diverging combinations (right-angled)
==========================  ==========  ===========  ======================

Symmetrical triangles, channels, rectangles and broadening structures are
*bilateral*: direction is assigned only from the actual breakout, never from shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.chart_patterns.base import (
    AnalysisContext, BreakoutEval, PatternDetector, build_result, evaluate_breakout,
)
from app.schemas.patterns import Direction, PatternCategory, PatternResult, PatternStage
from app.swing_detection.swings import Swing

NAMES = {
    "ascending_triangle": "Ascending Triangle", "descending_triangle": "Descending Triangle",
    "symmetrical_triangle": "Symmetrical Triangle", "rising_wedge": "Rising Wedge", "falling_wedge": "Falling Wedge",
    "ascending_channel": "Ascending Channel", "descending_channel": "Descending Channel",
    "rectangle": "Rectangle Consolidation", "horizontal_channel": "Horizontal Channel",
    "expanding_triangle": "Expanding Triangle", "broadening_top": "Broadening Top",
    "broadening_bottom": "Broadening Bottom", "broadening_formation": "Broadening Formation",
}
# expected direction (None = bilateral, resolved by breakout)
EXPECTED = {
    "ascending_triangle": Direction.BULLISH, "descending_triangle": Direction.BEARISH,
    "rising_wedge": Direction.BEARISH, "falling_wedge": Direction.BULLISH,
}
CATEGORY = {
    "broadening_top": PatternCategory.REVERSAL, "broadening_bottom": PatternCategory.REVERSAL,
    "rising_wedge": PatternCategory.REVERSAL, "falling_wedge": PatternCategory.REVERSAL,
    "symmetrical_triangle": PatternCategory.BILATERAL, "expanding_triangle": PatternCategory.BILATERAL,
    "broadening_formation": PatternCategory.BILATERAL, "horizontal_channel": PatternCategory.BILATERAL,
}


@dataclass
class Fit:
    first: int
    last: int
    mu: float
    bu: float
    ml: float
    bl: float
    highs: list[Swing]
    lows: list[Swing]
    atr: float
    max_resid: float
    violations: float

    def upper(self, j: float) -> float:
        return self.mu * j + self.bu

    def lower(self, j: float) -> float:
        return self.ml * j + self.bl

    @property
    def bars(self) -> int:
        return self.last - self.first


def _line(points: list[Swing]) -> tuple[float, float, float]:
    x = np.array([p.index for p in points], dtype=float)
    y = np.array([p.price for p in points], dtype=float)
    m, b = np.polyfit(x, y, 1)
    resid = float(np.max(np.abs(y - (m * x + b))))
    return float(m), float(b), resid


def fit_two_lines(ctx: AnalysisContext, seg: list[Swing], touch_tol_atr: float, max_violation_frac: float) -> Fit | None:
    highs = [s for s in seg if s.kind == "H"]
    lows = [s for s in seg if s.kind == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return None
    first, last = seg[0].index, seg[-1].index
    atr = float(np.median(ctx.atr[first:last + 1]))
    mu, bu, ru = _line(highs)
    ml, bl, rl = _line(lows)
    if max(ru, rl) > touch_tol_atr * atr:
        return None
    js = np.arange(first, last + 1)
    up, lo = mu * js + bu, ml * js + bl
    if np.any(up - lo <= 0):
        return None
    tol = 0.35 * atr
    c = ctx.c[first:last + 1]
    viol = float(np.mean((c > up + tol) | (c < lo - tol)))
    if viol > max_violation_frac:
        return None
    return Fit(first, last, mu, bu, ml, bl, highs, lows, atr, max(ru, rl), viol)


def classify(fit: Fit, prior_atr: float, cfg) -> str | None:
    flat = cfg.p("two_line", "flat_atr", 1.2)
    conv = cfg.p("two_line", "converge_ratio", 0.72)
    div = cfg.p("two_line", "diverge_ratio", 1.4)
    du = fit.mu * fit.bars / fit.atr
    dl = fit.ml * fit.bars / fit.atr
    su = "flat" if abs(du) <= flat else ("up" if du > 0 else "down")
    sl = "flat" if abs(dl) <= flat else ("up" if dl > 0 else "down")
    w0 = fit.upper(fit.first) - fit.lower(fit.first)
    w1 = fit.upper(fit.last) - fit.lower(fit.last)
    ratio = w1 / w0
    if ratio < conv:
        return {("flat", "up"): "ascending_triangle", ("down", "flat"): "descending_triangle",
                ("down", "up"): "symmetrical_triangle", ("up", "up"): "rising_wedge",
                ("down", "down"): "falling_wedge"}.get((su, sl))
    if ratio <= div:
        if su == sl == "up":
            return "ascending_channel"
        if su == sl == "down":
            return "descending_channel"
        if su == sl == "flat":
            return "rectangle" if abs(prior_atr) >= cfg.min_trend_atr else "horizontal_channel"
        return None
    # diverging
    if su == "up" and sl == "down":
        if prior_atr >= cfg.min_trend_atr:
            return "broadening_top"
        if prior_atr <= -cfg.min_trend_atr:
            return "broadening_bottom"
        if abs(abs(du) - abs(dl)) <= 0.5 * max(abs(du), abs(dl)):
            return "expanding_triangle"
        return "broadening_formation"
    if (su, sl) in (("up", "flat"), ("flat", "down")):
        return "broadening_formation"
    return None


def evaluate_two_sided(ctx: AnalysisContext, fit: Fit, t_up: float, t_down: float) -> tuple[BreakoutEval, Direction]:
    up = evaluate_breakout(ctx, fit.last + 1, fit.upper, Direction.BULLISH, None, t_up)
    dn = evaluate_breakout(ctx, fit.last + 1, fit.lower, Direction.BEARISH, None, t_down)
    cu, cd = up.confirmation_index, dn.confirmation_index
    if cu is not None and (cd is None or cu <= cd):
        return up, Direction.BULLISH
    if cd is not None:
        return dn, Direction.BEARISH
    for ev in (up, dn):
        if ev.provisional:
            return ev, Direction.BULLISH if ev is up else Direction.BEARISH
    if up.stage == PatternStage.INVALIDATED and dn.stage == PatternStage.INVALIDATED:
        return up, Direction.NEUTRAL
    if up.stage == PatternStage.APPROACHING_CONFIRMATION:
        return up, Direction.NEUTRAL
    if dn.stage == PatternStage.APPROACHING_CONFIRMATION:
        return dn, Direction.NEUTRAL
    return up, Direction.NEUTRAL


class TwoLineDetector(PatternDetector):
    pattern_ids = tuple(NAMES)
    category = PatternCategory.CONTINUATION

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        cfg = ctx.cfg
        touch_tol = cfg.p("two_line", "touch_tol_atr", 0.9)
        max_viol = cfg.p("two_line", "max_violation_frac", 0.08)
        min_bars = cfg.p("two_line", "min_bars", 12)
        max_bars = cfg.p("two_line", "max_bars", 220)
        max_anchor = cfg.p("two_line", "anchor_swings", 10)
        min_touches = cfg.p("two_line", "min_touches", 5)
        sw = ctx.confirmed_swings()
        out: list[PatternResult] = []
        for end in range(len(sw) - 1, max(-1, len(sw) - 1 - max_anchor), -1):
            for count in range(9, min_touches - 1, -1):
                start = end - count + 1
                if start < 0:
                    continue
                seg = sw[start:end + 1]
                bars = seg[-1].index - seg[0].index
                if not (min_bars <= bars <= max_bars):
                    continue
                fit = fit_two_lines(ctx, seg, touch_tol, max_viol)
                if fit is None:
                    continue
                prior = ctx.prior_extreme_move_atr(fit.first)
                pid = classify(fit, prior, cfg)
                if pid is None:
                    continue
                res = self._result(ctx, pid, fit, prior)
                if res is not None:
                    out.append(res)
                    break  # largest valid formation for this anchor
        return [r for r in out if r.end_index >= ctx.last - cfg.max_age_bars]

    def _result(self, ctx: AnalysisContext, pid: str, fit: Fit, prior: float) -> PatternResult | None:
        w0 = fit.upper(fit.first) - fit.lower(fit.first)
        w1 = fit.upper(fit.last) - fit.lower(fit.last)
        is_tri = pid.endswith("triangle") and pid != "expanding_triangle"
        apex = None
        if is_tri and fit.mu != fit.ml:
            apex = (fit.bl - fit.bu) / (fit.mu - fit.ml)
            if apex <= fit.last:
                return None
        if pid in ("rising_wedge", "falling_wedge"):
            t_up, t_down = fit.upper(fit.first), fit.lower(fit.first)
            measure = w0
        elif is_tri:
            measure = w0
            t_up, t_down = None, None
        else:
            measure = w1
            t_up, t_down = None, None
        expected = EXPECTED.get(pid)
        # breakout levels at the current bar (or confirmation bar, below)
        if expected is None:
            t_up_v = t_up if t_up is not None else fit.upper(ctx.last) + measure
            t_dn_v = t_down if t_down is not None else fit.lower(ctx.last) - measure
            ev, direction = evaluate_two_sided(ctx, fit, t_up_v, t_dn_v)
        elif expected == Direction.BULLISH:
            tgt = t_up if t_up is not None else fit.upper(ctx.last) + measure
            ev = evaluate_breakout(ctx, fit.last + 1, fit.upper, Direction.BULLISH, fit.lower, tgt)
            direction = Direction.BULLISH
        else:
            tgt = t_down if t_down is not None else fit.lower(ctx.last) - measure
            ev = evaluate_breakout(ctx, fit.last + 1, fit.lower, Direction.BEARISH, fit.upper, tgt)
            direction = Direction.BEARISH
        if apex is not None and ev.confirmation_index is None and ctx.last > apex:
            ev.stage = PatternStage.INVALIDATED
            ev.notes.append("Price drifted past the triangle apex without a breakout (pattern lost validity).")
        b_idx = ev.confirmation_index if ev.confirmation_index is not None else ctx.last
        if direction == Direction.BEARISH:
            level = fit.lower(b_idx)
            target = t_down if t_down is not None else level - measure
            invalid = fit.upper(b_idx)
        elif direction == Direction.BULLISH:
            level = fit.upper(b_idx)
            target = t_up if t_up is not None else level + measure
            invalid = fit.lower(b_idx)
        else:
            level, target, invalid = None, None, None
        touches = len(fit.highs) + len(fit.lows)
        comps = {
            "touches": min(1.0, 0.4 + 0.15 * (touches - 4)),
            "fit": 1 - fit.max_resid / (ctx.cfg.p("two_line", "touch_tol_atr", 0.9) * fit.atr),
            "containment": 1 - fit.violations / 0.08,
            "duration": 1.0 if 20 <= fit.bars <= 150 else 0.6,
        }
        if expected is not None or pid in ("rectangle", "ascending_channel", "descending_channel"):
            # continuation patterns should agree with the preceding trend
            bias = {"ascending_triangle": 1, "falling_wedge": 1, "descending_triangle": -1, "rising_wedge": -1}.get(pid)
            if bias is not None:
                comps["trend_context"] = 1.0 if bias * prior > 0 else 0.5
        if is_tri and apex is not None and ev.confirmation_index is not None:
            frac = (ev.confirmation_index - fit.first) / (apex - fit.first)
            comps["breakout_timing"] = 1.0 if 0.5 <= frac <= 0.85 else 0.6
        if ctx.volume_available:
            v = ctx.df["volume"].to_numpy()[fit.first:fit.last + 1]
            half = len(v) // 2
            if half > 2:
                comps["volume_contraction"] = 1.0 if np.nanmean(v[half:]) < np.nanmean(v[:half]) else 0.5
        end = ev.confirmation_index or ctx.last
        kps = [ctx.kp(f"R{k + 1}", s.index, s.price) for k, s in enumerate(fit.highs)] + \
              [ctx.kp(f"S{k + 1}", s.index, s.price) for k, s in enumerate(fit.lows)]
        lines = [ctx.line("Upper boundary", fit.first, fit.upper(fit.first), end, fit.upper(end)),
                 ctx.line("Lower boundary", fit.first, fit.lower(fit.first), end, fit.lower(end))]
        evidence = [
            f"{len(fit.highs)} upper / {len(fit.lows)} lower touches over {fit.bars} bars "
            f"(max residual {fit.max_resid / fit.atr:.2f} ATR)",
            f"Upper slope {fit.mu * fit.bars / fit.atr:+.1f} ATR, lower slope {fit.ml * fit.bars / fit.atr:+.1f} ATR "
            f"across formation; width {w0:.4g} -> {w1:.4g}",
            f"Preceding move {prior:+.1f} ATR",
        ]
        if expected is None:
            evidence.append("Bilateral formation: direction is taken only from the actual breakout, not the shape.")
            if direction != Direction.NEUTRAL and abs(prior) >= ctx.cfg.min_trend_atr:
                cont = (prior > 0) == (direction == Direction.BULLISH)
                evidence.append(f"Breakout {'continues' if cont else 'reverses'} the preceding trend.")
        category = CATEGORY.get(pid, PatternCategory.CONTINUATION)
        return build_result(
            ctx, pattern_id=pid, name=NAMES[pid], category=category, direction=direction, ev=ev,
            start=fit.first, end=fit.last, key_points=sorted(kps, key=lambda k: k.index), lines=lines,
            breakout_level=level, invalidation_level=invalid, target=target, components=comps, evidence=evidence,
        )
