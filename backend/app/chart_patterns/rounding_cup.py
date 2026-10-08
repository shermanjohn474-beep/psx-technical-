"""Rounding bottom/top and cup & handle (normal and inverted).

Computed in "bullish space" (tops are mirrored), so one implementation serves both.

Rounding bottom:
  * left rim = confirmed swing high ``R``; bottom = lowest low after ``R``.
  * the right side must recover to within ``approach_atr`` ATR of the rim (``e`` = that bar).
  * closes on ``[R, e]`` fit a quadratic with positive curvature, R^2 >= ``min_r2``,
    vertex in the middle ``[0.25, 0.75]`` of the span, and >= ``min_base_frac`` of bars
    in the lower third of the depth (rounded base, not a V).
  * depth >= ``min_depth_atr`` ATR and >= ``min_height_pct`` of price, <= ``max_depth_pct``.
  * confirmation: completed close above the rim. Invalidation: close below the bottom.
  * target: rim + depth.

Cup & handle: a rounding bottom whose right rim is followed (before breakout) by a
handle pullback of ``<= max_handle_frac`` of cup depth lasting ``[min_handle, max_handle]``
bars and staying in the upper half of the cup. Trigger = highest of the two rims.
"""
from __future__ import annotations

import numpy as np

from app.chart_patterns.base import AnalysisContext, PatternDetector, build_result, evaluate_breakout
from app.schemas.patterns import Direction, PatternCategory, PatternResult


def oriented(ctx: AnalysisContext, bull: bool):
    s = 1.0 if bull else -1.0
    H = ctx.h if bull else -ctx.l
    L = ctx.l if bull else -ctx.h
    return H, L, ctx.c * s, s


class RoundingCupDetector(PatternDetector):
    pattern_ids = ("rounding_bottom", "rounding_top", "cup_and_handle", "inverted_cup_and_handle")
    category = PatternCategory.REVERSAL

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        out = []
        for bull in (True, False):
            out += self._scan(ctx, bull)
        return [r for r in out if r.end_index >= ctx.last - ctx.cfg.max_age_bars]

    def _scan(self, ctx: AnalysisContext, bull: bool) -> list[PatternResult]:
        cfg = ctx.cfg
        g = lambda k, d: cfg.p("rounding", k, d)  # noqa: E731
        min_bars, max_bars = g("min_bars", 30), g("max_bars", 200)
        min_r2, min_base = g("min_r2", 0.7), g("min_base_frac", 0.42)
        min_depth_atr, max_depth_pct = g("min_depth_atr", 4.0), g("max_depth_pct", 50.0)
        max_handle_frac, min_handle, max_handle = g("max_handle_frac", 0.5), g("min_handle", 5), g("max_handle", 30)
        H, L, C, s = oriented(ctx, bull)
        rims = [sw for sw in ctx.confirmed_swings() if sw.kind == ("H" if bull else "L")]
        out = []
        for rim in rims:
            r0 = rim.index
            rim_px = H[r0]
            if ctx.last - r0 < min_bars:
                continue
            seg_end = min(ctx.last, r0 + max_bars)
            # bottom must be the low of the segment up to the right-rim bar
            e = None
            b = r0 + 1 + int(np.argmin(L[r0 + 1:seg_end + 1]))
            for j in range(b + 1, seg_end + 1):
                if H[j] >= rim_px - cfg.approach_atr * ctx.atr[j]:
                    e = j
                    break
            if e is None or e - r0 < min_bars:
                continue
            b = r0 + int(np.argmin(L[r0:e + 1]))
            depth = rim_px - L[b]
            atr = float(np.median(ctx.atr[r0:e + 1]))
            depth_pct = 100 * depth / abs(s * rim_px)
            if depth < min_depth_atr * atr or depth_pct < cfg.min_height_pct or depth_pct > max_depth_pct:
                continue
            x = np.arange(r0, e + 1, dtype=float)
            y = C[r0:e + 1]
            a2, a1, a0 = np.polyfit(x - r0, y, 2)
            if a2 <= 0:
                continue
            fitted = np.polyval([a2, a1, a0], x - r0)
            r2 = 1 - np.sum((y - fitted) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
            vertex = -a1 / (2 * a2) / (e - r0)
            base_frac = float(np.mean(y <= L[b] + depth / 3))
            if r2 < min_r2 or not (0.25 <= vertex <= 0.75) or base_frac < min_base:
                continue
            prior = s * ctx.prior_extreme_move_atr(r0)
            # handle search: from e, look for a shallow pullback then breakout above max rim
            # right rim = running high from e until a >= 1.5 ATR pullback begins (before any breakout)
            right_rim, pulled = e, False
            for j in range(e, min(ctx.last, e + max_handle) + 1):
                if C[j] > rim_px + cfg.breakout_tol_atr * ctx.atr[j]:
                    break
                if H[j] > H[right_rim]:
                    right_rim = j
                elif H[right_rim] - L[j] >= 1.5 * ctx.atr[j]:
                    pulled = True
                    break
            handle = None
            if pulled and ctx.last - right_rim >= min_handle:
                trigger = max(rim_px, H[right_rim])
                h_end = min(ctx.last, right_rim + max_handle)
                brk = None
                for j in range(right_rim + 1, h_end + 1):
                    if C[j] > trigger + cfg.breakout_tol_atr * ctx.atr[j]:
                        brk = j
                        break
                stop = brk if brk is not None else h_end + 1
                if stop - right_rim - 1 >= min_handle:
                    h_low_i = right_rim + 1 + int(np.argmin(L[right_rim + 1:stop]))
                    pull = trigger - L[h_low_i]
                    if 0 < pull <= max_handle_frac * depth and L[h_low_i] > L[b] + depth / 2:
                        handle = (right_rim, h_low_i, trigger, pull)
            for kind in (["cup"] if handle else []) + ["round"]:
                if kind == "cup":
                    rr, hl, trigger, pull = handle
                    start_eval = rr + 1
                    invalid = L[hl]
                    pid = "cup_and_handle" if bull else "inverted_cup_and_handle"
                    name = "Cup and Handle" if bull else "Inverted Cup and Handle"
                else:
                    trigger, start_eval, invalid = rim_px, e, L[b]
                    pid = "rounding_bottom" if bull else "rounding_top"
                    name = "Rounding Bottom" if bull else "Rounding Top"
                target = trigger + depth
                direction = Direction.BULLISH if bull else Direction.BEARISH
                ev = evaluate_breakout(ctx, start_eval, lambda j, t=trigger: s * t, direction, s * invalid, s * target)
                comps = {"fit_r2": (r2 - min_r2) / (1 - min_r2), "base_roundness": min(1.0, base_frac / 0.55),
                         "vertex_centred": 1 - abs(vertex - 0.5) / 0.25, "depth": min(1.0, depth / (8 * atr)),
                         "prior_trend": 1.0 if prior < 0 else 0.5}
                if kind == "cup":
                    comps["handle_depth"] = 1.0 if handle[3] <= depth / 3 else 0.6
                kps = [ctx.kp("Left rim", r0, s * rim_px), ctx.kp("Bottom" if bull else "Top", b, s * L[b]),
                       ctx.kp("Right rim", e, s * H[e])]
                if kind == "cup":
                    kps.append(ctx.kp("Handle low" if bull else "Handle high", handle[1], s * L[handle[1]]))
                end = ev.confirmation_index or ctx.last
                out.append(build_result(
                    ctx, pattern_id=pid, name=name, category=PatternCategory.REVERSAL, direction=direction, ev=ev,
                    start=r0, end=max(e, handle[1] if kind == "cup" else e), key_points=kps,
                    lines=[ctx.line("Rim / trigger", r0, s * trigger, end, s * trigger, "dashed")],
                    breakout_level=s * trigger, invalidation_level=s * invalid, target=s * target, components=comps,
                    evidence=[f"Quadratic base fit R^2={r2:.2f}, vertex at {vertex:.0%} of span, {base_frac:.0%} of bars in lower third",
                              f"Depth {depth_pct:.1f}% ({depth / atr:.1f} ATR) over {e - r0} bars"]
                    + ([f"Handle pullback {handle[3] / depth:.0%} of cup depth"] if kind == "cup" else []),
                ))
        return out
