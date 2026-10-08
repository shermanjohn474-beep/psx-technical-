"""Harmonic (XABCD / ABCD) and Elliott-wave heuristics. **Experimental.**

Harmonic ratio definitions (tolerance ``tol`` applied to every bound):

=========  ============  ============  ============  ===================
pattern    AB / XA       BC / AB       CD / BC       AD / XA
=========  ============  ============  ============  ===================
Gartley    0.618         0.382-0.886   1.272-1.618   0.786
Bat        0.382-0.50    0.382-0.886   1.618-2.618   0.886
Butterfly  0.786         0.382-0.886   1.618-2.24    1.27-1.618
Crab       0.382-0.618   0.382-0.886   2.24-3.618    1.618
=========  ============  ============  ============  ===================

Cypher: AB/XA 0.382-0.618, XC/XA 1.272-1.414, D at 0.786 retracement of XC.
ABCD: BC/AB 0.618-0.786, CD/BC 1.272-1.618.

A completed pattern is CONFIRMED once price reverses ``confirm_frac`` of CD away from
D; INVALIDATED if price closes beyond D by more than ``invalid_frac`` of CD (or beyond X).
Elliott counts check only the three hard impulse rules and are always reported with
the caveat that alternative counts are plausible.
"""
from __future__ import annotations

from app.chart_patterns.base import AnalysisContext, BreakoutEval, PatternDetector, build_result
from app.schemas.patterns import Direction, PatternCategory, PatternResult, PatternStage

SPECS = {
    "gartley": {"ab": (0.618, 0.618), "bc": (0.382, 0.886), "cd": (1.272, 1.618), "ad": (0.786, 0.786)},
    "bat": {"ab": (0.382, 0.5), "bc": (0.382, 0.886), "cd": (1.618, 2.618), "ad": (0.886, 0.886)},
    "butterfly": {"ab": (0.786, 0.786), "bc": (0.382, 0.886), "cd": (1.618, 2.24), "ad": (1.27, 1.618)},
    "crab": {"ab": (0.382, 0.618), "bc": (0.382, 0.886), "cd": (2.24, 3.618), "ad": (1.618, 1.618)},
}


def _within(v: float, lo: float, hi: float, tol: float) -> bool:
    return lo * (1 - tol) <= v <= hi * (1 + tol)


def _resolve(ctx: AnalysisContext, d_idx: int, d_px: float, cd: float, bullish: bool, x_px: float | None,
             confirm_frac: float = 0.382, invalid_frac: float = 0.25) -> BreakoutEval:
    ev = BreakoutEval(stage=PatternStage.FORMING)
    s = 1 if bullish else -1
    trig = d_px + s * confirm_frac * cd
    for j in range(d_idx + 1, ctx.n):
        c = ctx.c[j]
        if s * (d_px - c) > invalid_frac * cd or (x_px is not None and s * (x_px - c) > 0):
            ev.stage, ev.invalidation_index = PatternStage.INVALIDATED, j
            ev.notes.append("Price extended beyond the potential reversal zone.")
            return ev
        if s * (c - trig) > 0 and (j < ctx.last or ctx.last_candle_complete):
            ev.stage, ev.confirmation_index = PatternStage.CONFIRMED, j
            ev.rel_volume = ctx.volume_at(j)
            return ev
    ev.stage = PatternStage.APPROACHING_CONFIRMATION if s * (ctx.c[ctx.last] - d_px) > 0 else PatternStage.FORMING
    return ev


class HarmonicDetector(PatternDetector):
    pattern_ids = tuple(f"{p}_{d}" for p in (*SPECS, "cypher", "abcd") for d in ("bullish", "bearish"))
    category = PatternCategory.HARMONIC
    experimental = True

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        tol = ctx.cfg.p("harmonic", "tolerance", 0.05)
        sw = ctx.confirmed_swings()
        out = []
        for i in range(len(sw) - 4):
            X, A, B, C, D = sw[i:i + 5]
            xa, ab, bc, cd = abs(A.price - X.price), abs(B.price - A.price), abs(C.price - B.price), abs(D.price - C.price)
            if min(xa, ab, bc, cd) <= 0:
                continue
            bullish = D.kind == "L"
            r_ab, r_bc, r_cd, r_ad = ab / xa, bc / ab, cd / bc, abs(A.price - D.price) / xa
            for name, sp in SPECS.items():
                if _within(r_ab, *sp["ab"], tol) and _within(r_bc, *sp["bc"], tol) and _within(r_cd, *sp["cd"], tol) \
                        and _within(r_ad, *sp["ad"], tol):
                    out.append(self._result(ctx, name, [X, A, B, C, D], bullish, cd,
                                            f"AB/XA={r_ab:.3f} BC/AB={r_bc:.3f} CD/BC={r_cd:.3f} AD/XA={r_ad:.3f}"))
            xc = abs(C.price - X.price) / xa
            xd = abs(C.price - D.price) / max(abs(C.price - X.price), 1e-9)
            if _within(r_ab, 0.382, 0.618, tol) and _within(xc, 1.272, 1.414, tol) and _within(xd, 0.786, 0.786, tol):
                out.append(self._result(ctx, "cypher", [X, A, B, C, D], bullish, cd,
                                        f"AB/XA={r_ab:.3f} XC/XA={xc:.3f} CD/XC={xd:.3f}"))
        for i in range(len(sw) - 3):
            A, B, C, D = sw[i:i + 4]
            ab, bc, cd = abs(B.price - A.price), abs(C.price - B.price), abs(D.price - C.price)
            if min(ab, bc, cd) <= 0:
                continue
            if _within(bc / ab, 0.618, 0.786, tol) and _within(cd / bc, 1.272, 1.618, tol):
                out.append(self._result(ctx, "abcd", [A, B, C, D], D.kind == "L", cd,
                                        f"BC/AB={bc / ab:.3f} CD/BC={cd / bc:.3f} CD/AB={cd / ab:.3f}"))
        return [r for r in out if r.end_index >= ctx.last - ctx.cfg.max_age_bars]

    def _result(self, ctx, name, pts, bullish, cd, ratios) -> PatternResult:
        D = pts[-1]
        x_px = pts[0].price if len(pts) == 5 else None
        ev = _resolve(ctx, D.index, D.price, cd, bullish, x_px if name not in ("butterfly", "crab") else None)
        s = 1 if bullish else -1
        labels = ["X", "A", "B", "C", "D"][-len(pts):]
        return build_result(
            ctx, pattern_id=f"{name}_{'bullish' if bullish else 'bearish'}",
            name=f"{'Bullish' if bullish else 'Bearish'} {name.upper() if name == 'abcd' else name.title()}",
            category=PatternCategory.HARMONIC, direction=Direction.BULLISH if bullish else Direction.BEARISH, ev=ev,
            start=pts[0].index, end=D.index, key_points=[ctx.kp(lab, p.index, p.price) for lab, p in zip(labels, pts)],
            lines=[ctx.line(f"{labels[k]}{labels[k + 1]}", a.index, a.price, b.index, b.price) for k, (a, b) in enumerate(zip(pts, pts[1:]))],
            breakout_level=D.price + s * 0.382 * cd, invalidation_level=D.price - s * 0.25 * cd,
            target=D.price + s * 0.618 * cd, components={"ratio_fit": 0.7},
            evidence=[ratios, "Harmonic patterns are experimental: ratio matches are sensitive to swing selection."],
            experimental=True,
        )


class ElliottDetector(PatternDetector):
    """Impulse candidate on six alternating swings 0-1-2-3-4-5 satisfying the hard rules:
    wave 2 does not retrace beyond wave 0; wave 3 is not the shortest of 1/3/5; wave 4
    does not overlap wave 1's price territory. Always experimental."""

    pattern_ids = ("elliott_impulse_up", "elliott_impulse_down")
    category = PatternCategory.ADVANCED
    experimental = True

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        sw = ctx.confirmed_swings()
        out = []
        for i in range(max(0, len(sw) - 12), len(sw) - 5):
            p = sw[i:i + 6]
            up = p[0].kind == "L"
            s = 1 if up else -1
            v = [s * x.price for x in p]
            w1, w3, w5 = v[1] - v[0], v[3] - v[2], v[5] - v[4]
            if min(w1, w3, w5) <= 0:
                continue
            if v[2] <= v[0] or w3 < min(w1, w5) or v[4] <= v[1]:
                continue
            if w1 < 2 * ctx.atr[p[1].index]:
                continue
            ev = BreakoutEval(stage=PatternStage.FORMING, notes=[
                "Impulse rules satisfied; a corrective (ABC) phase may follow. Alternative counts are plausible."])
            out.append(build_result(
                ctx, pattern_id="elliott_impulse_up" if up else "elliott_impulse_down",
                name=f"Elliott Impulse Candidate ({'up' if up else 'down'})", category=PatternCategory.ADVANCED,
                direction=Direction.BEARISH if up else Direction.BULLISH, ev=ev, start=p[0].index, end=p[5].index,
                key_points=[ctx.kp(str(k), x.index, x.price) for k, x in enumerate(p)],
                lines=[ctx.line(f"W{k + 1}", a.index, a.price, b.index, b.price) for k, (a, b) in enumerate(zip(p, p[1:]))],
                breakout_level=p[4].price, invalidation_level=p[5].price,
                target=p[5].price - s * 0.382 * (v[5] - v[0]),
                components={"wave3_extension": min(1.0, w3 / max(w1, 1e-9) / 1.618)},
                evidence=[f"W1={w1:.4g} W3={w3:.4g} W5={w5:.4g} (price units)",
                          "Heuristic count only — never an objective confirmation of wave position."],
                experimental=True,
            ))
        return out
