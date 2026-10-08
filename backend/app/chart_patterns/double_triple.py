"""Double / triple tops and bottoms, plus Adam & Eve variants.

Geometric definition (double top; bottoms are the mirror image):

1. Swing sequence ``H1 - T - H2`` (alternating, confirmed except possibly ``H2``).
2. ``|H1 - H2| <= equal_level_atr x ATR`` (configurable).
3. Intervening trough depth ``min(H1, H2) - T >= min_depth_atr x ATR``.
4. Peak separation within ``[min_sep, max_sep]`` bars.
5. Preceding advance into ``H1`` of at least ``min_prior_atr`` ATR.
6. Neckline = trough ``T``. Confirmation = completed close below neckline by
   ``breakout_tol_atr``. Invalidation (pre-confirmation) = close above the higher peak.
7. Measured move target = neckline - (peak - neckline).
"""
from __future__ import annotations

import numpy as np

from app.chart_patterns.base import (
    AnalysisContext, PatternDetector, build_result, closeness, evaluate_breakout,
)
from app.schemas.patterns import Direction, PatternCategory, PatternResult, PatternStage
from app.swing_detection.swings import Swing


def _extreme_width(ctx: AnalysisContext, idx: int, top: bool, band_atr: float, max_span: int = 30) -> int:
    """Bars around ``idx`` whose extreme stays within ``band_atr`` ATR of the pivot price."""
    a = ctx.atr[idx]
    ref = ctx.h[idx] if top else ctx.l[idx]
    width = 1
    for step in (-1, 1):
        j = idx + step
        while 0 <= j <= ctx.last and abs(j - idx) <= max_span:
            v = ctx.h[j] if top else ctx.l[j]
            if (ref - v if top else v - ref) <= band_atr * a:
                width += 1
                j += step
            else:
                break
    return width


class DoubleTripleDetector(PatternDetector):
    pattern_ids = ("double_top", "double_bottom", "triple_top", "triple_bottom", "adam_eve_top", "adam_eve_bottom")
    category = PatternCategory.REVERSAL

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        out: list[PatternResult] = []
        sw = ctx.swings_with_provisional()
        for top in (True, False):
            out += self._double(ctx, sw, top)
            out += self._triple(ctx, sw, top)
        return [r for r in out if r.end_index >= ctx.last - ctx.cfg.max_age_bars]

    # ------------------------------------------------------------------
    def _params(self, ctx: AnalysisContext, pid: str) -> dict:
        p = ctx.cfg
        return {
            "eq": p.p(pid, "equal_level_atr", p.equal_level_atr),
            "depth": p.p(pid, "min_depth_atr", 2.5),
            "min_sep": p.p(pid, "min_sep", 8),
            "max_sep": p.p(pid, "max_sep", 120),
            "prior": p.p(pid, "min_prior_atr", 0.5 * p.min_trend_atr),
        }

    def _double(self, ctx: AnalysisContext, sw: list[Swing], top: bool) -> list[PatternResult]:
        pid = "double_top" if top else "double_bottom"
        P = self._params(ctx, pid)
        peak_kind = "H" if top else "L"
        s = 1.0 if top else -1.0
        out = []
        for i in range(len(sw) - 2):
            a, t, b = sw[i], sw[i + 1], sw[i + 2]
            if a.kind != peak_kind or t.kind == peak_kind or b.kind != peak_kind:
                continue
            atr = ctx.atr[b.index]
            tol = P["eq"] * atr
            diff = a.price - b.price
            if abs(diff) > tol:
                continue
            extreme = max(a.price, b.price) if top else min(a.price, b.price)
            depth = s * (min(a.price, b.price) if top else max(a.price, b.price)) - s * t.price
            if depth < P["depth"] * atr or depth < ctx.cfg.min_height_pct / 100 * abs(t.price):
                continue
            sep = b.index - a.index
            if not (P["min_sep"] <= sep <= P["max_sep"]):
                continue
            prior = s * ctx.prior_extreme_move_atr(a.index)
            if prior < P["prior"]:
                continue
            neck = t.price
            height = s * (extreme - neck)
            target = neck - s * height
            direction = Direction.BEARISH if top else Direction.BULLISH
            ev = evaluate_breakout(ctx, b.index + 1, lambda j: neck, direction, invalidation=extreme, target=target)
            if b.provisional and ev.confirmation_index is not None:
                continue  # cannot confirm on an unconfirmed pivot
            comps = {
                "symmetry": closeness(diff, tol),
                "depth": min(1.0, depth / (5 * atr)),
                "prior_trend": min(1.0, prior / (2 * ctx.cfg.min_trend_atr)),
                "spacing": 1.0 if 15 <= sep <= 80 else 0.6,
            }
            if ctx.volume_available:
                va = np.nanmean(ctx.df["volume"].to_numpy()[max(0, a.index - 2):a.index + 3])
                vb = np.nanmean(ctx.df["volume"].to_numpy()[max(0, b.index - 2):b.index + 3])
                comps["volume_divergence"] = 1.0 if vb < va else 0.5
            label = "Top" if top else "Bottom"
            end = ev.confirmation_index or ctx.last
            lines = [
                ctx.line("Neckline", a.index, neck, end, neck, "dashed"),
                ctx.line("Resistance" if top else "Support", a.index, a.price, b.index, b.price, "dotted"),
            ]
            kps = [ctx.kp(f"{label} 1", a.index, a.price), ctx.kp("Neckline", t.index, t.price), ctx.kp(f"{label} 2", b.index, b.price)]
            evidence = [
                f"Two {'highs' if top else 'lows'} {a.price:.4g} / {b.price:.4g} within {P['eq']} ATR ({sep} bars apart)",
                f"Intervening {'trough' if top else 'peak'} {neck:.4g}, depth {depth / atr:.1f} ATR",
                f"Prior {'advance' if top else 'decline'} {prior:.1f} ATR",
            ]
            if b.provisional:
                evidence.append(f"Second {label.lower()} is not yet a confirmed swing (still forming).")
            res = build_result(
                ctx, pattern_id=pid, name=f"Double {label}", category=PatternCategory.REVERSAL, direction=direction,
                ev=ev, start=a.index, end=b.index, key_points=kps, lines=lines, breakout_level=neck,
                invalidation_level=extreme, target=target, components=comps, evidence=evidence,
            )
            out.append(res)
            out += self._adam_eve(ctx, res, a, b, top, ev, comps, kps, lines, neck, extreme, target, evidence)
        return out

    def _adam_eve(self, ctx, res, a, b, top, ev, comps, kps, lines, neck, extreme, target, evidence):
        pid = "adam_eve_top" if top else "adam_eve_bottom"
        band = ctx.cfg.p(pid, "band_atr", 1.0)
        wa = _extreme_width(ctx, a.index, top, band)
        wb = _extreme_width(ctx, b.index, top, band)
        adam_max = ctx.cfg.p(pid, "adam_max_width", 5)
        if wa <= adam_max and wb >= max(2 * wa, ctx.cfg.p(pid, "eve_min_width", 6)):
            r = res.model_copy(deep=True)
            r.pattern_id = pid
            r.name = f"Adam & Eve {'Top' if top else 'Bottom'}"
            r.evidence = evidence + [f"First extreme narrow/V-shaped ({wa} bars), second rounded ({wb} bars)",
                                     "Adam & Eve classification is a width heuristic."]
            return [r]
        return []

    def _triple(self, ctx: AnalysisContext, sw: list[Swing], top: bool) -> list[PatternResult]:
        pid = "triple_top" if top else "triple_bottom"
        P = self._params(ctx, pid)
        peak_kind = "H" if top else "L"
        s = 1.0 if top else -1.0
        out = []
        for i in range(len(sw) - 4):
            seq = sw[i:i + 5]
            if [x.kind for x in seq[::2]] != [peak_kind] * 3 or any(x.kind == peak_kind for x in seq[1::2]):
                continue
            p1, t1, p2, t2, p3 = seq
            atr = ctx.atr[p3.index]
            tol = P["eq"] * atr * 1.2
            peaks = np.array([p1.price, p2.price, p3.price])
            if peaks.max() - peaks.min() > tol:
                continue
            neck = min(t1.price, t2.price) if top else max(t1.price, t2.price)
            extreme = peaks.max() if top else peaks.min()
            depth = s * (peaks.mean() - neck)
            if depth < P["depth"] * atr or depth < ctx.cfg.min_height_pct / 100 * abs(neck):
                continue
            span = p3.index - p1.index
            if not (2 * P["min_sep"] <= span <= 2 * P["max_sep"]):
                continue
            prior = s * ctx.prior_extreme_move_atr(p1.index)
            if prior < P["prior"]:
                continue
            target = neck - s * (s * (extreme - neck))
            direction = Direction.BEARISH if top else Direction.BULLISH
            ev = evaluate_breakout(ctx, p3.index + 1, lambda j: neck, direction, invalidation=extreme, target=target)
            if p3.provisional and ev.confirmation_index is not None:
                continue
            label = "Top" if top else "Bottom"
            comps = {"symmetry": closeness(peaks.max() - peaks.min(), tol), "depth": min(1.0, depth / (5 * atr)),
                     "prior_trend": min(1.0, prior / (2 * ctx.cfg.min_trend_atr)),
                     "trough_alignment": closeness(t1.price - t2.price, 2 * atr)}
            end = ev.confirmation_index or ctx.last
            out.append(build_result(
                ctx, pattern_id=pid, name=f"Triple {label}", category=PatternCategory.REVERSAL, direction=direction,
                ev=ev, start=p1.index, end=p3.index,
                key_points=[ctx.kp(f"{label} 1", p1.index, p1.price), ctx.kp("Trough 1" if top else "Peak 1", t1.index, t1.price),
                            ctx.kp(f"{label} 2", p2.index, p2.price), ctx.kp("Trough 2" if top else "Peak 2", t2.index, t2.price),
                            ctx.kp(f"{label} 3", p3.index, p3.price)],
                lines=[ctx.line("Neckline", p1.index, neck, end, neck, "dashed"),
                       ctx.line("Resistance" if top else "Support", p1.index, peaks.mean(), p3.index, peaks.mean(), "dotted")],
                breakout_level=neck, invalidation_level=extreme, target=target, components=comps,
                evidence=[f"Three {'highs' if top else 'lows'} within {tol / atr:.1f} ATR over {span} bars",
                          f"Neckline {neck:.4g}; prior trend {prior:.1f} ATR"],
            ))
        return out
