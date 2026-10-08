"""Head & Shoulders and Inverse Head & Shoulders.

Definition (top; inverse is mirrored):

* Five alternating swings ``LS(H) - T1(L) - HEAD(H) - T2(L) - RS(H)``.
* Head exceeds both shoulders by ``>= head_min_atr`` ATR.
* Shoulder heights differ by ``<= shoulder_tol_frac`` x head height above neckline.
* Time symmetry: ``(HEAD-LS)/(RS-HEAD)`` within ``[1/time_ratio, time_ratio]``.
* Neckline through T1 and T2; its rise/fall across the pattern ``<= neck_slope_frac``
  of the head height (sloped necklines are allowed but scored lower).
* Prior trend: advance into the left shoulder.
* Confirmation: completed close below the *extrapolated* neckline by tolerance.
* Pre-confirmation invalidation: close above the head. Trade stop reference: right shoulder.
* Target: neckline at break - (head - neckline at head).
"""
from __future__ import annotations

from app.chart_patterns.base import (
    AnalysisContext, PatternDetector, build_result, closeness, evaluate_breakout,
)
from app.schemas.patterns import Direction, PatternCategory, PatternResult


class HeadShouldersDetector(PatternDetector):
    pattern_ids = ("head_and_shoulders", "inverse_head_and_shoulders")
    category = PatternCategory.REVERSAL

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        sw = ctx.swings_with_provisional()
        out = []
        for top in (True, False):
            pid = "head_and_shoulders" if top else "inverse_head_and_shoulders"
            cfg = ctx.cfg
            head_min = cfg.p(pid, "head_min_atr", 1.0)
            sh_tol = cfg.p(pid, "shoulder_tol_frac", 0.5)
            time_ratio = cfg.p(pid, "time_ratio", 2.5)
            neck_frac = cfg.p(pid, "neck_slope_frac", 0.6)
            min_prior = cfg.p(pid, "min_prior_atr", 0.5 * cfg.min_trend_atr)
            pk = "H" if top else "L"
            s = 1.0 if top else -1.0
            for i in range(len(sw) - 4):
                seq = sw[i:i + 5]
                if [x.kind for x in seq] != ([pk, "L", pk, "L", pk] if top else [pk, "H", pk, "H", pk]):
                    continue
                ls, t1, hd, t2, rs = seq
                atr = ctx.atr[rs.index]
                if s * (hd.price - ls.price) < head_min * atr or s * (hd.price - rs.price) < head_min * atr:
                    continue
                slope = (t2.price - t1.price) / max(t2.index - t1.index, 1)

                def neck(j, t1=t1, slope=slope):
                    return t1.price + slope * (j - t1.index)

                head_h = s * (hd.price - neck(hd.index))
                if head_h <= 0 or head_h < cfg.min_height_pct / 100 * abs(neck(hd.index)):
                    continue
                if abs(ls.price - rs.price) > sh_tol * head_h:
                    continue
                if s * (rs.price - neck(rs.index)) <= 0 or s * (ls.price - neck(ls.index)) <= 0:
                    continue
                d1, d2 = hd.index - ls.index, rs.index - hd.index
                if d2 <= 0 or not (1 / time_ratio <= d1 / d2 <= time_ratio):
                    continue
                neck_change = abs(slope * (rs.index - ls.index))
                if neck_change > neck_frac * head_h:
                    continue
                prior = s * ctx.prior_extreme_move_atr(ls.index)
                if prior < min_prior:
                    continue
                direction = Direction.BEARISH if top else Direction.BULLISH
                # target is computed relative to the neckline at confirmation; first pass with current neckline
                ev = evaluate_breakout(ctx, rs.index + 1, neck, direction, invalidation=hd.price,
                                       target=neck(ctx.last) - s * head_h)
                if rs.provisional and ev.confirmation_index is not None:
                    continue
                brk_idx = ev.confirmation_index if ev.confirmation_index is not None else ctx.last
                trigger = neck(brk_idx)
                target = trigger - s * head_h
                if ev.confirmation_index is not None:
                    ev = evaluate_breakout(ctx, rs.index + 1, neck, direction, invalidation=hd.price, target=target)
                comps = {
                    "shoulder_symmetry": closeness(ls.price - rs.price, sh_tol * head_h),
                    "time_symmetry": closeness(1 - min(d1, d2) / max(d1, d2), 1.0),
                    "head_prominence": min(1.0, min(s * (hd.price - ls.price), s * (hd.price - rs.price)) / (3 * atr)),
                    "neckline_flatness": closeness(neck_change, neck_frac * head_h),
                    "prior_trend": min(1.0, prior / (2 * cfg.min_trend_atr)),
                }
                if ctx.volume_available:
                    v = ctx.df["volume"].to_numpy()

                    def vwin(k):
                        return float(v[max(0, k - 2):k + 3].mean())

                    comps["volume_profile"] = 1.0 if vwin(rs.index) < vwin(ls.index) else 0.5
                end = ev.confirmation_index or ctx.last
                lbl = "" if top else "Inverse "
                out.append(build_result(
                    ctx, pattern_id=pid, name=f"{lbl}Head & Shoulders", category=PatternCategory.REVERSAL,
                    direction=direction, ev=ev, start=ls.index, end=rs.index,
                    key_points=[ctx.kp("Left Shoulder", ls.index, ls.price), ctx.kp("Neck 1", t1.index, t1.price),
                                ctx.kp("Head", hd.index, hd.price), ctx.kp("Neck 2", t2.index, t2.price),
                                ctx.kp("Right Shoulder", rs.index, rs.price)],
                    lines=[ctx.line("Neckline", ls.index, neck(ls.index), end, neck(end), "dashed")],
                    breakout_level=trigger, invalidation_level=rs.price, target=target, components=comps,
                    evidence=[
                        f"Head {hd.price:.4g} vs shoulders {ls.price:.4g}/{rs.price:.4g}",
                        f"Neckline {'rising' if slope > 0 else 'falling' if slope < 0 else 'flat'}; value at trigger {trigger:.4g}",
                        f"Time symmetry {d1}:{d2} bars; prior trend {prior:.1f} ATR",
                        f"Pattern void on close {'above' if top else 'below'} head {hd.price:.4g}; trade stop reference = right shoulder.",
                    ],
                ))
        return [r for r in out if r.end_index >= ctx.last - ctx.cfg.max_age_bars]
