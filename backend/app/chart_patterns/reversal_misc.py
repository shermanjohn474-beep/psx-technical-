"""Diamond top/bottom, V / inverse-V reversals, bull traps and bear traps."""
from __future__ import annotations

import numpy as np

from app.chart_patterns.base import AnalysisContext, PatternDetector, build_result, evaluate_breakout
from app.schemas.patterns import Direction, PatternCategory, PatternResult


class DiamondDetector(PatternDetector):
    """Diamond: a broadening phase (higher high + lower low) followed by a contracting
    phase (lower highs + higher lows) around a central peak/trough.

    Uses 6 consecutive swings. Right-side boundaries run from the central extreme
    high/low to the last high/low. Diamond top (after an advance) confirms on a close
    below the right-side lower boundary; the bottom is mirrored. A break of the opposite
    boundary first invalidates the expected resolution."""

    pattern_ids = ("diamond_top", "diamond_bottom")
    category = PatternCategory.REVERSAL

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        sw = ctx.confirmed_swings()
        cfg = ctx.cfg
        out = []
        for i in range(len(sw) - 5):
            seq = sw[i:i + 6]
            highs = [x for x in seq if x.kind == "H"]
            lows = [x for x in seq if x.kind == "L"]
            if len(highs) != 3 or len(lows) != 3:
                continue
            (h1, h2, h3), (l1, l2, l3) = highs, lows
            # broadening into the middle, contracting after
            if not (h2.price > h1.price and h2.price > h3.price and l2.price < l1.price and l2.price < l3.price):
                continue
            atr = ctx.atr[seq[-1].index]
            width = h2.price - l2.price
            if width < 4 * atr or width < cfg.min_height_pct / 100 * h2.price:
                continue
            if min(h2.price - h3.price, l3.price - l2.price) < 0.5 * atr:
                continue
            prior = ctx.prior_extreme_move_atr(seq[0].index)
            top = prior >= 0
            mu = (h3.price - h2.price) / max(h3.index - h2.index, 1)
            ml = (l3.price - l2.price) / max(l3.index - l2.index, 1)

            def upper(j, h2=h2, mu=mu):
                return h2.price + mu * (j - h2.index)

            def lower(j, l2=l2, ml=ml):
                return l2.price + ml * (j - l2.index)

            last_touch = seq[-1].index
            if top:
                direction, level, inval = Direction.BEARISH, lower, upper
                tgt = lower(last_touch) - width
            else:
                direction, level, inval = Direction.BULLISH, upper, lower
                tgt = upper(last_touch) + width
            ev = evaluate_breakout(ctx, last_touch + 1, level, direction, inval, tgt)
            b = ev.confirmation_index if ev.confirmation_index is not None else ctx.last
            trig = level(b)
            target = trig - width if top else trig + width
            end = ev.confirmation_index or ctx.last
            pid = "diamond_top" if top else "diamond_bottom"
            out.append(build_result(
                ctx, pattern_id=pid, name="Diamond Top" if top else "Diamond Bottom", category=PatternCategory.REVERSAL,
                direction=direction, ev=ev, start=seq[0].index, end=last_touch,
                key_points=[ctx.kp(f"{x.kind}{k}", x.index, x.price) for k, x in enumerate(seq, 1)],
                lines=[ctx.line("Left upper", h1.index, h1.price, h2.index, h2.price),
                       ctx.line("Left lower", l1.index, l1.price, l2.index, l2.price),
                       ctx.line("Right upper", h2.index, h2.price, end, upper(end)),
                       ctx.line("Right lower", l2.index, l2.price, end, lower(end))],
                breakout_level=trig, invalidation_level=inval(b), target=target,
                components={"width": min(1.0, width / (10 * atr)), "prior_trend": min(1.0, abs(prior) / (2 * cfg.min_trend_atr)),
                            "symmetry": 1 - min(1.0, abs((h2.index - h1.index) - (h3.index - h2.index)) / max(h3.index - h1.index, 1))},
                evidence=[f"Broadening then contracting swings around {'peak' if top else 'trough'}; width {width / atr:.1f} ATR",
                          f"Prior trend {prior:+.1f} ATR", "Diamonds are rare and subjective; treat as lower-reliability."],
            ))
        return [r for r in out if r.end_index >= ctx.last - cfg.max_age_bars]


class VReversalDetector(PatternDetector):
    """V-shaped bottom (and inverse V top): a steep decline of ``>= min_drop_atr`` ATR at
    ``>= min_slope_atr`` ATR/bar into a swing low, followed by a recovery. Confirmed when a
    close retraces ``confirm_frac`` (default 61.8%) of the decline within
    ``1.5 x`` the decline duration. Target: full retracement to the decline start."""

    pattern_ids = ("v_bottom", "inverted_v_top")
    category = PatternCategory.REVERSAL

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        cfg = ctx.cfg
        min_drop = cfg.p("v_reversal", "min_drop_atr", 6.0)
        min_slope = cfg.p("v_reversal", "min_slope_atr", 0.5)
        frac = cfg.p("v_reversal", "confirm_frac", 0.618)
        max_bars = cfg.p("v_reversal", "max_leg_bars", 20)
        min_started = cfg.p("v_reversal", "min_recovery_frac", 0.382)
        sw = ctx.swings_with_provisional()
        out = []
        for a, b in zip(sw, sw[1:]):
            bars = b.index - a.index
            if bars < 2 or bars > max_bars:
                continue
            atr = ctx.atr[b.index]
            drop = abs(a.price - b.price)
            if drop < min_drop * atr or drop / bars < min_slope * atr or drop < cfg.min_height_pct / 100 * a.price:
                continue
            bottom = b.kind == "L"
            direction = Direction.BULLISH if bottom else Direction.BEARISH
            level = b.price + frac * (a.price - b.price)
            target = a.price
            ev = evaluate_breakout(ctx, b.index + 1, lambda j, v=level: v, direction, b.price, target,
                                   max_wait=int(max(3, 1.5 * bars)))
            if ev.confirmation_index is None and ev.stage.value == "invalidated":
                continue  # recovery never came (or new extreme): no V reversal
            if ev.confirmation_index is None:
                after = ctx.h[b.index + 1:] if bottom else ctx.l[b.index + 1:]
                if len(after) == 0:
                    continue
                rec = (after.max() - b.price) if bottom else (b.price - after.min())
                if rec < min_started * drop:
                    continue  # recovery has not meaningfully started
            pid = "v_bottom" if bottom else "inverted_v_top"
            out.append(build_result(
                ctx, pattern_id=pid, name="V-Shaped Bottom" if bottom else "Inverted V Top",
                category=PatternCategory.REVERSAL, direction=direction, ev=ev, start=a.index, end=b.index,
                key_points=[ctx.kp("Decline start" if bottom else "Rally start", a.index, a.price),
                            ctx.kp("V low" if bottom else "V high", b.index, b.price)],
                lines=[ctx.line("Leg", a.index, a.price, b.index, b.price),
                       ctx.line(f"{frac:.1%} retracement trigger", b.index, level, ev.confirmation_index or ctx.last, level, "dashed")],
                breakout_level=level, invalidation_level=b.price, target=target,
                components={"steepness": min(1.0, drop / bars / (1.5 * atr)), "size": min(1.0, drop / (12 * atr))},
                evidence=[f"{'Decline' if bottom else 'Rally'} of {drop / atr:.1f} ATR in {bars} bars",
                          "V reversals have no basing period: confirmation must come quickly or the reading lapses."],
            ))
        return [r for r in out if r.end_index >= ctx.last - cfg.max_age_bars]


class TrapDetector(PatternDetector):
    """Failed breakout (bull trap) / failed breakdown (bear trap).

    Level = highest high (lowest low) of the prior ``lookback`` bars, excluding the
    breakout bar. A completed close beyond it by tolerance, followed within ``trap_bars``
    by a completed close back on the other side by tolerance, confirms the trap. The trap
    is invalidated if price then closes beyond the extreme of the breakout excursion."""

    pattern_ids = ("bull_trap", "bear_trap")
    category = PatternCategory.REVERSAL

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        cfg = ctx.cfg
        lookback = cfg.p("trap", "lookback", 40)
        trap_bars = cfg.p("trap", "trap_bars", 5)
        scan = cfg.p("trap", "scan_bars", 120)
        out = []
        for i in range(max(lookback, ctx.last - scan), ctx.last + 1):
            if i < lookback:
                continue
            tol = cfg.breakout_tol_atr * ctx.atr[i]
            res_lvl = ctx.h[i - lookback:i].max()
            sup_lvl = ctx.l[i - lookback:i].min()
            for bull_break in (True, False):
                lvl = res_lvl if bull_break else sup_lvl
                broke = ctx.c[i] > lvl + tol if bull_break else ctx.c[i] < lvl - tol
                if not broke or (i == ctx.last and not ctx.last_candle_complete):
                    continue
                # first bar of the break (avoid re-detecting a continuing breakout)
                if bull_break and ctx.c[i - 1] > lvl + tol or (not bull_break and ctx.c[i - 1] < lvl - tol):
                    continue
                j_end = min(ctx.last, i + trap_bars)
                excursion = ctx.h[i:j_end + 1].max() if bull_break else ctx.l[i:j_end + 1].min()
                direction = Direction.BEARISH if bull_break else Direction.BULLISH
                opp = sup_lvl if bull_break else res_lvl
                ev = evaluate_breakout(ctx, i + 1, lambda j, v=lvl: v, direction, excursion, opp, max_wait=trap_bars)
                if ev.confirmation_index is None and ev.stage.value == "invalidated":
                    continue  # breakout held: no trap
                if ev.confirmation_index is None and not ev.provisional:
                    continue  # a trap only exists once price has closed back through the level
                pid = "bull_trap" if bull_break else "bear_trap"
                end = ev.confirmation_index or ctx.last
                out.append(build_result(
                    ctx, pattern_id=pid, name="Failed Breakout / Bull Trap" if bull_break else "Failed Breakdown / Bear Trap",
                    category=PatternCategory.REVERSAL, direction=direction, ev=ev, start=i - lookback, end=i,
                    key_points=[ctx.kp("Breakout bar" if bull_break else "Breakdown bar", i, ctx.c[i]),
                                ctx.kp("Excursion extreme", i + int(np.argmax(ctx.h[i:j_end + 1]) if bull_break else np.argmin(ctx.l[i:j_end + 1])), excursion)],
                    lines=[ctx.line("Broken level", i - lookback, lvl, end, lvl, "dashed")],
                    breakout_level=lvl, invalidation_level=excursion, target=opp,
                    components={"reversal_speed": 1.0 if (ev.confirmation_index or i + trap_bars) - i <= 3 else 0.6,
                                "level_significance": min(1.0, lookback / 40)},
                    evidence=[f"Close beyond {lookback}-bar {'high' if bull_break else 'low'} {lvl:.4g} on {ctx.ts(i).date()} "
                              f"then back inside within {trap_bars} bars"],
                ))
        return [r for r in out if r.end_index >= ctx.last - cfg.max_age_bars]
