"""Bullish/bearish flags and pennants.

Definition (bullish; bearish is mirrored):

* **Flagpole**: a move from the lowest low in ``pole_max_bars`` up to a local high
  ``P`` of at least ``pole_min_atr`` ATR, with average steepness
  ``>= pole_min_slope_atr`` ATR per bar.
* **Consolidation**: starts at ``P``; between ``min_cons`` and ``max_cons`` bars; no
  high above ``P``; retracement of the pole ``<= max_retrace``. Upper/lower
  boundaries are least-squares lines through highs/lows shifted to envelope the
  bars (so no earlier bar is already "outside").
* **Flag**: boundaries roughly parallel, drifting against the pole or sideways.
  **Pennant**: boundaries converging (width ratio ``< pennant_ratio``).
* **Confirmation**: completed close above the upper boundary by tolerance, in the
  pole's direction. Invalidation: retracement beyond ``max_retrace`` or expiry.
* **Target**: breakout level + pole height. Stop reference: consolidation low.
"""
from __future__ import annotations

import numpy as np

from app.chart_patterns.base import AnalysisContext, PatternDetector, build_result, evaluate_breakout, BreakoutEval
from app.schemas.patterns import Direction, PatternCategory, PatternResult, PatternStage


def _envelope(x: np.ndarray, y: np.ndarray, upper: bool) -> tuple[float, float]:
    if len(x) < 2:
        return 0.0, float(y[0])
    m, b = np.polyfit(x, y, 1)
    r = y - (m * x + b)
    b += r.max() if upper else r.min()
    return float(m), float(b)


def _touches(x: np.ndarray, y: np.ndarray, m: float, b: float, width: np.ndarray, upper: bool,
             frac: float = 0.4, min_gap: int = 3) -> int:
    """Count distinct boundary tests: minor pivots (3-bar extremes) lying in the outer
    ``frac`` of the channel next to the boundary, at least ``min_gap`` bars apart."""
    line = m * x + b
    groups, last = 0, None
    for k in range(len(x)):
        left = y[k - 1] if k > 0 else (-np.inf if upper else np.inf)
        right = y[k + 1] if k + 1 < len(x) else None
        if right is None:
            continue  # pivot needs the following bar (causal: bar must be inside the window)
        is_pivot = (y[k] >= left and y[k] >= right) if upper else (y[k] <= left and y[k] <= right)
        near = (line[k] - y[k] if upper else y[k] - line[k]) <= frac * width[k]
        if is_pivot and near and (last is None or x[k] - last >= min_gap):
            groups += 1
            last = x[k]
    return groups


class FlagPennantDetector(PatternDetector):
    pattern_ids = ("bull_flag", "bear_flag", "bull_pennant", "bear_pennant")
    category = PatternCategory.CONTINUATION

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        out = []
        for bull in (True, False):
            out += self._scan(ctx, bull)
        return out

    def _scan(self, ctx: AnalysisContext, bull: bool) -> list[PatternResult]:
        cfg = ctx.cfg
        pid0 = "flag"
        pole_max = cfg.p(pid0, "pole_max_bars", 20)
        pole_min_atr = cfg.p(pid0, "pole_min_atr", 5.0)
        pole_slope = cfg.p(pid0, "pole_min_slope_atr", 0.45)
        min_cons = cfg.p(pid0, "min_cons", 5)
        max_counter = cfg.p(pid0, "max_counter_slope_frac", 0.4)  # flag drift vs pole slope
        min_touch = cfg.p(pid0, "min_touches_each", 2)
        max_cons = cfg.p(pid0, "max_cons", 25)
        max_retrace = cfg.p(pid0, "max_retrace", 0.4)
        max_drift = cfg.p(pid0, "max_drift_atr_per_bar", 0.12)  # allowed drift *with* the pole
        pennant_ratio = cfg.p(pid0, "pennant_ratio", 0.55)
        scan = cfg.p(pid0, "scan_bars", 150)
        s = 1.0 if bull else -1.0
        hi = ctx.h if bull else -ctx.l
        lo = ctx.l if bull else -ctx.h
        cl = ctx.c * s
        out = []
        for t in range(max(pole_max, ctx.last - scan), ctx.last - min_cons + 1):
            win_start = t - pole_max
            if hi[t] < hi[win_start:t + 1].max():
                continue
            if hi[t + 1:t + min_cons + 1].max(initial=-np.inf) > hi[t]:
                continue
            base = win_start + int(np.argmin(lo[win_start:t + 1]))
            height = hi[t] - lo[base]
            atr = ctx.atr[t]
            bars = t - base
            if bars < 2 or height < pole_min_atr * atr or height / bars < pole_slope * atr:
                continue
            # walk the consolidation forward
            brk = None
            stop_reason = None
            j_end = min(t + max_cons, ctx.last)
            fit = None
            for j in range(t + min_cons, j_end + 1):
                x = np.arange(t, j)
                if hi[t + 1:j].max(initial=-np.inf) > hi[t]:
                    stop_reason = "New extreme beyond the pole tip: not a flag."
                    break
                if hi[t] - lo[t:j].min() > max_retrace * height:
                    stop_reason = f"Retraced more than {max_retrace:.0%} of the pole."
                    break
                mu, bu = _envelope(x, hi[t:j], True)
                ml, bl = _envelope(x, lo[t:j], False)
                mid = (mu + ml) / 2
                shape_ok = (mid <= max_drift * atr and -mid <= max_counter * height / bars
                            and (mu * t + bu) - (ml * t + bl) <= 0.75 * height
                            and _touches(x, hi[t:j], mu, bu, (mu - ml) * x + bu - bl, True) >= min_touch
                            and _touches(x, lo[t:j], ml, bl, (mu - ml) * x + bu - bl, False) >= min_touch)
                if not shape_ok:
                    continue
                fit = (mu, bu, ml, bl, j - 1)
                if cl[j] > mu * j + bu + cfg.breakout_tol_atr * ctx.atr[j]:
                    brk = j
                    break
            if fit is None:
                continue
            mu, bu, ml, bl, wend = fit
            if stop_reason and stop_reason.startswith("New extreme"):
                continue  # the pole simply extended: not a flag
            cons_bars = wend - t + 1
            mid_slope = (mu + ml) / 2 / atr
            if mid_slope > max_drift:
                continue
            w0 = (mu * t + bu) - (ml * t + bl)
            w1 = (mu * wend + bu) - (ml * wend + bl)
            if w0 <= 0 or w1 <= 0 or w0 > 0.75 * height:
                continue
            pennant = (w1 / w0) < pennant_ratio and mu <= 0 <= ml
            kind = ("bull_" if bull else "bear_") + ("pennant" if pennant else "flag")
            direction = Direction.BULLISH if bull else Direction.BEARISH

            def level(j, mu=mu, bu=bu):
                return s * (mu * j + bu)

            breakout_px = level(brk if brk is not None else ctx.last)
            target = breakout_px + s * height
            cons_extreme = s * lo[t:wend + 1].min()
            if brk is not None:
                ev = evaluate_breakout(ctx, brk, level, direction, None, target)
            else:
                ev = BreakoutEval(stage=PatternStage.FORMING)
                if ctx.last - t > max_cons:
                    ev.stage = PatternStage.INVALIDATED
                    ev.notes.append(f"Consolidation exceeded {max_cons} bars without breakout.")
                elif s * (breakout_px - ctx.c[ctx.last]) <= cfg.approach_atr * ctx.atr[ctx.last]:
                    ev.stage = PatternStage.APPROACHING_CONFIRMATION
                if stop_reason:
                    ev.stage = PatternStage.INVALIDATED
                    ev.notes.append(stop_reason)
            comps = {
                "pole_strength": min(1.0, height / (10 * atr)),
                "pole_steepness": min(1.0, (height / bars) / (1.0 * atr)),
                "tightness": 1 - min(1.0, w0 / (0.75 * height)),
                "retrace": 1 - (hi[t] - lo[t:wend + 1].min()) / (max_retrace * height),
                "duration": 1.0 if 5 <= cons_bars <= 20 else 0.6,
            }
            if ctx.volume_available:
                v = ctx.df["volume"].to_numpy()
                comps["volume_dry_up"] = 1.0 if np.nanmean(v[t:wend + 1]) < np.nanmean(v[base:t + 1]) else 0.4
            end = ev.confirmation_index or wend
            name = ("Bullish " if bull else "Bearish ") + ("Pennant" if pennant else "Flag")
            out.append(build_result(
                ctx, pattern_id=kind, name=name, category=PatternCategory.CONTINUATION, direction=direction, ev=ev,
                start=base, end=wend,
                key_points=[ctx.kp("Pole start", base, s * lo[base]), ctx.kp("Pole top" if bull else "Pole bottom", t, s * hi[t])],
                lines=[ctx.line("Flagpole", base, s * lo[base], t, s * hi[t]),
                       ctx.line("Upper boundary" if bull else "Lower boundary", t, s * (mu * t + bu), end, s * (mu * end + bu)),
                       ctx.line("Lower boundary" if bull else "Upper boundary", t, s * (ml * t + bl), end, s * (ml * end + bl))],
                breakout_level=breakout_px, invalidation_level=cons_extreme, target=target, components=comps,
                evidence=[f"Pole {height / atr:.1f} ATR in {bars} bars",
                          f"Consolidation {cons_bars} bars, retrace {(hi[t] - lo[t:wend + 1].min()) / height:.0%} of pole",
                          "Converging boundaries (pennant)" if pennant else "Parallel/counter-trend boundaries (flag)"],
            ))
        return [r for r in out if r.end_index >= ctx.last - cfg.max_age_bars]
