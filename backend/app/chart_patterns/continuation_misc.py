"""Measured move, price-compression breakout and Volatility Contraction Pattern (VCP)."""
from __future__ import annotations

import numpy as np

from app.chart_patterns.base import (
    AnalysisContext, BreakoutEval, PatternDetector, build_result, evaluate_breakout,
)
from app.indicators.volatility import bollinger, bollinger_squeeze
from app.schemas.patterns import Direction, PatternCategory, PatternResult, PatternStage


class MeasuredMoveDetector(PatternDetector):
    """A-B-C measured move: impulse leg A->B (>= ``min_leg_atr`` ATR), corrective leg B->C
    retracing ``[min_retrace, max_retrace]`` of AB without breaching A. Confirmation =
    completed close beyond B (leg two under way). Target D = C + (B - A).
    Invalidation = close beyond C."""

    pattern_ids = ("measured_move_up", "measured_move_down")
    category = PatternCategory.CONTINUATION

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        cfg = ctx.cfg
        min_leg = cfg.p("measured_move", "min_leg_atr", 5.0)
        rmin, rmax = cfg.p("measured_move", "min_retrace", 0.3), cfg.p("measured_move", "max_retrace", 0.7)
        sw = ctx.swings_with_provisional()
        out = []
        for a, b, c in zip(sw, sw[1:], sw[2:]):
            up = a.kind == "L"
            leg = abs(b.price - a.price)
            atr = ctx.atr[b.index]
            if leg < min_leg * atr or leg < cfg.min_height_pct / 100 * a.price:
                continue
            retr = abs(b.price - c.price) / leg
            if not (rmin <= retr <= rmax):
                continue
            direction = Direction.BULLISH if up else Direction.BEARISH
            target = c.price + (b.price - a.price)
            ev = evaluate_breakout(ctx, c.index + 1, lambda j, v=b.price: v, direction, c.price, target)
            if c.provisional and ev.confirmation_index is not None:
                continue
            pid = "measured_move_up" if up else "measured_move_down"
            out.append(build_result(
                ctx, pattern_id=pid, name="Measured Move Up" if up else "Measured Move Down",
                category=PatternCategory.CONTINUATION, direction=direction, ev=ev, start=a.index, end=c.index,
                key_points=[ctx.kp("A", a.index, a.price), ctx.kp("B", b.index, b.price), ctx.kp("C", c.index, c.price)],
                lines=[ctx.line("Leg AB", a.index, a.price, b.index, b.price), ctx.line("Leg BC", b.index, b.price, c.index, c.price),
                       ctx.line("Trigger (B)", b.index, b.price, ev.confirmation_index or ctx.last, b.price, "dashed")],
                breakout_level=b.price, invalidation_level=c.price, target=target,
                components={"leg_size": min(1.0, leg / (10 * atr)), "retrace_quality": 1 - abs(retr - 0.5) / 0.25},
                evidence=[f"AB leg {leg / atr:.1f} ATR; BC retraced {retr:.0%}",
                          "Measured-move target assumes leg CD equals AB (a symmetry heuristic, not a forecast)."],
            ))
        return [r for r in out if r.end_index >= ctx.last - cfg.max_age_bars]


class CompressionBreakoutDetector(PatternDetector):
    """Price compression: Bollinger bandwidth in the lowest ``squeeze_pct`` of its trailing
    ``lookback`` window for ``>= min_len`` consecutive bars. The compression range is the
    high/low of those bars. Bilateral: direction comes from the first completed close
    outside the range (by tolerance)."""

    pattern_ids = ("compression_breakout",)
    category = PatternCategory.BILATERAL

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        cfg = ctx.cfg
        min_len = cfg.p("compression", "min_len", 6)
        lookback = cfg.p("compression", "lookback", 120)
        pct = cfg.p("compression", "squeeze_pct", 0.15)
        if ctx.n < lookback // 2 + 20:
            return []
        contraction = cfg.p("compression", "max_width_ratio", 0.55)
        sq = bollinger_squeeze(ctx.df["close"], 20, 2.0, lookback, pct).fillna(False).to_numpy()
        bw = bollinger(ctx.df["close"], 20, 2.0)["bb_width"]
        bw_med = bw.rolling(lookback, min_periods=lookback // 2).median().to_numpy()
        bwv = bw.to_numpy()
        out = []
        i = 0
        runs = []
        while i < ctx.n:
            if sq[i]:
                j = i
                while j + 1 < ctx.n and sq[j + 1]:
                    j += 1
                if j - i + 1 >= min_len:
                    runs.append((i, j))
                i = j + 1
            else:
                i += 1
        for s0, e0 in runs[-3:]:
            ratio = float(np.nanmean(bwv[s0:e0 + 1]) / bw_med[s0]) if np.isfinite(bw_med[s0]) and bw_med[s0] > 0 else 1.0
            if ratio > contraction:
                continue  # percentile squeeze without a genuine volatility contraction
            hi, lo = ctx.h[s0:e0 + 1].max(), ctx.l[s0:e0 + 1].min()
            width = hi - lo
            up = evaluate_breakout(ctx, e0 + 1, lambda j, v=hi: v, Direction.BULLISH, None, hi + width)
            dn = evaluate_breakout(ctx, e0 + 1, lambda j, v=lo: v, Direction.BEARISH, None, lo - width)
            cu, cd = up.confirmation_index, dn.confirmation_index
            if cu is not None and (cd is None or cu <= cd):
                ev, direction, lvl, tgt, inv = up, Direction.BULLISH, hi, hi + width, lo
            elif cd is not None:
                ev, direction, lvl, tgt, inv = dn, Direction.BEARISH, lo, lo - width, hi
            else:
                ev = up if up.stage != PatternStage.INVALIDATED else dn
                if e0 == ctx.last:
                    ev = BreakoutEval(stage=PatternStage.FORMING, notes=["Compression still in progress."])
                direction, lvl, tgt, inv = Direction.NEUTRAL, None, None, None
            out.append(build_result(
                ctx, pattern_id="compression_breakout", name="Price Compression & Breakout",
                category=PatternCategory.BILATERAL, direction=direction, ev=ev, start=s0, end=e0,
                key_points=[ctx.kp("Compression start", s0, ctx.c[s0]), ctx.kp("Compression end", e0, ctx.c[e0])],
                lines=[ctx.line("Range high", s0, hi, ev.confirmation_index or ctx.last, hi, "dashed"),
                       ctx.line("Range low", s0, lo, ev.confirmation_index or ctx.last, lo, "dashed")],
                breakout_level=lvl, invalidation_level=inv, target=tgt,
                components={"duration": min(1.0, (e0 - s0 + 1) / 15), "tightness": 1 - min(1.0, width / (6 * ctx.atr[e0]))},
                evidence=[f"Bollinger bandwidth in lowest {pct:.0%} of trailing {lookback} bars for {e0 - s0 + 1} bars "
                          f"({ratio:.0%} of median width)",
                          f"Range {lo:.4g}-{hi:.4g}; direction only from breakout"],
            ))
        return [r for r in out if r.end_index >= ctx.last - cfg.max_age_bars]


class VCPDetector(PatternDetector):
    """Volatility Contraction Pattern (Minervini-style, heuristic).

    In an established uptrend (close above SMA50 and prior advance >= ``min_trend_atr``),
    find >= ``min_contractions`` successive pullbacks (swing high -> next swing low) whose
    percentage depth shrinks by at least ``shrink`` each time, with highs staying within
    ``pivot_band_pct`` of each other. Pivot = last contraction high. Confirmation = close
    above pivot. Invalidation = close below the last contraction low."""

    pattern_ids = ("vcp",)
    category = PatternCategory.CONTINUATION

    def detect(self, ctx: AnalysisContext) -> list[PatternResult]:
        cfg = ctx.cfg
        min_c = cfg.p("vcp", "min_contractions", 2)
        shrink = cfg.p("vcp", "shrink", 0.8)
        band = cfg.p("vcp", "pivot_band_pct", 8.0)
        sma50 = ctx.df["close"].rolling(50).mean().to_numpy()
        sw = ctx.confirmed_swings()
        out = []
        for end in range(len(sw) - 1, max(-1, len(sw) - 6), -1):
            if sw[end].kind != "L":
                continue
            pulls = []
            k = end
            while k - 1 >= 0 and sw[k].kind == "L" and sw[k - 1].kind == "H":
                h, l = sw[k - 1], sw[k]
                pulls.insert(0, (h, l, (h.price - l.price) / h.price * 100))
                k -= 2
                if len(pulls) >= 5:
                    break
            # keep the longest suffix that contracts
            seq = [pulls[-1]] if pulls else []
            for p in reversed(pulls[:-1]):
                if seq[0][2] <= shrink * p[2]:
                    seq.insert(0, p)
                else:
                    break
            if len(seq) < min_c:
                continue
            highs = [h.price for h, _, _ in seq]
            if (max(highs) - min(highs)) / max(highs) * 100 > band:
                continue
            first_h = seq[0][0]
            if np.isnan(sma50[first_h.index]) or ctx.c[first_h.index] < sma50[first_h.index]:
                continue
            prior = ctx.prior_extreme_move_atr(first_h.index, 80)
            if prior < cfg.min_trend_atr:
                continue
            pivot = seq[-1][0].price
            last_low = seq[-1][1].price
            depth0 = seq[0][0].price - seq[0][1].price
            ev = evaluate_breakout(ctx, seq[-1][1].index + 1, lambda j, v=pivot: v, Direction.BULLISH, last_low,
                                   pivot + depth0)
            comps = {"contractions": min(1.0, len(seq) / 4), "final_tightness": 1 - min(1.0, seq[-1][2] / 10)}
            if ctx.volume_available:
                v = ctx.df["volume"].to_numpy()
                v_first = np.nanmean(v[seq[0][0].index:seq[0][1].index + 1])
                v_last = np.nanmean(v[seq[-1][0].index:seq[-1][1].index + 1])
                comps["volume_dry_up"] = 1.0 if v_last < 0.8 * v_first else 0.4
            kps = []
            for n_, (h, l, _) in enumerate(seq, 1):
                kps += [ctx.kp(f"C{n_} high", h.index, h.price), ctx.kp(f"C{n_} low", l.index, l.price)]
            out.append(build_result(
                ctx, pattern_id="vcp", name="Volatility Contraction Pattern", category=PatternCategory.CONTINUATION,
                direction=Direction.BULLISH, ev=ev, start=first_h.index, end=seq[-1][1].index, key_points=kps,
                lines=[ctx.line("Pivot", first_h.index, pivot, ev.confirmation_index or ctx.last, pivot, "dashed")],
                breakout_level=pivot, invalidation_level=last_low, target=pivot + depth0, components=comps,
                evidence=["Pullback depths " + " -> ".join(f"{d:.1f}%" for _, _, d in seq),
                          "VCP is a heuristic model; contraction counts are sensitive to swing settings."],
                experimental=True,
            ))
            break
        return [r for r in out if r.end_index >= ctx.last - cfg.max_age_bars]
