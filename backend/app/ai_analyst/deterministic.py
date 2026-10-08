"""Deterministic, template-based analyst commentary.

Built only from fields of the validated report, so it can never introduce a price,
indicator or pattern that the quantitative engine did not produce. It is the
default commentary and the fallback whenever no AI provider is configured.
"""
from __future__ import annotations

from app.schemas.patterns import Direction, PatternStage
from app.schemas.report import AnalysisReport, Commentary


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:,.4g}"


def deterministic_commentary(r: AnalysisReport) -> Commentary:
    sc = r.score
    bias = sc.rating
    bull = [p for p in r.patterns if p.direction == Direction.BULLISH and p.is_active]
    bear = [p for p in r.patterns if p.direction == Direction.BEARISH and p.is_active]
    failed = [p for p in r.patterns if p.stage == PatternStage.FAILED]
    comps = sorted([c for c in sc.components if c.value is not None], key=lambda c: -abs(c.value) * c.weight)
    strongest = []
    for c in comps[:3]:
        lean = "bullish" if c.value > 0.15 else "bearish" if c.value < -0.15 else "neutral"
        strongest.append(f"{c.name.replace('_', ' ').title()} ({lean}): " + "; ".join(c.reasons[:2]))
    conflicts = []
    signs = {c.name: c.value for c in comps if abs(c.value) >= 0.2}
    pos = [k for k, v in signs.items() if v > 0]
    neg = [k for k, v in signs.items() if v < 0]
    if pos and neg:
        conflicts.append(f"Bullish factors ({', '.join(pos)}) conflict with bearish factors ({', '.join(neg)}).")
    if bull and bear:
        conflicts.append(f"Competing pattern readings: {bull[0].name} ({bull[0].stage.value}) vs "
                         f"{bear[0].name} ({bear[0].stage.value}).")
    for p in failed[:2]:
        conflicts.append(f"{p.name} failed after confirmation — evidence against its original direction.")
    missing = [c.name for c in sc.components if c.value is None]
    if missing:
        conflicts.append("Unavailable factors (excluded, not assumed favourable): " + ", ".join(missing) + ".")
    if r.trend:
        conflicts += [n for n in r.trend.commentary if "countertrend" in n or "disagree" in n or "pullback" in n]

    ls, ss = r.long_setup, r.short_setup
    change, confirm = [], []
    if r.key_levels.breakout_trigger:
        change.append(f"A completed close above {_fmt(r.key_levels.breakout_trigger)} would strengthen the bullish case.")
    if r.key_levels.breakdown_trigger:
        change.append(f"A completed close below {_fmt(r.key_levels.breakdown_trigger)} would strengthen the bearish case.")
    if ls.valid:
        confirm.append(f"Long: {ls.confirmation_required}")
    if ss.valid:
        confirm.append(f"Bearish: {ss.confirmation_required}")

    preferred = None
    cands = [s for s in (ls, ss) if s.valid and s.status in ("actionable", "watch") and s.meets_min_rr]
    if cands:
        best = max(cands, key=lambda s: s.confidence)
        aligned = (best.side == "long" and sc.score >= 55) or (best.side == "short" and sc.score <= 45)
        if aligned:
            preferred = (f"{best.side.title()} ({best.basis}): {best.entry_trigger} Stop {_fmt(best.stop_loss)}; "
                         f"primary R:R {best.primary_rr}.")
    stay_neutral = preferred is None or sc.coverage < 0.5 or bias == "Neutral"
    ext = next((s for s in (ls, ss) if s.valid and s.status == "extended"
                and ((s.side == "long" and sc.score >= 60) or (s.side == "short" and sc.score <= 40))), None)
    price = r.overview.price
    head = f"{r.overview.symbol} ({r.overview.timeframe}) last {_fmt(price)}."
    if r.trend:
        head += f" Multi-timeframe alignment: {r.trend.alignment.replace('_', ' ')}."
    if r.patterns:
        top = r.patterns[0]
        head += f" Most relevant pattern: {top.name}, {top.stage.value.replace('_', ' ')}"
        if top.breakout_level is not None and top.confirmation_index is None:
            head += f" (trigger {_fmt(top.breakout_level)})"
        head += "."
    else:
        head += " No current chart pattern meets the detection criteria."
    head += f" Confluence score {sc.score:.0f}/100 ({bias}, coverage {sc.coverage:.0%})."
    if stay_neutral and ext is not None:
        conclusion = (f"{bias.upper()} BIAS — NO NEW ENTRY AT CURRENT PRICE. {ext.basis} is confirmed but price is "
                      f"extended beyond the trigger; a retest of {_fmt(ext.entry_zone[0])}-{_fmt(ext.entry_zone[1])} "
                      f"that holds on a closing basis would offer a defined-risk entry (stop {_fmt(ext.stop_loss)}).")
    elif stay_neutral:
        conclusion = "NEUTRAL — NO TRADE — AWAITING CONFIRMATION. " + (
            "Data coverage is insufficient for a directional call." if sc.coverage < 0.5 else
            "Evidence is mixed or no setup meets the 1:2 reward/risk filter; wait for the confirmation conditions above.")
    else:
        conclusion = (f"Conditional {bias.lower()} bias: {preferred} The view is invalidated by "
                      f"{(ls if preferred.startswith('Long') else ss).technical_invalidation or 'a break of the stop level'}")
    if any("SYNTHETIC" in w for w in r.warnings):
        conclusion += " (Computed on synthetic demo data.)"
    return Commentary(
        engine="deterministic-template", bias=bias, summary=head, strongest_evidence=strongest,
        conflicting_signals=conflicts, what_changes_view=change, confirmation_needed=confirm,
        preferred_setup=preferred, stay_neutral=stay_neutral, conclusion=conclusion,
        validation={"numbers_sourced_from_report": True},
    )
