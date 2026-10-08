"""Long and short/bearish trade-scenario construction.

Rules:
* Stops come from technical invalidation (pattern or structure); an ATR stop is
  shown as an alternative. Stops/targets are **never** moved to manufacture a
  better reward/risk.
* Long R:R = (target - entry) / (entry - stop); short R:R = (entry - target) / (stop - entry).
* 1:2 is the preferred minimum: setups below it are still shown but flagged
  ``meets_min_rr=False`` (lower-quality candidate).
* A bearish *analytical* scenario is distinguished from an executable short:
  executability is only asserted for symbols the operator lists in
  ``SHORT_ELIGIBLE_SYMBOLS``, and even then must be verified with the broker.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.schemas.patterns import Direction, PatternResult, PatternStage
from app.support_resistance.levels import LevelsResult


class Target(BaseModel):
    label: str
    price: float
    rr: float | None
    source: str


class TradeSetup(BaseModel):
    side: str  # long | short
    valid: bool = True
    status: str = "watch"  # actionable | watch | extended | invalid | target_reached
    basis: str  # pattern name or "structure"
    pattern_id: str | None = None
    pattern_stage: str | None = None
    entry_trigger: str
    entry_price: float | None = None
    entry_zone: tuple[float, float] | None = None
    confirmation_required: str
    stop_loss: float | None = None
    stop_basis: str | None = None
    atr_stop_alternative: float | None = None
    technical_invalidation: str | None = None
    targets: list[Target] = Field(default_factory=list)
    primary_rr: float | None = None
    meets_min_rr: bool = False
    confidence: float = 0.0  # 0..100 quality score, NOT a probability
    key_risks: list[str] = Field(default_factory=list)
    execution: str | None = None  # for shorts: eligibility statement
    notes: list[str] = Field(default_factory=list)


@dataclass
class RiskConfig:
    min_rr: float = 2.0
    atr_stop_mult: float = 2.0
    max_stop_atr: float = 6.0
    extended_atr: float = 2.0
    daily_limit_pct: float = 7.5  # PSX price-limit assumption; verify against current rules
    stop_buffer_atr: float = 0.2


def reward_risk(side: str, entry: float, stop: float, target: float) -> float | None:
    if side == "long":
        risk, reward = entry - stop, target - entry
    else:
        risk, reward = stop - entry, entry - target
    if risk <= 0 or reward <= 0:
        return None
    return round(reward / risk, 2)


ACTIVE = (PatternStage.FORMING, PatternStage.APPROACHING_CONFIRMATION, PatternStage.CONFIRMED, PatternStage.RETESTING)
STAGE_W = {PatternStage.CONFIRMED: 1.0, PatternStage.RETESTING: 1.0, PatternStage.APPROACHING_CONFIRMATION: 0.8,
           PatternStage.FORMING: 0.55}


def _pick_pattern(patterns: list[PatternResult], direction: Direction, last: int, recent: int = 30,
                  close: float | None = None, atr: float | None = None, extended_atr: float = 2.0) -> PatternResult | None:
    cands = [p for p in patterns if p.direction == direction and p.stage in ACTIVE and p.breakout_level is not None
             and p.invalidation_level is not None and p.target is not None and not p.target_reached
             and (p.confirmation_index is None or last - p.confirmation_index <= recent)]
    if not cands:
        return None
    sgn = 1 if direction == Direction.BULLISH else -1

    def rank(p: PatternResult) -> float:
        r = p.quality_score * STAGE_W[p.stage] * (0.8 if p.experimental else 1.0)
        if close is not None and atr and p.confirmation_index is not None and \
                sgn * (close - p.breakout_level) > extended_atr * atr:
            r *= 0.6  # already extended: less actionable
        return r

    return max(cands, key=rank)


def build_setup(side: str, close: float, atr: float, patterns: list[PatternResult], levels: LevelsResult,
                last_index: int, fib_ext: list[float] | None = None, cfg: RiskConfig | None = None,
                short_eligible: bool = False, symbol: str = "") -> TradeSetup:
    cfg = cfg or RiskConfig()
    long = side == "long"
    sgn = 1 if long else -1
    direction = Direction.BULLISH if long else Direction.BEARISH
    pat = _pick_pattern(patterns, direction, last_index, close=close, atr=atr, extended_atr=cfg.extended_atr)
    zones_ahead = sorted([z for z in levels.zones if sgn * (z.mid - close) > 0], key=lambda z: sgn * z.mid)
    zones_behind = sorted([z for z in levels.zones if sgn * (close - z.mid) > 0], key=lambda z: -sgn * z.mid)
    notes: list[str] = []
    risks: list[str] = []
    if pat is not None:
        trigger = pat.breakout_level
        confirmed = pat.stage in (PatternStage.CONFIRMED, PatternStage.RETESTING)
        stop = pat.invalidation_level - sgn * cfg.stop_buffer_atr * atr
        stop_basis = f"{pat.name} invalidation ({pat.invalidation_level:.4g}) with {cfg.stop_buffer_atr} ATR buffer"
        if confirmed:
            entry_zone = tuple(sorted((trigger, trigger + sgn * 0.5 * atr)))
            if sgn * (close - trigger) > cfg.extended_atr * atr:
                status = "extended"
                entry = trigger + sgn * 0.25 * atr
                trig_text = (f"Confirmed; price is extended {abs(close - trigger) / atr:.1f} ATR beyond the trigger. "
                             f"Prefer a retest of {entry_zone[0]:.4g}-{entry_zone[1]:.4g}.")
            else:
                status = "actionable"
                entry = close
                trig_text = (f"Confirmed {pat.name}; entry on hold of {trigger:.4g} "
                             f"(retest zone {entry_zone[0]:.4g}-{entry_zone[1]:.4g}).")
            confirm = "Already confirmed by a completed close; require the retest zone to hold on a closing basis."
        else:
            status = "watch"
            entry = trigger + sgn * 0.1 * atr
            entry_zone = tuple(sorted((trigger, trigger + sgn * 0.5 * atr)))
            trig_text = f"{'Buy' if long else 'Sell'}-stop on a completed close {'above' if long else 'below'} {trigger:.4g} ({pat.name} trigger)."
            confirm = (f"Completed candle close {'above' if long else 'below'} {trigger:.4g} by >= 0.25 ATR, "
                       "ideally on relative volume >= 1.5x.")
        basis, pid, stage = pat.name, pat.pattern_id, pat.stage.value
        measured = pat.target
        inval_text = f"Close {'below' if long else 'above'} {pat.invalidation_level:.4g} invalidates the {pat.name}."
    else:
        # structure-based fallback
        basis, pid, stage, measured = "structure", None, None, None
        ahead = zones_ahead[0] if zones_ahead else None
        behind = zones_behind[0] if zones_behind else None
        if ahead is None or behind is None:
            return TradeSetup(side=side, valid=False, status="invalid", basis="structure", entry_trigger="n/a",
                              confirmation_required="n/a",
                              notes=["No supporting pattern, and no support/resistance zone on both sides of price "
                                     "to define a structural setup."])
        trigger = ahead.high if long else ahead.low
        entry = trigger + sgn * 0.1 * atr
        entry_zone = tuple(sorted((trigger, trigger + sgn * 0.5 * atr)))
        stop = (behind.low if long else behind.high) - sgn * cfg.stop_buffer_atr * atr
        stop_basis = f"Beyond {'support' if long else 'resistance'} zone {behind.low:.4g}-{behind.high:.4g}"
        status = "watch"
        trig_text = f"No qualifying pattern. Structural {'breakout above' if long else 'breakdown below'} zone {ahead.low:.4g}-{ahead.high:.4g}."
        confirm = f"Completed close {'above' if long else 'below'} {trigger:.4g}."
        inval_text = f"Loss of zone {behind.low:.4g}-{behind.high:.4g} on a closing basis."
        notes.append("Structure-only setup: lower conviction than a pattern-based setup.")

    atr_stop = entry - sgn * cfg.atr_stop_mult * atr
    if sgn * (entry - stop) <= 0:
        return TradeSetup(side=side, valid=False, status="invalid", basis=basis, pattern_id=pid, pattern_stage=stage,
                          entry_trigger=trig_text, confirmation_required=confirm, entry_price=round(entry, 4),
                          stop_loss=round(stop, 4), notes=["Stop is on the wrong side of entry (gap/extension): setup invalid."])
    stop_dist_atr = abs(entry - stop) / atr
    if stop_dist_atr > cfg.max_stop_atr:
        risks.append(f"Technical stop is {stop_dist_atr:.1f} ATR away (wide); position size accordingly or use the "
                     f"ATR alternative {atr_stop:.4g} with the understanding that it sits inside the pattern.")
    if abs(entry - stop) / entry * 100 > cfg.daily_limit_pct:
        risks.append(f"Stop distance exceeds the assumed {cfg.daily_limit_pct}% daily price limit: a limit-locked "
                     "session could prevent exiting at the stop.")

    # targets: nearest opposing zone, measured move, fib extension, next zone
    cand: list[tuple[str, float, str]] = []
    for z in zones_ahead[:2]:
        px = z.low if long else z.high
        if sgn * (px - entry) > 0.5 * atr:
            cand.append(("zone", px, f"{'Resistance' if long else 'Support'} zone {z.low:.4g}-{z.high:.4g}"))
    if measured is not None and sgn * (measured - entry) > 0:
        cand.append(("measured", measured, f"{basis} measured move"))
    for f in fib_ext or []:
        if sgn * (f - entry) > 0:
            cand.append(("fib", f, "Fibonacci extension"))
            break
    seen: list[float] = []
    targets: list[Target] = []
    for _, px, src in sorted(cand, key=lambda x: sgn * x[1]):
        if sgn * (px - entry) < 0.5 * atr or any(abs(px - s) < 0.5 * atr for s in seen):
            continue
        seen.append(px)
        targets.append(Target(label=f"T{len(targets) + 1}", price=round(px, 4),
                              rr=reward_risk(side, entry, stop, px), source=src))
        if len(targets) == 3:
            break
    if pat is not None and sgn * (close - pat.target) >= 0:
        status = "target_reached"
        notes.append("Measured-move target already reached: setup is stale.")
    primary = next((t for t in targets if t.source.endswith("measured move")), targets[-1] if targets else None)
    primary_rr = primary.rr if primary else None
    meets = primary_rr is not None and primary_rr >= cfg.min_rr
    if not targets:
        notes.append("No target with positive reward identified.")
    elif not meets:
        notes.append(f"Primary reward/risk {primary_rr} is below the preferred 1:{cfg.min_rr:g} minimum: lower-quality candidate. "
                     "Stops and targets were not adjusted to improve it.")
    if zones_ahead and targets and sgn * (zones_ahead[0].mid - entry) < sgn * (targets[0].price - entry):
        risks.append("Opposing zone sits before the first target.")
    risks.append("Thin liquidity, trading halts or gaps can cause fills worse than the levels shown.")
    execution = None
    if not long:
        execution = ("Eligible for short selling per operator configuration — verify current eligibility, uptick/price "
                     "rules and account permissions with your broker." if short_eligible else
                     "ANALYTICAL ONLY: shorting on PSX is restricted to eligible securities/instruments and permitted "
                     "accounts. Eligibility for this symbol is not verified; use as a bearish/exit scenario.")
    conf = (pat.quality_score * STAGE_W[pat.stage] if pat else 35.0)
    if not meets:
        conf *= 0.8
    return TradeSetup(
        side=side, status=status, basis=basis, pattern_id=pid, pattern_stage=stage, entry_trigger=trig_text,
        entry_price=round(entry, 4), entry_zone=(round(entry_zone[0], 4), round(entry_zone[1], 4)),
        confirmation_required=confirm, stop_loss=round(stop, 4), stop_basis=stop_basis,
        atr_stop_alternative=round(atr_stop, 4), technical_invalidation=inval_text, targets=targets,
        primary_rr=primary_rr, meets_min_rr=meets, confidence=round(conf, 1), key_risks=risks, execution=execution,
        notes=notes,
    )
