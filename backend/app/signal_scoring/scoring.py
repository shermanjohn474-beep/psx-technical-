"""Transparent confluence scoring.

Each component produces a directional value in [-1, +1] (+1 = fully bullish
evidence) with a written reason. The bias score is ``50 + 50 x weighted mean`` over
*available* components (0 = strong bearish, 50 = neutral, 100 = strong bullish).
Missing components are excluded and reported; coverage = available weight / total.

These weights are initial heuristics, **not calibrated probabilities**: a score of
80 does not mean an 80% chance of success. Historical reliability is measured
separately by the backtesting module on out-of-sample data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from app.divergences.divergence import Divergence
from app.multi_timeframe.mtf import MTFResult
from app.risk_management.setups import TradeSetup
from app.schemas.patterns import Direction, PatternResult, PatternStage
from app.support_resistance.levels import LevelsResult

DEFAULT_WEIGHTS = {"htf_trend": 20, "pattern": 20, "momentum": 15, "volume": 15, "sr_alignment": 15, "risk_reward": 15}


class ScoreComponent(BaseModel):
    name: str
    weight: float
    value: float | None  # -1..1, None = unavailable
    points: float | None  # contribution on a 0..weight scale (bullish orientation)
    reasons: list[str] = Field(default_factory=list)


class SignalScore(BaseModel):
    score: float  # 0..100, 50 = neutral
    rating: str
    coverage: float
    components: list[ScoreComponent]
    disclaimer: str = ("Heuristic confluence score; not a probability of success. Historical reliability must be "
                       "evaluated separately via out-of-sample backtests.")


@dataclass
class ScoreConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    min_coverage_for_rating: float = 0.5


STAGE_FACTOR = {PatternStage.CONFIRMED: 1.0, PatternStage.RETESTING: 1.0, PatternStage.APPROACHING_CONFIRMATION: 0.6,
                PatternStage.FORMING: 0.3, PatternStage.FAILED: -0.5, PatternStage.INVALIDATED: 0.0}


def rating_for(score: float, coverage: float, cfg: ScoreConfig) -> str:
    if coverage < cfg.min_coverage_for_rating:
        return "Neutral"
    if score >= 75:
        return "Strong Bullish"
    if score >= 60:
        return "Bullish"
    if score <= 25:
        return "Strong Bearish"
    if score <= 40:
        return "Bearish"
    return "Neutral"


def compute_score(*, mtf: MTFResult | None, patterns: list[PatternResult], snapshot: dict,
                  divergences: list[Divergence], levels: LevelsResult, long_setup: TradeSetup | None,
                  short_setup: TradeSetup | None, last_index: int, close: float, atr: float,
                  cfg: ScoreConfig | None = None) -> SignalScore:
    cfg = cfg or ScoreConfig()
    W = cfg.weights
    comps: list[ScoreComponent] = []

    # 1. Higher-timeframe trend
    reasons: list[str] = []
    val = None
    if mtf is not None:
        htf = [t for t in mtf.timeframes if t.timeframe in ("1M", "1w", "1d") and t.trend != "unavailable"]
        if htf:
            wts = {"1M": 1.0, "1w": 1.5, "1d": 1.0}
            val = sum(wts[t.timeframe] * t.score / 100 for t in htf) / sum(wts[t.timeframe] for t in htf)
            reasons = [f"{t.label}: {t.trend} ({t.score:+.0f})" for t in htf] + [f"Alignment: {mtf.alignment}"]
    comps.append(_comp("htf_trend", W, val, reasons or ["No timeframe trend available."]))

    # 2. Pattern geometry & confirmation (recent/active only)
    recent = [p for p in patterns if p.end_index >= last_index - 40 and p.direction != Direction.NEUTRAL]
    val, reasons = None, []
    if recent:
        tot = 0.0
        for p in sorted(recent, key=lambda p: -p.quality_score)[:5]:
            sgn = 1 if p.direction == Direction.BULLISH else -1
            f = STAGE_FACTOR[p.stage] * (0.6 if p.experimental else 1.0)
            if p.target_reached:
                f *= 0.3
            tot += sgn * f * p.quality_score / 100
            if f:
                reasons.append(f"{p.name} ({p.stage.value}, quality {p.quality_score:.0f})"
                               + (" — failed, counts against its direction" if p.stage == PatternStage.FAILED else ""))
        val = float(np.clip(tot, -1, 1))
    comps.append(_comp("pattern", W, val, reasons or ["No recent directional pattern."]))
    if not recent:
        comps[-1].value, comps[-1].points = 0.0, W["pattern"] / 2

    # 3. Momentum
    val, reasons, parts = None, [], []
    r = snapshot.get("rsi")
    if r:
        rv = r["value"]
        x = float(np.clip((rv - 50) / 20, -1, 1))
        if rv >= 75 or rv <= 25:
            x *= 0.5
            reasons.append(f"RSI {rv:.1f} at an extreme ({r['zone']}): momentum strong but stretched")
        else:
            reasons.append(f"RSI {rv:.1f} ({r['direction']})")
        parts.append(x)
    m = snapshot.get("macd")
    if m:
        x = 0.5 * (1 if m["state"] == "above_signal" else -1) + 0.5 * (1 if m["zero_line"] == "above" else -1)
        parts.append(x)
        reasons.append(f"MACD {m['state'].replace('_', ' ')}, {m['zero_line']} zero" + (f", {m['cross']}" if m["cross"] else ""))
    recent_div = [d for d in divergences if d.signal_index >= last_index - 15 and d.type.startswith("regular")]
    for d in recent_div[-2:]:
        parts.append(0.8 if d.direction == "bullish" else -0.8)
        reasons.append(f"{d.type.replace('_', ' ')} divergence on {d.indicator} ({d.signal_time[:10]})")
    if parts:
        val = float(np.clip(np.mean(parts), -1, 1))
    comps.append(_comp("momentum", W, val, reasons or ["Momentum indicators unavailable."]))

    # 4. Volume confirmation
    v = snapshot.get("volume")
    val, reasons = None, []
    if v:
        parts = []
        if v.get("obv_direction"):
            parts.append({"rising": 0.6, "falling": -0.6}.get(v["obv_direction"], 0.0))
            reasons.append(f"OBV {v['obv_direction']} over 10 bars")
        if v.get("cmf") is not None:
            parts.append(float(np.clip(v["cmf"] * 4, -1, 1)))
            reasons.append(f"Chaikin money flow {v['cmf']:+.2f}")
        conf = [p for p in recent if p.confirmation_index is not None and p.volume_confirmation is not None]
        for p in conf[:2]:
            sgn = 1 if p.direction == Direction.BULLISH else -1
            parts.append(sgn * (0.8 if p.volume_confirmation else -0.2))
            reasons.append(f"{p.name} breakout {'with' if p.volume_confirmation else 'without'} volume confirmation"
                           + (f" ({p.relative_volume}x)" if p.relative_volume else ""))
        if parts:
            val = float(np.clip(np.mean(parts), -1, 1))
    comps.append(_comp("volume", W, val, reasons or ["Volume data unavailable: factor excluded (not treated as favourable)."]))

    # 5. Support/resistance alignment
    val, reasons = None, []
    sup, res = levels.immediate_support, levels.immediate_resistance
    if atr > 0 and (sup or res):
        ds = (close - sup.high) / atr if sup else None
        dr = (res.low - close) / atr if res else None
        if ds is not None and dr is not None:
            val = float(np.clip((dr - ds) / max(dr + ds, 1e-9), -1, 1))
            reasons.append(f"{max(ds, 0):.1f} ATR above support {sup.low:.4g}-{sup.high:.4g}; "
                           f"{max(dr, 0):.1f} ATR below resistance {res.low:.4g}-{res.high:.4g}")
        elif dr is None:
            val = 0.5
            reasons.append("No overhead resistance in analysed history (price discovery).")
        else:
            val = -0.5
            reasons.append("No support below in analysed history.")
    comps.append(_comp("sr_alignment", W, val, reasons or ["No support/resistance zones."]))

    # 6. Risk/reward & volatility
    val, reasons = None, []
    lr = long_setup.primary_rr if long_setup and long_setup.valid else None
    sr = short_setup.primary_rr if short_setup and short_setup.valid else None
    if lr is not None or sr is not None:
        val = float(np.clip(((lr or 0) - (sr or 0)) / 3, -1, 1))
        reasons.append(f"Long R:R {lr if lr is not None else 'n/a'} vs short R:R {sr if sr is not None else 'n/a'}")
        atr_pct = snapshot.get("atr", {}).get("pct_of_price") if snapshot.get("atr") else None
        if atr_pct and atr_pct > 5:
            val *= 0.6
            reasons.append(f"High volatility (ATR {atr_pct:.1f}% of price) reduces conviction")
    comps.append(_comp("risk_reward", W, val, reasons or ["No valid setups for reward/risk comparison."]))

    avail = [c for c in comps if c.value is not None]
    total_w = sum(W.values())
    coverage = sum(c.weight for c in avail) / total_w if total_w else 0.0
    if avail:
        mean = sum(c.value * c.weight for c in avail) / sum(c.weight for c in avail)
    else:
        mean = 0.0
    score = round(50 + 50 * mean, 1)
    return SignalScore(score=score, rating=rating_for(score, coverage, cfg), coverage=round(coverage, 2), components=comps)


def _comp(name: str, W: dict, value: float | None, reasons: list[str]) -> ScoreComponent:
    w = W.get(name, 0)
    return ScoreComponent(name=name, weight=w, value=None if value is None else round(value, 3),
                          points=None if value is None else round(w * (value + 1) / 2, 2), reasons=reasons)
