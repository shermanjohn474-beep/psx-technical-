"""LLM analyst layer: explains validated findings; never adds new facts.

Flow: report -> compact evidence JSON -> provider.complete_json(schema) ->
schema validation (pydantic) -> evidence validation (every number in the text
must match a number in the evidence within tolerance; every named pattern must
be in the report) -> accepted commentary, or rejection with the deterministic
commentary kept. Every call is written to the audit log with model id, prompt
and evidence hashes, output and validation result.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.ai_analyst.providers import AIProviderError, LLMProvider, NullProvider
from app.chart_patterns.registry import DETECTORS
from app.config.settings import get_settings
from app.schemas.report import AnalysisReport, Commentary

PROMPT_VERSION = "analyst-v1"

SYSTEM = """You are a disciplined Chartered Market Technician writing for an institutional research desk covering the Pakistan Stock Exchange.
You receive VALIDATED quantitative findings as JSON. Rules:
- Use ONLY numbers, levels, indicators and patterns present in the evidence. Never invent prices, dates, indicator values, volumes or patterns.
- Quote prices exactly as given in the evidence (you may round to 2 decimals).
- Distinguish confirmed patterns from forming/approaching ones. Never call an unconfirmed pattern confirmed.
- Present both bullish and bearish evidence. Use conditional language ("if price closes above X..."). No guarantees, no probabilities of success.
- Confluence scores are heuristics, not probabilities.
- If evidence is mixed, insufficient or reward/risk is poor, say so; "NEUTRAL - NO TRADE - AWAITING CONFIRMATION" is a valid conclusion.
- A bearish scenario is analytical unless the evidence states the symbol is short-eligible.
Return JSON matching the schema."""

SCHEMA = {
    "type": "object",
    "properties": {
        "bias": {"type": "string", "enum": ["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"]},
        "summary": {"type": "string"},
        "chart_reading": {"type": "string"},
        "pattern_assessment": {"type": "string"},
        "bullish_confirmation": {"type": "string"},
        "bearish_confirmation": {"type": "string"},
        "key_levels": {"type": "string"},
        "trade_plan": {"type": "string"},
        "strongest_evidence": {"type": "array", "items": {"type": "string"}},
        "conflicting_signals": {"type": "array", "items": {"type": "string"}},
        "what_changes_view": {"type": "array", "items": {"type": "string"}},
        "stay_neutral": {"type": "boolean"},
        "conclusion": {"type": "string"},
    },
    "required": ["bias", "summary", "chart_reading", "pattern_assessment", "bullish_confirmation", "bearish_confirmation",
                 "key_levels", "trade_plan", "strongest_evidence", "conflicting_signals", "what_changes_view",
                 "stay_neutral", "conclusion"],
    "additionalProperties": False,
}


class AICommentary(BaseModel):
    bias: str
    summary: str
    chart_reading: str
    pattern_assessment: str
    bullish_confirmation: str
    bearish_confirmation: str
    key_levels: str
    trade_plan: str
    strongest_evidence: list[str]
    conflicting_signals: list[str]
    what_changes_view: list[str]
    stay_neutral: bool
    conclusion: str


def build_evidence(r: AnalysisReport) -> dict:
    def setup(s):
        return None if not s.valid else {
            "side": s.side, "status": s.status, "basis": s.basis, "entry_trigger": s.entry_trigger,
            "entry_price": s.entry_price, "entry_zone": s.entry_zone, "stop_loss": s.stop_loss,
            "targets": [{"label": t.label, "price": t.price, "rr": t.rr} for t in s.targets],
            "primary_rr": s.primary_rr, "meets_min_rr_1_to_2": s.meets_min_rr, "execution": s.execution,
            "risks": s.key_risks}
    return {
        "symbol": r.overview.symbol, "company": r.overview.company_name, "timeframe": r.overview.timeframe,
        "last_price": r.overview.price, "change_pct": r.overview.change_pct,
        "data_timestamp": r.overview.data_timestamp.isoformat(),
        "data_source": {"provider": r.overview.data_source.provider, "synthetic": r.overview.data_source.is_synthetic},
        "data_quality": {"rating": r.data_quality.rating if r.data_quality else None,
                         "issues": [i.message for i in (r.data_quality.issues if r.data_quality else [])][:6]},
        "timeframes": [{"tf": t.label, "trend": t.trend, "score": t.score, "structure": t.structure}
                       for t in (r.trend.timeframes if r.trend else []) if t.trend != "unavailable"],
        "alignment": r.trend.alignment if r.trend else None,
        "indicators": {k: v for k, v in r.indicators.items() if k != "unavailable"},
        "patterns": [{"name": p.name, "stage": p.stage.value, "direction": p.direction.value,
                      "trigger": p.breakout_level, "target": p.target, "invalidation": p.invalidation_level,
                      "quality": p.quality_score, "volume_confirmed": p.volume_confirmation,
                      "experimental": p.experimental} for p in r.patterns[:8]],
        "candlesticks": [{"name": c.name, "date": c.timestamp[:10], "score": c.score, "confirmation": c.confirmation}
                         for c in r.candlesticks[-5:]],
        "divergences": [{"type": d.type, "indicator": d.indicator, "date": d.signal_time[:10]} for d in r.divergences[-4:]],
        "key_levels": r.key_levels.model_dump(),
        "long_setup": setup(r.long_setup), "short_setup": setup(r.short_setup),
        "confluence": {"score": r.score.score, "rating": r.score.rating, "coverage": r.score.coverage,
                       "components": {c.name: {"value": c.value, "reasons": c.reasons} for c in r.score.components}},
        "warnings": r.warnings,
    }


_NUM = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*\.?\d*")


def _numbers(obj) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        for m in _NUM.findall(obj):
            try:
                out.append(float(m.replace(",", "")))
            except ValueError:
                pass
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out += _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers(v)
    return out


def validate_against_evidence(c: AICommentary, evidence: dict, rel_tol: float = 0.006) -> dict:
    allowed = _numbers(evidence)
    text_fields = [c.summary, c.chart_reading, c.pattern_assessment, c.bullish_confirmation, c.bearish_confirmation,
                   c.key_levels, c.trade_plan, c.conclusion, *c.strongest_evidence, *c.conflicting_signals,
                   *c.what_changes_view]
    unsupported = []
    for txt in text_fields:
        for v in _numbers(txt):
            if abs(v) <= 10 and float(v).is_integer():
                continue  # small counts / ordinals ("two tops", "1:2")
            if v in (14, 20, 26, 50, 100, 200, 12, 9):  # standard indicator periods
                continue
            if not any(abs(v - a) <= max(rel_tol * abs(a), 0.011) for a in allowed):
                unsupported.append(v)
    known = {p["name"].lower() for p in evidence.get("patterns", [])} | {x["name"].lower() for x in evidence.get("candlesticks", [])}
    vocab = set()
    for d in DETECTORS:
        for pid in d.pattern_ids:
            vocab.add(pid.replace("_", " "))
    text = " ".join(text_fields).lower()
    unknown_patterns = [v for v in vocab if v in text and not any(v in k or k in v for k in known)
                        and v not in ("rectangle", "measured move up", "measured move down")]
    bias_ok = c.bias == evidence["confluence"]["rating"] or c.stay_neutral
    return {"unsupported_numbers": sorted(set(unsupported))[:20], "unknown_patterns": sorted(unknown_patterns),
            "bias_consistent": bias_ok,
            "passed": not unsupported and not unknown_patterns}


def _audit(record: dict) -> None:
    path = Path(get_settings().data_dir) / "audit"
    path.mkdir(parents=True, exist_ok=True)
    with (path / "ai_audit.jsonl").open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    try:  # also persist to the database when available
        from app.database.repository import record_ai_audit
        record_ai_audit(record)
    except Exception:
        pass


def ai_commentary(report: AnalysisReport, provider: LLMProvider) -> tuple[Commentary, dict]:
    """Return (commentary, audit_record). Falls back to deterministic commentary on any failure."""
    evidence = build_evidence(report)
    ev_json = json.dumps(evidence, sort_keys=True, default=str)
    prompt = ("Write the analyst commentary for this validated evidence. Cover: what the chart shows, the pattern and "
              "whether it is confirmed, what confirms the bullish and bearish scenarios, key levels, potential entries, "
              "stops/invalidation, targets and reward/risk, and conflicting or missing evidence.\n\nEVIDENCE:\n" + ev_json)
    record = {"ts": datetime.now(timezone.utc).isoformat(), "report_id": report.report_id, "symbol": report.overview.symbol,
              "prompt_version": PROMPT_VERSION, "provider": provider.name, "model": getattr(provider, "model", None),
              "evidence_sha256": hashlib.sha256(ev_json.encode()).hexdigest(),
              "prompt_sha256": hashlib.sha256((SYSTEM + prompt).encode()).hexdigest()}
    fallback = report.commentary
    if isinstance(provider, NullProvider):
        record.update(status="skipped", reason="no AI provider configured")
        return fallback, record
    try:
        res = provider.complete_json(SYSTEM, prompt, SCHEMA)
        record.update(model=res.model, latency_s=res.latency_s, usage=res.usage, output=res.data)
        c = AICommentary.model_validate(res.data)
    except (AIProviderError, ValidationError) as exc:
        record.update(status="error", error=str(exc)[:300])
        _audit(record)
        fallback.validation = {**fallback.validation, "ai_error": str(exc)[:200]}
        return fallback, record
    val = validate_against_evidence(c, evidence)
    record["validation"] = val
    if not val["passed"]:
        record["status"] = "rejected"
        _audit(record)
        fallback.validation = {**fallback.validation, "ai_rejected": val}
        return fallback, record
    record["status"] = "accepted"
    _audit(record)
    return Commentary(
        engine=f"{res.provider}:{res.model}", bias=c.bias,
        summary=" ".join([c.summary, c.chart_reading, c.pattern_assessment]),
        strongest_evidence=c.strongest_evidence, conflicting_signals=c.conflicting_signals,
        what_changes_view=c.what_changes_view + [c.bullish_confirmation, c.bearish_confirmation],
        confirmation_needed=[c.bullish_confirmation, c.bearish_confirmation], preferred_setup=c.trade_plan,
        stay_neutral=c.stay_neutral, conclusion=c.conclusion,
        validation={**val, "deterministic_bias": report.score.rating, "key_levels_text": c.key_levels},
    ), record
