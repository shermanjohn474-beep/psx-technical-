"""Alert evaluation on completed candles, with duplicate suppression, stale-data
safeguards and pluggable notification adapters.

Monitoring is only "active" when the scheduler is enabled AND a data source that
is actually updated is configured. ``scheduler_status()`` reports exactly that.
"""
from __future__ import annotations

import smtplib
import threading
from datetime import datetime, timezone
from email.message import EmailMessage

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.analysis.service import AnalysisOptions, analyze_frame
from app.config.settings import get_settings
from app.data_providers.base import DataUnavailable, MarketDataProvider
from app.database.models import AlertEvent, AlertRule
from app.database.repository import session_scope
from app.schemas.market import Timeframe
from app.schemas.patterns import Direction, PatternStage

CONDITIONS = {
    "break_resistance": "Completed close above the immediate resistance zone",
    "break_support": "Completed close below the immediate support zone",
    "price_above": "Completed close above params.level",
    "price_below": "Completed close below params.level",
    "pattern_confirmed": "A pattern (optionally params.pattern_id) confirmed on the latest completed bar",
    "pattern_invalidated": "A pattern failed or was invalidated on the latest completed bar",
    "bullish_rsi_divergence": "Regular bullish RSI divergence signalled on the latest completed bar",
    "bearish_macd_divergence": "Regular bearish MACD divergence signalled on the latest completed bar",
    "cross_sma50": "Close crossed SMA50 (either direction)",
    "cross_sma200": "Close crossed SMA200 (either direction)",
    "ma_crossover": "SMA(params.fast, default 50) crossed SMA(params.slow, default 200)",
    "unusual_volume": "Relative volume >= params.mult (default 2.0)",
    "breakout_retest": "Price is retesting a confirmed breakout level",
    "bullish_reversal": "A bullish reversal pattern confirmed on the latest completed bar",
    "bearish_reversal": "A bearish reversal pattern confirmed on the latest completed bar",
}


class AlertHit(BaseModel):
    symbol: str
    timeframe: str
    condition: str
    description: str
    trigger_level: float | None
    bar_timestamp: str
    data_as_of: str
    evidence: list[str] = Field(default_factory=list)
    link: str


def completed_frame(df: pd.DataFrame, last_complete: bool) -> pd.DataFrame:
    return df if last_complete else df.iloc[:-1]


def evaluate_condition(df: pd.DataFrame, rule: AlertRule, report=None) -> tuple[bool, float | None, list[str]]:
    c = df["close"]
    last, prev = float(c.iloc[-1]), float(c.iloc[-2]) if len(c) > 1 else None
    p = rule.params or {}
    cond = rule.condition
    n_last = len(df) - 1
    if cond in ("price_above", "price_below"):
        lvl = float(p["level"])
        hit = (last > lvl >= prev) if cond == "price_above" else (last < lvl <= prev)
        return hit, lvl, [f"Close {last:.4g} vs level {lvl:.4g}"]
    if cond in ("cross_sma50", "cross_sma200", "ma_crossover"):
        if cond == "ma_crossover":
            f, s = c.rolling(int(p.get("fast", 50))).mean(), c.rolling(int(p.get("slow", 200))).mean()
        else:
            f, s = c, c.rolling(50 if cond == "cross_sma50" else 200).mean()
        if np.isnan(s.iloc[-1]) or np.isnan(s.iloc[-2]):
            return False, None, ["Insufficient history for moving average"]
        d0, d1 = f.iloc[-2] - s.iloc[-2], f.iloc[-1] - s.iloc[-1]
        hit = d0 * d1 < 0
        return hit, float(s.iloc[-1]), [f"{'Bullish' if d1 > 0 else 'Bearish'} cross; MA {s.iloc[-1]:.4g}"]
    if cond == "unusual_volume":
        v = df["volume"]
        avg = v.shift(1).rolling(20).mean().iloc[-1]
        if not np.isfinite(avg) or avg <= 0:
            return False, None, ["Volume unavailable"]
        rel = float(v.iloc[-1] / avg)
        return rel >= float(p.get("mult", 2.0)), None, [f"Relative volume {rel:.2f}x"]
    rep = report
    if cond == "break_resistance":
        # zones are computed on data excluding the latest bar, so the broken zone is the prior one
        return (prev is not None and _prior_zone_break(df, up=True)), None, ["Close above prior resistance zone"]
    if cond == "break_support":
        return (prev is not None and _prior_zone_break(df, up=False)), None, ["Close below prior support zone"]
    if cond in ("pattern_confirmed", "bullish_reversal", "bearish_reversal"):
        for pt in rep.patterns + rep.historical_patterns:
            if pt.confirmation_index != n_last:
                continue
            if cond == "pattern_confirmed" and p.get("pattern_id") and pt.pattern_id != p["pattern_id"]:
                continue
            if cond == "bullish_reversal" and not (pt.direction == Direction.BULLISH and pt.category.value == "reversal"):
                continue
            if cond == "bearish_reversal" and not (pt.direction == Direction.BEARISH and pt.category.value == "reversal"):
                continue
            return True, pt.breakout_level, [f"{pt.name} confirmed (quality {pt.quality_score:.0f})"] + pt.evidence[:3]
        return False, None, []
    if cond == "pattern_invalidated":
        for pt in rep.patterns:
            if pt.stage in (PatternStage.FAILED, PatternStage.INVALIDATED) and pt.end_index == n_last:
                return True, pt.invalidation_level, [f"{pt.name} {pt.stage.value}"] + pt.evidence[-2:]
        return False, None, []
    if cond == "breakout_retest":
        for pt in rep.patterns:
            if pt.stage == PatternStage.RETESTING:
                return True, pt.breakout_level, [f"{pt.name} retesting {pt.breakout_level}"]
        return False, None, []
    if cond in ("bullish_rsi_divergence", "bearish_macd_divergence"):
        want = ("rsi", "regular_bullish") if cond == "bullish_rsi_divergence" else ("macd", "regular_bearish")
        for d in rep.divergences:
            if d.signal_index == n_last and (d.indicator, d.type) == want:
                return True, d.price2, [f"{d.type} {d.indicator}: pivots {d.pivot1_time[:10]} / {d.pivot2_time[:10]}"]
        return False, None, []
    raise ValueError(f"Unknown condition {cond}")


def _prior_zone_break(df: pd.DataFrame, up: bool) -> bool:
    from app.support_resistance.levels import detect_levels

    lv = detect_levels(df.iloc[:-1])
    last, prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
    if up:
        z = lv.immediate_resistance
        return z is not None and last > z.high >= prev
    z = lv.immediate_support
    return z is not None and last < z.low <= prev


def evaluate_rule(rule: AlertRule, provider: MarketDataProvider, now: datetime | None = None) -> tuple[str, AlertHit | None]:
    s = get_settings()
    tf = Timeframe(rule.timeframe)
    try:
        res = provider.get_ohlcv(rule.symbol, tf)
    except DataUnavailable as exc:
        return f"no_data: {exc}"[:200], None
    q = res.quality
    if q.stale and not res.source.is_synthetic:
        return f"stale_data: latest bar {q.last_timestamp} ({q.staleness_days} days old) - not evaluated", None
    df = completed_frame(res.frame, q.last_candle_complete)
    if len(df) < 30:
        return "insufficient_data", None
    rep = None
    if rule.condition not in ("price_above", "price_below", "cross_sma50", "cross_sma200", "ma_crossover", "unusual_volume",
                              "break_resistance", "break_support"):
        rep = analyze_frame(df, symbol=rule.symbol, timeframe=tf, source=res.source, quality=q,
                            options=AnalysisOptions(with_mtf=False))
    hit, level, evidence = evaluate_condition(df, rule, rep)
    if not hit:
        return "evaluated: no trigger", None
    bar_ts = df.index[-1].isoformat()
    if res.source.is_synthetic:
        evidence.append("SYNTHETIC DEMO DATA")
    return "triggered", AlertHit(symbol=rule.symbol, timeframe=rule.timeframe, condition=rule.condition,
                                 description=CONDITIONS[rule.condition], trigger_level=level, bar_timestamp=bar_ts,
                                 data_as_of=str(q.last_timestamp), evidence=evidence,
                                 link=f"{s.public_base_url}/chart/{rule.symbol}?tf={rule.timeframe}")


# ---------------------------------------------------------------- adapters
def deliver(hit: AlertHit, channels: list[dict]) -> list[dict]:
    out = []
    for ch in channels or []:
        typ = ch.get("type")
        try:
            if typ == "webhook":
                r = httpx.post(ch["url"], json=hit.model_dump(), timeout=10)
                out.append({"type": typ, "ok": r.status_code < 400, "status": r.status_code})
            elif typ == "email":
                out.append(_email(hit, ch["to"]))
            elif typ == "whatsapp":
                out.append(_whatsapp(hit, ch["to"]))
            else:
                out.append({"type": typ, "ok": False, "error": "unknown channel"})
        except Exception as exc:  # delivery failures are recorded, never raised
            out.append({"type": typ, "ok": False, "error": str(exc)[:200]})
    return out


def _text(hit: AlertHit) -> str:
    return (f"[{hit.symbol} {hit.timeframe}] {hit.description}\nTrigger level: {hit.trigger_level}\n"
            f"Bar: {hit.bar_timestamp} | Data as of: {hit.data_as_of}\nEvidence: " + "; ".join(hit.evidence) +
            f"\nChart: {hit.link}\nNot investment advice.")


def _email(hit: AlertHit, to: str) -> dict:
    s = get_settings()
    if not (s.smtp_host and s.smtp_from):
        return {"type": "email", "ok": False, "error": "SMTP not configured"}
    msg = EmailMessage()
    msg["Subject"] = f"PSX alert: {hit.symbol} {hit.condition}"
    msg["From"], msg["To"] = s.smtp_from, to
    msg.set_content(_text(hit))
    with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=15) as smtp:
        smtp.starttls()
        if s.smtp_user:
            smtp.login(s.smtp_user, s.smtp_password or "")
        smtp.send_message(msg)
    return {"type": "email", "ok": True}


def _whatsapp(hit: AlertHit, to: str) -> dict:
    s = get_settings()
    if not (s.whatsapp_api_url and s.whatsapp_api_token):
        return {"type": "whatsapp", "ok": False, "error": "No authorized WhatsApp provider configured"}
    r = httpx.post(s.whatsapp_api_url, json={"to": to, "text": _text(hit)},
                   headers={"Authorization": f"Bearer {s.whatsapp_api_token}"}, timeout=10)
    return {"type": "whatsapp", "ok": r.status_code < 400, "status": r.status_code}


def run_alerts(provider: MarketDataProvider, rule_ids: list[int] | None = None) -> list[dict]:
    """Evaluate enabled rules once; store new (non-duplicate) events and deliver them."""
    out = []
    with session_scope() as s:
        q = select(AlertRule).where(AlertRule.enabled.is_(True))
        if rule_ids:
            q = q.where(AlertRule.id.in_(rule_ids))
        rules = list(s.scalars(q))
    for rule in rules:
        status, hit = evaluate_rule(rule, provider)
        entry = {"rule_id": rule.id, "symbol": rule.symbol, "condition": rule.condition, "status": status}
        if hit is not None:
            key = f"{rule.id}|{rule.condition}|{hit.bar_timestamp}"
            try:
                with session_scope() as s:
                    ev = AlertEvent(rule_id=rule.id, dedup_key=key, bar_timestamp=hit.bar_timestamp, payload=hit.model_dump())
                    s.add(ev)
                    s.flush()
                    ev_id = ev.id
                deliveries = deliver(hit, rule.channels)
                with session_scope() as s:
                    s.get(AlertEvent, ev_id).deliveries = deliveries
                entry.update(status="triggered", event_id=ev_id, deliveries=deliveries, hit=hit.model_dump())
            except IntegrityError:
                entry["status"] = "duplicate_suppressed"
        with session_scope() as s:
            r = s.get(AlertRule, rule.id)
            r.last_evaluated_at = datetime.now(timezone.utc)
            r.last_status = entry["status"][:200]
        out.append(entry)
    return out


class _Scheduler:
    def __init__(self):
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.last_run: str | None = None
        self.last_error: str | None = None

    def start(self, provider: MarketDataProvider, interval: int) -> None:
        if self.thread and self.thread.is_alive():
            return

        def loop():
            while not self.stop.is_set():
                try:
                    run_alerts(provider)
                    self.last_run = datetime.now(timezone.utc).isoformat()
                except Exception as exc:  # pragma: no cover
                    self.last_error = str(exc)[:200]
                self.stop.wait(interval)

        self.thread = threading.Thread(target=loop, daemon=True, name="alert-scheduler")
        self.thread.start()

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())


SCHEDULER = _Scheduler()


def scheduler_status(provider) -> dict:
    s = get_settings()
    live = bool(getattr(provider, "extra", []))
    msg = []
    if not SCHEDULER.running:
        msg.append("Scheduler not running: alerts are evaluated only on demand (POST /api/alerts/evaluate).")
    if not live:
        msg.append("No live/authorized data feed configured: alerts evaluate against local files and only fire "
                   "when those files are updated. Continuous market monitoring is NOT active.")
    return {"scheduler_enabled": s.alerts_scheduler_enabled, "scheduler_running": SCHEDULER.running,
            "interval_seconds": s.alerts_interval_seconds, "last_run": SCHEDULER.last_run,
            "last_error": SCHEDULER.last_error, "live_data_source_configured": live,
            "monitoring_active": SCHEDULER.running and live, "notes": msg}
