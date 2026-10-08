"""REST API routes."""
from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.ai_analyst.analyst import ai_commentary
from app.ai_analyst.providers import get_ai_provider
from app.alerts.engine import CONDITIONS, SCHEDULER, run_alerts, scheduler_status
from app.analysis.service import AnalysisOptions, analyze_symbol
from app.api.deps import data_error, provider, require_api_key
from app.backtesting.engine import CostModel, run_backtest, walk_forward
from app.backtesting.strategies import StrategySpec
from app.chart_patterns.base import PatternConfig
from app.chart_patterns.registry import DETECTORS
from app.config.settings import get_settings
from app.data_providers.base import DataUnavailable, ProviderError
from app.data_providers.symbols import known_symbols, resolve_symbol
from app.database.models import AlertEvent, AlertRule, ScreenPreset, Watchlist
from app.database.repository import recent_audit, session_scope
from app.indicators.engine import compute_indicators
from app.reporting.html_report import chart_png, render_html
from app.scanner.scanner import FLAGS, ScanRules, scan
from app.schemas.market import Timeframe
from app.screenshot_analysis.extract import AxisCalibration
from app.screenshot_analysis.service import analyze_screenshots
from app.signal_scoring.scoring import DEFAULT_WEIGHTS, ScoreConfig

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])
MAX_IMAGE = 10 * 1024 * 1024
MAX_TABLE = 25 * 1024 * 1024


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


# ------------------------------------------------------------------ meta / data
@router.get("/symbols")
def symbols(q: str | None = None, prov=Depends(provider)):
    have = set(prov.list_symbols())
    reg = {s.symbol: s for s in known_symbols()}
    out = []
    for sym in sorted(have | set(reg)):
        info = reg.get(sym) or resolve_symbol(sym)
        if q and q.upper() not in sym and (not info.name or q.upper() not in info.name.upper()):
            continue
        out.append({**info.model_dump(), "has_data": sym in have,
                    "timeframes": [t.value for t in prov.available_timeframes(sym)] if sym in have else [],
                    "synthetic": sym.startswith("SYN-")})
    return out


@router.get("/symbols/resolve/{query}")
def resolve(query: str):
    return resolve_symbol(query)


@router.post("/data/upload")
async def upload(file: UploadFile = File(...), symbol: str = Form(...), timeframe: Timeframe = Form(Timeframe.D1),
                 attribution: str | None = Form(None), adjusted: bool | None = Form(None), prov=Depends(provider)):
    content = await file.read()
    if len(content) > MAX_TABLE:
        raise HTTPException(413, "File too large.")
    if symbol.upper().startswith("SYN-"):
        raise HTTPException(400, "The SYN- prefix is reserved for synthetic demo data.")
    try:
        res = prov.user_files.save_upload(symbol, timeframe, content, file.filename or "upload.csv", attribution, adjusted)
    except (DataUnavailable, ValueError) as exc:
        raise data_error(exc)
    return {"symbol": res.symbol, "timeframe": res.timeframe.value, "quality": res.quality, "source": res.source}


@router.get("/ohlcv/{symbol}")
def ohlcv(symbol: str, timeframe: Timeframe = Timeframe.D1, start: str | None = None, end: str | None = None,
          indicators: str = Query("", description="comma-separated indicator columns"), prov=Depends(provider)):
    try:
        res = prov.get_ohlcv(resolve_symbol(symbol).symbol, timeframe, _dt(start), _dt(end))
    except (DataUnavailable, ProviderError) as exc:
        raise data_error(exc)
    df = res.frame
    candles = [{"time": int(ts.timestamp()), "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                "volume": None if r.volume != r.volume else r.volume} for ts, r in df.iterrows()]
    out = {"symbol": res.symbol, "timeframe": timeframe.value, "candles": candles, "quality": res.quality,
           "source": res.source, "notes": res.notes}
    wanted = [x.strip() for x in indicators.split(",") if x.strip()]
    if wanted:
        ind = compute_indicators(df, timeframe)
        out["indicators"] = {k: [None if v != v else round(float(v), 6) for v in ind.frame[k].to_numpy()]
                             for k in wanted if k in ind.frame}
        out["indicators_unavailable"] = {k: ind.unavailable.get(k, "unknown indicator") for k in wanted if k not in ind.frame}
    return out


# ------------------------------------------------------------------ analysis
def _options(weights: str | None = None, include_experimental: bool = True) -> AnalysisOptions:
    opt = AnalysisOptions(include_experimental=include_experimental)
    if weights:
        w = {**DEFAULT_WEIGHTS, **json.loads(weights)}
        opt.score_cfg = ScoreConfig(weights=w)
    return opt


@router.get("/analysis/{symbol}")
def analysis(symbol: str, timeframe: Timeframe = Timeframe.D1, start: str | None = None, end: str | None = None,
             ai: bool = False, weights: str | None = None, include_experimental: bool = True, prov=Depends(provider)):
    try:
        rep = analyze_symbol(prov, symbol, timeframe, _dt(start), _dt(end), _options(weights, include_experimental))
    except (DataUnavailable, ProviderError, ValueError) as exc:
        raise data_error(exc)
    audit = None
    if ai:
        rep.commentary, audit = ai_commentary(rep, get_ai_provider())
    return {"report": rep, "ai_audit": audit}


@router.get("/analysis/{symbol}/chart.png")
def analysis_chart(symbol: str, timeframe: Timeframe = Timeframe.D1, prov=Depends(provider)):
    try:
        rep = analyze_symbol(prov, symbol, timeframe, options=AnalysisOptions(with_mtf=False))
        df = prov.get_ohlcv(rep.overview.symbol, timeframe).frame
    except (DataUnavailable, ProviderError) as exc:
        raise data_error(exc)
    return Response(chart_png(df, rep), media_type="image/png")


@router.get("/report/{symbol}", response_class=HTMLResponse)
def report_html(symbol: str, timeframe: Timeframe = Timeframe.D1, ai: bool = False, prov=Depends(provider)):
    try:
        rep = analyze_symbol(prov, symbol, timeframe)
        df = prov.get_ohlcv(rep.overview.symbol, timeframe).frame
    except (DataUnavailable, ProviderError) as exc:
        raise data_error(exc)
    if ai:
        rep.commentary, _ = ai_commentary(rep, get_ai_provider())
    return HTMLResponse(render_html(rep, df))


# ------------------------------------------------------------------ screenshots
@router.post("/screenshot")
async def screenshot(files: list[UploadFile] = File(...), symbol: str | None = Form(None),
                     timeframe: Timeframe | None = Form(None), calibration: str | None = Form(None),
                     log_scale: bool | None = Form(None), use_vision: bool = Form(True), prov=Depends(provider)):
    items = []
    for f in files[:6]:
        data = await f.read()
        if len(data) > MAX_IMAGE:
            raise HTTPException(413, f"{f.filename}: image too large (max 10 MB).")
        items.append((data, f.filename or "image"))
    cal = AxisCalibration.model_validate_json(calibration) if calibration else None
    try:
        out = analyze_screenshots(items, symbol=resolve_symbol(symbol).symbol if symbol else None, timeframe=timeframe,
                                  calibration=cal, vision_provider=get_ai_provider() if use_vision else None,
                                  data_provider=prov, log_scale=log_scale)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return out


# ------------------------------------------------------------------ scanner / watchlists
class ScanRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    watchlist: str | None = None
    timeframe: Timeframe = Timeframe.D1
    rules: ScanRules = Field(default_factory=ScanRules)
    with_mtf: bool = True


@router.post("/scan")
def run_scan(req: ScanRequest, prov=Depends(provider)):
    syms = list(req.symbols)
    if req.watchlist:
        with session_scope() as s:
            wl = s.scalar(select(Watchlist).where(Watchlist.name == req.watchlist))
            if wl is None:
                raise HTTPException(404, "Watchlist not found.")
            syms += wl.symbols
    if not syms:
        syms = prov.list_symbols()
    return scan(prov, sorted(set(s.upper() for s in syms)), req.timeframe, req.rules, req.with_mtf)


@router.get("/scan/flags")
def scan_flags():
    return {"flags": FLAGS, "patterns": [pid for d in DETECTORS for pid in d.pattern_ids]}


class WatchlistIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    symbols: list[str]


@router.get("/watchlists")
def list_watchlists():
    with session_scope() as s:
        return [{"id": w.id, "name": w.name, "symbols": w.symbols} for w in s.scalars(select(Watchlist))]


@router.post("/watchlists")
def save_watchlist(w: WatchlistIn):
    with session_scope() as s:
        cur = s.scalar(select(Watchlist).where(Watchlist.name == w.name))
        syms = sorted({resolve_symbol(x).symbol for x in w.symbols})
        if cur:
            cur.symbols = syms
        else:
            s.add(Watchlist(name=w.name, symbols=syms))
    return {"name": w.name, "symbols": syms}


@router.delete("/watchlists/{name}")
def delete_watchlist(name: str):
    with session_scope() as s:
        cur = s.scalar(select(Watchlist).where(Watchlist.name == name))
        if not cur:
            raise HTTPException(404, "Watchlist not found.")
        s.delete(cur)
    return {"deleted": name}


class PresetIn(BaseModel):
    name: str
    rules: ScanRules


@router.get("/scan/presets")
def list_presets():
    with session_scope() as s:
        return [{"name": p.name, "rules": p.rules} for p in s.scalars(select(ScreenPreset))]


@router.post("/scan/presets")
def save_preset(p: PresetIn):
    with session_scope() as s:
        cur = s.scalar(select(ScreenPreset).where(ScreenPreset.name == p.name))
        if cur:
            cur.rules = p.rules.model_dump()
        else:
            s.add(ScreenPreset(name=p.name, rules=p.rules.model_dump()))
    return p


# ------------------------------------------------------------------ backtests
class BacktestRequest(BaseModel):
    symbols: list[str]
    timeframe: Timeframe = Timeframe.D1
    strategy: StrategySpec = Field(default_factory=StrategySpec)
    costs: CostModel = Field(default_factory=CostModel)
    walk_forward: bool = False
    folds: int = 4
    start: str | None = None
    end: str | None = None


@router.post("/backtest")
def backtest(req: BacktestRequest, prov=Depends(provider)):
    results, errors = [], {}
    for sym in req.symbols[:20]:
        try:
            df = prov.get_ohlcv(resolve_symbol(sym).symbol, req.timeframe, _dt(req.start), _dt(req.end)).frame
            r = run_backtest(df, req.strategy, symbol=sym.upper(), timeframe=req.timeframe.value, costs=req.costs)
            item = {"result": r}
            if req.walk_forward:
                item["walk_forward"] = walk_forward(df, req.strategy, symbol=sym.upper(), costs=req.costs, folds=req.folds)
            results.append(item)
        except (DataUnavailable, ProviderError, ValueError) as exc:
            errors[sym] = str(exc)[:200]
    by_symbol = [{"symbol": x["result"].symbol, "trades": x["result"].metrics.trades,
                  "expectancy_r": x["result"].metrics.expectancy_r, "win_rate": x["result"].metrics.win_rate,
                  "profit_factor": x["result"].metrics.profit_factor} for x in results]
    return {"results": results, "by_symbol": by_symbol, "errors": errors,
            "disclaimer": "Simulated results on historical data; not live performance."}


# ------------------------------------------------------------------ alerts
class AlertIn(BaseModel):
    symbol: str
    timeframe: Timeframe = Timeframe.D1
    condition: str
    params: dict = Field(default_factory=dict)
    channels: list[dict] = Field(default_factory=list)
    enabled: bool = True


@router.get("/alerts/conditions")
def alert_conditions():
    return CONDITIONS


@router.get("/alerts")
def list_alerts():
    with session_scope() as s:
        return [{"id": a.id, "symbol": a.symbol, "timeframe": a.timeframe, "condition": a.condition, "params": a.params,
                 "channels": a.channels, "enabled": a.enabled, "last_evaluated_at": a.last_evaluated_at,
                 "last_status": a.last_status} for a in s.scalars(select(AlertRule))]


@router.post("/alerts")
def create_alert(a: AlertIn):
    if a.condition not in CONDITIONS:
        raise HTTPException(400, f"Unknown condition. Options: {sorted(CONDITIONS)}")
    if a.condition in ("price_above", "price_below") and "level" not in a.params:
        raise HTTPException(400, "params.level is required.")
    for ch in a.channels:
        if ch.get("type") not in ("webhook", "email", "whatsapp"):
            raise HTTPException(400, "Channel type must be webhook, email or whatsapp.")
    with session_scope() as s:
        r = AlertRule(symbol=resolve_symbol(a.symbol).symbol, timeframe=a.timeframe.value, condition=a.condition,
                      params=a.params, channels=a.channels, enabled=a.enabled)
        s.add(r)
        s.flush()
        return {"id": r.id}


@router.delete("/alerts/{rule_id}")
def delete_alert(rule_id: int):
    with session_scope() as s:
        r = s.get(AlertRule, rule_id)
        if not r:
            raise HTTPException(404, "Alert not found.")
        s.delete(r)
    return {"deleted": rule_id}


@router.post("/alerts/evaluate")
def evaluate_alerts(prov=Depends(provider)):
    return {"results": run_alerts(prov), "status": scheduler_status(prov)}


@router.get("/alerts/events")
def alert_events(limit: int = 100):
    with session_scope() as s:
        return [{"id": e.id, "rule_id": e.rule_id, "triggered_at": e.triggered_at, "bar_timestamp": e.bar_timestamp,
                 "payload": e.payload, "deliveries": e.deliveries}
                for e in s.scalars(select(AlertEvent).order_by(AlertEvent.id.desc()).limit(limit))]


@router.get("/alerts/status")
def alert_status(prov=Depends(provider)):
    return scheduler_status(prov)


# ------------------------------------------------------------------ config / audit
@router.get("/config")
def config():
    s = get_settings()
    pc = PatternConfig()
    return {
        "ai_provider": s.ai_provider if (s.anthropic_api_key or s.openai_api_key) else "none",
        "ai_model": s.anthropic_model if s.ai_provider == "anthropic" else s.openai_model if s.ai_provider == "openai" else None,
        "auth_required": bool(s.api_keys),
        "scoring_weights": DEFAULT_WEIGHTS,
        "pattern_defaults": {k: v for k, v in pc.__dict__.items() if k not in ("swing", "params")},
        "swing_defaults": pc.swing.__dict__,
        "detectors": [{"class": type(d).__name__, "patterns": list(d.pattern_ids), "experimental": d.experimental}
                      for d in DETECTORS],
        "short_eligible_symbols": s.short_eligible_symbols,
        "timezone": s.timezone,
    }


@router.get("/audit")
def audit(limit: int = 100):
    return [{"ts": a.ts, "actor": a.actor, "method": a.method, "path": a.path, "status": a.status} for a in recent_audit(limit)]


@router.post("/alerts/scheduler/start")
def start_scheduler(prov=Depends(provider)):
    s = get_settings()
    if not s.alerts_scheduler_enabled:
        raise HTTPException(400, "Scheduler disabled (set ALERTS_SCHEDULER_ENABLED=true).")
    SCHEDULER.start(prov, s.alerts_interval_seconds)
    return scheduler_status(prov)
