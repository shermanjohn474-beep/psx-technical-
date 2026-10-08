"""FastAPI application entry point: ``uvicorn app.main:app``."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.alerts.engine import SCHEDULER
from app.api.routes import router
from app.config.settings import get_settings
from app.data_providers.manager import get_provider
from app.database.repository import get_engine, record_audit

log = logging.getLogger("psx")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    get_engine()
    if s.alerts_scheduler_enabled:
        SCHEDULER.start(get_provider(), s.alerts_interval_seconds)
    yield
    SCHEDULER.stop.set()


app = FastAPI(title=get_settings().app_name, version="0.4.0", lifespan=lifespan,
              description="PSX technical analysis & pattern recognition API. Research output only; not investment advice.")
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def audit_mw(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api") and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        try:
            record_audit(getattr(request.state, "actor", "anonymous"), request.method, request.url.path, response.status_code)
        except Exception as exc:  # audit failures must not break requests
            log.warning("audit log failed: %s", exc)
    return response


@app.get("/health")
def health():
    s = get_settings()
    p = get_provider()
    return {"status": "ok", "symbols_with_data": len(p.list_symbols()),
            "ai_provider_configured": s.ai_provider != "none" and bool(s.anthropic_api_key or s.openai_api_key),
            "live_data_provider_configured": bool(p.extra)}


app.include_router(router)
