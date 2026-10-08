"""Shared API dependencies: auth, provider, error mapping."""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request

from app.config.settings import get_settings
from app.data_providers.base import DataUnavailable, ProviderError
from app.data_providers.manager import get_provider


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> str:
    """If APP_API_KEYS is configured, every /api request must carry a valid X-API-Key."""
    keys = get_settings().api_keys
    if not keys:
        request.state.actor = "dev-open"
        return "dev-open"
    if x_api_key and any(hmac.compare_digest(x_api_key, k) for k in keys):
        request.state.actor = f"key:{x_api_key[:4]}…"
        return request.state.actor
    raise HTTPException(status_code=401, detail="Missing or invalid API key (X-API-Key).")


def provider():
    return get_provider()


def data_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DataUnavailable):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ProviderError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))
