"""Application settings.

All settings are read from environment variables (optionally loaded from a
``.env`` file by the process manager / docker-compose). Secrets such as AI
provider API keys are only ever read server-side and never returned by any
API endpoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str]:
    value = _env(name)
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


@dataclass(frozen=True)
class Settings:
    app_name: str = "PSX Technical Research Platform"
    timezone: str = "Asia/Karachi"
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", f"sqlite:///{BACKEND_ROOT / 'psx_dev.sqlite3'}")
    )
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(BACKEND_ROOT / "data"))))
    sample_data_dir: Path = field(default_factory=lambda: BACKEND_ROOT / "fixtures" / "sample_data")
    holidays_file: Path = field(
        default_factory=lambda: Path(_env("PSX_HOLIDAYS_FILE", str(BACKEND_ROOT / "app" / "config" / "psx_holidays.csv")))
    )
    # PSX regular-session close used to decide if today's daily candle is complete.
    # Verify against the current PSX trading-hours notice (Friday hours differ).
    session_close_hhmm: str = field(default_factory=lambda: _env("PSX_SESSION_CLOSE", "15:30"))
    friday_session_close_hhmm: str = field(default_factory=lambda: _env("PSX_FRIDAY_SESSION_CLOSE", "16:30"))
    max_data_staleness_days: int = field(default_factory=lambda: int(_env("MAX_DATA_STALENESS_DAYS", "4")))

    # Optional authorized HTTP market-data provider (disabled unless configured).
    http_provider_url_template: str | None = field(default_factory=lambda: _env("MARKET_DATA_URL_TEMPLATE"))
    http_provider_api_key: str | None = field(default_factory=lambda: _env("MARKET_DATA_API_KEY"))
    http_provider_name: str = field(default_factory=lambda: _env("MARKET_DATA_PROVIDER_NAME", "configured-http-provider"))

    # AI providers (keys are server-side only).
    ai_provider: str = field(default_factory=lambda: _env("AI_PROVIDER", "none"))  # none|anthropic|openai
    anthropic_api_key: str | None = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-4-8"))
    openai_api_key: str | None = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o"))
    ai_timeout_seconds: float = field(default_factory=lambda: float(_env("AI_TIMEOUT_SECONDS", "90")))

    # API authentication: if empty, the API runs open (development only).
    api_keys: list[str] = field(default_factory=lambda: _env_list("APP_API_KEYS"))
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("CORS_ORIGINS") or ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # Alerts
    alerts_scheduler_enabled: bool = field(default_factory=lambda: _env_bool("ALERTS_SCHEDULER_ENABLED", False))
    alerts_interval_seconds: int = field(default_factory=lambda: int(_env("ALERTS_INTERVAL_SECONDS", "900")))
    smtp_host: str | None = field(default_factory=lambda: _env("SMTP_HOST"))
    smtp_port: int = field(default_factory=lambda: int(_env("SMTP_PORT", "587")))
    smtp_user: str | None = field(default_factory=lambda: _env("SMTP_USER"))
    smtp_password: str | None = field(default_factory=lambda: _env("SMTP_PASSWORD"))
    smtp_from: str | None = field(default_factory=lambda: _env("SMTP_FROM"))
    whatsapp_api_url: str | None = field(default_factory=lambda: _env("WHATSAPP_API_URL"))
    whatsapp_api_token: str | None = field(default_factory=lambda: _env("WHATSAPP_API_TOKEN"))
    public_base_url: str = field(default_factory=lambda: _env("PUBLIC_BASE_URL", "http://localhost:5173"))

    # Symbols the user has verified as eligible for regulated short selling
    # through their broker. Empty = no short is treated as executable.
    short_eligible_symbols: list[str] = field(default_factory=lambda: _env_list("SHORT_ELIGIBLE_SYMBOLS"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
