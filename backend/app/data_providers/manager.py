"""Composite provider: tries providers in priority order.

Synthetic demo data is only served for symbols that are explicitly synthetic
(prefixed ``SYN-``). A real PSX symbol never silently falls back to simulated
quotes.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from app.config.settings import get_settings
from app.data_providers.base import DataUnavailable, MarketDataProvider, OHLCVResult, ProviderError, ProviderNotConfigured
from app.data_providers.file_provider import FileProvider
from app.data_providers.http_provider import HTTPJSONProvider
from app.schemas.market import Timeframe

SYNTHETIC_PREFIX = "SYN-"


class ProviderManager(MarketDataProvider):
    name = "composite"

    def __init__(self, user_files: FileProvider, synthetic: FileProvider, extra: list[MarketDataProvider] | None = None):
        self.user_files = user_files
        self.synthetic = synthetic
        self.extra = extra or []

    def _chain(self, symbol: str) -> list[MarketDataProvider]:
        if symbol.upper().startswith(SYNTHETIC_PREFIX):
            return [self.synthetic]
        return [self.user_files, *self.extra]

    def available_timeframes(self, symbol: str) -> list[Timeframe]:
        tfs: set[Timeframe] = set()
        for p in self._chain(symbol):
            try:
                tfs.update(p.available_timeframes(symbol))
            except ProviderError:
                continue
        return sorted(tfs, key=lambda t: t.rank)

    def list_symbols(self) -> list[str]:
        return sorted(set(self.user_files.list_symbols()) | set(self.synthetic.list_symbols()))

    def get_ohlcv(self, symbol: str, timeframe: Timeframe, start: datetime | None = None,
                  end: datetime | None = None) -> OHLCVResult:
        errors = []
        for p in self._chain(symbol):
            try:
                return p.get_ohlcv(symbol, timeframe, start, end)
            except ProviderNotConfigured:
                continue
            except DataUnavailable as exc:
                errors.append(f"{p.name}: {exc}")
        raise DataUnavailable(
            f"No {timeframe.value} data available for {symbol}. "
            + (" | ".join(errors) if errors else "")
            + " Upload a CSV/Excel OHLCV file or configure an authorized data provider."
        )


@lru_cache
def get_provider() -> ProviderManager:
    s = get_settings()
    extra: list[MarketDataProvider] = []
    if s.http_provider_url_template:
        extra.append(HTTPJSONProvider(s.http_provider_url_template, s.http_provider_api_key, s.http_provider_name,
                                      max_staleness_days=s.max_data_staleness_days))
    return ProviderManager(
        user_files=FileProvider(s.data_dir, max_staleness_days=s.max_data_staleness_days),
        synthetic=FileProvider(s.sample_data_dir, synthetic=True),
        extra=extra,
    )
