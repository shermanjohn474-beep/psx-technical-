"""Generic adapter for an *authorized* HTTP market-data source.

No PSX endpoints are hard-coded. The operator must configure a URL template for a
feed they are licensed/permitted to use, e.g.::

    MARKET_DATA_URL_TEMPLATE=https://vendor.example/api/ohlcv?symbol={symbol}&tf={timeframe}&from={start}&to={end}

The endpoint must return JSON: either a list of objects with
``timestamp|date, open, high, low, close, volume`` keys or ``{"data": [...]}``.
Publicly viewable charts (including dps.psx.com.pk) are NOT assumed to grant API or
scraping rights; check the site's terms before pointing this adapter at it.
"""
from __future__ import annotations

from datetime import datetime

import httpx
import pandas as pd

from app.data_providers.base import (
    DataUnavailable, MarketDataProvider, OHLCVResult, ProviderError, ProviderNotConfigured,
)
from app.data_validation.validator import validate_ohlcv
from app.schemas.market import DataSourceInfo, Timeframe


class HTTPJSONProvider(MarketDataProvider):
    def __init__(self, url_template: str | None, api_key: str | None = None, name: str = "configured-http-provider",
                 timeframes: list[Timeframe] | None = None, timeout: float = 20.0, delayed: bool | None = None,
                 max_staleness_days: float | None = 4):
        self.url_template = url_template
        self.api_key = api_key
        self.name = name
        self._timeframes = timeframes or [Timeframe.D1]
        self.timeout = timeout
        self.delayed = delayed
        self.max_staleness_days = max_staleness_days

    @property
    def configured(self) -> bool:
        return bool(self.url_template)

    def available_timeframes(self, symbol: str) -> list[Timeframe]:
        return list(self._timeframes) if self.configured else []

    def get_ohlcv(self, symbol: str, timeframe: Timeframe, start: datetime | None = None,
                  end: datetime | None = None) -> OHLCVResult:
        if not self.configured:
            raise ProviderNotConfigured("No authorized HTTP market-data provider configured (MARKET_DATA_URL_TEMPLATE).")
        if timeframe not in self._timeframes:
            raise DataUnavailable(f"{self.name} does not provide {timeframe.value} data.")
        url = self.url_template.format(
            symbol=symbol, timeframe=timeframe.value,
            start=start.date().isoformat() if start else "", end=end.date().isoformat() if end else "",
        )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = httpx.get(url, headers=headers, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} request failed: {exc.__class__.__name__}") from exc
        if resp.status_code == 404:
            raise DataUnavailable(f"{symbol} not found at {self.name}.")
        if resp.status_code >= 400:
            raise ProviderError(f"{self.name} returned HTTP {resp.status_code}.")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name} returned non-JSON payload.") from exc
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            raise DataUnavailable(f"{self.name} returned no candles for {symbol}.")
        frame, quality = validate_ohlcv(pd.DataFrame(rows), symbol, timeframe, max_staleness_days=self.max_staleness_days)
        source = DataSourceInfo(provider=self.name, attribution=f"Data from {self.name}", is_live=False,
                                delayed=self.delayed, adjusted=None, as_of=quality.last_timestamp)
        return OHLCVResult(symbol, timeframe, frame, source, quality)
