"""Provider-agnostic market-data interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from app.schemas.market import DataQualityReport, DataSourceInfo, Timeframe


class ProviderError(RuntimeError):
    pass


class ProviderNotConfigured(ProviderError):
    pass


class DataUnavailable(ProviderError):
    """Requested symbol/timeframe is not available from this provider."""


@dataclass
class OHLCVResult:
    symbol: str
    timeframe: Timeframe
    frame: pd.DataFrame
    source: DataSourceInfo
    quality: DataQualityReport
    notes: list[str] = field(default_factory=list)


class MarketDataProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def get_ohlcv(
        self, symbol: str, timeframe: Timeframe, start: datetime | None = None, end: datetime | None = None
    ) -> OHLCVResult:
        """Return validated OHLCV. Must raise DataUnavailable rather than fabricate data."""

    @abstractmethod
    def available_timeframes(self, symbol: str) -> list[Timeframe]:
        ...

    def list_symbols(self) -> list[str]:
        return []


def slice_range(df: pd.DataFrame, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    if start is not None:
        s = pd.Timestamp(start)
        s = s.tz_localize(df.index.tz) if s.tzinfo is None else s.tz_convert(df.index.tz)
        df = df[df.index >= s]
    if end is not None:
        e = pd.Timestamp(end)
        e = e.tz_localize(df.index.tz) if e.tzinfo is None else e.tz_convert(df.index.tz)
        df = df[df.index <= e]
    return df
