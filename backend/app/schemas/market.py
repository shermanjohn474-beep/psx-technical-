"""Market-data schemas: timeframes, data metadata and data-quality reports."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Timeframe(str, Enum):
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

    @property
    def is_intraday(self) -> bool:
        return self in (Timeframe.M5, Timeframe.M15, Timeframe.H1)

    @property
    def minutes(self) -> int | None:
        return {Timeframe.M5: 5, Timeframe.M15: 15, Timeframe.H1: 60}.get(self)

    @property
    def rank(self) -> int:
        """Higher rank = higher timeframe."""
        return [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.D1, Timeframe.W1, Timeframe.MN1].index(self)


class DataIssue(BaseModel):
    severity: str  # info | warning | error
    code: str
    message: str
    count: int | None = None
    examples: list[str] = Field(default_factory=list)


class DataQualityReport(BaseModel):
    """Result of validating an OHLCV series."""

    symbol: str
    timeframe: Timeframe
    rows_in: int
    rows_out: int
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    duplicates_removed: int = 0
    invalid_ohlc_rows: int = 0
    missing_candles: int = 0
    has_volume: bool = True
    zero_volume_bars: int = 0
    last_candle_complete: bool = True
    stale: bool = False
    staleness_days: float | None = None
    issues: list[DataIssue] = Field(default_factory=list)
    rating: str = "unknown"  # high | medium | low | insufficient
    score: float = 0.0  # 0..100

    def add(self, severity: str, code: str, message: str, count: int | None = None, examples: list[str] | None = None) -> None:
        self.issues.append(DataIssue(severity=severity, code=code, message=message, count=count, examples=examples or []))


class DataSourceInfo(BaseModel):
    provider: str
    attribution: str
    is_synthetic: bool = False
    is_live: bool = False
    delayed: bool | None = None
    adjusted: bool | None = None  # None = unknown adjustment status
    as_of: datetime | None = None
    notes: list[str] = Field(default_factory=list)


class SymbolInfo(BaseModel):
    symbol: str
    name: str | None = None
    sector: str | None = None
    is_index: bool = False
    active: bool | None = None  # None = unknown
    source: str = "static-registry"
