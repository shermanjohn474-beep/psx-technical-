"""Schemas shared by all pattern detectors."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PatternStage(str, Enum):
    FORMING = "forming"
    APPROACHING_CONFIRMATION = "approaching_confirmation"
    CONFIRMED = "confirmed"
    RETESTING = "retesting"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class PatternCategory(str, Enum):
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    BILATERAL = "bilateral"
    CANDLESTICK = "candlestick"
    HARMONIC = "harmonic"
    STRUCTURE = "structure"
    ADVANCED = "advanced"


class KeyPoint(BaseModel):
    label: str
    index: int
    timestamp: datetime
    price: float


class LinePoint(BaseModel):
    index: int
    timestamp: datetime
    price: float


class PatternLine(BaseModel):
    """A line segment to draw (neckline, trendline, boundary, level)."""

    label: str
    start: LinePoint
    end: LinePoint
    style: str = "solid"  # solid | dashed | dotted


class PatternResult(BaseModel):
    pattern_id: str
    name: str
    category: PatternCategory
    direction: Direction
    stage: PatternStage
    timeframe: str | None = None
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    key_points: list[KeyPoint] = Field(default_factory=list)
    lines: list[PatternLine] = Field(default_factory=list)
    breakout_level: float | None = None
    invalidation_level: float | None = None
    target: float | None = None
    target_reached: bool = False
    confirmation_index: int | None = None
    confirmation_time: datetime | None = None
    volume_confirmation: bool | None = None  # None = volume unavailable / not applicable
    relative_volume: float | None = None
    quality_score: float = 0.0  # 0..100 geometric/contextual quality (NOT a probability)
    quality_components: dict[str, float] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    provisional: bool = False
    experimental: bool = False
    detected_at_index: int | None = None  # last bar index of the data used for detection

    @property
    def is_active(self) -> bool:
        return self.stage in (
            PatternStage.FORMING,
            PatternStage.APPROACHING_CONFIRMATION,
            PatternStage.CONFIRMED,
            PatternStage.RETESTING,
        )


class SwingPointModel(BaseModel):
    index: int
    timestamp: datetime
    price: float
    kind: str  # "H" | "L"
    confirmed_index: int | None = None
    label: str | None = None  # HH/HL/LH/LL once structure is classified
