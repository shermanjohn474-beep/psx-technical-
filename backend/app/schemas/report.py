"""Structured technical analysis report (sections A-G)."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.candlestick_patterns.engine import CandleSignal
from app.divergences.divergence import Divergence
from app.harmonic_patterns.fibonacci import FibAnalysis
from app.market_structure.structure import StructureResult
from app.multi_timeframe.mtf import MTFResult
from app.risk_management.setups import TradeSetup
from app.schemas.market import DataQualityReport, DataSourceInfo
from app.schemas.patterns import PatternResult
from app.signal_scoring.scoring import SignalScore
from app.support_resistance.levels import LevelsResult


class StockOverview(BaseModel):  # A
    symbol: str
    company_name: str | None
    price: float
    change_pct: float | None
    data_timestamp: datetime
    data_source: DataSourceInfo
    timeframe: str
    bars_analyzed: int


class KeyLevels(BaseModel):  # E
    immediate_support: str | None
    major_support: str | None
    immediate_resistance: str | None
    major_resistance: str | None
    breakout_trigger: float | None
    breakdown_trigger: float | None


class Commentary(BaseModel):  # G
    engine: str  # "deterministic-template" or "<provider>:<model>"
    bias: str
    summary: str
    strongest_evidence: list[str] = Field(default_factory=list)
    conflicting_signals: list[str] = Field(default_factory=list)
    what_changes_view: list[str] = Field(default_factory=list)
    confirmation_needed: list[str] = Field(default_factory=list)
    preferred_setup: str | None = None
    stay_neutral: bool = False
    conclusion: str
    validation: dict = Field(default_factory=dict)


class AnalysisReport(BaseModel):
    report_id: str
    generated_at: datetime
    engine_version: str
    overview: StockOverview
    data_quality: DataQualityReport
    trend: MTFResult  # B
    indicators: dict  # C
    patterns: list[PatternResult]  # D (current)
    historical_patterns: list[PatternResult] = Field(default_factory=list)
    candlesticks: list[CandleSignal] = Field(default_factory=list)
    divergences: list[Divergence] = Field(default_factory=list)
    structure: StructureResult | None = None
    fibonacci: FibAnalysis | None = None
    levels: LevelsResult
    key_levels: KeyLevels  # E
    long_setup: TradeSetup  # F
    short_setup: TradeSetup
    score: SignalScore
    commentary: Commentary | None = None  # G
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = ("Technical analysis output for research purposes. Not investment advice; no outcome is "
                       "guaranteed. Scores are heuristic, not probabilities.")
