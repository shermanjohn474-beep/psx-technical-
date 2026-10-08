"""Detector registry and the single entry point used by API, scanner and backtester."""
from __future__ import annotations

import pandas as pd

from app.chart_patterns.base import AnalysisContext, PatternConfig, PatternDetector, build_context, dedupe
from app.chart_patterns.double_triple import DoubleTripleDetector
from app.chart_patterns.flags import FlagPennantDetector
from app.chart_patterns.head_shoulders import HeadShouldersDetector
from app.chart_patterns.two_line import TwoLineDetector
from app.schemas.patterns import PatternResult

DETECTORS: list[PatternDetector] = [
    HeadShouldersDetector(),
    DoubleTripleDetector(),
    TwoLineDetector(),
    FlagPennantDetector(),
]


def register(detector: PatternDetector) -> None:
    if not any(type(d) is type(detector) for d in DETECTORS):
        DETECTORS.append(detector)


def all_pattern_ids() -> list[str]:
    return [pid for d in DETECTORS for pid in d.pattern_ids]


def detect_patterns(
    df: pd.DataFrame | None = None,
    cfg: PatternConfig | None = None,
    *,
    timeframe: str | None = None,
    include: set[str] | None = None,
    last_candle_complete: bool = True,
    ctx: AnalysisContext | None = None,
) -> list[PatternResult]:
    if ctx is None:
        if df is None or len(df) < 30:
            return []
        ctx = build_context(df, cfg, timeframe, last_candle_complete)
    out: list[PatternResult] = []
    for det in DETECTORS:
        if include is not None and not (set(det.pattern_ids) & include):
            continue
        res = det.detect(ctx)
        if include is not None:
            res = [r for r in res if r.pattern_id in include]
        out += res
    return dedupe(out)
