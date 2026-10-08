"""Orchestrates the full deterministic analysis pipeline for one symbol/timeframe.

All numbers in the report come from here. The AI analyst layer only *explains*
these findings and is validated against them.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from app.candlestick_patterns.engine import detect_candles
from app.chart_patterns.base import PatternConfig, build_context
from app.chart_patterns.registry import detect_patterns
from app.config.settings import get_settings
from app.data_providers.base import DataUnavailable, MarketDataProvider, OHLCVResult
from app.data_providers.symbols import resolve_symbol
from app.data_validation.calendar import now_pkt
from app.data_validation.validator import is_last_candle_complete
from app.divergences.divergence import detect_divergences
from app.harmonic_patterns.fibonacci import fibonacci
from app.indicators.engine import compute_indicators, indicator_snapshot
from app.market_structure.structure import analyze_structure
from app.multi_timeframe.mtf import ORDER, analyze_mtf
from app.risk_management.setups import RiskConfig, build_setup
from app.schemas.market import DataSourceInfo, Timeframe
from app.schemas.patterns import PatternStage
from app.schemas.report import AnalysisReport, KeyLevels, StockOverview
from app.signal_scoring.scoring import ScoreConfig, compute_score
from app.support_resistance.levels import detect_levels

ENGINE_VERSION = "0.2.0"


@dataclass
class AnalysisOptions:
    pattern_cfg: PatternConfig | None = None
    risk_cfg: RiskConfig = field(default_factory=RiskConfig)
    score_cfg: ScoreConfig = field(default_factory=ScoreConfig)
    include_experimental: bool = True
    current_window: int = 40  # patterns ending/confirming within N bars are "current"
    with_mtf: bool = True


def default_pattern_cfg(tf: Timeframe) -> PatternConfig:
    cfg = PatternConfig()
    if tf.is_intraday:
        cfg.min_height_pct = 0.8
    elif tf in (Timeframe.W1, Timeframe.MN1):
        cfg.min_height_pct = 6.0
    return cfg


def analyze_frame(df: pd.DataFrame, *, symbol: str, timeframe: Timeframe, source: DataSourceInfo, quality,
                  mtf_frames: dict[Timeframe, pd.DataFrame] | None = None, options: AnalysisOptions | None = None,
                  as_of: datetime | None = None) -> AnalysisReport:
    opt = options or AnalysisOptions()
    warnings: list[str] = []
    if len(df) < 30:
        raise DataUnavailable(f"Only {len(df)} bars for {symbol} {timeframe.value}: at least 30 are required.")
    now = as_of or now_pkt()
    complete = quality.last_candle_complete if quality else is_last_candle_complete(df.index[-1], timeframe, now)
    pcfg = opt.pattern_cfg or default_pattern_cfg(timeframe)
    ctx = build_context(df, pcfg, timeframe.value, complete)
    ind = compute_indicators(df, timeframe)
    snap = indicator_snapshot(df, ind)
    levels = detect_levels(df, ctx.swings)
    all_patterns = detect_patterns(ctx=ctx, include_experimental=opt.include_experimental)
    last = ctx.last
    current, historical = [], []
    for p in all_patterns:
        recent_conf = p.confirmation_index is not None and last - p.confirmation_index <= opt.current_window
        active_recent = p.is_active and p.end_index >= last - opt.current_window
        resolved_recent = p.stage in (PatternStage.FAILED, PatternStage.INVALIDATED) and p.end_index >= last - 10
        (current if (recent_conf or active_recent or resolved_recent) else historical).append(p)
    current.sort(key=lambda p: (-int(p.is_active), -p.quality_score))
    candles = detect_candles(df, levels.zones, last_candle_complete=complete)
    divs_inputs = {"rsi": ind.frame["rsi"], "macd": ind.frame["macd"]}
    if "obv" in ind.frame:
        divs_inputs["obv"] = ind.frame["obv"]
    if "mfi" in ind.frame:
        divs_inputs["mfi"] = ind.frame["mfi"]
    divergences = detect_divergences(df, ctx.swings, divs_inputs)
    structure = analyze_structure(df, ctx.swings)
    fib = fibonacci(df, ctx.swings)
    close = float(df["close"].iloc[-1])
    atr = float(ctx.atr[-1])
    s = get_settings()
    fib_levels = [lv.price for lv in fib.levels] if fib else []
    long_setup = build_setup("long", close, atr, current, levels, last,
                             sorted(p for p in fib_levels if p > close), opt.risk_cfg)
    short_setup = build_setup("short", close, atr, current, levels, last,
                              sorted((p for p in fib_levels if p < close), reverse=True), opt.risk_cfg,
                              short_eligible=symbol.upper() in s.short_eligible_symbols, symbol=symbol)
    mtf = None
    if opt.with_mtf:
        frames = dict(mtf_frames or {})
        frames.setdefault(timeframe, df)
        mtf = analyze_mtf(frames)
    score = compute_score(mtf=mtf, patterns=current, snapshot=snap, divergences=divergences, levels=levels,
                          long_setup=long_setup, short_setup=short_setup, last_index=last, close=close, atr=atr,
                          cfg=opt.score_cfg)
    if source.is_synthetic:
        warnings.append("SYNTHETIC DEMO DATA: not real PSX prices. For demonstration and testing only.")
    if quality and quality.stale:
        warnings.append(f"Data is stale ({quality.staleness_days} days old): levels may not reflect the current market.")
    if not complete:
        warnings.append("Latest candle is incomplete: any signal on it is provisional.")
    if quality and not quality.has_volume:
        warnings.append("No volume data: volume confirmation factors excluded.")
    if source.adjusted is None:
        warnings.append("Corporate-action adjustment status unknown: historical levels across splits/bonuses may be distorted.")
    prev = float(df["close"].iloc[-2]) if len(df) > 1 else None
    info = resolve_symbol(symbol)

    def zr(z):
        return None if z is None else f"{z.low:.4g} - {z.high:.4g} ({z.touches} touches, strength {z.strength:.0f})"

    bull_trig = next((p.breakout_level for p in current if p.direction.value == "bullish" and p.is_active
                      and p.confirmation_index is None), None) or (levels.immediate_resistance.high if levels.immediate_resistance else None)
    bear_trig = next((p.breakout_level for p in current if p.direction.value == "bearish" and p.is_active
                      and p.confirmation_index is None), None) or (levels.immediate_support.low if levels.immediate_support else None)
    report = AnalysisReport(
        report_id=uuid.uuid4().hex[:12], generated_at=datetime.now(now.tzinfo), engine_version=ENGINE_VERSION,
        overview=StockOverview(symbol=symbol.upper(), company_name=info.name, price=round(close, 4),
                               change_pct=None if prev is None else round(100 * (close / prev - 1), 2),
                               data_timestamp=df.index[-1].to_pydatetime(), data_source=source, timeframe=timeframe.value,
                               bars_analyzed=len(df)),
        data_quality=quality, trend=mtf, indicators=snap, patterns=current, historical_patterns=historical[-25:],
        candlesticks=candles[-15:], divergences=divergences[-10:], structure=structure, fibonacci=fib, levels=levels,
        key_levels=KeyLevels(immediate_support=zr(levels.immediate_support), major_support=zr(levels.major_support),
                             immediate_resistance=zr(levels.immediate_resistance), major_resistance=zr(levels.major_resistance),
                             breakout_trigger=None if bull_trig is None else round(bull_trig, 4),
                             breakdown_trigger=None if bear_trig is None else round(bear_trig, 4)),
        long_setup=long_setup, short_setup=short_setup, score=score, warnings=warnings,
    )
    from app.ai_analyst.deterministic import deterministic_commentary  # local import avoids cycle
    report.commentary = deterministic_commentary(report)
    return report


def gather_mtf_frames(provider: MarketDataProvider, symbol: str, base: Timeframe,
                      end: datetime | None = None) -> dict[Timeframe, pd.DataFrame]:
    frames = {}
    avail = set(provider.available_timeframes(symbol))
    for tf in ORDER:
        if tf == base or tf not in avail:
            continue
        try:
            frames[tf] = provider.get_ohlcv(symbol, tf, None, end).frame
        except DataUnavailable:
            continue
    return frames


def analyze_symbol(provider: MarketDataProvider, symbol: str, timeframe: Timeframe = Timeframe.D1,
                   start: datetime | None = None, end: datetime | None = None,
                   options: AnalysisOptions | None = None) -> AnalysisReport:
    info = resolve_symbol(symbol)
    res: OHLCVResult = provider.get_ohlcv(info.symbol, timeframe, start, end)
    frames = gather_mtf_frames(provider, info.symbol, timeframe, end) if (options is None or options.with_mtf) else {}
    rep = analyze_frame(res.frame, symbol=info.symbol, timeframe=timeframe, source=res.source, quality=res.quality,
                        mtf_frames=frames, options=options)
    rep.warnings += res.notes
    return rep
