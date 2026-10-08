"""PSX universe scanner with configurable screening rules."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from app.analysis.service import AnalysisOptions, analyze_symbol
from app.data_providers.base import DataUnavailable, MarketDataProvider, ProviderError
from app.indicators.momentum import rsi
from app.schemas.market import Timeframe
from app.schemas.patterns import Direction, PatternStage
from app.schemas.report import AnalysisReport

FLAGS = (
    "emerging_pattern", "confirmed_breakout", "confirmed_breakdown", "breakout_retest", "near_major_level",
    "rsi_divergence", "macd_divergence", "unusual_volume", "golden_cross", "death_cross", "oversold_recovery",
    "bullish_reversal_after_downtrend", "bearish_reversal_after_advance",
)


class ScanRules(BaseModel):
    min_score: float | None = None
    max_score: float | None = None
    directions: list[str] = Field(default_factory=list)       # bullish / bearish / neutral
    stages: list[str] = Field(default_factory=list)           # pattern stages
    pattern_ids: list[str] = Field(default_factory=list)
    flags_any: list[str] = Field(default_factory=list)
    min_rr: float | None = None
    include_experimental: bool = False
    unusual_volume_mult: float = 2.0
    near_level_atr: float = 1.0


class ScanRow(BaseModel):
    symbol: str
    price: float
    data_time: str
    trend: str
    pattern: str | None
    pattern_id: str | None
    pattern_stage: str | None
    breakout_level: float | None
    volume_confirmation: bool | None
    rsi: float | None
    macd: str | None
    direction: str
    signal_score: float
    rating: str
    risk_reward: float | None
    flags: list[str]
    data_quality: str
    synthetic: bool


class ScanResult(BaseModel):
    timeframe: str
    rows: list[ScanRow]
    errors: dict[str, str] = Field(default_factory=dict)
    scanned: int
    matched: int


def flags_for(rep: AnalysisReport, rules: ScanRules, df=None) -> list[str]:
    f: set[str] = set()
    pats = rep.patterns if rules.include_experimental else [p for p in rep.patterns if not p.experimental]
    for p in pats:
        if p.stage in (PatternStage.FORMING, PatternStage.APPROACHING_CONFIRMATION):
            f.add("emerging_pattern")
        if p.stage == PatternStage.CONFIRMED and p.confirmation_index is not None and \
                rep.overview.bars_analyzed - 1 - p.confirmation_index <= 5:
            f.add("confirmed_breakout" if p.direction == Direction.BULLISH else "confirmed_breakdown" if p.direction == Direction.BEARISH else "")
        if p.stage == PatternStage.RETESTING:
            f.add("breakout_retest")
        if df is not None and p.stage in (PatternStage.CONFIRMED, PatternStage.RETESTING) and p.category.value == "reversal":
            c = df["close"].to_numpy()
            j = max(0, p.start_index - 40)
            atr_v = (rep.indicators.get("atr") or {}).get("value") or 1.0
            prior = (c[p.start_index] - c[j]) / atr_v
            if p.direction == Direction.BULLISH and prior <= -3:
                f.add("bullish_reversal_after_downtrend")
            if p.direction == Direction.BEARISH and prior >= 3:
                f.add("bearish_reversal_after_advance")
    atr = (rep.indicators.get("atr") or {}).get("value")
    if atr:
        for z in (rep.levels.major_support, rep.levels.major_resistance):
            if z is not None and abs(z.mid - rep.overview.price) <= rules.near_level_atr * atr:
                f.add("near_major_level")
    n = rep.overview.bars_analyzed
    for d in rep.divergences:
        if d.signal_index >= n - 10 and d.type.startswith("regular"):
            f.add("rsi_divergence" if d.indicator == "rsi" else "macd_divergence" if d.indicator == "macd" else "")
    vol = rep.indicators.get("volume") or {}
    if vol.get("relative") is not None and vol["relative"] >= rules.unusual_volume_mult:
        f.add("unusual_volume")
    f.discard("")
    return sorted(f)


def _extra_flags(df, f: list[str]) -> list[str]:
    c = df["close"]
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    if len(c) > 205:
        d = (s50 - s200).iloc[-6:]
        if (d.iloc[-1] > 0) and (d.iloc[0] <= 0):
            f.append("golden_cross")
        if (d.iloc[-1] < 0) and (d.iloc[0] >= 0):
            f.append("death_cross")
    r = rsi(c).iloc[-6:]
    if len(r.dropna()) == 6 and r.iloc[:-1].min() < 30 <= r.iloc[-1]:
        f.append("oversold_recovery")
    return sorted(set(f))


def scan(provider: MarketDataProvider, symbols: list[str], timeframe: Timeframe = Timeframe.D1,
         rules: ScanRules | None = None, with_mtf: bool = True, workers: int = 4) -> ScanResult:
    rules = rules or ScanRules()
    rows: list[ScanRow] = []
    errors: dict[str, str] = {}

    def one(sym: str):
        try:
            rep = analyze_symbol(provider, sym, timeframe, options=AnalysisOptions(
                with_mtf=with_mtf, include_experimental=rules.include_experimental))
            df = provider.get_ohlcv(rep.overview.symbol, timeframe).frame
            return sym, rep, df, None
        except (DataUnavailable, ProviderError, ValueError) as exc:
            return sym, None, None, str(exc)[:200]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, symbols))
    for sym, rep, df, err in results:
        if err:
            errors[sym] = err
            continue
        flags = _extra_flags(df, flags_for(rep, rules, df))
        pats = [p for p in rep.patterns if (rules.include_experimental or not p.experimental)]
        if rules.pattern_ids:
            pats = [p for p in pats if p.pattern_id in rules.pattern_ids]
        if rules.stages:
            pats = [p for p in pats if p.stage.value in rules.stages]
        top = pats[0] if pats else None
        direction = top.direction.value if top else ("bullish" if rep.score.score >= 60 else "bearish" if rep.score.score <= 40 else "neutral")
        rr = None
        for s in (rep.long_setup, rep.short_setup):
            if s.valid and top is not None and s.pattern_id == top.pattern_id:
                rr = s.primary_rr
        d_tf = next((t for t in rep.trend.timeframes if t.timeframe == timeframe.value), None) if rep.trend else None
        row = ScanRow(
            symbol=rep.overview.symbol, price=rep.overview.price, data_time=rep.overview.data_timestamp.isoformat(),
            trend=d_tf.trend if d_tf else "n/a", pattern=top.name if top else None, pattern_id=top.pattern_id if top else None,
            pattern_stage=top.stage.value if top else None, breakout_level=top.breakout_level if top else None,
            volume_confirmation=top.volume_confirmation if top else None,
            rsi=(rep.indicators.get("rsi") or {}).get("value"),
            macd=(rep.indicators.get("macd") or {}).get("state"), direction=direction, signal_score=rep.score.score,
            rating=rep.score.rating, risk_reward=rr, flags=flags,
            data_quality=rep.data_quality.rating if rep.data_quality else "unknown",
            synthetic=rep.overview.data_source.is_synthetic,
        )
        if (rules.pattern_ids or rules.stages) and top is None:
            continue
        if rules.min_score is not None and row.signal_score < rules.min_score:
            continue
        if rules.max_score is not None and row.signal_score > rules.max_score:
            continue
        if rules.directions and row.direction not in rules.directions:
            continue
        if rules.flags_any and not set(rules.flags_any) & set(flags):
            continue
        if rules.min_rr is not None and (row.risk_reward is None or row.risk_reward < rules.min_rr):
            continue
        rows.append(row)
    rows.sort(key=lambda r: -abs(r.signal_score - 50))
    return ScanResult(timeframe=timeframe.value, rows=rows, errors=errors, scanned=len(symbols), matched=len(rows))
