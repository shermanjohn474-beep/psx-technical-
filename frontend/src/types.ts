// Types mirroring the backend pydantic schemas (subset used by the UI).

export type Stage = "forming" | "approaching_confirmation" | "confirmed" | "retesting" | "failed" | "invalidated";
export type Dir = "bullish" | "bearish" | "neutral";

export interface LinePoint { index: number; timestamp: string; price: number }
export interface PatternLine { label: string; start: LinePoint; end: LinePoint; style: string }
export interface KeyPoint { label: string; index: number; timestamp: string; price: number }

export interface Pattern {
  pattern_id: string; name: string; category: string; direction: Dir; stage: Stage; timeframe: string | null;
  start_index: number; end_index: number; start_time: string; end_time: string;
  key_points: KeyPoint[]; lines: PatternLine[];
  breakout_level: number | null; invalidation_level: number | null; target: number | null; target_reached: boolean;
  confirmation_time: string | null; volume_confirmation: boolean | null; relative_volume: number | null;
  quality_score: number; quality_components: Record<string, number>; evidence: string[];
  provisional: boolean; experimental: boolean;
}

export interface Zone { low: number; high: number; mid: number; kind: string; touches: number; strength: number; last_touch_time: string | null }
export interface Target { label: string; price: number; rr: number | null; source: string }
export interface Setup {
  side: "long" | "short"; valid: boolean; status: string; basis: string; pattern_stage: string | null;
  entry_trigger: string; entry_price: number | null; entry_zone: [number, number] | null; confirmation_required: string;
  stop_loss: number | null; stop_basis: string | null; atr_stop_alternative: number | null; technical_invalidation: string | null;
  targets: Target[]; primary_rr: number | null; meets_min_rr: boolean; confidence: number; key_risks: string[];
  execution: string | null; notes: string[];
}
export interface ScoreComponent { name: string; weight: number; value: number | null; points: number | null; reasons: string[] }
export interface TFTrend { timeframe: string; label: string; trend: string; score: number; strength: string; structure: string; ma_structure: string | null; notes: string[]; bars: number }
export interface DataIssue { severity: string; code: string; message: string; count: number | null }
export interface Quality { rating: string; score: number; issues: DataIssue[]; stale: boolean; staleness_days: number | null; last_candle_complete: boolean; has_volume: boolean; rows_out: number }
export interface Source { provider: string; attribution: string; is_synthetic: boolean; is_live: boolean; adjusted: boolean | null; as_of: string | null; notes: string[] }
export interface Candle { pattern_id: string; name: string; timestamp: string; direction: Dir; score: number; confirmation: string; evidence: string[] }
export interface Divergence { indicator: string; type: string; direction: Dir; pivot1_time: string; pivot2_time: string; price1: number; price2: number; signal_time: string }
export interface Commentary {
  engine: string; bias: string; summary: string; strongest_evidence: string[]; conflicting_signals: string[];
  what_changes_view: string[]; confirmation_needed: string[]; preferred_setup: string | null; stay_neutral: boolean;
  conclusion: string; validation: Record<string, unknown>;
}
export interface Fib { direction: string; levels: { ratio: number; price: number; kind: string }[]; anchor_start_time: string; anchor_end_time: string }

export interface Report {
  report_id: string; generated_at: string; engine_version: string;
  overview: { symbol: string; company_name: string | null; price: number; change_pct: number | null; data_timestamp: string; data_source: Source; timeframe: string; bars_analyzed: number };
  data_quality: Quality;
  trend: { timeframes: TFTrend[]; alignment: string; alignment_score: number; commentary: string[] } | null;
  indicators: Record<string, any>;
  patterns: Pattern[]; historical_patterns: Pattern[]; candlesticks: Candle[]; divergences: Divergence[];
  structure: { trend: string; events: { type: string; direction: string; time: string; level: number }[]; fvgs: any[]; order_blocks: any[]; wyckoff: any[] } | null;
  fibonacci: Fib | null;
  levels: { zones: Zone[]; immediate_support: Zone | null; immediate_resistance: Zone | null; major_support: Zone | null; major_resistance: Zone | null; auxiliary: { price: number; source: string }[] };
  key_levels: { immediate_support: string | null; major_support: string | null; immediate_resistance: string | null; major_resistance: string | null; breakout_trigger: number | null; breakdown_trigger: number | null };
  long_setup: Setup; short_setup: Setup;
  score: { score: number; rating: string; coverage: number; components: ScoreComponent[]; disclaimer: string };
  commentary: Commentary | null; warnings: string[]; disclaimer: string;
}

export interface OHLCVResponse {
  symbol: string; timeframe: string;
  candles: { time: number; open: number; high: number; low: number; close: number; volume: number | null }[];
  quality: Quality; source: Source; notes: string[];
  indicators?: Record<string, (number | null)[]>; indicators_unavailable?: Record<string, string>;
}

export interface SymbolRow { symbol: string; name: string | null; sector: string | null; has_data: boolean; timeframes: string[]; synthetic: boolean }

export interface ScanRow {
  symbol: string; price: number; data_time: string; trend: string; pattern: string | null; pattern_id: string | null;
  pattern_stage: string | null; breakout_level: number | null; volume_confirmation: boolean | null; rsi: number | null;
  macd: string | null; direction: Dir; signal_score: number; rating: string; risk_reward: number | null; flags: string[];
  data_quality: string; synthetic: boolean;
}
