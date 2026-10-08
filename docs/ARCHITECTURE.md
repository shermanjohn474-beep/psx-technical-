# Architecture

```
frontend (React + TS + Vite + Lightweight Charts)
        │  /api (JSON, multipart)            X-API-Key (optional)
backend (FastAPI)
  api/            routes, auth dependency, audit middleware
  analysis/       orchestration -> AnalysisReport (sections A–G)
  data_providers/ provider interface, file provider, authorized-HTTP adapter, resampling, symbols, synthetic
  data_validation/calendar (Asia/Karachi, PSX holidays), validator (quality report)
  indicators/     trend, momentum, volatility, volume, engine (+snapshot)
  swing_detection causal ATR zigzag, fractals, find_peaks
  chart_patterns/ base (context, lifecycle, scoring), detectors, registry
  candlestick_patterns/, divergences/, market_structure/, harmonic_patterns/ (fib, harmonics, Elliott)
  support_resistance/, multi_timeframe/, signal_scoring/, risk_management/
  screenshot_analysis/ render (fixtures + report charts), extract (OpenCV), service (tiers)
  ai_analyst/     providers (Anthropic SDK / OpenAI / null), analyst (evidence, validation, audit), deterministic
  scanner/, backtesting/ (strategies, engine, walk-forward), alerts/, reporting/ (HTML)
  database/       SQLAlchemy models + repository (SQLite dev, Postgres-ready)
```

## Data flow for `GET /api/analysis/{symbol}`
1. `ProviderManager` resolves the symbol: user files → configured authorized feed. `SYN-`
   symbols are served only from the synthetic fixture directory. Missing data raises
   `DataUnavailable` (HTTP 404) — never a substitute series.
2. `validate_ohlcv` normalises and produces a `DataQualityReport`.
3. `build_context` computes causal ATR and zigzag swings once; all detectors share it.
4. Indicators, S/R zones, patterns, candlesticks, divergences, structure and Fibonacci run on
   the same frame. Patterns are split into *current* and *historical*.
5. Other timeframes are fetched (or aggregated on calendar boundaries) for the MTF engine.
6. Setups are built from the best active pattern per direction (structure fallback).
7. The confluence score combines six components with coverage reporting.
8. Deterministic commentary is attached; optionally the LLM analyst replaces it if its output
   passes evidence validation.

## Pattern engine rules
* Every detector produces `PatternResult` with key points (index + timestamp + price), drawable
  lines, trigger, invalidation, measured-move target, quality components, evidence and stage.
* `evaluate_breakout` (in `chart_patterns/base.py`) is the single lifecycle implementation:
  invalidation before confirmation, completed-close confirmation with ATR tolerance, failure
  when price closes back through the *breakout price* within `fail_lookahead_bars`, retest
  detection, target tracking, expiry after `max_wait_bars`.
* Bilateral formations (symmetrical triangle, rectangles, channels, broadening, compression)
  get a direction only from the actual breakout.
* All thresholds live in `PatternConfig` (global) or `PatternConfig.params[detector_id]`.
* Geometric definitions are documented in each detector module's docstring.

## Look-ahead safeguards
* Zigzag pivots carry `confirmed_index`; detectors only use pivots with
  `confirmed_index <= last` plus a single provisional extreme that can never confirm a pattern.
* Tests: zigzag causality, indicator causality, divergence causality, pattern confirmation
  visibility at time *t*, backtest invariance to garbage future data, and a spy strategy that
  asserts it only ever receives `df.iloc[:t+1]`.

## Screenshot tiers
| Tier | Requirement | Output units |
|---|---|---|
| visual_only | candle extraction succeeded | pixels (no prices/dates) |
| estimated | + axis calibration (user or vision model) | prices ± uncertainty |
| data_verified | + symbol & timeframe + data match (price ρ ≥ 0.95 and return ρ ≥ 0.8) | real prices, full report |
