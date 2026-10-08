# Implementation status

Legend: ✅ implemented and tested · 🟡 implemented with stated limitations / not fully verified · ⛔ not implemented

Test suite: **276 tests passing** (`cd backend && python -m pytest`). Frontend: `tsc` clean,
production build succeeds, and the main flows were exercised end-to-end in headless Chromium
(dashboard, chart analysis, scanner, screenshot upload, backtest with walk-forward, alerts,
mobile viewport).

## Phase 1 — Core quantitative engine
| Item | Status | Notes |
|---|---|---|
| Project structure, typed schemas | ✅ | `backend/app/*` modules per the requested layout |
| Market-data models, provider interface | ✅ | `MarketDataProvider.get_ohlcv(symbol, timeframe, start, end)` |
| CSV/Excel importer | ✅ | Excel needs `openpyxl` (optional dependency) |
| Data validation | ✅ | tz normalisation (Asia/Karachi), duplicates, invalid OHLC, missing candles vs PSX calendar, gaps/corporate-action heuristic, suspensions/illiquidity, staleness, incomplete candle, quality rating |
| Swing detection | ✅ | causal ATR zigzag (default), fractals, `find_peaks` (non-causal, visual only) |
| SMA, EMA, RSI, MACD, Bollinger, ATR (+ADX, Stoch, CCI, W%R, ROC, MOM, OBV, A/D, MFI, SAR) | ✅ | verified against TA-Lib |
| Support/resistance zones | ✅ | clustered zones scored on touches, recency, rejection, volume, role reversal, retests |
| H&S, inverse H&S, double top/bottom, asc/desc/sym triangles, bull/bear flags | ✅ | each tested for valid, incomplete, invalid geometry, false breakout, failed, noise |
| Working example | ✅ | `scripts/phase1_demo.py` |

## Phase 2 — Advanced technical analysis
| Item | Status | Notes |
|---|---|---|
| Remaining reversal patterns (triple, rounding, cup & handle ±, broadening top/bottom, diamonds, V/inverse V, Adam & Eve, bull/bear traps) | ✅ | Adam & Eve is a width heuristic; diamonds flagged lower reliability |
| Remaining continuation patterns (pennants, rectangle, wedges, channels, expanding triangle, broadening formation, measured move, compression breakout, VCP) | ✅ | VCP marked experimental |
| Candlestick engine (34 patterns) with context filters | ✅ | trend context, location at S/R, size, volume, next-bar confirmation; TA-Lib cross-check |
| Divergences (regular/hidden, RSI/MACD/OBV/MFI) | ✅ | aligned pivots, causal; tested |
| Fibonacci retracements/extensions | ✅ | anchored on dominant recent leg |
| Harmonics (ABCD, Gartley, Bat, Butterfly, Crab, Cypher), Elliott impulse | 🟡 | experimental; Elliott only checks the three hard rules |
| Market structure: HH/HL/LH/LL, BOS, CHoCH, FVG | ✅ | |
| Order blocks, supply/demand, Wyckoff spring/upthrust | 🟡 | heuristic candidates, labelled as such; no Wyckoff phase labelling |
| Volume analysis | ✅ | relative volume, OBV, A/D, CMF, MFI, dry-up, volume/price divergence, breakout confirmation |
| Multi-timeframe engine | ✅ | M/W/D/H/15m where data exists; counter-trend commentary |
| Signal scoring with coverage | ✅ | configurable weights; missing factors excluded and reported |
| Long/short setups and R:R | ✅ | technical stops never moved for R:R; <1:2 flagged; short executability not assumed |

## Phase 3 — Screenshot analysis & AI
| Item | Status | Notes |
|---|---|---|
| Image upload (PNG/JPG/WEBP), multi-screenshot | ✅ | |
| Chart region / multi-panel detection | ✅ | colour segmentation; tallest band = price panel |
| Candle extraction, axis calibration (linear/log) | ✅ | tested on rendered charts incl. log scale, overlays, dark theme |
| Three-tier confidence (visual-only / estimated / data-verified) | ✅ | data matching requires both price and return correlation |
| Vision-model integration (Anthropic, OpenAI) | 🟡 | implemented with strict JSON schemas and tested with a fake provider; **not exercised against live APIs here (no keys)** |
| Annotated output image | ✅ | |
| AI analyst with schema + evidence validation, audit trail | ✅ | rejects invented numbers/patterns; JSONL + DB audit |
| Real broker/TradingView screenshot accuracy | 🟡 | measured only on synthetic renders; hollow/monochrome candles, Heikin-Ashi and line charts are not extracted (falls back to visual-only) |
| OCR of axis labels | ⛔ | no OCR binary; axis reading relies on the vision model or user calibration |

## Phase 4 — Frontend
| Item | Status | Notes |
|---|---|---|
| Dashboard (search, watchlists, data status, emerging/confirmed lists, upload) | ✅ | |
| Chart analysis (candles, volume, indicators, patterns, S/R, Fibonacci, setups, timeframes, date range) | ✅ | |
| Screenshot page (drag-drop, preview, correction of symbol/timeframe/axis, tiers) | ✅ | |
| Scanner (sortable, filters, presets, watchlists) | ✅ | |
| Research report, PDF/image export | 🟡 | PDF via browser print of the server HTML report; PNG export of the chart; no server-side PDF library |
| Sherman Securities theme, responsive layout | ✅ | checked at 1440px and 390px widths |

## Phase 5 — Scanners, backtesting, alerts
| Item | Status | Notes |
|---|---|---|
| Universe scanner | ✅ | thread-pooled; flags for all requested scan conditions |
| Backtester with PSX constraints | ✅ | costs, slippage/spread, gap skips, limit-lock skips, liquidity cap, stop-first intrabar |
| Out-of-sample split and walk-forward | ✅ | |
| PSX fee schedule accuracy | 🟡 | **placeholder rates**; verify brokerage, CVT, SECP/PSX/NCCPL/CDC levies with your broker |
| Price limits | 🟡 | 7.5% assumption; verify current PSX circuit-breaker rules |
| Alerts (15 conditions), dedup, stale guard, webhook/email/WhatsApp | ✅ | email/WhatsApp require configured providers; monitoring status is reported honestly |
| Authorized PSX data integration | ⛔ | generic authorized-HTTP adapter only; no PSX feed licensed/configured. dps.psx.com.pk is not scraped. |

## Phase 6 — Production readiness
| Item | Status | Notes |
|---|---|---|
| Core logic test coverage | ✅ | 276 tests incl. look-ahead tests |
| Provider error handling | ✅ | 404 for unavailable data, 502 for provider errors |
| API-key auth, audit log, server-side key management | ✅ | single-tenant keys; no per-user accounts/RBAC |
| Docker / compose | 🟡 | files provided and `docker compose config` validated; **images not built here (no Docker daemon in this environment)** |
| Deployment docs | ✅ | `docs/DEPLOYMENT.md` |
| Performance | 🟡 | full analysis ≈0.1–0.3 s per symbol on ~400 daily bars (with MTF); pattern-strategy backtests ≈2 s per 365 bars (per-bar detection). Large universes or long intraday histories would benefit from process pools / caching. |
| PostgreSQL | 🟡 | models avoid SQLite-only types; not run against Postgres here |

## Known limitations (by design or pending)
* Real-market detector accuracy has not been measured — only the synthetic benchmark in
  `DETECTOR_EVALUATION.md`. Random walks frequently contain "confirmed" patterns, so pattern
  presence alone should not be treated as an edge without out-of-sample testing.
* Corporate actions are *flagged* (large gaps), never auto-adjusted; adjusted series must come
  from the data source.
* Holiday calendar contains fixed-date holidays only; add Eid and other moving holidays.
* Sharpe ratio is withheld below 20 trades / 120 bars.
