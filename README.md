# PSX Technical Research Platform

An AI-assisted technical-analysis and pattern-recognition platform for the Pakistan Stock
Exchange (PSX). It analyses historical OHLCV data with a numerical engine (indicators, swing
detection, 40+ geometric pattern detectors, candlesticks, divergences, market structure,
support/resistance zones, multi-timeframe alignment, confluence scoring, long/short setups),
interprets chart screenshots with a three-tier confidence model, scans a symbol universe,
backtests strategies with PSX-style execution costs and walk-forward validation, and runs
completed-candle alerts.

> **Research output only — not investment advice.** No signal is guaranteed. Confluence and
> quality scores are heuristics, not probabilities. Backtests are simulations, not live results.

## What makes it different

* **Deterministic first, AI second.** Every price, level and pattern comes from the numerical
  engine. The optional LLM analyst only explains validated findings; its output is rejected
  if it contains a number or pattern that is not in the evidence (and the deterministic
  commentary is used instead). Every AI call is audit-logged with model id and hashes.
* **No fabricated data.** Real PSX symbols are analysed only from user-uploaded files or an
  authorized feed you configure. Demo data uses the reserved `SYN-` prefix and is labelled
  synthetic everywhere. Intraday candles are never manufactured from daily data.
* **Honest lifecycle.** Patterns move through *forming → approaching confirmation →
  confirmed → retesting / failed / invalidated*. Confirmation requires a **completed**
  candle close beyond the trigger; a break on a live candle is marked provisional.
* **No look-ahead.** Swing detection is causal (each pivot records the bar on which it became
  knowable); indicators, divergences, pattern confirmations and backtests are tested against
  future-data leakage.

## Quick start (local)

Requirements: Python 3.12+, Node 20+.

```bash
# backend
cd backend
pip install -r requirements.txt          # TA-Lib is optional (used as a test reference)
python scripts/generate_sample_data.py   # (re)creates the SYN- demo datasets
python -m pytest                         # 276 tests
uvicorn app.main:app --reload --port 8000

# frontend (second terminal)
cd frontend
npm ci
npm run dev                              # http://localhost:5173 (proxies /api to :8000)
```

Command-line demo of the engine:

```bash
cd backend
python scripts/phase1_demo.py fixtures/sample_data/SYN-HS_1d.csv
python scripts/evaluate_detectors.py --seeds 15   # writes docs/DETECTOR_EVALUATION.md
```

### Using real PSX data

Upload a CSV/Excel file (columns `date|timestamp, open, high, low, close, volume`; timestamps
are interpreted as Asia/Karachi) from the dashboard, or:

```bash
curl -F file=@OGDC.csv -F symbol=OGDC -F timeframe=1d -F attribution="Broker export" \
     http://localhost:8000/api/data/upload
```

Weekly/monthly views are aggregated from daily data on calendar boundaries (PSX week ends
Friday). For an automated feed, set `MARKET_DATA_URL_TEMPLATE` to an endpoint you are
licensed to use — no PSX endpoint is hard-coded or assumed scrapeable.

### Docker

```bash
cp .env.example .env      # edit as needed
docker compose up --build # frontend http://localhost:8080, API http://localhost:8000
```

## Configuration

All settings are environment variables — see [`.env.example`](.env.example). Key ones:

| Variable | Purpose |
|---|---|
| `APP_API_KEYS` | Comma-separated API keys; when set every `/api` call needs `X-API-Key`. Empty = open (dev only). |
| `AI_PROVIDER`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | Server-side AI provider (`anthropic` default model `claude-opus-5-5`, or `openai`). Keys never reach the browser. |
| `MARKET_DATA_URL_TEMPLATE` | Optional authorized JSON OHLCV feed. |
| `PSX_HOLIDAYS_FILE` | Holiday calendar CSV (moving Islamic holidays must be added from the PSX notice). |
| `SHORT_ELIGIBLE_SYMBOLS` | Symbols you verified as short-eligible; otherwise bearish setups are "analytical only". |
| `ALERTS_SCHEDULER_ENABLED` | Enables the background alert loop (on-demand evaluation always works). |

Pattern tolerances are in `PatternConfig` (`backend/app/chart_patterns/base.py`) and per-detector
overrides via `PatternConfig.params`; scoring weights can be overridden per request
(`/api/analysis/{symbol}?weights={"pattern":30}`).

## API overview

Interactive docs at `http://localhost:8000/docs`.

| Endpoint | Description |
|---|---|
| `GET /api/symbols` | Registry + symbols with data and available timeframes |
| `POST /api/data/upload` | Import CSV/Excel OHLCV |
| `GET /api/ohlcv/{symbol}` | Validated candles, quality report, selectable indicator series |
| `GET /api/analysis/{symbol}` | Full report (sections A–G); `ai=true` adds validated LLM commentary |
| `GET /api/analysis/{symbol}/chart.png` | Annotated chart image |
| `GET /api/report/{symbol}` | Printable HTML report (browser “Save as PDF”) |
| `POST /api/screenshot` | Chart screenshot(s) → tiered analysis + annotated image |
| `POST /api/scan` | Universe scanner with rules; `GET/POST /api/scan/presets`, `/api/watchlists` |
| `POST /api/backtest` | Strategy backtests (+ walk-forward) |
| `GET/POST/DELETE /api/alerts`, `POST /api/alerts/evaluate`, `GET /api/alerts/status` | Alerts |
| `GET /api/config`, `GET /api/audit` | Effective configuration (no secrets) and audit log |

## Documentation

* [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) — what is done, partial, or not done
* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — modules, data flow, pattern definitions, design rules
* [docs/DETECTOR_EVALUATION.md](docs/DETECTOR_EVALUATION.md) — synthetic recall / false-positive benchmark
* [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — production checklist
