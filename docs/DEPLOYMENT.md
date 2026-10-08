# Deployment checklist

1. **Secrets**: copy `.env.example` to `.env`; set `APP_API_KEYS` (long random values) and any
   AI provider key. Keys are read server-side only; `/api/config` never returns them.
2. **CORS**: set `CORS_ORIGINS` to the frontend origin(s).
3. **Database**: SQLite is fine for a single instance. For PostgreSQL, add a driver
   (e.g. `psycopg[binary]`) to `requirements.txt` and set
   `DATABASE_URL=postgresql+psycopg://user:pass@host:5432/psx`. Tables are created on start-up
   (add Alembic migrations before schema changes in production).
4. **Data**: mount a persistent volume at `DATA_DIR` (uploaded OHLCV, audit JSONL). Configure
   `MARKET_DATA_URL_TEMPLATE` only for feeds you are licensed to use.
5. **Calendar**: maintain `PSX_HOLIDAYS_FILE` with the official PSX holiday notice each year.
6. **Costs & rules**: review `CostModel` defaults and the 7.5% price-limit assumption against
   your broker and current PSX/SECP/NCCPL/CDC schedules. Populate `SHORT_ELIGIBLE_SYMBOLS`
   only after verifying eligibility.
7. **Alerts**: set `ALERTS_SCHEDULER_ENABLED=true` and SMTP / WhatsApp provider settings if
   needed. Run a single API worker when the in-process scheduler is enabled (or move the
   scheduler to a separate process) to avoid duplicate loops; duplicate *events* are already
   prevented by a unique key per rule and bar.
8. **TLS / reverse proxy**: terminate TLS at your proxy; the bundled nginx config proxies
   `/api` to the backend and allows 30 MB uploads.
9. **Monitoring**: `/health` reports data/AI/feed configuration; `/api/audit` lists mutating
   requests; `data/audit/ai_audit.jsonl` and the `ai_audit` table record every AI call.
10. **Run**: `docker compose up --build -d` (frontend :8080, API :8000).
