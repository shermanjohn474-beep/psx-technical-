"""PSX exchange calendar helpers (Asia/Karachi).

PSX trades Monday-Friday. Holidays come from a CSV file; fixed-date national
holidays are pre-populated but moving Islamic holidays must be supplied from the
official PSX notice. Anything not in the file is treated as a trading day, so an
unlisted holiday appears as a *missing candle* warning, never as fabricated data.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from app.config.settings import get_settings

PKT = ZoneInfo("Asia/Karachi")


@lru_cache
def load_holidays(path: str | None = None) -> frozenset[date]:
    p = Path(path) if path else get_settings().holidays_file
    out: set[date] = set()
    if not p.exists():
        return frozenset()
    with p.open() as fh:
        rows = (line for line in fh if line.strip() and not line.lstrip().startswith("#"))
        for row in csv.DictReader(rows):
            try:
                out.add(date.fromisoformat(row["date"].strip()))
            except (KeyError, ValueError):
                continue
    return frozenset(out)


def is_trading_day(d: date, holidays: frozenset[date] | None = None) -> bool:
    holidays = load_holidays() if holidays is None else holidays
    return d.weekday() < 5 and d not in holidays


def expected_trading_days(start: date, end: date, holidays: frozenset[date] | None = None) -> pd.DatetimeIndex:
    holidays = load_holidays() if holidays is None else holidays
    days = pd.bdate_range(start, end)
    return pd.DatetimeIndex([d for d in days if d.date() not in holidays])


def session_close(d: date) -> time:
    s = get_settings()
    hhmm = s.friday_session_close_hhmm if d.weekday() == 4 else s.session_close_hhmm
    h, m = (int(x) for x in hhmm.split(":"))
    return time(h, m)


def now_pkt() -> datetime:
    return datetime.now(PKT)
