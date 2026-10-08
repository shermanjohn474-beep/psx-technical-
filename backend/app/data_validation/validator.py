"""OHLCV normalization and validation.

Canonical frame used everywhere in the engine:

* ``DatetimeIndex`` named ``timestamp``, tz-aware ``Asia/Karachi``, strictly increasing
* float columns ``open, high, low, close`` and ``volume`` (NaN volume = unavailable)

The validator never invents candles. Missing candles are *reported*; invalid
rows are *dropped and reported*.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from app.data_validation.calendar import PKT, expected_trading_days, load_holidays, now_pkt, session_close
from app.schemas.market import DataQualityReport, Timeframe

REQUIRED = ["open", "high", "low", "close"]

_ALIASES = {
    "date": "timestamp", "datetime": "timestamp", "time": "timestamp", "timestamp": "timestamp",
    "o": "open", "open": "open", "h": "high", "high": "high", "l": "low", "low": "low",
    "c": "close", "close": "close", "price": "close", "adj close": "adj_close", "adj_close": "adj_close",
    "v": "volume", "vol": "volume", "volume": "volume",
}


class DataValidationError(ValueError):
    pass


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in _ALIASES:
            rename[c] = _ALIASES[key]
    out = df.rename(columns=rename)
    if out.index.name and str(out.index.name).lower() in _ALIASES and "timestamp" not in out.columns:
        out = out.reset_index().rename(columns={out.index.name: "timestamp"})
    return out


def to_pkt_index(ts: pd.Series | pd.Index, assume_tz: str = "Asia/Karachi") -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(ts, errors="coerce"))
    if idx.tz is None:
        idx = idx.tz_localize(assume_tz, ambiguous="NaT", nonexistent="NaT")
    return idx.tz_convert(PKT)


def validate_ohlcv(
    raw: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    *,
    assume_tz: str = "Asia/Karachi",
    as_of: datetime | None = None,
    max_staleness_days: float | None = None,
    check_calendar: bool = True,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Normalize + validate. Returns (clean_frame, report)."""
    report = DataQualityReport(symbol=symbol, timeframe=timeframe, rows_in=len(raw), rows_out=0)
    if raw is None or len(raw) == 0:
        report.add("error", "empty", "No rows supplied.")
        report.rating, report.score = "insufficient", 0.0
        return _empty_frame(), report

    df = normalize_columns(raw.copy())
    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        else:
            raise DataValidationError("No timestamp/date column found.")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    df["timestamp"] = to_pkt_index(df["timestamp"], assume_tz)
    bad_ts = int(df["timestamp"].isna().sum())
    if bad_ts:
        report.add("warning", "bad_timestamp", f"{bad_ts} rows with unparseable/missing timestamps dropped.", bad_ts)
        df = df[df["timestamp"].notna()]

    for c in REQUIRED + (["volume"] if "volume" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = np.nan
        report.has_volume = False
        report.add("warning", "no_volume", "Volume column missing: volume-based confirmation unavailable.")

    if timeframe in (Timeframe.D1, Timeframe.W1, Timeframe.MN1):
        # Daily+ bars are keyed by calendar date in Karachi.
        df["timestamp"] = df["timestamp"].dt.normalize()

    df = df.sort_values("timestamp")
    dup_mask = df["timestamp"].duplicated(keep="last")
    report.duplicates_removed = int(dup_mask.sum())
    if report.duplicates_removed:
        report.add("warning", "duplicates", "Duplicate timestamps removed (kept last occurrence).", report.duplicates_removed)
        df = df[~dup_mask]

    o, h, l, c = (df[k] for k in REQUIRED)
    invalid = (
        o.isna() | h.isna() | l.isna() | c.isna()
        | (h < l) | (h < np.maximum(o, c) - 1e-9) | (l > np.minimum(o, c) + 1e-9)
        | (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    )
    if "volume" in df.columns:
        invalid |= df["volume"] < 0
    report.invalid_ohlc_rows = int(invalid.sum())
    if report.invalid_ohlc_rows:
        ex = [str(t.date()) for t in df.loc[invalid, "timestamp"].head(5)]
        report.add("warning", "invalid_ohlc", "Rows with inconsistent/non-positive OHLC dropped.", report.invalid_ohlc_rows, ex)
        df = df[~invalid]

    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)
    df.index.name = "timestamp"

    if df["volume"].notna().any():
        report.zero_volume_bars = int((df["volume"] == 0).sum())
        if report.zero_volume_bars:
            report.add("info", "zero_volume", "Bars with zero volume (possible suspension/no trades).", report.zero_volume_bars)
        frac_flat = float(((df["high"] == df["low"]) & (df["volume"].fillna(0) == 0)).mean()) if len(df) else 0.0
        if frac_flat > 0.05:
            report.add("warning", "illiquid", f"{frac_flat:.0%} of bars are flat with no volume: low liquidity / suspended periods.")
    else:
        report.has_volume = False

    # Gap/split heuristics: large overnight jumps are flagged, never auto-adjusted.
    if len(df) > 1:
        ret = df["open"] / df["close"].shift(1) - 1
        big = ret.abs() > 0.30
        if big.any():
            ex = [str(t.date()) for t in df.index[big][:5]]
            report.add(
                "warning", "possible_corporate_action",
                "Open-vs-prior-close jump >30%: possible split/bonus/rights adjustment issue or data error. "
                "Verify adjustment status before trusting levels across this date.",
                int(big.sum()), ex,
            )

    if check_calendar and timeframe == Timeframe.D1 and len(df) > 1:
        expected = expected_trading_days(df.index[0].date(), df.index[-1].date(), load_holidays())
        have = set(df.index.date)
        miss = [d for d in expected if d.date() not in have]
        report.missing_candles = len(miss)
        if miss:
            report.add(
                "info" if len(miss) / max(len(expected), 1) < 0.03 else "warning",
                "missing_candles",
                "Expected PSX trading days without a candle (holiday not in calendar, suspension, or data gap). "
                "No candles were synthesized.",
                len(miss), [str(d.date()) for d in miss[:5]],
            )
    elif check_calendar and timeframe.is_intraday and len(df) > 1:
        step = pd.Timedelta(minutes=timeframe.minutes or 0)
        diffs = df.index.to_series().diff()
        same_day = df.index.to_series().dt.date == df.index.to_series().shift(1).dt.date
        gaps = int(((diffs > step * 1.5) & same_day).sum())
        report.missing_candles = gaps
        if gaps:
            report.add("info", "intraday_gaps", "Intra-session gaps detected (no trades or missing data).", gaps)

    now = as_of or now_pkt()
    if len(df):
        report.first_timestamp = df.index[0].to_pydatetime()
        report.last_timestamp = df.index[-1].to_pydatetime()
        report.last_candle_complete = is_last_candle_complete(df.index[-1], timeframe, now)
        if not report.last_candle_complete:
            report.add("info", "incomplete_candle", "Last candle may still be forming; signals on it are provisional.")
        staleness = (now - df.index[-1].to_pydatetime()).total_seconds() / 86400
        report.staleness_days = round(staleness, 2)
        limit = max_staleness_days
        if limit is not None and staleness > limit:
            report.stale = True
            report.add("warning", "stale", f"Latest candle is {staleness:.1f} days old: not current market data.")

    report.rows_out = len(df)
    _rate(report)
    return df, report


def is_last_candle_complete(ts: pd.Timestamp, timeframe: Timeframe, now: datetime) -> bool:
    ts = ts.tz_convert(PKT) if ts.tzinfo else ts.tz_localize(PKT)
    now = pd.Timestamp(now).tz_convert(PKT) if pd.Timestamp(now).tzinfo else pd.Timestamp(now).tz_localize(PKT)
    if timeframe.is_intraday:
        return now >= ts + pd.Timedelta(minutes=timeframe.minutes or 0)
    d = ts.date()
    close_dt = pd.Timestamp.combine(d, session_close(d)).tz_localize(PKT)
    if timeframe == Timeframe.D1:
        return now >= close_dt
    if timeframe == Timeframe.W1:
        week_end = (ts + pd.offsets.Week(weekday=4)).normalize() if ts.weekday() != 4 else ts.normalize()
        return now >= pd.Timestamp.combine(week_end.date(), session_close(week_end.date())).tz_localize(PKT)
    if timeframe == Timeframe.MN1:
        month_end = (ts + pd.offsets.MonthEnd(0)).normalize()
        return now.date() > month_end.date()
    return True


def _rate(report: DataQualityReport) -> None:
    if report.rows_out == 0:
        report.rating, report.score = "insufficient", 0.0
        return
    score = 100.0
    if report.rows_out < 60:
        score -= 35
        report.add("warning", "short_history", f"Only {report.rows_out} bars: long-period indicators (e.g. SMA200) unavailable.")
    elif report.rows_out < 200:
        score -= 10
    if not report.has_volume:
        score -= 15
    score -= min(20, 100 * report.invalid_ohlc_rows / max(report.rows_in, 1))
    score -= min(15, 100 * report.missing_candles / max(report.rows_out, 1))
    if report.stale:
        score -= 20
    if any(i.code == "possible_corporate_action" for i in report.issues):
        score -= 10
    report.score = max(0.0, round(score, 1))
    report.rating = "high" if score >= 80 else "medium" if score >= 60 else "low" if score >= 30 else "insufficient"


def _empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz=PKT, name="timestamp")
    return pd.DataFrame({c: pd.Series(dtype=float) for c in ["open", "high", "low", "close", "volume"]}, index=idx)
