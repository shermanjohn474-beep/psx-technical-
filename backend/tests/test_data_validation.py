import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from app.data_providers.base import DataUnavailable
from app.data_providers.file_provider import FileProvider
from app.data_providers.manager import ProviderManager
from app.data_providers.resample import ResampleNotAllowed, resample_ohlcv
from app.data_providers.synthetic import make_ohlcv, random_walk
from app.data_validation.calendar import PKT
from app.data_validation.validator import DataValidationError, is_last_candle_complete, validate_ohlcv
from app.schemas.market import Timeframe


def raw_frame(n=60):
    df = random_walk(n, seed=3)
    out = df.reset_index()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    return out.rename(columns={"timestamp": "Date", "close": "Close", "open": "Open", "high": "High", "low": "Low",
                               "volume": "Volume"})


def test_normalizes_columns_and_timezone():
    df, rep = validate_ohlcv(raw_frame(), "TEST", Timeframe.D1, as_of=datetime(2030, 1, 1, tzinfo=PKT))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "Asia/Karachi"
    assert df.index.is_monotonic_increasing
    assert rep.rows_out == 60


def test_empty_data():
    df, rep = validate_ohlcv(pd.DataFrame(), "X", Timeframe.D1)
    assert df.empty and rep.rating == "insufficient"


def test_missing_required_column():
    raw = raw_frame().drop(columns=["High"])
    with pytest.raises(DataValidationError):
        validate_ohlcv(raw, "X", Timeframe.D1)


def test_duplicates_and_invalid_rows_removed():
    raw = raw_frame()
    raw = pd.concat([raw, raw.iloc[[5]]], ignore_index=True)
    raw.loc[10, "High"] = raw.loc[10, "Low"] - 1  # high < low
    raw.loc[11, "Close"] = -5
    df, rep = validate_ohlcv(raw, "X", Timeframe.D1)
    assert rep.duplicates_removed == 1
    assert rep.invalid_ohlc_rows == 2
    assert rep.rows_out == 58
    assert {i.code for i in rep.issues} >= {"duplicates", "invalid_ohlc"}


def test_missing_timestamps_dropped():
    raw = raw_frame()
    raw.loc[3, "Date"] = None
    raw.loc[4, "Date"] = "not a date"
    df, rep = validate_ohlcv(raw, "X", Timeframe.D1)
    assert rep.rows_out == 58
    assert any(i.code == "bad_timestamp" for i in rep.issues)


def test_missing_volume_reported():
    raw = raw_frame().drop(columns=["Volume"])
    df, rep = validate_ohlcv(raw, "X", Timeframe.D1)
    assert not rep.has_volume and df["volume"].isna().all()


def test_missing_candles_detected_not_filled():
    raw = raw_frame()
    raw = raw.drop(index=[20, 21, 22])
    df, rep = validate_ohlcv(raw, "X", Timeframe.D1)
    assert rep.missing_candles >= 3
    assert len(df) == 57  # nothing synthesized


def test_price_gap_flagged_as_possible_corporate_action():
    raw = raw_frame()
    raw.loc[30:, ["Open", "High", "Low", "Close"]] *= 0.5  # e.g. unadjusted 1:1 bonus
    raw.loc[30, "High"] = max(raw.loc[30, "High"], raw.loc[30, "Open"], raw.loc[30, "Close"])
    df, rep = validate_ohlcv(raw, "X", Timeframe.D1)
    assert any(i.code == "possible_corporate_action" for i in rep.issues)


def test_suspended_flat_bars_flagged():
    raw = raw_frame(100)
    for k in range(40, 60):
        p = raw.loc[39, "Close"]
        raw.loc[k, ["Open", "High", "Low", "Close"]] = p
        raw.loc[k, "Volume"] = 0
    df, rep = validate_ohlcv(raw, "X", Timeframe.D1)
    assert rep.zero_volume_bars == 20
    assert any(i.code == "illiquid" for i in rep.issues)


def test_staleness_flag():
    raw = raw_frame()
    df, rep = validate_ohlcv(raw, "X", Timeframe.D1, as_of=datetime(2030, 1, 1, tzinfo=PKT), max_staleness_days=4)
    assert rep.stale
    assert any(i.code == "stale" for i in rep.issues)


def test_incomplete_daily_candle():
    ts = pd.Timestamp("2025-03-04", tz=PKT)
    assert not is_last_candle_complete(ts, Timeframe.D1, datetime(2025, 3, 4, 11, 0, tzinfo=PKT))
    assert is_last_candle_complete(ts, Timeframe.D1, datetime(2025, 3, 4, 16, 0, tzinfo=PKT))
    assert not is_last_candle_complete(pd.Timestamp("2025-03-04 10:00", tz=PKT), Timeframe.H1,
                                       datetime(2025, 3, 4, 10, 30, tzinfo=PKT))


def test_weekly_monthly_resample_calendar_boundaries():
    df = make_ohlcv([(0, 100), (120, 130)])
    w = resample_ohlcv(df, Timeframe.D1, Timeframe.W1)
    m = resample_ohlcv(df, Timeframe.D1, Timeframe.MN1)
    # every weekly bar contains days of one Mon-Fri week only
    for ts, row in w.iterrows():
        week = df[(df.index >= ts) & (df.index < ts + pd.Timedelta(days=7))]
        assert row["high"] == pytest.approx(week["high"].max())
    assert m.index[0].month == df.index[0].month
    assert m["volume"].sum() == pytest.approx(df["volume"].sum())


def test_intraday_never_derived_from_daily():
    df = make_ohlcv([(0, 100), (50, 120)])
    with pytest.raises(ResampleNotAllowed):
        resample_ohlcv(df, Timeframe.D1, Timeframe.H1)


def test_provider_never_substitutes_synthetic_for_real(tmp_path):
    synth_dir = tmp_path / "synth"
    synth_dir.mkdir()
    make_ohlcv([(0, 100), (80, 120)]).reset_index().to_csv(synth_dir / "OGDC_1d.csv", index=False)
    mgr = ProviderManager(FileProvider(tmp_path / "user"), FileProvider(synth_dir, synthetic=True))
    with pytest.raises(DataUnavailable):
        mgr.get_ohlcv("OGDC", Timeframe.D1)


def test_upload_roundtrip_and_resample(tmp_path):
    fp = FileProvider(tmp_path)
    csv = raw_frame(120).to_csv(index=False).encode()
    res = fp.save_upload("ogdc", Timeframe.D1, csv, "ogdc.csv", attribution="test file")
    assert res.symbol == "OGDC" and not res.source.is_synthetic
    weekly = fp.get_ohlcv("OGDC", Timeframe.W1)
    assert len(weekly.frame) < 30 and "aggregated" in weekly.notes[0]
    with pytest.raises(DataUnavailable):
        fp.get_ohlcv("OGDC", Timeframe.M15)
