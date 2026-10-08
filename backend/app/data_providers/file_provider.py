"""Local file provider: user-supplied CSV/Excel OHLCV files.

Files are stored as ``<SYMBOL>_<timeframe>.csv`` (e.g. ``OGDC_1d.csv``) with an
optional sidecar ``<SYMBOL>_<timeframe>.meta.json`` recording attribution and
adjustment status supplied by the user at upload time.
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.data_providers.base import DataUnavailable, MarketDataProvider, OHLCVResult, slice_range
from app.data_providers.resample import can_resample, resample_ohlcv
from app.data_validation.validator import validate_ohlcv
from app.schemas.market import DataSourceInfo, Timeframe

_FNAME = re.compile(r"^(?P<sym>[A-Z0-9\-\.]+)_(?P<tf>5m|15m|1h|1d|1w|1M)\.csv$")


def read_tabular(content: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(content))
        except ImportError as exc:  # openpyxl not installed
            raise DataUnavailable("Excel import requires 'openpyxl' (pip install openpyxl).") from exc
    text = content.decode("utf-8-sig", errors="replace")
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(lines)))


class FileProvider(MarketDataProvider):
    name = "local-files"

    def __init__(self, directory: Path, *, synthetic: bool = False, provider_label: str | None = None,
                 max_staleness_days: float | None = None):
        self.directory = Path(directory)
        self.synthetic = synthetic
        self.provider_label = provider_label or ("synthetic-demo" if synthetic else "user-upload")
        self.max_staleness_days = max_staleness_days
        self._cache: dict[tuple[str, Timeframe], tuple[float, OHLCVResult]] = {}

    def _files(self) -> dict[tuple[str, Timeframe], Path]:
        out = {}
        if not self.directory.exists():
            return out
        for p in self.directory.glob("*.csv"):
            m = _FNAME.match(p.name)
            if m:
                out[(m["sym"], Timeframe(m["tf"]))] = p
        return out

    def list_symbols(self) -> list[str]:
        return sorted({s for s, _ in self._files()})

    def native_timeframes(self, symbol: str) -> list[Timeframe]:
        return sorted([tf for (s, tf) in self._files() if s == symbol.upper()], key=lambda t: t.rank)

    def available_timeframes(self, symbol: str) -> list[Timeframe]:
        native = self.native_timeframes(symbol)
        tfs = set(native)
        for src in native:
            for dst in Timeframe:
                if can_resample(src, dst):
                    tfs.add(dst)
        return sorted(tfs, key=lambda t: t.rank)

    def _load_native(self, symbol: str, tf: Timeframe) -> OHLCVResult:
        path = self._files()[(symbol, tf)]
        mtime = path.stat().st_mtime
        cached = self._cache.get((symbol, tf))
        if cached and cached[0] == mtime:
            return cached[1]
        meta = {}
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        raw = read_tabular(path.read_bytes(), path.name)
        frame, quality = validate_ohlcv(raw, symbol, tf, max_staleness_days=None if self.synthetic else self.max_staleness_days)
        attribution = meta.get("attribution") or (
            "SYNTHETIC DEMO DATA - generated for testing; NOT real PSX prices."
            if self.synthetic else f"User-supplied file {path.name} (source not independently verified)."
        )
        source = DataSourceInfo(
            provider=self.provider_label,
            attribution=attribution,
            is_synthetic=self.synthetic or bool(meta.get("synthetic", False)),
            is_live=False,
            delayed=None,
            adjusted=meta.get("adjusted"),
            as_of=quality.last_timestamp,
            notes=list(meta.get("notes", [])),
        )
        if source.adjusted is None:
            source.notes.append("Corporate-action adjustment status unknown (splits/bonus/rights).")
        result = OHLCVResult(symbol, tf, frame, source, quality)
        self._cache[(symbol, tf)] = (mtime, result)
        return result

    def get_ohlcv(self, symbol: str, timeframe: Timeframe, start: datetime | None = None,
                  end: datetime | None = None) -> OHLCVResult:
        symbol = symbol.upper()
        native = self.native_timeframes(symbol)
        if not native:
            raise DataUnavailable(f"No local data for {symbol}.")
        if timeframe in native:
            base = self._load_native(symbol, timeframe)
            frame = slice_range(base.frame, start, end)
            return OHLCVResult(symbol, timeframe, frame, base.source, base.quality)
        sources = [tf for tf in native if can_resample(tf, timeframe)]
        if not sources:
            raise DataUnavailable(
                f"{timeframe.value} data for {symbol} is unavailable (have: {[t.value for t in native]}). "
                "Intraday candles are never derived from daily data."
            )
        src = max(sources, key=lambda t: t.rank)
        base = self._load_native(symbol, src)
        frame = resample_ohlcv(base.frame, src, timeframe)
        frame = slice_range(frame, start, end)
        _, quality = validate_ohlcv(frame.reset_index(), symbol, timeframe, check_calendar=False,
                                    max_staleness_days=None if self.synthetic else self.max_staleness_days)
        quality.issues = [i for i in base.quality.issues if i.code != "short_history"] + quality.issues
        note = f"{timeframe.value} candles aggregated from {src.value} data on calendar boundaries (Asia/Karachi)."
        source = base.source.model_copy(update={"notes": base.source.notes + [note]})
        return OHLCVResult(symbol, timeframe, frame, source, quality, notes=[note])

    def save_upload(self, symbol: str, timeframe: Timeframe, content: bytes, filename: str,
                    attribution: str | None = None, adjusted: bool | None = None) -> OHLCVResult:
        symbol = symbol.upper()
        if not re.fullmatch(r"[A-Z0-9\-\.]{1,20}", symbol):
            raise ValueError("Invalid symbol format.")
        raw = read_tabular(content, filename)
        frame, quality = validate_ohlcv(raw, symbol, timeframe)
        if frame.empty:
            raise DataUnavailable("Uploaded file contained no valid OHLCV rows.")
        self.directory.mkdir(parents=True, exist_ok=True)
        out = self.directory / f"{symbol}_{timeframe.value}.csv"
        frame.reset_index().assign(timestamp=lambda d: d["timestamp"].astype(str)).to_csv(out, index=False)
        meta = {"attribution": attribution or f"User upload '{filename}'", "adjusted": adjusted,
                "uploaded_at": datetime.now(timezone.utc).isoformat(), "original_filename": filename}
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        self._cache.pop((symbol, timeframe), None)
        return self._load_native(symbol, timeframe)
