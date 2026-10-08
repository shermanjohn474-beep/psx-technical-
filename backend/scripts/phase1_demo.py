"""Phase-1 demo: validate an OHLCV CSV and run the quantitative engine.

    python scripts/phase1_demo.py fixtures/sample_data/SYN-DBOT_1d.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chart_patterns.base import build_context  # noqa: E402
from app.chart_patterns.registry import detect_patterns  # noqa: E402
from app.data_providers.file_provider import read_tabular  # noqa: E402
from app.data_validation.validator import validate_ohlcv  # noqa: E402
from app.indicators.engine import compute_indicators, indicator_snapshot  # noqa: E402
from app.schemas.market import Timeframe  # noqa: E402
from app.support_resistance.levels import detect_levels  # noqa: E402


def main(path: str) -> None:
    p = Path(path)
    raw = read_tabular(p.read_bytes(), p.name)
    df, q = validate_ohlcv(raw, p.stem.split("_")[0], Timeframe.D1)
    print(f"Data: {q.rows_out} bars {q.first_timestamp:%Y-%m-%d} -> {q.last_timestamp:%Y-%m-%d}  quality={q.rating} ({q.score})")
    for i in q.issues:
        print(f"  [{i.severity}] {i.code}: {i.message}")
    snap = indicator_snapshot(df, compute_indicators(df))
    print(f"Close {snap['close']:.2f} | RSI {snap['rsi']['value']} ({snap['rsi']['zone']}) | MACD {snap['macd']['state']}"
          f" | ADX {snap['adx']['adx']} ({snap['adx']['strength']}) | ATR {snap['atr']['value']}")
    ctx = build_context(df, timeframe="1d", last_candle_complete=q.last_candle_complete)
    lv = detect_levels(df, ctx.swings)
    for label in ("immediate_support", "major_support", "immediate_resistance", "major_resistance"):
        z = getattr(lv, label)
        print(f"  {label:22s}: " + (f"{z.low:.2f}-{z.high:.2f} touches={z.touches} strength={z.strength}" if z else "n/a"))
    print("Patterns:")
    for r in detect_patterns(ctx=ctx):
        print(f"  {r.name:28s} {r.stage.value:26s} {r.direction.value:8s} trigger={r.breakout_level} "
              f"target={r.target} invalidation={r.invalidation_level} quality={r.quality_score}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures/sample_data/SYN-DBOT_1d.csv")
