"""Geometric candle extraction from chart screenshots (OpenCV).

Pipeline:
1. Colour segmentation (HSV) for up/down candle colours. Defaults cover the common
   green/teal vs red palettes; others can be supplied.
2. Panel separation: the row-occupancy profile of candle-coloured pixels is split
   at empty horizontal bands; the tallest band is the price panel, others are
   reported as sub-panels (volume/indicators) and ignored for price extraction.
3. Candle extraction: connected components per colour inside the price panel.
   The wick is the full vertical extent; the body is the rows where the
   component is at least ``body_width_frac`` of its maximum width.
4. Output pixel-space OHLC (y inverted so "up" is larger) and, if an axis
   calibration is available, price-space OHLC with a per-value uncertainty.

Limitations (reported, not hidden): hollow/monochrome candles, heavy overlays
drawn in candle colours, Heikin-Ashi or line charts, and touching same-colour
candles at very high density reduce extraction quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class AxisCalibration(BaseModel):
    """Two y-pixel positions with their prices (from the user or a vision model)."""

    y1: float
    price1: float
    y2: float
    price2: float
    log_scale: bool = False
    source: str = "user"  # user | vision_model

    def price(self, y: np.ndarray | float):
        y = np.asarray(y, dtype=float)
        t = (y - self.y1) / (self.y2 - self.y1)
        if self.log_scale:
            a, b = np.log(self.price1), np.log(self.price2)
            return np.exp(a + t * (b - a))
        return self.price1 + t * (self.price2 - self.price1)

    def price_per_pixel(self, y: float) -> float:
        return float(abs(self.price(y + 0.5) - self.price(y - 0.5)))


class Panel(BaseModel):
    top: int
    bottom: int
    left: int
    right: int
    role: str  # price | sub_panel


class ExtractionResult(BaseModel):
    ok: bool
    image_width: int
    image_height: int
    panels: list[Panel] = Field(default_factory=list)
    price_panel: Panel | None = None
    candles: int = 0
    median_candle_width: float | None = None
    extraction_confidence: float = 0.0  # 0..1
    issues: list[str] = Field(default_factory=list)
    x_centers: list[float] = Field(default_factory=list)


@dataclass
class ColorProfile:
    up_ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = field(default_factory=lambda: [
        ((35, 60, 60), (100, 255, 255)),     # green / teal
    ])
    down_ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = field(default_factory=lambda: [
        ((0, 70, 70), (10, 255, 255)), ((165, 70, 70), (180, 255, 255)),  # red
    ])


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image (supported: PNG, JPG, JPEG, WEBP).")
    return img


def _mask(hsv: np.ndarray, ranges) -> np.ndarray:
    m = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    return m


def _bands(occ: np.ndarray, min_gap: int, min_height: int) -> list[tuple[int, int]]:
    rows = np.flatnonzero(occ > 0)
    if len(rows) == 0:
        return []
    bands, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - prev > min_gap:
            bands.append((start, prev))
            start = r
        prev = r
    bands.append((start, prev))
    return [b for b in bands if b[1] - b[0] >= min_height]


def extract_candles(img: np.ndarray, colors: ColorProfile | None = None,
                    body_width_frac: float = 0.6) -> tuple[pd.DataFrame | None, ExtractionResult]:
    colors = colors or ColorProfile()
    h, w = img.shape[:2]
    res = ExtractionResult(ok=False, image_width=w, image_height=h)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    up, dn = _mask(hsv, colors.up_ranges), _mask(hsv, colors.down_ranges)
    both = up | dn
    if both.sum() / 255 < 200:
        res.issues.append("No candle-coloured pixels found (monochrome, hollow candles, line chart or unusual palette).")
        return None, res
    occ = (both > 0).sum(axis=1)
    bands = _bands(occ, min_gap=max(4, h // 120), min_height=max(10, h // 25))
    if not bands:
        res.issues.append("Could not isolate a chart panel.")
        return None, res
    price_band = max(bands, key=lambda b: b[1] - b[0])
    cols = np.flatnonzero((both[price_band[0]:price_band[1] + 1] > 0).any(axis=0))
    left, right = int(cols.min()), int(cols.max())
    for b in bands:
        res.panels.append(Panel(top=int(b[0]), bottom=int(b[1]), left=left, right=right,
                                role="price" if b == price_band else "sub_panel"))
    if len(bands) > 1:
        res.issues.append(f"{len(bands) - 1} additional panel(s) detected (volume/indicators) and excluded from price extraction.")
    pp = next(p for p in res.panels if p.role == "price")
    res.price_panel = pp
    rows = []
    for color_mask, bull in ((up, True), (dn, False)):
        sub = color_mask[pp.top:pp.bottom + 1, :]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        for k in range(1, n):
            x, y, cw, ch, area = stats[k]
            if area < 3 or ch < 2:
                continue
            comp = labels[y:y + ch, x:x + cw] == k
            widths = comp.sum(axis=1)
            wmax = widths.max()
            body_rows = np.flatnonzero(widths >= max(1, body_width_frac * wmax))
            if wmax <= 2 and cw <= 2:
                body_rows = np.array([0, ch - 1])  # pure wick-like: treat as doji line
            top_wick, bot_wick = y, y + ch - 1
            body_top, body_bot = y + body_rows.min(), y + body_rows.max()
            xc = x + cw / 2
            rows.append((xc, cw, top_wick + pp.top, bot_wick + pp.top, body_top + pp.top, body_bot + pp.top, bull, area))
    if len(rows) < 10:
        res.issues.append(f"Only {len(rows)} candle components found: insufficient for analysis.")
        return None, res
    cand = pd.DataFrame(rows, columns=["x", "w", "wick_top", "wick_bot", "body_top", "body_bot", "bull", "area"])
    med_w = float(cand["w"].median())
    # drop text/marker blobs far wider than candles and specks
    cand = cand[(cand["w"] <= 3.5 * med_w + 2)].sort_values("x").reset_index(drop=True)
    # merge fragments with nearly identical x (e.g. wick broken by a grid line)
    merged = []
    for _, r in cand.iterrows():
        if merged and abs(r["x"] - merged[-1]["x"]) < max(1.0, 0.4 * med_w) and r["bull"] == merged[-1]["bull"]:
            m = merged[-1]
            m["wick_top"] = min(m["wick_top"], r["wick_top"])
            m["wick_bot"] = max(m["wick_bot"], r["wick_bot"])
            m["body_top"] = min(m["body_top"], r["body_top"])
            m["body_bot"] = max(m["body_bot"], r["body_bot"])
        else:
            merged.append(r.to_dict())
    cand = pd.DataFrame(merged)
    spacing = np.diff(cand["x"].to_numpy())
    med_sp = float(np.median(spacing)) if len(spacing) else 0.0
    irregular = float(np.mean(np.abs(spacing - med_sp) > 0.5 * med_sp)) if med_sp > 0 else 1.0
    if irregular > 0.15:
        res.issues.append(f"Irregular candle spacing ({irregular:.0%} of gaps): possible merged/missing candles or gaps.")
    H = float(h)
    out = pd.DataFrame({
        "open": np.where(cand["bull"], H - cand["body_bot"], H - cand["body_top"]),
        "close": np.where(cand["bull"], H - cand["body_top"], H - cand["body_bot"]),
        "high": H - cand["wick_top"],
        "low": H - cand["wick_bot"],
    })
    out["volume"] = np.nan
    out["y_open"] = np.where(cand["bull"], cand["body_bot"], cand["body_top"])
    out["y_close"] = np.where(cand["bull"], cand["body_top"], cand["body_bot"])
    out["y_high"] = cand["wick_top"].to_numpy()
    out["y_low"] = cand["wick_bot"].to_numpy()
    res.x_centers = [float(x) for x in cand["x"]]
    res.candles = len(out)
    res.median_candle_width = med_w
    res.ok = True
    conf = 1.0 - 0.6 * irregular - (0.2 if len(out) < 30 else 0.0) - (0.1 if len(bands) > 3 else 0.0)
    res.extraction_confidence = round(float(np.clip(conf, 0.05, 1.0)), 2)
    return out, res


def to_price_frame(pixel: pd.DataFrame, cal: AxisCalibration) -> tuple[pd.DataFrame, float]:
    """Convert pixel OHLC to price OHLC; returns (frame, typical +/- uncertainty in price)."""
    out = pd.DataFrame({
        "open": cal.price(pixel["y_open"].to_numpy()),
        "high": cal.price(pixel["y_high"].to_numpy()),
        "low": cal.price(pixel["y_low"].to_numpy()),
        "close": cal.price(pixel["y_close"].to_numpy()),
    })
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    out["volume"] = np.nan
    unc = 1.5 * float(np.median([cal.price_per_pixel(y) for y in pixel["y_close"].to_numpy()]))
    return out, unc
