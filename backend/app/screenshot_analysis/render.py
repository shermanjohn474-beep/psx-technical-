"""Candlestick chart renderer (OpenCV).

Used to (a) render annotated report charts server-side and (b) generate
screenshot test fixtures with *known* patterns in TradingView-like styling,
optionally with a volume panel, log scale and overlapping labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

UP = (154, 166, 38)      # BGR of #26a69a
DOWN = (80, 83, 239)     # BGR of #ef5350
BRAND = (223, 146, 0)    # BGR of #0092DF
GRID = (235, 235, 235)
TEXT = (60, 50, 40)


@dataclass
class RenderSpec:
    width: int = 1200
    height: int = 700
    margin_left: int = 20
    margin_right: int = 90
    margin_top: int = 40
    margin_bottom: int = 40
    volume_panel: bool = True
    volume_frac: float = 0.2
    log_scale: bool = False
    title: str | None = None
    labels_overlay: bool = False
    dark: bool = False


@dataclass
class ChartGeometry:
    x0: int
    x1: int
    y_top: int
    y_bot: int
    pmin: float
    pmax: float
    n: int
    log_scale: bool

    def x(self, i: float) -> int:
        step = (self.x1 - self.x0) / self.n
        return int(self.x0 + step * (i + 0.5))

    def y(self, price: float) -> int:
        if self.log_scale:
            a, b, p = np.log(self.pmin), np.log(self.pmax), np.log(max(price, 1e-9))
        else:
            a, b, p = self.pmin, self.pmax, price
        return int(self.y_bot - (p - a) / (b - a) * (self.y_bot - self.y_top))


def render_chart(df: pd.DataFrame, spec: RenderSpec | None = None) -> tuple[np.ndarray, ChartGeometry]:
    spec = spec or RenderSpec()
    bg = (24, 20, 19) if spec.dark else (255, 255, 255)
    img = np.full((spec.height, spec.width, 3), bg, dtype=np.uint8)
    n = len(df)
    vol_h = int((spec.height - spec.margin_top - spec.margin_bottom) * spec.volume_frac) if spec.volume_panel else 0
    gap = 12 if spec.volume_panel else 0
    y_top, y_bot = spec.margin_top, spec.height - spec.margin_bottom - vol_h - gap
    x0, x1 = spec.margin_left, spec.width - spec.margin_right
    pmin, pmax = float(df["low"].min()), float(df["high"].max())
    pad = (pmax - pmin) * 0.05
    pmin, pmax = max(pmin - pad, 1e-6), pmax + pad
    g = ChartGeometry(x0, x1, y_top, y_bot, pmin, pmax, n, spec.log_scale)
    # grid + right axis labels
    for k in range(6):
        if spec.log_scale:
            p = float(np.exp(np.log(pmin) + (np.log(pmax) - np.log(pmin)) * k / 5))
        else:
            p = pmin + (pmax - pmin) * k / 5
        yy = g.y(p)
        cv2.line(img, (x0, yy), (x1, yy), GRID if not spec.dark else (50, 50, 50), 1)
        cv2.putText(img, f"{p:.2f}", (x1 + 8, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT if not spec.dark else (200, 200, 200), 1, cv2.LINE_AA)
    step = (x1 - x0) / n
    bw = max(1, int(step * 0.65))
    for i, (_, r) in enumerate(df.iterrows()):
        col = UP if r["close"] >= r["open"] else DOWN
        xc = g.x(i)
        cv2.line(img, (xc, g.y(r["high"])), (xc, g.y(r["low"])), col, 1)
        top, bot = sorted((g.y(r["open"]), g.y(r["close"])))
        cv2.rectangle(img, (xc - bw // 2, top), (xc - bw // 2 + bw - 1, max(bot, top + 1)), col, -1)
    if spec.volume_panel and "volume" in df and df["volume"].notna().any():
        vmax = float(df["volume"].max()) or 1.0
        vb = spec.height - spec.margin_bottom
        for i, (_, r) in enumerate(df.iterrows()):
            col = tuple(int(c * 0.55 + 255 * 0.45) for c in (UP if r["close"] >= r["open"] else DOWN))
            hgt = int(vol_h * float(r["volume"]) / vmax)
            xc = g.x(i)
            cv2.rectangle(img, (xc - bw // 2, vb - hgt), (xc - bw // 2 + bw - 1, vb), col, -1)
    if spec.title:
        cv2.putText(img, spec.title, (x0, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT if not spec.dark else (230, 230, 230), 2, cv2.LINE_AA)
    if spec.labels_overlay:
        for k, txt in enumerate(("RSI 14  55.2", "MACD 12 26 9", "SMA 50")):
            cv2.putText(img, txt, (x0 + 10, y_top + 20 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    return img, g


def annotate_patterns(img: np.ndarray, g: ChartGeometry, patterns, zones=None, setups=None, max_patterns: int = 4) -> np.ndarray:
    """Draw pattern key points/lines, S/R zones and setup levels using a ChartGeometry."""
    out = img.copy()
    overlay = out.copy()
    for z in (zones or [])[:8]:
        y1, y2 = g.y(z.high), g.y(z.low)
        col = (120, 200, 120) if z.kind == "support" else (120, 120, 230)
        cv2.rectangle(overlay, (g.x0, y1), (g.x1, max(y2, y1 + 2)), col, -1)
    out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
    for p in patterns[:max_patterns]:
        for ln in p.lines:
            style_col = BRAND
            cv2.line(out, (g.x(ln.start.index), g.y(ln.start.price)), (g.x(ln.end.index), g.y(ln.end.price)), style_col, 2, cv2.LINE_AA)
        for k in p.key_points:
            pt = (g.x(k.index), g.y(k.price))
            cv2.circle(out, pt, 4, BRAND, -1, cv2.LINE_AA)
            cv2.putText(out, k.label, (pt[0] + 5, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BRAND, 1, cv2.LINE_AA)
        lab = f"{p.name} [{p.stage.value}]"
        x = g.x(p.start_index)
        cv2.putText(out, lab, (x, max(g.y_top + 12, g.y(max(k.price for k in p.key_points)) - 18) if p.key_points else g.y_top + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 40, 0), 1, cv2.LINE_AA)
    for s in setups or []:
        if not getattr(s, "valid", False) or s.entry_price is None:
            continue
        for lvl, col, name in ((s.entry_price, (0, 140, 255), "entry"), (s.stop_loss, (40, 40, 220), "stop")) + \
                tuple((t.price, (60, 160, 60), t.label) for t in s.targets):
            if lvl is None or not (g.pmin <= lvl <= g.pmax):
                continue
            yy = g.y(lvl)
            for xx in range(g.x0, g.x1, 12):
                cv2.line(out, (xx, yy), (min(xx + 6, g.x1), yy), col, 1)
            cv2.putText(out, f"{s.side} {name} {lvl:.2f}", (g.x1 - 160, yy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return out


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()
