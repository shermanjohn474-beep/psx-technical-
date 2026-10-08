"""Market structure: swing labels, BOS/CHoCH, fair value gaps, order blocks,
supply/demand zones and Wyckoff spring/upthrust candidates.

HH/HL/LH/LL labelling and BOS/CHoCH are objective given the swing settings.
Order blocks, supply/demand zones and Wyckoff events are **heuristic** concepts
with competing definitions; they are labelled ``heuristic=True`` and must not be
presented as confirmed facts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.swing_detection.swings import SwingSet


class StructureEvent(BaseModel):
    type: str  # BOS | CHoCH
    direction: str
    index: int
    time: str
    level: float
    broken_swing_index: int


class Zone2(BaseModel):
    type: str
    direction: str
    low: float
    high: float
    index: int
    time: str
    status: str = "open"  # open | partially_filled | filled | mitigated
    heuristic: bool = True
    note: str | None = None


class StructureResult(BaseModel):
    labels: list[dict] = Field(default_factory=list)
    trend: str = "undetermined"  # uptrend | downtrend | range | undetermined
    events: list[StructureEvent] = Field(default_factory=list)
    fvgs: list[Zone2] = Field(default_factory=list)
    order_blocks: list[Zone2] = Field(default_factory=list)
    supply_demand: list[Zone2] = Field(default_factory=list)
    wyckoff: list[Zone2] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def label_swings(swings) -> list[dict]:
    out = []
    last_h = last_l = None
    for s in swings:
        if s.kind == "H":
            lab = None if last_h is None else ("HH" if s.price > last_h.price else "LH")
            last_h = s
        else:
            lab = None if last_l is None else ("HL" if s.price > last_l.price else "LL")
            last_l = s
        out.append({"index": s.index, "kind": s.kind, "price": round(s.price, 4), "label": lab,
                    "confirmed_index": s.confirmed_index})
    return out


def classify_trend(labels: list[dict]) -> str:
    recent = [x["label"] for x in labels[-4:] if x["label"]]
    if len(recent) < 3:
        return "undetermined"
    up = sum(lab in ("HH", "HL") for lab in recent)
    dn = sum(lab in ("LH", "LL") for lab in recent)
    if up >= 3 and dn == 0 or (up == 3 and len(recent) == 4):
        return "uptrend"
    if dn >= 3 and up == 0 or (dn == 3 and len(recent) == 4):
        return "downtrend"
    return "range"


def analyze_structure(df: pd.DataFrame, swings: SwingSet, recent: int = 200) -> StructureResult:
    n = len(df)
    res = StructureResult()
    if n < 20:
        res.notes.append("Insufficient data for market structure.")
        return res
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr = swings.atr.to_numpy()
    ts = df.index
    sw = [s for s in swings.swings if s.confirmed_index <= n - 1]
    res.labels = label_swings(sw)
    res.trend = classify_trend(res.labels)

    # BOS / CHoCH: walk bars; reference swings are those already confirmed at bar t
    state = 0  # +1 up, -1 down
    k = 0
    known_h = known_l = None
    used_h = used_l = None
    for t in range(n):
        while k < len(sw) and sw[k].confirmed_index <= t:
            if sw[k].kind == "H":
                known_h = sw[k]
            else:
                known_l = sw[k]
            k += 1
        if known_h is not None and known_h is not used_h and c[t] > known_h.price:
            typ = "BOS" if state >= 0 else "CHoCH"
            res.events.append(StructureEvent(type=typ, direction="bullish", index=t, time=ts[t].isoformat(),
                                             level=round(known_h.price, 4), broken_swing_index=known_h.index))
            state, used_h = 1, known_h
        if known_l is not None and known_l is not used_l and c[t] < known_l.price:
            typ = "BOS" if state <= 0 else "CHoCH"
            res.events.append(StructureEvent(type=typ, direction="bearish", index=t, time=ts[t].isoformat(),
                                             level=round(known_l.price, 4), broken_swing_index=known_l.index))
            state, used_l = -1, known_l
    res.events = [e for e in res.events if e.index >= n - recent]

    # Fair value gaps (3-candle imbalance) with fill tracking
    for i in range(max(2, n - recent), n):
        for bull in (True, False):
            gap_lo, gap_hi = (h[i - 2], l[i]) if bull else (h[i], l[i - 2])
            if gap_hi - gap_lo <= 0.1 * atr[i]:
                continue
            status = "open"
            fut_l, fut_h = l[i + 1:], h[i + 1:]
            if len(fut_l):
                if bull:
                    status = "filled" if fut_l.min() <= gap_lo else "partially_filled" if fut_l.min() < gap_hi else "open"
                else:
                    status = "filled" if fut_h.max() >= gap_hi else "partially_filled" if fut_h.max() > gap_lo else "open"
            res.fvgs.append(Zone2(type="FVG", direction="bullish" if bull else "bearish", low=round(gap_lo, 4),
                                  high=round(gap_hi, 4), index=i - 1, time=ts[i - 1].isoformat(), status=status,
                                  heuristic=False, note="3-candle imbalance"))
    res.fvgs = [z for z in res.fvgs if z.status != "filled"][-10:]

    # Order-block candidates: last opposite candle before a BOS/CHoCH impulse
    for e in res.events:
        bull = e.direction == "bullish"
        lo_i = max(0, e.broken_swing_index - 30)
        for j in range(e.index - 1, lo_i, -1):
            if (bull and c[j] < o[j]) or (not bull and c[j] > o[j]):
                after = slice(e.index + 1, n)
                mitig = bool(e.index + 1 < n and (l[after].min() <= h[j] if bull else h[after].max() >= l[j]))
                res.order_blocks.append(Zone2(type="order_block", direction=e.direction, low=round(l[j], 4),
                                              high=round(h[j], 4), index=j, time=ts[j].isoformat(),
                                              status="mitigated" if mitig else "open",
                                              note=f"Last opposite candle before {e.type} on {e.time[:10]} (heuristic)"))
                break
    res.order_blocks = res.order_blocks[-6:]

    # Supply/demand: tight base (<=3 small bars) followed by a >= 2.5 ATR move within 3 bars
    body = np.abs(c - o)
    for i in range(max(3, n - recent), n - 3):
        base = slice(i - 2, i + 1)
        if body[base].max() > 0.6 * atr[i]:
            continue
        move = c[i + 3] - c[i]
        if abs(move) >= 2.5 * atr[i]:
            demand = move > 0
            res.supply_demand.append(Zone2(type="demand" if demand else "supply", direction="bullish" if demand else "bearish",
                                           low=round(l[base].min(), 4), high=round(h[base].max(), 4), index=i,
                                           time=ts[i].isoformat(), note="Base before impulsive departure (heuristic)"))
    res.supply_demand = res.supply_demand[-6:]

    # Wyckoff spring / upthrust candidates inside a trading range
    win = 40
    for i in range(max(win, n - recent), n):
        rng_hi, rng_lo = h[i - win:i].max(), l[i - win:i].min()
        if (rng_hi - rng_lo) > 8 * atr[i]:
            continue  # not a trading range
        if l[i] < rng_lo - 0.2 * atr[i] and c[i] > rng_lo:
            res.wyckoff.append(Zone2(type="spring", direction="bullish", low=round(l[i], 4), high=round(rng_lo, 4),
                                     index=i, time=ts[i].isoformat(),
                                     note="Wyckoff spring candidate: undercut of range low closed back inside (heuristic)"))
        if h[i] > rng_hi + 0.2 * atr[i] and c[i] < rng_hi:
            res.wyckoff.append(Zone2(type="upthrust", direction="bearish", low=round(rng_hi, 4), high=round(h[i], 4),
                                     index=i, time=ts[i].isoformat(),
                                     note="Wyckoff upthrust candidate: poke above range high closed back inside (heuristic)"))
    res.wyckoff = res.wyckoff[-6:]
    if res.wyckoff:
        res.notes.append("Wyckoff events are candidates only; phase interpretation (accumulation vs distribution) is subjective.")
    return res
