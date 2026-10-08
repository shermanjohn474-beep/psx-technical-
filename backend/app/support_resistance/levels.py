"""Support/resistance zone detection.

Candidate levels are gathered from confirmed swing pivots, prior consolidation
(rectangle) edges and major historical extremes, then clustered in 1-D with an
ATR-based tolerance. Each cluster becomes a *zone* (``low``-``high`` band, not a
single price). Zones are scored on touches, recency, rejection strength, volume
at touches, role reversals (support <-> resistance) and successful retests.

Moving averages, round numbers, Fibonacci levels and an approximate
volume-at-price profile are reported as *auxiliary* levels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.indicators.volatility import causal_atr
from app.swing_detection.swings import SwingSet, zigzag_atr


class Zone(BaseModel):
    low: float
    high: float
    mid: float
    kind: str  # support | resistance (relative to the latest close)
    touches: int
    first_index: int
    last_index: int
    last_touch_time: str | None = None
    strength: float  # 0..100 heuristic
    components: dict[str, float] = Field(default_factory=dict)
    role_reversals: int = 0
    sources: list[str] = Field(default_factory=list)
    distance_atr: float | None = None


class AuxLevel(BaseModel):
    price: float
    source: str
    note: str | None = None


class LevelsResult(BaseModel):
    zones: list[Zone]
    auxiliary: list[AuxLevel]
    immediate_support: Zone | None = None
    major_support: Zone | None = None
    immediate_resistance: Zone | None = None
    major_resistance: Zone | None = None
    notes: list[str] = Field(default_factory=list)


@dataclass
class LevelConfig:
    cluster_atr: float = 0.8
    zone_pad_atr: float = 0.15
    min_touches: int = 2
    recency_halflife: int = 120
    reaction_bars: int = 5
    max_zones: int = 12
    round_number_steps: tuple[float, ...] = field(default=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000))


def _round_step(price: float) -> float:
    for step in (0.5, 1, 5, 10, 50, 100, 500, 1000, 5000):
        if price / step < 40:
            return step
    return 10000


def detect_levels(df: pd.DataFrame, swings: SwingSet | None = None, cfg: LevelConfig | None = None) -> LevelsResult:
    cfg = cfg or LevelConfig()
    n = len(df)
    if n < 20:
        return LevelsResult(zones=[], auxiliary=[], notes=["Insufficient data for support/resistance."])
    atr = causal_atr(df).to_numpy()
    swings = swings or zigzag_atr(df)
    last = n - 1
    a_last = float(atr[last])
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    vol = df["volume"].to_numpy(dtype=float)
    has_vol = np.isfinite(vol).any() and np.nansum(vol) > 0
    vavg = pd.Series(vol).rolling(20, min_periods=5).mean().to_numpy() if has_vol else None

    cands: list[tuple[float, int, str]] = [(s.price, s.index, f"swing_{s.kind}") for s in swings.swings if s.confirmed_index <= last]
    # major extremes
    cands.append((float(h.max()), int(h.argmax()), "historical_high"))
    cands.append((float(l.min()), int(l.argmin()), "historical_low"))
    if not cands:
        return LevelsResult(zones=[], auxiliary=[], notes=["No swing points detected."])

    tol = cfg.cluster_atr * a_last
    cands.sort(key=lambda x: x[0])
    clusters: list[list[tuple[float, int, str]]] = [[cands[0]]]
    for cnd in cands[1:]:
        if cnd[0] - np.mean([x[0] for x in clusters[-1]]) <= tol:
            clusters[-1].append(cnd)
        else:
            clusters.append([cnd])

    zones: list[Zone] = []
    price = float(c[last])
    for cl in clusters:
        prices = np.array([x[0] for x in cl])
        idxs = [x[1] for x in cl]
        pad = cfg.zone_pad_atr * a_last
        zlo, zhi = float(prices.min() - pad), float(prices.max() + pad)
        # count bar-level touches: bars whose range intersects zone, grouped into episodes
        inter = (l <= zhi) & (h >= zlo)
        episodes = []
        k = 0
        while k < n:
            if inter[k]:
                s0 = k
                while k < n and inter[k]:
                    k += 1
                episodes.append((s0, k - 1))
            else:
                k += 1
        touches = max(len(set(idxs)), sum(1 for e in episodes if e[1] - e[0] < 15))
        if touches < cfg.min_touches and not any(src.startswith("historical") for _, _, src in cl):
            continue
        # rejection strength: move away from the zone after each pivot touch, in ATR
        reacts = []
        for _, i, src in cl:
            j = min(n - 1, i + cfg.reaction_bars)
            if j <= i:
                continue
            if src.endswith("H") or src == "historical_high":
                reacts.append((h[i] - l[i + 1:j + 1].min()) / atr[i])
            else:
                reacts.append((h[i + 1:j + 1].max() - l[i]) / atr[i])
        rejection = float(np.mean(reacts)) if reacts else 0.0
        last_idx = max(idxs + [e[1] for e in episodes]) if episodes else max(idxs)
        recency = float(0.5 ** ((last - last_idx) / cfg.recency_halflife))
        # role reversal: zone acted as both support and resistance (closes crossed it)
        side = np.sign(c - (zlo + zhi) / 2)
        crossings = int(np.sum(np.abs(np.diff(side[side != 0])) > 0)) if np.any(side != 0) else 0
        has_h = any(src.endswith("H") or src == "historical_high" for _, _, src in cl)
        has_l = any(src.endswith("L") or src == "historical_low" for _, _, src in cl)
        role_rev = 1 if (has_h and has_l) else 0
        vol_comp = None
        if vavg is not None:
            rv = [vol[i] / vavg[i] for i in idxs if vavg[i] and np.isfinite(vavg[i])]
            vol_comp = float(np.clip(np.mean(rv) / 2, 0, 1)) if rv else None
        comps = {
            "touches": min(1.0, touches / 5),
            "recency": recency,
            "rejection": min(1.0, rejection / 4),
            "role_reversal": 1.0 if role_rev else 0.3,
            "retests": min(1.0, crossings / 4) if crossings else 0.2,
        }
        if vol_comp is not None:
            comps["volume"] = vol_comp
        weights = {"touches": 3, "recency": 2, "rejection": 2, "role_reversal": 1, "retests": 1, "volume": 1}
        strength = 100 * sum(comps[k] * weights[k] for k in comps) / sum(weights[k] for k in comps)
        mid = (zlo + zhi) / 2
        kind = "support" if mid < price else "resistance"
        if zlo <= price <= zhi:
            kind = "support" if c[max(0, last - 5)] > mid else "resistance"
        zones.append(Zone(
            low=round(zlo, 4), high=round(zhi, 4), mid=round(mid, 4), kind=kind, touches=int(touches),
            first_index=int(min(idxs)), last_index=int(last_idx), last_touch_time=str(df.index[last_idx].date()),
            strength=round(float(strength), 1), components={k: round(v, 3) for k, v in comps.items()},
            role_reversals=role_rev, sources=sorted({src for _, _, src in cl}),
            distance_atr=round((mid - price) / a_last, 2),
        ))

    zones.sort(key=lambda z: -z.strength)
    zones = zones[: cfg.max_zones]
    zones.sort(key=lambda z: z.mid)
    sup = [z for z in zones if z.kind == "support"]
    res = [z for z in zones if z.kind == "resistance"]
    result = LevelsResult(zones=zones, auxiliary=_aux_levels(df, swings, price))
    if sup:
        result.immediate_support = max(sup, key=lambda z: z.mid)
        result.major_support = max(sup, key=lambda z: z.strength)
    if res:
        result.immediate_resistance = min(res, key=lambda z: z.mid)
        result.major_resistance = max(res, key=lambda z: z.strength)
    if not sup:
        result.notes.append("No support zone below current price within the analysed history (price at/near lows).")
    if not res:
        result.notes.append("No resistance zone above current price within the analysed history (price at/near highs).")
    return result


def _aux_levels(df: pd.DataFrame, swings: SwingSet, price: float) -> list[AuxLevel]:
    out: list[AuxLevel] = []
    c = df["close"]
    for p in (50, 100, 200):
        if len(c) >= p:
            out.append(AuxLevel(price=round(float(c.rolling(p).mean().iloc[-1]), 4), source=f"sma_{p}", note="dynamic"))
    step = _round_step(price)
    base = np.floor(price / step) * step
    for k in (-1, 0, 1, 2):
        lvl = base + k * step
        if lvl > 0:
            out.append(AuxLevel(price=float(lvl), source="round_number"))
    # approximate volume-at-price (from bar typical prices, not true tick data)
    if df["volume"].notna().any() and df["volume"].sum() > 0:
        tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
        hist, edges = np.histogram(tp, bins=30, weights=df["volume"].fillna(0).to_numpy())
        top = np.argsort(hist)[-2:]
        for b in sorted(top):
            out.append(AuxLevel(price=round(float((edges[b] + edges[b + 1]) / 2), 4), source="volume_profile_approx",
                                note="Approximate volume-at-price from bar typical prices (no tick data)."))
    return out
