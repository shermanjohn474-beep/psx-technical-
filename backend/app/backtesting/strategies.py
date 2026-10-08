"""Backtest strategies. Each strategy is called bar-by-bar with ONLY the data up to
and including the current bar, and returns entry signals to be executed at the
next bar's open. Strategies never receive future bars.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.chart_patterns.base import PatternConfig, build_context
from app.chart_patterns.registry import detect_patterns
from app.indicators.core import sma
from app.indicators.momentum import macd, rsi
from app.indicators.volatility import causal_atr
from app.schemas.patterns import Direction


@dataclass
class Signal:
    side: str  # long | short
    stop: float
    target: float | None
    reason: str
    max_hold: int | None = None


class StrategySpec(BaseModel):
    """Serializable strategy definition (API input)."""

    type: str = "pattern"  # pattern | sma_cross | rsi_recovery | macd_cross
    pattern_ids: list[str] = Field(default_factory=lambda: ["double_bottom", "inverse_head_and_shoulders", "ascending_triangle", "bull_flag"])
    allow_short: bool = False
    fast: int = 20
    slow: int = 50
    rsi_level: float = 30.0
    stop_atr: float = 2.0
    stop_buffer_atr: float = 0.2
    target_mode: str = "pattern"  # pattern | r_multiple
    target_r: float = 2.0
    max_hold: int = 40
    require_trend_filter: bool = False  # close above SMA200 for longs (below for shorts)
    require_volume_confirmation: bool = False
    lookback: int = 250
    params: dict = Field(default_factory=dict)  # pattern-config overrides


class Strategy(ABC):
    def __init__(self, spec: StrategySpec):
        self.spec = spec

    def prepare(self, df: pd.DataFrame) -> None:
        """Optional causal precomputation (indicators whose value at t uses only bars <= t)."""

    @abstractmethod
    def on_bar(self, hist: pd.DataFrame, t: int) -> list[Signal]:
        ...

    def _trend_ok(self, hist: pd.DataFrame, side: str) -> bool:
        if not self.spec.require_trend_filter:
            return True
        s200 = hist["close"].rolling(200).mean().iloc[-1]
        if np.isnan(s200):
            return False
        return hist["close"].iloc[-1] > s200 if side == "long" else hist["close"].iloc[-1] < s200


class PatternStrategy(Strategy):
    def on_bar(self, hist: pd.DataFrame, t: int) -> list[Signal]:
        sp = self.spec
        window = hist.iloc[-sp.lookback:]
        if len(window) < 40:
            return []
        cfg = PatternConfig(params=sp.params)
        ctx = build_context(window, cfg)
        res = detect_patterns(ctx=ctx, include=set(sp.pattern_ids))
        out = []
        last = ctx.last
        atr = float(ctx.atr[-1])
        for p in res:
            if p.confirmation_index != last or p.provisional:
                continue  # only act on the bar the pattern is confirmed by a completed close
            side = "long" if p.direction == Direction.BULLISH else "short" if p.direction == Direction.BEARISH else None
            if side is None or (side == "short" and not sp.allow_short):
                continue
            if sp.require_volume_confirmation and p.volume_confirmation is not True:
                continue
            if not self._trend_ok(hist, side):
                continue
            sgn = 1 if side == "long" else -1
            stop = (p.invalidation_level or (p.breakout_level - sgn * sp.stop_atr * atr)) - sgn * sp.stop_buffer_atr * atr
            entry_ref = float(hist["close"].iloc[-1])
            if sp.target_mode == "r_multiple" or p.target is None:
                target = entry_ref + sgn * sp.target_r * abs(entry_ref - stop)
            else:
                target = p.target
            out.append(Signal(side, stop, target, f"{p.name} confirmed", sp.max_hold))
        return out


class SMACrossStrategy(Strategy):
    def prepare(self, df):
        self.f = sma(df["close"], self.spec.fast).to_numpy()
        self.s = sma(df["close"], self.spec.slow).to_numpy()
        self.atr = causal_atr(df).to_numpy()

    def on_bar(self, hist, t):
        if t < 1 or np.isnan(self.s[t]) or np.isnan(self.s[t - 1]):
            return []
        up = self.f[t] > self.s[t] and self.f[t - 1] <= self.s[t - 1]
        dn = self.f[t] < self.s[t] and self.f[t - 1] >= self.s[t - 1]
        return _atr_signal(self, hist, t, up, dn, f"SMA{self.spec.fast}/{self.spec.slow} cross")


class RSIRecoveryStrategy(Strategy):
    def prepare(self, df):
        self.r = rsi(df["close"]).to_numpy()
        self.atr = causal_atr(df).to_numpy()

    def on_bar(self, hist, t):
        lv = self.spec.rsi_level
        if t < 1 or np.isnan(self.r[t - 1]):
            return []
        up = self.r[t - 1] < lv <= self.r[t]
        dn = self.r[t - 1] > 100 - lv >= self.r[t]
        return _atr_signal(self, hist, t, up, dn, f"RSI recovery through {lv:g}")


class MACDCrossStrategy(Strategy):
    def prepare(self, df):
        m = macd(df["close"])
        self.h = m["macd_hist"].to_numpy()
        self.atr = causal_atr(df).to_numpy()

    def on_bar(self, hist, t):
        if t < 1 or np.isnan(self.h[t - 1]):
            return []
        return _atr_signal(self, hist, t, self.h[t] > 0 >= self.h[t - 1], self.h[t] < 0 <= self.h[t - 1], "MACD signal cross")


def _atr_signal(strat: Strategy, hist, t, up: bool, dn: bool, reason: str) -> list[Signal]:
    sp = strat.spec
    close = float(hist["close"].iloc[-1])
    a = float(strat.atr[t])
    out = []
    if up and strat._trend_ok(hist, "long"):
        stop = close - sp.stop_atr * a
        out.append(Signal("long", stop, close + sp.target_r * (close - stop), reason, sp.max_hold))
    if dn and sp.allow_short and strat._trend_ok(hist, "short"):
        stop = close + sp.stop_atr * a
        out.append(Signal("short", stop, close - sp.target_r * (stop - close), reason, sp.max_hold))
    return out


def make_strategy(spec: StrategySpec) -> Strategy:
    return {"pattern": PatternStrategy, "sma_cross": SMACrossStrategy, "rsi_recovery": RSIRecoveryStrategy,
            "macd_cross": MACDCrossStrategy}[spec.type](spec)
