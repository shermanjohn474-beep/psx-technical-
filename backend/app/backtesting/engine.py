"""Event-driven backtester with PSX-style execution constraints.

* Signals are generated from ``df.iloc[:t+1]`` only and filled at bar ``t+1``'s open.
* Fills include slippage (bps) and half the bid/ask spread (tick-based).
* Entries are skipped when the next open gaps beyond the stop or target, when the
  bar is locked at the assumed daily price limit, or when volume is zero.
* Size is capped at ``max_participation`` of the entry bar's volume.
* Exits: stop/target checked intrabar (stop assumed first if both are hit; gap
  fills at the open), time exit after ``max_hold`` bars, forced exit at data end.
* Costs: brokerage (max of % and per-share minimum) + sales tax on brokerage +
  combined exchange/clearing/CVT levies, on both sides. **Default rates are
  placeholders — verify with your broker and current PSX/SECP/NCCPL/CDC schedules.**

Results are simulations: they are not live performance.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.backtesting.strategies import Signal, Strategy, StrategySpec, make_strategy


class CostModel(BaseModel):
    commission_pct: float = 0.15          # % of traded value
    commission_min_per_share: float = 0.03  # PKR per share minimum
    sales_tax_on_commission_pct: float = 15.0
    levies_pct: float = 0.02              # combined CVT/SECP/PSX/NCCPL/CDC (placeholder)
    slippage_bps: float = 10.0
    tick_size: float = 0.01
    spread_ticks: float = 2.0
    daily_limit_pct: float = 7.5          # PSX price-limit assumption; verify
    max_participation: float = 0.1        # fraction of bar volume
    note: str = "Default PSX cost assumptions are placeholders; verify with your broker and current fee schedules."

    def fees(self, price: float, shares: int) -> float:
        value = price * shares
        brokerage = max(value * self.commission_pct / 100, shares * self.commission_min_per_share)
        return brokerage * (1 + self.sales_tax_on_commission_pct / 100) + value * self.levies_pct / 100


class Trade(BaseModel):
    symbol: str
    side: str
    signal_time: str
    entry_time: str
    entry: float
    exit_time: str
    exit: float
    shares: int
    stop: float
    target: float | None
    exit_reason: str
    bars_held: int
    pnl: float
    return_pct: float
    r_multiple: float
    fees: float
    reason: str


class Metrics(BaseModel):
    trades: int
    win_rate: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    profit_factor: float | None
    max_drawdown_pct: float | None
    sharpe: float | None
    sharpe_note: str | None = None
    avg_r: float | None
    expectancy_r: float | None
    expectancy_pct: float | None
    avg_reward_risk_planned: float | None
    signals_per_100_bars: float | None
    avg_holding_bars: float | None
    total_return_pct: float | None
    skipped_entries: dict = Field(default_factory=dict)


class BacktestResult(BaseModel):
    symbol: str
    timeframe: str
    strategy: StrategySpec
    costs: CostModel
    start: str
    end: str
    bars: int
    trades: list[Trade]
    metrics: Metrics
    equity_curve: list[tuple[str, float]] = Field(default_factory=list)
    disclaimer: str = "Simulated results on historical data; not indicative of live performance."


@dataclass
class _Pos:
    side: str
    entry_i: int
    entry: float
    shares: int
    stop: float
    target: float | None
    max_hold: int | None
    signal_i: int
    reason: str
    fees: float
    planned_rr: float | None


def run_backtest(df: pd.DataFrame, spec: StrategySpec, *, symbol: str = "", timeframe: str = "1d",
                 costs: CostModel | None = None, capital: float = 1_000_000.0, risk_per_trade: float = 0.01,
                 strategy: Strategy | None = None, eval_range: tuple[int, int] | None = None) -> BacktestResult:
    costs = costs or CostModel()
    strat = strategy or make_strategy(spec)
    strat.prepare(df)
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    v = df["volume"].to_numpy(dtype=float)
    n = len(df)
    lo_i, hi_i = eval_range or (0, n)
    equity = capital
    pos: _Pos | None = None
    trades: list[Trade] = []
    skipped: dict[str, int] = {}
    eq_curve = []
    pending: Signal | None = None
    pending_i = None
    planned = []
    half_spread = costs.spread_ticks * costs.tick_size / 2

    def slip(px: float, buy: bool) -> float:
        adj = px * costs.slippage_bps / 1e4 + half_spread
        return px + adj if buy else px - adj

    def close_pos(i: int, px: float, reason: str):
        nonlocal equity, pos
        buy_to_close = pos.side == "short"
        fill = slip(px, buy_to_close)
        fees = costs.fees(fill, pos.shares)
        sgn = 1 if pos.side == "long" else -1
        pnl = sgn * (fill - pos.entry) * pos.shares - fees - pos.fees
        risk = abs(pos.entry - pos.stop) * pos.shares
        trades.append(Trade(
            symbol=symbol, side=pos.side, signal_time=df.index[pos.signal_i].isoformat(),
            entry_time=df.index[pos.entry_i].isoformat(), entry=round(pos.entry, 4), exit_time=df.index[i].isoformat(),
            exit=round(fill, 4), shares=pos.shares, stop=round(pos.stop, 4),
            target=None if pos.target is None else round(pos.target, 4), exit_reason=reason, bars_held=i - pos.entry_i,
            pnl=round(pnl, 2), return_pct=round(100 * pnl / (pos.entry * pos.shares), 3),
            r_multiple=round(pnl / risk, 3) if risk > 0 else 0.0, fees=round(fees + pos.fees, 2), reason=pos.reason))
        equity += pnl
        pos = None

    for t in range(n):
        # 1) execute pending entry at this bar's open
        if pending is not None and pos is None and t > pending_i:
            sig = pending
            pending = None
            buy = sig.side == "long"
            sgn = 1 if buy else -1
            prev_c = c[t - 1]
            locked = h[t] == l[t] and abs(o[t] / prev_c - 1) * 100 >= 0.98 * costs.daily_limit_pct
            if np.isfinite(v[t]) and v[t] == 0:
                skipped["zero_volume"] = skipped.get("zero_volume", 0) + 1
            elif locked and ((buy and o[t] > prev_c) or (not buy and o[t] < prev_c)):
                skipped["price_limit_locked"] = skipped.get("price_limit_locked", 0) + 1
            elif sgn * (o[t] - sig.stop) <= 0:
                skipped["gap_through_stop"] = skipped.get("gap_through_stop", 0) + 1
            elif sig.target is not None and sgn * (o[t] - sig.target) >= 0:
                skipped["gap_through_target"] = skipped.get("gap_through_target", 0) + 1
            else:
                fill = slip(o[t], buy)
                per_share_risk = abs(fill - sig.stop)
                shares = int(equity * risk_per_trade / per_share_risk) if per_share_risk > 0 else 0
                shares = min(shares, int(equity / fill))
                if np.isfinite(v[t]):
                    shares = min(shares, int(costs.max_participation * v[t]))
                if shares <= 0:
                    skipped["size_zero_liquidity"] = skipped.get("size_zero_liquidity", 0) + 1
                else:
                    fees = costs.fees(fill, shares)
                    rr = abs(sig.target - fill) / per_share_risk if sig.target is not None else None
                    planned.append(rr)
                    pos = _Pos(sig.side, t, fill, shares, sig.stop, sig.target, sig.max_hold, pending_i, sig.reason, fees, rr)
        # 2) manage open position on this bar (entry bar included: stop can trigger same day)
        if pos is not None:
            sgn = 1 if pos.side == "long" else -1
            hit_stop = (l[t] <= pos.stop) if sgn > 0 else (h[t] >= pos.stop)
            hit_tgt = pos.target is not None and ((h[t] >= pos.target) if sgn > 0 else (l[t] <= pos.target))
            if hit_stop:
                gap = (o[t] < pos.stop) if sgn > 0 else (o[t] > pos.stop)
                close_pos(t, o[t] if gap and t > pos.entry_i else pos.stop, "stop")
            elif hit_tgt:
                gap = (o[t] > pos.target) if sgn > 0 else (o[t] < pos.target)
                close_pos(t, o[t] if gap and t > pos.entry_i else pos.target, "target")
            elif pos.max_hold is not None and t - pos.entry_i >= pos.max_hold:
                close_pos(t, c[t], "time")
        # 3) generate signals from data up to and including bar t (never beyond)
        if pos is None and pending is None and lo_i <= t < hi_i - 1 and t < n - 1:
            sigs = strat.on_bar(df.iloc[: t + 1], t)
            if sigs:
                pending, pending_i = sigs[0], t
        mtm = equity
        if pos is not None:
            sgn = 1 if pos.side == "long" else -1
            mtm += sgn * (c[t] - pos.entry) * pos.shares
        eq_curve.append((df.index[t].isoformat(), round(mtm, 2)))
    if pos is not None:
        close_pos(n - 1, c[n - 1], "end_of_data")
        eq_curve[-1] = (eq_curve[-1][0], round(equity, 2))
    if eval_range:
        eq_curve = eq_curve[lo_i:hi_i]
    m = compute_metrics(trades, eq_curve, capital, max(1, (hi_i - lo_i)), planned, skipped)
    return BacktestResult(symbol=symbol, timeframe=timeframe, strategy=spec, costs=costs,
                          start=df.index[lo_i].isoformat(), end=df.index[hi_i - 1].isoformat(), bars=hi_i - lo_i,
                          trades=trades, metrics=m, equity_curve=eq_curve[:: max(1, len(eq_curve) // 400)])


def compute_metrics(trades: list[Trade], eq_curve, capital: float, bars: int, planned: list, skipped: dict,
                    periods_per_year: int = 245) -> Metrics:
    n = len(trades)
    if n == 0:
        return Metrics(trades=0, win_rate=None, avg_win_pct=None, avg_loss_pct=None, profit_factor=None,
                       max_drawdown_pct=None, sharpe=None, sharpe_note="No trades.", avg_r=None, expectancy_r=None,
                       expectancy_pct=None, avg_reward_risk_planned=None, signals_per_100_bars=0.0,
                       avg_holding_bars=None, total_return_pct=0.0, skipped_entries=skipped)
    rets = np.array([t.return_pct for t in trades])
    rs = np.array([t.r_multiple for t in trades])
    pnl = np.array([t.pnl for t in trades])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    gross_win, gross_loss = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    eq = np.array([e for _, e in eq_curve]) if eq_curve else np.array([capital])
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min() * 100) if len(eq) else None
    sharpe, note = None, None
    daily = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
    if n >= 20 and len(daily) >= 120 and daily.std() > 0:
        sharpe = round(float(daily.mean() / daily.std() * math.sqrt(periods_per_year)), 2)
    else:
        note = "Sharpe not reported: fewer than 20 trades or 120 bars (not statistically meaningful)."
    pl = [p for p in planned if p is not None]
    return Metrics(
        trades=n, win_rate=round(float(len(wins) / n), 3), avg_win_pct=round(float(wins.mean()), 3) if len(wins) else None,
        avg_loss_pct=round(float(losses.mean()), 3) if len(losses) else None,
        profit_factor=round(float(gross_win / gross_loss), 2) if gross_loss > 0 else None,
        max_drawdown_pct=None if mdd is None else round(mdd, 2), sharpe=sharpe, sharpe_note=note,
        avg_r=round(float(rs.mean()), 3), expectancy_r=round(float(rs.mean()), 3),
        expectancy_pct=round(float(rets.mean()), 3), avg_reward_risk_planned=round(float(np.mean(pl)), 2) if pl else None,
        signals_per_100_bars=round(100 * n / bars, 2), avg_holding_bars=round(float(np.mean([t.bars_held for t in trades])), 1),
        total_return_pct=round(100 * (eq[-1] / capital - 1), 2) if len(eq) else None, skipped_entries=skipped,
    )


class WalkForwardFold(BaseModel):
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    chosen_params: dict
    train_expectancy_r: float | None
    test_metrics: Metrics


class WalkForwardResult(BaseModel):
    symbol: str
    folds: list[WalkForwardFold]
    oos_metrics: Metrics
    in_sample_split: dict
    notes: list[str] = Field(default_factory=list)


def walk_forward(df: pd.DataFrame, spec: StrategySpec, *, symbol: str = "", costs: CostModel | None = None,
                 folds: int = 4, train_frac: float = 0.6, grid: dict[str, list] | None = None,
                 min_train_trades: int = 3) -> WalkForwardResult:
    """Anchored walk-forward: optimise a small grid on each training window by expectancy,
    apply the chosen parameters to the following unseen test window."""
    grid = grid or {"stop_atr": [1.5, 2.0, 3.0], "max_hold": [20, 40], "target_r": [2.0, 3.0]}
    n = len(df)
    first_test = int(n * train_frac)
    step = max(1, (n - first_test) // folds)
    out_folds, oos_trades, notes = [], [], []
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    for k in range(folds):
        ts, te = first_test + k * step, min(n, first_test + (k + 1) * step)
        if te - ts < 10:
            break
        best, best_exp = None, -np.inf
        for cmb in combos:
            sp = spec.model_copy(update=cmb)
            r = run_backtest(df.iloc[:ts], sp, symbol=symbol, costs=costs)
            if r.metrics.trades >= min_train_trades and (r.metrics.expectancy_r or -np.inf) > best_exp:
                best, best_exp = cmb, r.metrics.expectancy_r
        if best is None:
            best = {kk: getattr(spec, kk) for kk in keys}
            notes.append(f"Fold {k + 1}: too few training trades; default parameters used.")
        sp = spec.model_copy(update=best)
        # test: history is visible for indicator warm-up but signals only inside [ts, te)
        r = run_backtest(df.iloc[:te], sp, symbol=symbol, costs=costs, eval_range=(ts, te))
        oos_trades += r.trades
        out_folds.append(WalkForwardFold(fold=k + 1, train_start=df.index[0].isoformat(), train_end=df.index[ts - 1].isoformat(),
                                         test_start=df.index[ts].isoformat(), test_end=df.index[te - 1].isoformat(),
                                         chosen_params=best,
                                         train_expectancy_r=None if best_exp == -np.inf else round(best_exp, 3),
                                         test_metrics=r.metrics))
    oos = compute_metrics(oos_trades, [], 1_000_000.0, max(1, n - first_test), [], {})
    split = int(n * 0.7)
    ins = run_backtest(df.iloc[:split], spec, symbol=symbol, costs=costs).metrics
    outs = run_backtest(df, spec, symbol=symbol, costs=costs, eval_range=(split, n)).metrics
    notes.append("Out-of-sample metrics aggregate only trades generated inside unseen test windows.")
    if oos.trades < 20:
        notes.append(f"Only {oos.trades} out-of-sample trades: results are not statistically robust.")
    return WalkForwardResult(symbol=symbol, folds=out_folds, oos_metrics=oos,
                             in_sample_split={"split_time": df.index[split].isoformat(), "in_sample": ins.model_dump(),
                                              "out_of_sample": outs.model_dump()}, notes=notes)
