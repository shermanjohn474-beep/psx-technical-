import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { post, fmt } from "../api";

const STRATS = ["pattern", "sma_cross", "rsi_recovery", "macd_cross"];

export default function Backtest() {
  const [sp] = useSearchParams();
  const [symbols, setSymbols] = useState(sp.get("symbol") ?? "SYN-DBOT, SYN-HS, SYN-ASCT");
  const [spec, setSpec] = useState<any>({ type: "pattern", pattern_ids: ["double_bottom", "inverse_head_and_shoulders", "ascending_triangle", "bull_flag"],
    allow_short: false, stop_atr: 2, target_mode: "pattern", target_r: 2, max_hold: 40, require_trend_filter: false, require_volume_confirmation: false, fast: 20, slow: 50 });
  const [costs, setCosts] = useState<any>({ commission_pct: 0.15, commission_min_per_share: 0.03, sales_tax_on_commission_pct: 15, levies_pct: 0.02, slippage_bps: 10, spread_ticks: 2, max_participation: 0.1 });
  const [wf, setWf] = useState(true);
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setBusy(true); setErr(null);
    try {
      setRes(await post("/api/backtest", { symbols: symbols.split(",").map((s) => s.trim()).filter(Boolean), strategy: spec, costs, walk_forward: wf, folds: 3 }));
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  const num = (obj: any, set: (x: any) => void, k: string, label: string, step = "0.01") => (
    <label className="field" key={k}>{label}<input type="number" step={step} value={obj[k]} style={{ width: 90 }} onChange={(e) => set({ ...obj, [k]: Number(e.target.value) })} /></label>);

  return (
    <div>
      <h1>Backtesting & pattern research</h1>
      <div className="card soft">
        <div className="row">
          <label className="field">Symbols<input value={symbols} onChange={(e) => setSymbols(e.target.value)} style={{ width: 280 }} /></label>
          <label className="field">Strategy<select value={spec.type} onChange={(e) => setSpec({ ...spec, type: e.target.value })}>{STRATS.map((s) => <option key={s}>{s}</option>)}</select></label>
          {spec.type === "pattern" && <label className="field">Pattern ids (comma)<input value={spec.pattern_ids.join(",")} style={{ width: 300 }}
            onChange={(e) => setSpec({ ...spec, pattern_ids: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} /></label>}
          {spec.type === "sma_cross" && <>{num(spec, setSpec, "fast", "Fast", "1")}{num(spec, setSpec, "slow", "Slow", "1")}</>}
        </div>
        <div className="row" style={{ marginTop: 6 }}>
          {num(spec, setSpec, "stop_atr", "Stop (ATR)", "0.1")}{num(spec, setSpec, "target_r", "Target (R)", "0.1")}{num(spec, setSpec, "max_hold", "Max hold (bars)", "1")}
          <label className="field">Target mode<select value={spec.target_mode} onChange={(e) => setSpec({ ...spec, target_mode: e.target.value })}><option>pattern</option><option>r_multiple</option></select></label>
          <span className="toggle"><input type="checkbox" checked={spec.require_trend_filter} onChange={() => setSpec({ ...spec, require_trend_filter: !spec.require_trend_filter })} />SMA200 trend filter</span>
          <span className="toggle"><input type="checkbox" checked={spec.require_volume_confirmation} onChange={() => setSpec({ ...spec, require_volume_confirmation: !spec.require_volume_confirmation })} />volume confirmation</span>
          <span className="toggle"><input type="checkbox" checked={spec.allow_short} onChange={() => setSpec({ ...spec, allow_short: !spec.allow_short })} />allow shorts (eligibility not verified)</span>
        </div>
        <h3>PSX execution costs (placeholders — verify with your broker)</h3>
        <div className="row">
          {num(costs, setCosts, "commission_pct", "Brokerage %")}{num(costs, setCosts, "commission_min_per_share", "Min/share PKR")}
          {num(costs, setCosts, "sales_tax_on_commission_pct", "Sales tax on brokerage %", "0.5")}{num(costs, setCosts, "levies_pct", "Levies %", "0.005")}
          {num(costs, setCosts, "slippage_bps", "Slippage bps", "1")}{num(costs, setCosts, "spread_ticks", "Spread ticks", "1")}
          {num(costs, setCosts, "max_participation", "Max % of bar vol")}
          <span className="toggle"><input type="checkbox" checked={wf} onChange={() => setWf(!wf)} />walk-forward</span>
          <button onClick={run} disabled={busy}>{busy ? "Running…" : "Run backtest"}</button>
        </div>
      </div>
      {err && <div className="err-box">{err}</div>}
      {res && <>
        <div className="warn-box">{res.disclaimer}</div>
        {Object.keys(res.errors).length > 0 && <div className="warn-box">Skipped: {Object.entries(res.errors).map(([s, e]: any) => `${s}: ${e.slice(0, 80)}`).join(" | ")}</div>}
        {res.results.map((x: any) => {
          const m = x.result.metrics;
          return (
            <div key={x.result.symbol} className="card">
              <h2>{x.result.symbol} · {x.result.start.slice(0, 10)} → {x.result.end.slice(0, 10)} ({x.result.bars} bars)</h2>
              <div className="kpis">
                {[["Trades", m.trades], ["Win rate", m.win_rate === null ? "—" : `${(m.win_rate * 100).toFixed(0)}%`], ["Profit factor", fmt(m.profit_factor)],
                  ["Expectancy (R)", fmt(m.expectancy_r, 3)], ["Avg win %", fmt(m.avg_win_pct)], ["Avg loss %", fmt(m.avg_loss_pct)],
                  ["Max DD %", fmt(m.max_drawdown_pct)], ["Sharpe", m.sharpe ?? "n/a"], ["Planned R:R", fmt(m.avg_reward_risk_planned)],
                  ["Signals /100 bars", fmt(m.signals_per_100_bars)], ["Avg hold (bars)", fmt(m.avg_holding_bars, 1)], ["Total return %", fmt(m.total_return_pct)]]
                  .map(([l, v]) => <div key={l as string} className="kpi"><div className="l">{l}</div><div className="v" style={{ fontSize: 16 }}>{v}</div></div>)}
              </div>
              {m.sharpe_note && <p className="muted">{m.sharpe_note}</p>}
              {Object.keys(m.skipped_entries).length > 0 && <p className="muted">Skipped entries: {JSON.stringify(m.skipped_entries)}</p>}
              <details><summary className="muted">Trades ({x.result.trades.length})</summary>
                <table><thead><tr><th>Entry</th><th>Exit</th><th>Side</th><th>Entry px</th><th>Exit px</th><th>Exit</th><th>R</th><th>Ret %</th><th>Fees</th><th>Reason</th></tr></thead>
                  <tbody>{x.result.trades.map((t: any, k: number) => <tr key={k}><td>{t.entry_time.slice(0, 10)}</td><td>{t.exit_time.slice(0, 10)}</td><td>{t.side}</td>
                    <td>{fmt(t.entry)}</td><td>{fmt(t.exit)}</td><td>{t.exit_reason}</td><td className={t.r_multiple >= 0 ? "bullish-t" : "bearish-t"}>{fmt(t.r_multiple)}</td>
                    <td>{fmt(t.return_pct)}</td><td>{fmt(t.fees)}</td><td className="muted">{t.reason}</td></tr>)}</tbody></table></details>
              {x.walk_forward && <div className="card soft" style={{ marginTop: 10 }}>
                <h3>Walk-forward (out-of-sample)</h3>
                <p>OOS trades {x.walk_forward.oos_metrics.trades} · expectancy {fmt(x.walk_forward.oos_metrics.expectancy_r, 3)}R · win rate {x.walk_forward.oos_metrics.win_rate === null ? "—" : `${(x.walk_forward.oos_metrics.win_rate * 100).toFixed(0)}%`}</p>
                <table><thead><tr><th>Fold</th><th>Test window</th><th>Chosen params</th><th>Train exp. R</th><th>Test trades</th><th>Test exp. R</th></tr></thead>
                  <tbody>{x.walk_forward.folds.map((f: any) => <tr key={f.fold}><td>{f.fold}</td><td>{f.test_start.slice(0, 10)} → {f.test_end.slice(0, 10)}</td>
                    <td className="muted">{JSON.stringify(f.chosen_params)}</td><td>{fmt(f.train_expectancy_r, 3)}</td><td>{f.test_metrics.trades}</td><td>{fmt(f.test_metrics.expectancy_r, 3)}</td></tr>)}</tbody></table>
                <ul className="tight">{x.walk_forward.notes.map((n: string) => <li key={n} className="muted">{n}</li>)}</ul>
              </div>}
            </div>);
        })}
      </>}
    </div>
  );
}
