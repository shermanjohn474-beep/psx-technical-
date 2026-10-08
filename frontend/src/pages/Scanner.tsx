import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { get, post, fmt, stageLabel } from "../api";
import type { ScanRow } from "../types";
import { DirBadge, StageBadge } from "../components/ReportPanels";

type SortKey = keyof ScanRow;
const COLS: [SortKey, string][] = [["symbol", "Symbol"], ["price", "Price"], ["trend", "Trend"], ["pattern", "Pattern"],
  ["pattern_stage", "Pattern stage"], ["breakout_level", "Breakout level"], ["volume_confirmation", "Volume conf."],
  ["rsi", "RSI"], ["macd", "MACD"], ["direction", "Direction"], ["signal_score", "Signal score"], ["risk_reward", "R:R"]];

export default function Scanner() {
  const [rows, setRows] = useState<ScanRow[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [meta, setMeta] = useState<{ flags: string[]; patterns: string[] }>({ flags: [], patterns: [] });
  const [lists, setLists] = useState<{ name: string }[]>([]);
  const [presets, setPresets] = useState<{ name: string; rules: any }[]>([]);
  const [watchlist, setWatchlist] = useState("");
  const [symbols, setSymbols] = useState("");
  const [tf, setTf] = useState("1d");
  const [rules, setRules] = useState<any>({ directions: [], stages: [], pattern_ids: [], flags_any: [], min_score: null, min_rr: null, include_experimental: false });
  const [sort, setSort] = useState<{ k: SortKey; asc: boolean }>({ k: "signal_score", asc: false });
  const [busy, setBusy] = useState(false);
  const [presetName, setPresetName] = useState("");

  useEffect(() => {
    get<any>("/api/scan/flags").then(setMeta);
    get<any[]>("/api/watchlists").then(setLists);
    get<any[]>("/api/scan/presets").then(setPresets);
  }, []);

  const run = async () => {
    setBusy(true);
    try {
      const r: any = await post("/api/scan", {
        timeframe: tf, watchlist: watchlist || null,
        symbols: symbols.split(",").map((s) => s.trim()).filter(Boolean), rules,
      });
      setRows(r.rows); setErrors(r.errors);
    } finally { setBusy(false); }
  };

  const sorted = useMemo(() => [...rows].sort((a, b) => {
    const x = a[sort.k] as any, y = b[sort.k] as any;
    if (x === y) return 0;
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    return (x > y ? 1 : -1) * (sort.asc ? 1 : -1);
  }), [rows, sort]);

  const multi = (key: string, opts: string[]) => (
    <label className="field">{key.replace("_", " ")}
      <select multiple value={rules[key]} style={{ minWidth: 160, height: 90 }}
        onChange={(e) => setRules({ ...rules, [key]: Array.from(e.target.selectedOptions).map((o) => o.value) })}>
        {opts.map((o) => <option key={o} value={o}>{o.replace(/_/g, " ")}</option>)}
      </select></label>
  );

  return (
    <div>
      <h1>PSX pattern scanner</h1>
      <div className="card soft no-print">
        <div className="row">
          <label className="field">Watchlist<select value={watchlist} onChange={(e) => setWatchlist(e.target.value)}>
            <option value="">(all symbols with data)</option>{lists.map((l) => <option key={l.name}>{l.name}</option>)}</select></label>
          <label className="field">Extra symbols<input value={symbols} onChange={(e) => setSymbols(e.target.value)} placeholder="OGDC, LUCK" /></label>
          <label className="field">Timeframe<select value={tf} onChange={(e) => setTf(e.target.value)}>{["15m", "1h", "1d", "1w"].map((t) => <option key={t}>{t}</option>)}</select></label>
          <label className="field">Min score<input type="number" value={rules.min_score ?? ""} style={{ width: 80 }}
            onChange={(e) => setRules({ ...rules, min_score: e.target.value === "" ? null : Number(e.target.value) })} /></label>
          <label className="field">Min R:R<input type="number" step="0.1" value={rules.min_rr ?? ""} style={{ width: 80 }}
            onChange={(e) => setRules({ ...rules, min_rr: e.target.value === "" ? null : Number(e.target.value) })} /></label>
          <span className="toggle"><input type="checkbox" checked={rules.include_experimental}
            onChange={() => setRules({ ...rules, include_experimental: !rules.include_experimental })} /> experimental models</span>
        </div>
        <div className="row" style={{ marginTop: 8, alignItems: "flex-start" }}>
          {multi("directions", ["bullish", "bearish", "neutral"])}
          {multi("stages", ["forming", "approaching_confirmation", "confirmed", "retesting", "failed"])}
          {multi("flags_any", meta.flags)}
          {multi("pattern_ids", meta.patterns)}
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={run} disabled={busy}>{busy ? "Scanning…" : "Run scan"}</button>
          <select onChange={(e) => { const p = presets.find((x) => x.name === e.target.value); if (p) setRules({ ...rules, ...p.rules }); }} defaultValue="">
            <option value="">Load preset…</option>{presets.map((p) => <option key={p.name}>{p.name}</option>)}</select>
          <input placeholder="Preset name" value={presetName} onChange={(e) => setPresetName(e.target.value)} style={{ width: 120 }} />
          <button className="secondary" disabled={!presetName}
            onClick={() => post("/api/scan/presets", { name: presetName, rules }).then(() => get<any[]>("/api/scan/presets").then(setPresets))}>Save rules</button>
        </div>
      </div>
      {Object.keys(errors).length > 0 && <div className="warn-box">Skipped: {Object.entries(errors).map(([s, e]) => `${s} (${e.slice(0, 60)}…)`).join("; ")}</div>}
      <div className="card table-wrap">
        <table>
          <thead><tr>{COLS.map(([k, l]) => (
            <th key={k} className="sortable" onClick={() => setSort({ k, asc: sort.k === k ? !sort.asc : false })}>
              {l}{sort.k === k ? (sort.asc ? " ▲" : " ▼") : ""}</th>))}<th>Flags</th></tr></thead>
          <tbody>{sorted.map((r) => (
            <tr key={r.symbol}>
              <td><Link to={`/chart/${r.symbol}?tf=${tf}`}>{r.symbol}</Link> {r.synthetic && <span className="badge warn">syn</span>}</td>
              <td>{fmt(r.price)}</td><td>{r.trend === "n/a" ? "—" : <DirBadge d={r.trend} />}</td><td>{r.pattern ?? "—"}</td>
              <td>{r.pattern_stage ? <StageBadge s={r.pattern_stage} /> : "—"}</td><td>{fmt(r.breakout_level)}</td>
              <td>{r.volume_confirmation === null ? "n/a" : r.volume_confirmation ? "yes" : "no"}</td><td>{fmt(r.rsi, 1)}</td>
              <td className="muted">{stageLabel(r.macd)}</td><td><DirBadge d={r.direction} /></td>
              <td><b>{fmt(r.signal_score, 0)}</b> <span className="muted">{r.rating}</span></td><td>{r.risk_reward ?? "—"}</td>
              <td>{r.flags.map((f) => <span key={f} className="badge" style={{ marginRight: 3 }}>{f.replace(/_/g, " ")}</span>)}</td>
            </tr>))}</tbody>
        </table>
        {!rows.length && <p className="muted">Run a scan to see results.</p>}
      </div>
      <p className="muted">Signal score is a heuristic confluence score (50 = neutral), not a probability of success.</p>
    </div>
  );
}
