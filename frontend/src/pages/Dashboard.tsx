import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { get, post, postForm, fmt, stageLabel } from "../api";
import type { ScanRow, SymbolRow } from "../types";
import { DirBadge } from "../components/ReportPanels";

export default function Dashboard() {
  const [symbols, setSymbols] = useState<SymbolRow[]>([]);
  const [q, setQ] = useState("");
  const [scan, setScan] = useState<ScanRow[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [watch, setWatch] = useState<{ name: string; symbols: string[] }[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const nav = useNavigate();

  const load = () => {
    get<SymbolRow[]>("/api/symbols").then(setSymbols).catch((e) => setErr(e.message));
    get("/health").then(setHealth).catch(() => setHealth(null));
    get<any[]>("/api/watchlists").then(setWatch).catch(() => undefined);
    post<{ rows: ScanRow[] }>("/api/scan", { with_mtf: false }).then((r) => setScan(r.rows)).catch(() => undefined);
  };
  useEffect(load, []);

  const withData = symbols.filter((s) => s.has_data);
  const emerging = scan.filter((r) => r.flags.includes("emerging_pattern"));
  const confirmed = scan.filter((r) => r.flags.includes("confirmed_breakout") || r.flags.includes("confirmed_breakdown"));

  return (
    <div>
      <div className="topbar">
        <div><h1>Technical research dashboard</h1><p className="muted">Pakistan Stock Exchange · Asia/Karachi</p></div>
        <form className="row" onSubmit={(e) => { e.preventDefault(); if (q) nav(`/chart/${q.toUpperCase()}`); }}>
          <input placeholder="Symbol (e.g. OGDC, LUCK, SYN-ASCT)" value={q} onChange={(e) => setQ(e.target.value)} list="sym-list" />
          <datalist id="sym-list">{symbols.map((s) => <option key={s.symbol} value={s.symbol}>{s.name ?? ""}</option>)}</datalist>
          <button>Analyse</button>
        </form>
      </div>
      {err && <div className="err-box">{err}</div>}
      <div className="card soft">
        <h2>Market data status</h2>
        <div className="row">
          <span className="badge">{withData.length} symbols with data</span>
          <span className={`badge ${health?.live_data_provider_configured ? "bullish" : "warn"}`}>
            {health?.live_data_provider_configured ? "authorized data feed configured" : "no live feed — local/uploaded files only"}</span>
          <span className={`badge ${health?.ai_provider_configured ? "bullish" : "neutral"}`}>AI provider {health?.ai_provider_configured ? "configured" : "not configured"}</span>
        </div>
        <p className="muted">Symbols prefixed SYN- are synthetic demo series, not PSX prices. Real symbols require an uploaded OHLCV file or an authorized provider; prices are never simulated.</p>
      </div>
      <div className="grid two">
        <div className="card"><h2>Top emerging patterns</h2><MiniTable rows={emerging} /></div>
        <div className="card"><h2>Recently confirmed breakouts / breakdowns</h2><MiniTable rows={confirmed} /></div>
      </div>
      <div className="grid two">
        <div className="card"><h2>Symbols with data</h2>
          <table><thead><tr><th>Symbol</th><th>Name</th><th>Timeframes</th></tr></thead>
            <tbody>{withData.map((s) => (
              <tr key={s.symbol}><td><Link to={`/chart/${s.symbol}`}>{s.symbol}</Link> {s.synthetic && <span className="badge warn">synthetic</span>}</td>
                <td className="muted">{s.name ?? "—"}</td><td className="muted">{s.timeframes.join(", ")}</td></tr>))}</tbody></table>
        </div>
        <div><Upload onDone={load} /><Watchlists lists={watch} onChange={load} /></div>
      </div>
    </div>
  );
}

function MiniTable({ rows }: { rows: ScanRow[] }) {
  if (!rows.length) return <p className="muted">None in the current universe.</p>;
  return (
    <table><thead><tr><th>Symbol</th><th>Pattern</th><th>Stage</th><th>Dir</th><th>Score</th></tr></thead>
      <tbody>{rows.slice(0, 8).map((r) => (
        <tr key={r.symbol}><td><Link to={`/chart/${r.symbol}`}>{r.symbol}</Link></td><td>{r.pattern ?? "—"}</td>
          <td className="muted">{stageLabel(r.pattern_stage)}</td><td><DirBadge d={r.direction} /></td><td>{fmt(r.signal_score, 0)}</td></tr>))}</tbody></table>
  );
}

function Upload({ onDone }: { onDone: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [sym, setSym] = useState("");
  const [tf, setTf] = useState("1d");
  const [attr, setAttr] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const submit = async () => {
    if (!file || !sym) return;
    const f = new FormData();
    f.append("file", file); f.append("symbol", sym); f.append("timeframe", tf);
    if (attr) f.append("attribution", attr);
    try {
      const r: any = await postForm("/api/data/upload", f);
      setMsg(`Imported ${r.quality.rows_out} bars for ${r.symbol} (quality ${r.quality.rating}). ` + r.quality.issues.map((i: any) => i.message).join(" "));
      onDone();
    } catch (e: any) { setMsg(`Upload failed: ${e.message}`); }
  };
  return (
    <div className="card"><h2>Import OHLCV (CSV / Excel)</h2>
      <div className="row">
        <input placeholder="Symbol" value={sym} onChange={(e) => setSym(e.target.value)} style={{ width: 110 }} />
        <select value={tf} onChange={(e) => setTf(e.target.value)}>{["5m", "15m", "1h", "1d", "1w", "1M"].map((t) => <option key={t}>{t}</option>)}</select>
        <input type="file" accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </div>
      <div className="row" style={{ marginTop: 6 }}>
        <input placeholder="Source / attribution (e.g. broker export)" value={attr} onChange={(e) => setAttr(e.target.value)} style={{ flex: 1 }} />
        <button onClick={submit} disabled={!file || !sym}>Import</button>
      </div>
      <p className="muted">Columns: date/timestamp, open, high, low, close, volume (optional). Timestamps are interpreted as Asia/Karachi.</p>
      {msg && <div className="warn-box">{msg}</div>}
    </div>
  );
}

function Watchlists({ lists, onChange }: { lists: { name: string; symbols: string[] }[]; onChange: () => void }) {
  const [name, setName] = useState("");
  const [syms, setSyms] = useState("");
  return (
    <div className="card"><h2>Watchlists</h2>
      {lists.map((w) => <div key={w.name} className="row" style={{ marginBottom: 4 }}><b>{w.name}</b>
        {w.symbols.map((s) => <Link key={s} to={`/chart/${s}`} className="badge brand">{s}</Link>)}</div>)}
      <div className="row" style={{ marginTop: 8 }}>
        <input placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} style={{ width: 100 }} />
        <input placeholder="Symbols, comma separated" value={syms} onChange={(e) => setSyms(e.target.value)} style={{ flex: 1 }} />
        <button className="secondary" disabled={!name || !syms}
          onClick={() => post("/api/watchlists", { name, symbols: syms.split(",").map((s) => s.trim()).filter(Boolean) }).then(onChange)}>Save</button>
      </div>
    </div>
  );
}
