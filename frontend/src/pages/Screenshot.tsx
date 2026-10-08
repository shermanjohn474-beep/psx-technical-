import { useState } from "react";
import { Link } from "react-router-dom";
import { postForm, fmt } from "../api";
import { DirBadge, PatternTable, StageBadge } from "../components/ReportPanels";

interface Result {
  analysis_id: string; filename: string; tier: string; preliminary: boolean; needs_confirmation: string[];
  symbol: string | null; timeframe: string | null; confidence: number; units: string; price_uncertainty: number | null;
  extraction: { ok: boolean; candles: number; extraction_confidence: number; issues: string[]; panels: { role: string }[] };
  vision: any; vision_status: string; pattern_claims: { name: string; direction: string; stage: string; description: string; geometry_supported: boolean; matched_detection: string | null }[];
  patterns: any[]; levels: { low: number; high: number; kind: string; touches: number }[];
  data_match: { matched: boolean; price_correlation: number | null; return_correlation: number | null; end_timestamp: string | null; note: string | null } | null;
  report: any; annotated_png_b64: string | null; notes: string[]; dates_are_placeholders: boolean;
}

const TIER: Record<string, [string, string]> = {
  visual_only: ["Visual-only", "Geometry in pixel space. No prices, dates or indicator values inferred."],
  estimated: ["Estimated extraction", "Prices estimated from the axis calibration, with uncertainty."],
  data_verified: ["Data-verified", "Screenshot matched to OHLCV data; full numerical engine applied."],
};

export default function Screenshot() {
  const [files, setFiles] = useState<File[]>([]);
  const [over, setOver] = useState(false);
  const [symbol, setSymbol] = useState("");
  const [tf, setTf] = useState("");
  const [cal, setCal] = useState({ y1: "", price1: "", y2: "", price2: "", log: false });
  const [res, setRes] = useState<{ results: Result[]; notes: string[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const add = (fl: FileList | null) => {
    if (!fl) return;
    const ok = Array.from(fl).filter((f) => /\.(png|jpe?g|webp)$/i.test(f.name));
    setFiles((cur) => [...cur, ...ok].slice(0, 6));
  };

  const analyze = async () => {
    setBusy(true); setErr(null);
    const f = new FormData();
    files.forEach((x) => f.append("files", x));
    if (symbol) f.append("symbol", symbol);
    if (tf) f.append("timeframe", tf);
    if (cal.y1 && cal.price1 && cal.y2 && cal.price2)
      f.append("calibration", JSON.stringify({ y1: +cal.y1, price1: +cal.price1, y2: +cal.y2, price2: +cal.price2, log_scale: cal.log }));
    if (cal.log) f.append("log_scale", "true");
    try { setRes(await postForm("/api/screenshot", f)); } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  return (
    <div>
      <h1>AI chart screenshot analysis</h1>
      <p className="muted">TradingView, SCS Trade, PSX charts or broker terminals. PNG / JPG / WEBP, up to 6 screenshots (e.g. several timeframes of one stock).</p>
      <div className={`dropzone ${over ? "over" : ""}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); add(e.dataTransfer.files); }}
        onClick={() => document.getElementById("shot-input")?.click()}>
        Drag & drop chart screenshots here, or click to choose
        <input id="shot-input" type="file" accept=".png,.jpg,.jpeg,.webp" multiple hidden onChange={(e) => add(e.target.files)} />
      </div>
      {files.length > 0 && (
        <div className="grid three" style={{ marginTop: 10 }}>
          {files.map((f, i) => (
            <div key={i} className="card"><img src={URL.createObjectURL(f)} alt={f.name} style={{ width: "100%", borderRadius: 6 }} />
              <div className="row"><span className="muted">{f.name}</span><button className="secondary" onClick={() => setFiles(files.filter((_, k) => k !== i))}>Remove</button></div></div>))}
        </div>
      )}
      <div className="card soft">
        <h2>Optional: confirm symbol / timeframe / price axis</h2>
        <div className="row">
          <label className="field">Symbol<input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="e.g. LUCK" /></label>
          <label className="field">Timeframe<select value={tf} onChange={(e) => setTf(e.target.value)}><option value="">unknown</option>
            {["5m", "15m", "1h", "1d", "1w", "1M"].map((t) => <option key={t}>{t}</option>)}</select></label>
          <label className="field">y-pixel A<input value={cal.y1} onChange={(e) => setCal({ ...cal, y1: e.target.value })} style={{ width: 80 }} /></label>
          <label className="field">price A<input value={cal.price1} onChange={(e) => setCal({ ...cal, price1: e.target.value })} style={{ width: 90 }} /></label>
          <label className="field">y-pixel B<input value={cal.y2} onChange={(e) => setCal({ ...cal, y2: e.target.value })} style={{ width: 80 }} /></label>
          <label className="field">price B<input value={cal.price2} onChange={(e) => setCal({ ...cal, price2: e.target.value })} style={{ width: 90 }} /></label>
          <span className="toggle"><input type="checkbox" checked={cal.log} onChange={() => setCal({ ...cal, log: !cal.log })} /> log scale</span>
          <button onClick={analyze} disabled={!files.length || busy}>{busy ? "Analysing…" : "Analyse"}</button>
        </div>
        <p className="muted">Supplying the symbol and timeframe (with matching OHLCV loaded) enables data-verified analysis. Two price-axis readings enable estimated prices.</p>
      </div>
      {err && <div className="err-box">{err}</div>}
      {res?.notes.map((n) => <div key={n} className="warn-box">{n}</div>)}
      {res?.results.map((r) => (
        <div key={r.analysis_id} className="card">
          <div className="topbar">
            <h2>{r.filename} {r.symbol && <>· {r.symbol}</>} {r.timeframe && <>· {r.timeframe}</>}</h2>
            <div className="row">
              <span className={`badge ${r.tier === "data_verified" ? "bullish" : r.tier === "estimated" ? "brand" : "warn"}`}>{TIER[r.tier][0]}</span>
              {r.preliminary && <span className="badge warn">PRELIMINARY</span>}
              <span className="badge">confidence {(r.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
          <p className="muted">{TIER[r.tier][1]} Units: {r.units}{r.price_uncertainty ? ` (±${fmt(r.price_uncertainty)})` : ""}.</p>
          {r.needs_confirmation.length > 0 && <div className="warn-box">Please confirm: {r.needs_confirmation.join(", ")}.</div>}
          <div className="grid two">
            <div>{r.annotated_png_b64 && <img alt="annotated" src={`data:image/png;base64,${r.annotated_png_b64}`} style={{ width: "100%", border: "1px solid var(--line)", borderRadius: 6 }} />}</div>
            <div>
              <h3>Source quality</h3>
              <ul className="tight">
                <li>Candles extracted: {r.extraction.candles} (extraction confidence {(r.extraction.extraction_confidence * 100).toFixed(0)}%)</li>
                <li>Panels: {r.extraction.panels.map((p) => p.role).join(", ") || "—"}</li>
                <li>Vision model: {r.vision_status}</li>
                {r.data_match && <li>Data match: {r.data_match.matched ? `yes (price ρ ${r.data_match.price_correlation}, returns ρ ${r.data_match.return_correlation}, ending ${r.data_match.end_timestamp?.slice(0, 10)})` : r.data_match.note}</li>}
                {[...r.extraction.issues, ...r.notes].map((n) => <li key={n} className="muted">{n}</li>)}
              </ul>
              {r.pattern_claims.length > 0 && <>
                <h3>Vision-model pattern claims</h3>
                <table><tbody>{r.pattern_claims.map((c) => (
                  <tr key={c.name}><td>{c.name}</td><td><DirBadge d={c.direction} /></td><td><StageBadge s={c.stage} /></td>
                    <td>{c.geometry_supported ? <span className="badge bullish">geometry-supported</span> : <span className="badge warn">unverified</span>}</td></tr>))}</tbody></table></>}
              {r.vision && (r.vision.bullish_scenario || r.vision.bearish_scenario) && <>
                <h3>Visual scenarios (model, unverified)</h3>
                <p className="muted"><b>Bullish:</b> {r.vision.bullish_scenario}<br /><b>Bearish:</b> {r.vision.bearish_scenario}</p></>}
            </div>
          </div>
          <PatternTable patterns={r.patterns} title={`Detected patterns (${r.units})`} />
          {r.tier !== "data_verified" && r.levels.length > 0 && (
            <div className="card soft"><h3>Support / resistance zones ({r.units})</h3>
              {r.levels.slice(0, 8).map((z, k) => <span key={k} className={`badge ${z.kind === "support" ? "bullish" : "bearish"}`} style={{ marginRight: 4 }}>
                {z.kind} {fmt(z.low)}–{fmt(z.high)} ({z.touches}x)</span>)}</div>)}
          {r.report && <p><Link to={`/chart/${r.report.overview.symbol}?tf=${r.report.overview.timeframe}`}>Open full data-verified analysis →</Link></p>}
          {r.dates_are_placeholders && <p className="muted">Bar dates in this result are placeholders (screenshots do not carry reliable dates).</p>}
        </div>
      ))}
    </div>
  );
}
