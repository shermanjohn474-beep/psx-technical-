import { useEffect, useState } from "react";
import { del, get, post } from "../api";

export default function Alerts() {
  const [conds, setConds] = useState<Record<string, string>>({});
  const [rules, setRules] = useState<any[]>([]);
  const [events, setEvents] = useState<any[]>([]);
  const [status, setStatus] = useState<any>(null);
  const [form, setForm] = useState({ symbol: "", timeframe: "1d", condition: "pattern_confirmed", params: "{}", webhook: "", email: "" });
  const [msg, setMsg] = useState<string | null>(null);

  const load = () => {
    get<any>("/api/alerts/conditions").then(setConds);
    get<any[]>("/api/alerts").then(setRules);
    get<any[]>("/api/alerts/events").then(setEvents);
    get<any>("/api/alerts/status").then(setStatus);
  };
  useEffect(load, []);

  const create = async () => {
    const channels = [];
    if (form.webhook) channels.push({ type: "webhook", url: form.webhook });
    if (form.email) channels.push({ type: "email", to: form.email });
    try {
      await post("/api/alerts", { symbol: form.symbol, timeframe: form.timeframe, condition: form.condition, params: JSON.parse(form.params || "{}"), channels });
      setMsg("Alert created."); load();
    } catch (e: any) { setMsg(e.message); }
  };

  return (
    <div>
      <h1>Alerts & technical monitoring</h1>
      {status && <div className={status.monitoring_active ? "card soft" : "warn-box"}>
        <b>Monitoring {status.monitoring_active ? "active" : "NOT active"}.</b> Scheduler {status.scheduler_running ? "running" : "not running"};
        live data source {status.live_data_source_configured ? "configured" : "not configured"}.
        {status.notes.map((n: string) => <div key={n}>{n}</div>)}
      </div>}
      <div className="card soft">
        <div className="row">
          <label className="field">Symbol<input value={form.symbol} onChange={(e) => setForm({ ...form, symbol: e.target.value })} style={{ width: 110 }} /></label>
          <label className="field">TF<select value={form.timeframe} onChange={(e) => setForm({ ...form, timeframe: e.target.value })}>{["15m", "1h", "1d", "1w"].map((t) => <option key={t}>{t}</option>)}</select></label>
          <label className="field">Condition<select value={form.condition} onChange={(e) => setForm({ ...form, condition: e.target.value })}>
            {Object.keys(conds).map((c) => <option key={c} value={c}>{c.replace(/_/g, " ")}</option>)}</select></label>
          <label className="field">Params (JSON)<input value={form.params} onChange={(e) => setForm({ ...form, params: e.target.value })} style={{ width: 160 }} /></label>
          <label className="field">Webhook URL<input value={form.webhook} onChange={(e) => setForm({ ...form, webhook: e.target.value })} /></label>
          <label className="field">Email<input value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} /></label>
          <button onClick={create} disabled={!form.symbol}>Create</button>
          <button className="secondary" onClick={() => post("/api/alerts/evaluate", {}).then(load)}>Evaluate now</button>
        </div>
        <p className="muted">{conds[form.condition]}. Alerts evaluate on completed candles only, suppress duplicates per bar, and skip stale data.</p>
        {msg && <div className="warn-box">{msg}</div>}
      </div>
      <div className="card"><h2>Rules</h2>
        <table><thead><tr><th>Symbol</th><th>TF</th><th>Condition</th><th>Params</th><th>Channels</th><th>Last status</th><th /></tr></thead>
          <tbody>{rules.map((r) => <tr key={r.id}><td>{r.symbol}</td><td>{r.timeframe}</td><td>{r.condition}</td><td className="muted">{JSON.stringify(r.params)}</td>
            <td className="muted">{r.channels.map((c: any) => c.type).join(", ") || "—"}</td><td className="muted">{r.last_status ?? "never evaluated"}</td>
            <td><button className="secondary" onClick={() => del(`/api/alerts/${r.id}`).then(load)}>Delete</button></td></tr>)}</tbody></table>
      </div>
      <div className="card"><h2>Recent events</h2>
        <table><thead><tr><th>When</th><th>Symbol</th><th>Condition</th><th>Level</th><th>Bar</th><th>Evidence</th><th>Delivery</th></tr></thead>
          <tbody>{events.map((e) => <tr key={e.id}><td>{String(e.triggered_at).slice(0, 19)}</td><td>{e.payload.symbol}</td><td>{e.payload.condition}</td>
            <td>{e.payload.trigger_level ?? "—"}</td><td>{e.bar_timestamp.slice(0, 10)}</td><td className="muted">{e.payload.evidence.join("; ")}</td>
            <td className="muted">{e.deliveries.map((d: any) => `${d.type}:${d.ok ? "ok" : d.error ?? d.status}`).join(", ") || "—"}</td></tr>)}</tbody></table>
      </div>
    </div>
  );
}
