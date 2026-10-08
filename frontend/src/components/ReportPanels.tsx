import type { Pattern, Report, Setup } from "../types";
import { fmt, stageLabel } from "../api";

export function DirBadge({ d }: { d: string }) {
  return <span className={`badge ${d}`}>{d}</span>;
}

export function StageBadge({ s }: { s: string | null }) {
  const cls = s === "confirmed" || s === "retesting" ? "brand" : s === "failed" || s === "invalidated" ? "warn" : "neutral";
  return <span className={`badge ${cls}`}>{stageLabel(s)}</span>;
}

export function DataStatus({ r }: { r: Report }) {
  const src = r.overview.data_source;
  const q = r.data_quality;
  return (
    <div className="row">
      {src.is_synthetic && <span className="badge warn">SYNTHETIC DEMO DATA</span>}
      <span className="badge">{src.provider}</span>
      <span className={`badge ${q.rating === "high" ? "bullish" : q.rating === "medium" ? "neutral" : "bearish"}`}>data quality: {q.rating}</span>
      {q.stale && <span className="badge bearish">stale</span>}
      {!q.last_candle_complete && <span className="badge warn">live candle — provisional</span>}
      <span className="muted">as of {new Date(r.overview.data_timestamp).toLocaleString("en-GB", { timeZone: "Asia/Karachi" })} PKT</span>
    </div>
  );
}

export function Warnings({ items }: { items: string[] }) {
  return <>{items.map((w) => <div key={w} className="warn-box">{w}</div>)}</>;
}

export function ScoreCard({ r }: { r: Report }) {
  const s = r.score;
  return (
    <div className="card">
      <h2>Confluence score</h2>
      <div className="kpis">
        <div className="kpi"><div className="l">Score</div><div className="v">{s.score.toFixed(0)}<span className="muted">/100</span></div></div>
        <div className="kpi"><div className="l">Rating</div><div className={`v ${s.rating.includes("Bull") ? "bullish-t" : s.rating.includes("Bear") ? "bearish-t" : ""}`}>{s.rating}</div></div>
        <div className="kpi"><div className="l">Coverage</div><div className="v">{(s.coverage * 100).toFixed(0)}%</div></div>
      </div>
      <table style={{ marginTop: 10 }}>
        <thead><tr><th>Component</th><th>Wt</th><th>Points</th><th>Reasons</th></tr></thead>
        <tbody>{s.components.map((c) => (
          <tr key={c.name}><td>{c.name.replace(/_/g, " ")}</td><td>{c.weight}</td>
            <td>{c.points === null ? <span className="badge warn">n/a</span> : c.points.toFixed(1)}</td>
            <td className="muted">{c.reasons.join("; ")}</td></tr>))}</tbody>
      </table>
      <p className="muted">{s.disclaimer}</p>
    </div>
  );
}

export function TrendTable({ r }: { r: Report }) {
  if (!r.trend) return null;
  return (
    <div className="card">
      <h2>B. Multi-timeframe trend <span className="badge brand">{r.trend.alignment.replace(/_/g, " ")}</span></h2>
      <table><thead><tr><th>Timeframe</th><th>Trend</th><th>Strength</th><th>Structure</th><th>MA structure</th></tr></thead>
        <tbody>{r.trend.timeframes.map((t) => (
          <tr key={t.timeframe}><td>{t.label}</td><td>{t.trend === "unavailable" ? <span className="muted">no data</span> : <DirBadge d={t.trend} />}</td>
            <td>{t.strength}</td><td>{t.structure}</td><td className="muted">{t.ma_structure ?? "—"}</td></tr>))}</tbody></table>
      <ul className="tight">{r.trend.commentary.map((c) => <li key={c} className="muted">{c}</li>)}</ul>
    </div>
  );
}

export function IndicatorTable({ r }: { r: Report }) {
  const i = r.indicators;
  const rows: [string, string, string][] = [];
  for (const k of ["sma_50", "sma_100", "sma_200", "ema_50", "ema_100", "ema_200"]) {
    const v = i.moving_averages?.[k];
    rows.push([k.toUpperCase().replace("_", " "), v ? fmt(v.value) : "—", v ? `${v.price_above ? "above" : "below"} (${v.distance_pct > 0 ? "+" : ""}${v.distance_pct}%)` : i.unavailable?.[k] ?? "unavailable"]);
  }
  if (i.rsi) rows.push(["RSI 14", fmt(i.rsi.value), `${i.rsi.zone}, ${i.rsi.direction}`]);
  if (i.macd) rows.push(["MACD 12/26/9", `${fmt(i.macd.macd, 3)} / ${fmt(i.macd.signal, 3)}`, `${i.macd.state.replace("_", " ")}${i.macd.cross ? ", " + i.macd.cross.replace(/_/g, " ") : ""}`]);
  if (i.bollinger) rows.push(["Bollinger 20/2", `%B ${fmt(i.bollinger.pct_b)}`, `${i.bollinger.position.replace("_", " ")}${i.bollinger.squeeze ? " · squeeze" : ""}`]);
  if (i.atr) rows.push(["ATR 14", fmt(i.atr.value), `${i.atr.pct_of_price}% of price`]);
  rows.push(["Volume / 20-avg", i.volume ? `${fmt(i.volume.relative)}x` : "—", i.volume ? `OBV ${i.volume.obv_direction ?? "—"}` : "volume unavailable"]);
  if (i.adx) rows.push(["ADX 14", fmt(i.adx.adx), `${i.adx.strength} (+DI ${fmt(i.adx.plus_di, 1)} / −DI ${fmt(i.adx.minus_di, 1)})`]);
  return (
    <div className="card"><h2>C. Technical indicators</h2>
      <table><thead><tr><th>Indicator</th><th>Value</th><th>State</th></tr></thead>
        <tbody>{rows.map(([a, b, c]) => <tr key={a}><td>{a}</td><td>{b}</td><td className="muted">{c}</td></tr>)}</tbody></table>
    </div>
  );
}

export function PatternTable({ patterns, title = "D. Detected chart patterns" }: { patterns: Pattern[]; title?: string }) {
  return (
    <div className="card"><h2>{title}</h2>
      <div className="table-wrap"><table>
        <thead><tr><th>Pattern</th><th>Dir</th><th>Stage</th><th>Trigger</th><th>Invalidation</th><th>Target</th><th>Vol conf.</th><th>Quality</th></tr></thead>
        <tbody>{patterns.length === 0 ? <tr><td colSpan={8} className="muted">No current pattern meets the detection criteria.</td></tr> :
          patterns.map((p, k) => (
            <tr key={p.pattern_id + k} title={p.evidence.join("\n")}>
              <td>{p.name} {p.experimental && <span className="badge warn">experimental</span>} {p.provisional && <span className="badge warn">provisional</span>}</td>
              <td><DirBadge d={p.direction} /></td><td><StageBadge s={p.stage} /></td>
              <td>{fmt(p.breakout_level)}</td><td>{fmt(p.invalidation_level)}</td><td>{fmt(p.target)}{p.target_reached ? " ✓" : ""}</td>
              <td>{p.volume_confirmation === null ? "n/a" : p.volume_confirmation ? "yes" : "no"}</td>
              <td><div className="bar" style={{ width: 70 }}><div style={{ width: `${p.quality_score}%` }} /></div><span className="muted">{p.quality_score.toFixed(0)}</span></td>
            </tr>))}</tbody></table></div>
      <p className="muted">Hover a row for the geometric evidence. Quality is a geometric/contextual score, not a probability.</p>
    </div>
  );
}

export function LevelsCard({ r }: { r: Report }) {
  const k = r.key_levels;
  return (
    <div className="card"><h2>E. Key technical levels</h2>
      <table><tbody>
        <tr><th>Immediate support</th><td>{k.immediate_support ?? "—"}</td></tr>
        <tr><th>Major support</th><td>{k.major_support ?? "—"}</td></tr>
        <tr><th>Immediate resistance</th><td>{k.immediate_resistance ?? "—"}</td></tr>
        <tr><th>Major resistance</th><td>{k.major_resistance ?? "—"}</td></tr>
        <tr><th>Breakout trigger</th><td>{fmt(k.breakout_trigger)}</td></tr>
        <tr><th>Breakdown trigger</th><td>{fmt(k.breakdown_trigger)}</td></tr>
      </tbody></table>
    </div>
  );
}

function SetupBlock({ s }: { s: Setup }) {
  const title = s.side === "long" ? "Long setup" : "Short / bearish scenario";
  if (!s.valid) return <div className="card soft"><h3>{title}</h3><p className="muted">{s.notes.join(" ") || "No valid setup."}</p></div>;
  return (
    <div className="card soft">
      <h3>{title} <span className="badge">{s.status}</span> {!s.meets_min_rr && <span className="badge warn">below 1:2 R:R</span>}</h3>
      <p style={{ fontSize: 13, margin: "4px 0" }}>{s.entry_trigger}</p>
      <table><tbody>
        <tr><th>Entry zone</th><td>{s.entry_zone ? `${fmt(s.entry_zone[0])} – ${fmt(s.entry_zone[1])}` : "—"}</td></tr>
        <tr><th>Stop-loss</th><td>{fmt(s.stop_loss)} <span className="muted">{s.stop_basis}</span></td></tr>
        <tr><th>ATR stop alt.</th><td>{fmt(s.atr_stop_alternative)}</td></tr>
        {s.targets.map((t) => <tr key={t.label}><th>{t.label}</th><td>{fmt(t.price)} <span className="muted">R:R {t.rr ?? "—"} · {t.source}</span></td></tr>)}
        <tr><th>Confidence</th><td>{s.confidence.toFixed(0)}/100 <span className="muted">(quality, not probability)</span></td></tr>
      </tbody></table>
      <p className="muted"><b>Confirmation:</b> {s.confirmation_required}</p>
      {s.technical_invalidation && <p className="muted"><b>Invalidation:</b> {s.technical_invalidation}</p>}
      {s.execution && <div className="warn-box">{s.execution}</div>}
      <ul className="tight">{[...s.key_risks, ...s.notes].map((x) => <li key={x} className="muted">{x}</li>)}</ul>
    </div>
  );
}

export function SetupsCard({ r }: { r: Report }) {
  return (
    <div className="card"><h2>F. Trading report card</h2>
      <div className="grid two"><SetupBlock s={r.long_setup} /><SetupBlock s={r.short_setup} /></div>
    </div>
  );
}

export function CommentaryCard({ r }: { r: Report }) {
  const c = r.commentary;
  if (!c) return null;
  return (
    <div className="card"><h2>G. Analyst commentary <span className="badge">{c.engine}</span></h2>
      <p style={{ fontSize: 14 }}>{c.summary}</p>
      <div className="grid two">
        <div><h3>Strongest evidence</h3><ul className="tight">{c.strongest_evidence.map((x) => <li key={x}>{x}</li>)}</ul></div>
        <div><h3>Conflicting / missing evidence</h3><ul className="tight">{c.conflicting_signals.map((x) => <li key={x}>{x}</li>)}</ul></div>
        <div><h3>What would change the view</h3><ul className="tight">{c.what_changes_view.map((x) => <li key={x}>{x}</li>)}</ul></div>
        <div><h3>Confirmation needed</h3><ul className="tight">{c.confirmation_needed.map((x) => <li key={x}>{x}</li>)}</ul></div>
      </div>
      <div className="conclusion">{c.conclusion}</div>
      {"ai_rejected" in c.validation && <p className="muted">AI commentary was rejected by evidence validation; deterministic commentary shown.</p>}
    </div>
  );
}

export function ExtrasCard({ r }: { r: Report }) {
  return (
    <div className="grid two">
      <div className="card"><h2>Candlestick signals</h2>
        <table><thead><tr><th>Date</th><th>Pattern</th><th>Dir</th><th>Context score</th><th>Next bar</th></tr></thead>
          <tbody>{r.candlesticks.slice(-8).reverse().map((c, k) => (
            <tr key={k} title={c.evidence.join("\n")}><td>{c.timestamp.slice(0, 10)}</td><td>{c.name}</td><td><DirBadge d={c.direction} /></td>
              <td>{c.score.toFixed(0)}</td><td className="muted">{c.confirmation.replace("_", " ")}</td></tr>))}</tbody></table>
      </div>
      <div className="card"><h2>Divergences & structure</h2>
        <table><tbody>{r.divergences.slice(-5).reverse().map((d, k) => (
          <tr key={k}><td>{d.signal_time.slice(0, 10)}</td><td>{d.type.replace(/_/g, " ")}</td><td>{d.indicator.toUpperCase()}</td></tr>))}
          {r.structure && <tr><th>Structure</th><td colSpan={2}>{r.structure.trend}; {r.structure.events.slice(-3).map((e) => `${e.type} ${e.direction} ${e.time.slice(0, 10)}`).join(" · ")}</td></tr>}
        </tbody></table>
        <p className="muted">Order blocks, supply/demand and Wyckoff events are heuristic; Elliott/harmonic counts are experimental.</p>
      </div>
    </div>
  );
}
