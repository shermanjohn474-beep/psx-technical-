import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router-dom";
import type { IChartApi } from "lightweight-charts";
import { get, reportUrl } from "../api";
import type { OHLCVResponse, Report } from "../types";
import PriceChart, { Overlays } from "../components/PriceChart";
import {
  CommentaryCard, DataStatus, ExtrasCard, IndicatorTable, LevelsCard, PatternTable, ScoreCard, SetupsCard, TrendTable, Warnings,
} from "../components/ReportPanels";

const TFS = ["5m", "15m", "1h", "1d", "1w", "1M"];
const IND = "sma_50,sma_200,ema_20,bb_upper,bb_lower";

export default function ChartAnalysis() {
  const { symbol = "SYN-ASCT" } = useParams();
  const [sp, setSp] = useSearchParams();
  const tf = sp.get("tf") ?? "1d";
  const nav = useNavigate();
  const [data, setData] = useState<OHLCVResponse | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [tfs, setTfs] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [ai, setAi] = useState(false);
  const [start, setStart] = useState(sp.get("start") ?? "");
  const [end, setEnd] = useState(sp.get("end") ?? "");
  const [ov, setOv] = useState<Overlays>({ sma50: true, sma200: true, ema20: false, bollinger: false, patterns: true, levels: true, fib: false, setups: true });
  const chartRef = useRef<IChartApi | null>(null);
  const onChart = useCallback((c: IChartApi | null) => { chartRef.current = c; }, []);

  useEffect(() => {
    get<{ symbol: string; timeframes: string[] }[]>(`/api/symbols?q=${encodeURIComponent(symbol)}`)
      .then((rows) => setTfs(rows.find((r) => r.symbol === symbol.toUpperCase())?.timeframes ?? [])).catch(() => setTfs([]));
  }, [symbol]);

  useEffect(() => {
    let cancel = false;
    setLoading(true); setErr(null);
    const range = `${start ? `&start=${start}` : ""}${end ? `&end=${end}` : ""}`;
    Promise.all([
      get<OHLCVResponse>(`/api/ohlcv/${encodeURIComponent(symbol)}?timeframe=${tf}&indicators=${IND}${range}`),
      get<{ report: Report }>(`/api/analysis/${encodeURIComponent(symbol)}?timeframe=${tf}&ai=${ai}${range}`),
    ]).then(([o, a]) => { if (!cancel) { setData(o); setReport(a.report); } })
      .catch((e) => { if (!cancel) { setErr(String(e.message ?? e)); setData(null); setReport(null); } })
      .finally(() => !cancel && setLoading(false));
    return () => { cancel = true; };
  }, [symbol, tf, ai, start, end]);

  const exportPng = () => {
    const c = chartRef.current?.takeScreenshot();
    if (!c) return;
    const a = document.createElement("a");
    a.href = c.toDataURL("image/png"); a.download = `${symbol}_${tf}.png`; a.click();
  };
  const toggle = (k: keyof Overlays) => setOv((o) => ({ ...o, [k]: !o[k] }));

  return (
    <div>
      <div className="topbar">
        <div>
          <h1>{report?.overview.symbol ?? symbol.toUpperCase()} <span className="muted">{report?.overview.company_name}</span></h1>
          {report && <DataStatus r={report} />}
        </div>
        <div className="row no-print">
          {TFS.map((t) => (
            <button key={t} className={t === tf ? "" : "secondary"} disabled={tfs.length > 0 && !tfs.includes(t)}
              title={tfs.length > 0 && !tfs.includes(t) ? "No source data for this timeframe" : ""}
              onClick={() => setSp({ tf: t })}>{t}</button>))}
        </div>
      </div>
      <div className="row no-print" style={{ marginBottom: 10 }}>
        <label className="field">From<input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></label>
        <label className="field">To<input type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></label>
        <span className="toggle"><input type="checkbox" checked={ai} onChange={() => setAi(!ai)} /> AI commentary (if configured)</span>
        <button className="secondary" onClick={exportPng}>Export chart PNG</button>
        <a href={reportUrl(symbol, tf)} target="_blank" rel="noreferrer"><button className="secondary">Printable report / PDF</button></a>
        <button className="secondary" onClick={() => window.print()}>Print page</button>
        <button className="secondary" onClick={() => nav(`/backtest?symbol=${symbol}`)}>Backtest</button>
      </div>
      {err && <div className="err-box">{err}</div>}
      {loading && <p className="muted">Analysing…</p>}
      {report && <Warnings items={report.warnings} />}
      {data && (
        <div className="card">
          <div className="row no-print" style={{ marginBottom: 6 }}>
            {(Object.keys(ov) as (keyof Overlays)[]).map((k) => (
              <span key={k} className="toggle"><input type="checkbox" checked={ov[k]} onChange={() => toggle(k)} />{k}</span>))}
          </div>
          <PriceChart data={data} report={report} overlays={ov} onChart={onChart} />
        </div>
      )}
      {report && (
        <>
          <div className="grid two"><ScoreCard r={report} /><div><TrendTable r={report} /><LevelsCard r={report} /></div></div>
          <PatternTable patterns={report.patterns} />
          <SetupsCard r={report} />
          <div className="grid two"><IndicatorTable r={report} /><CommentaryCard r={report} /></div>
          <ExtrasCard r={report} />
          {report.historical_patterns.length > 0 && <PatternTable patterns={report.historical_patterns.slice(-10)} title="Historical patterns (resolved)" />}
          <p className="muted">{report.disclaimer}</p>
        </>
      )}
    </div>
  );
}
