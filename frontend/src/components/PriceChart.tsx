import { useEffect, useRef } from "react";
import {
  CandlestickSeries, ColorType, createChart, createSeriesMarkers, HistogramSeries, IChartApi, ISeriesApi,
  LineSeries, LineStyle, SeriesMarker, Time, UTCTimestamp,
} from "lightweight-charts";
import type { OHLCVResponse, Report } from "../types";

export interface Overlays {
  sma50: boolean; sma200: boolean; ema20: boolean; bollinger: boolean;
  patterns: boolean; levels: boolean; fib: boolean; setups: boolean;
}

const PATTERN_COLORS = ["#0092DF", "#8e44ad", "#e67e22", "#16a085"];
const ts = (iso: string) => Math.floor(Date.parse(iso) / 1000) as UTCTimestamp;

export default function PriceChart({ data, report, overlays, onChart }: {
  data: OHLCVResponse; report: Report | null; overlays: Overlays; onChart?: (c: IChartApi | null) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = createChart(ref.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "#ffffff" }, textColor: "#1b2430", fontFamily: "Inter, sans-serif",
        panes: { separatorColor: "#dfe4ea" } },
      grid: { vertLines: { color: "#f0f2f5" }, horzLines: { color: "#f0f2f5" } },
      rightPriceScale: { borderColor: "#dfe4ea" },
      timeScale: { borderColor: "#dfe4ea", timeVisible: data.timeframe.endsWith("m") || data.timeframe === "1h" },
      crosshair: { mode: 0 },
    });
    onChart?.(chart);
    const narrow = (ref.current.clientWidth || 800) < 600;
    const lastClose = data.candles.length ? data.candles[data.candles.length - 1].close : 0;
    const title = (t: string) => (narrow ? "" : t);
    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a", downColor: "#ef5350", borderVisible: false, wickUpColor: "#26a69a", wickDownColor: "#ef5350",
    });
    candles.setData(data.candles.map((c) => ({ time: c.time as UTCTimestamp, open: c.open, high: c.high, low: c.low, close: c.close })));

    if (data.candles.some((c) => c.volume !== null)) {
      const vol = chart.addSeries(HistogramSeries, { priceFormat: { type: "volume" }, priceScaleId: "vol" }, 1);
      vol.setData(data.candles.map((c) => ({ time: c.time as UTCTimestamp, value: c.volume ?? 0,
        color: c.close >= c.open ? "rgba(38,166,154,0.5)" : "rgba(239,83,80,0.5)" })));
      chart.panes()[1]?.setHeight(110);
    }

    const ind = data.indicators ?? {};
    const addLine = (key: string, color: string, width: 1 | 2 = 1, style: LineStyle = LineStyle.Solid) => {
      const vals = ind[key];
      if (!vals) return;
      const s = chart.addSeries(LineSeries, { color, lineWidth: width, lineStyle: style, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false, autoscaleInfoProvider: () => null });
      s.setData(data.candles.map((c, i) => (vals[i] === null ? { time: c.time as UTCTimestamp } : { time: c.time as UTCTimestamp, value: vals[i] as number })));
    };
    if (overlays.sma50) addLine("sma_50", "#f39c12", 2);
    if (overlays.sma200) addLine("sma_200", "#8e44ad", 2);
    if (overlays.ema20) addLine("ema_20", "#2c3e50", 1);
    if (overlays.bollinger) { addLine("bb_upper", "#7f8c8d", 1, LineStyle.Dashed); addLine("bb_lower", "#7f8c8d", 1, LineStyle.Dashed); }

    const markers: SeriesMarker<Time>[] = [];
    if (report && overlays.patterns) {
      let triggers = 0;
      report.patterns.filter((p) => p.stage !== "invalidated").slice(0, 4).forEach((p, k) => {
        const color = PATTERN_COLORS[k % PATTERN_COLORS.length];
        p.lines.forEach((ln) => {
          const a = ts(ln.start.timestamp), b = ts(ln.end.timestamp);
          if (a === b) return;
          const s = chart.addSeries(LineSeries, { color, lineWidth: 2, lineStyle: ln.style === "dashed" ? LineStyle.Dashed : ln.style === "dotted" ? LineStyle.Dotted : LineStyle.Solid,
            priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
          s.setData(a < b ? [{ time: a, value: ln.start.price }, { time: b, value: ln.end.price }] : [{ time: b, value: ln.end.price }, { time: a, value: ln.start.price }]);
        });
        p.key_points.forEach((kp) => markers.push({ time: ts(kp.timestamp), position: "atPriceMiddle", price: kp.price, shape: "circle", color, text: kp.label, size: 0.8 } as SeriesMarker<Time>));
        if (p.breakout_level !== null && p.confirmation_time === null && triggers < 2) {
          triggers += 1;
          candles.createPriceLine({ price: p.breakout_level, color, lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: title(`${p.name} trigger`) });
        }
      });
    }
    if (report && overlays.levels) {
      report.levels.zones
        .filter((z) => z.strength >= 50 && Math.abs(z.mid - lastClose) / lastClose <= 0.15)
        .sort((a, b) => Math.abs(a.mid - lastClose) - Math.abs(b.mid - lastClose)).slice(0, 4).forEach((z) => {
        const col = z.kind === "support" ? "rgba(19,134,90,0.55)" : "rgba(192,57,43,0.55)";
        candles.createPriceLine({ price: z.high, color: col, lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: false, title: "" });
        candles.createPriceLine({ price: z.low, color: col, lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: false, title: title(`${z.kind} ${z.touches}x`) });
      });
    }
    if (report?.fibonacci && overlays.fib) {
      report.fibonacci.levels.filter((l) => l.kind === "retracement").forEach((l) =>
        candles.createPriceLine({ price: l.price, color: "rgba(0,146,223,0.45)", lineWidth: 1, lineStyle: LineStyle.SparseDotted, axisLabelVisible: false, title: title(`Fib ${l.ratio}`) }));
    }
    if (report && overlays.setups) {
      for (const s of [report.long_setup, report.short_setup]) {
        if (!s.valid || s.entry_price === null) continue;
        const side = s.side === "long" ? "L" : "S";
        candles.createPriceLine({ price: s.entry_price, color: "#0072b0", lineWidth: 1, lineStyle: LineStyle.Solid, axisLabelVisible: true, title: title(`${side} entry`) });
        if (s.stop_loss !== null) candles.createPriceLine({ price: s.stop_loss, color: "#c0392b", lineWidth: 1, lineStyle: LineStyle.Solid, axisLabelVisible: true, title: title(`${side} stop`) });
        s.targets.slice(0, 1).forEach((t) => candles.createPriceLine({ price: t.price, color: "#13865a", lineWidth: 1, lineStyle: LineStyle.Solid, axisLabelVisible: true, title: title(`${side} ${t.label}`) }));
      }
    }
    markers.sort((a, b) => (a.time as number) - (b.time as number));
    createSeriesMarkers(candles as ISeriesApi<"Candlestick">, markers);
    const n = data.candles.length;
    if (n > 160) chart.timeScale().setVisibleLogicalRange({ from: n - 160, to: n + 3 });
    else chart.timeScale().fitContent();
    return () => { onChart?.(null); chart.remove(); };
  }, [data, report, overlays, onChart]);

  return <div ref={ref} className="chart-box" />;
}
