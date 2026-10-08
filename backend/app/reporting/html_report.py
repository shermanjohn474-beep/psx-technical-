"""Standalone printable HTML research report (sections A-G) with an embedded
annotated chart. "Save as PDF" from the browser print dialog produces the PDF
export; the stylesheet is print-optimised (A4)."""
from __future__ import annotations

import base64
from html import escape

import pandas as pd

from app.schemas.report import AnalysisReport
from app.screenshot_analysis.render import RenderSpec, annotate_patterns, encode_png, render_chart


def chart_png(df: pd.DataFrame, report: AnalysisReport, bars: int = 160, width: int = 1200, height: int = 620) -> bytes:
    off = max(0, len(df) - bars)
    view = df.iloc[off:]
    img, g = render_chart(view, RenderSpec(width=width, height=height, title=f"{report.overview.symbol} {report.overview.timeframe}"))
    shifted = []
    for p in report.patterns:
        if p.start_index < off or p.stage.value == "invalidated":
            continue
        q = p.model_copy(deep=True)
        for k in q.key_points:
            k.index -= off
        for ln in q.lines:
            ln.start.index -= off
            ln.end.index -= off
        q.start_index -= off
        shifted.append(q)
    zones = [z for z in report.levels.zones if z.strength >= 50]
    return encode_png(annotate_patterns(img, g, shifted, zones, [report.long_setup, report.short_setup], max_patterns=3))


def _f(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return escape(str(x))


def render_html(report: AnalysisReport, df: pd.DataFrame | None = None) -> str:
    r = report
    img = ""
    if df is not None:
        img = f'<img class="chart" alt="Annotated chart" src="data:image/png;base64,{base64.b64encode(chart_png(df, r)).decode()}">'
    trend_rows = "".join(
        f"<tr><td>{escape(t.label)}</td><td class='{t.trend}'>{escape(t.trend)}</td><td>{escape(t.strength)}</td>"
        f"<td>{escape(t.structure)}</td><td>{escape(t.ma_structure or '—')}</td></tr>" for t in (r.trend.timeframes if r.trend else []))
    ind = r.indicators
    ma = ind.get("moving_averages", {})
    ind_rows = "".join(f"<tr><td>{k.upper().replace('_', ' ')}</td><td>{_f(v['value'])}</td><td>{'above' if v['price_above'] else 'below'} ({v['distance_pct']:+.2f}%)</td></tr>"
                       for k, v in ma.items() if v and k in ("sma_50", "sma_100", "sma_200", "ema_50", "ema_100", "ema_200"))
    if ind.get("rsi"):
        ind_rows += f"<tr><td>RSI (14)</td><td>{_f(ind['rsi']['value'])}</td><td>{ind['rsi']['zone']}, {ind['rsi']['direction']}</td></tr>"
    if ind.get("macd"):
        m = ind["macd"]
        ind_rows += f"<tr><td>MACD (12,26,9)</td><td>{_f(m['macd'], 3)} / {_f(m['signal'], 3)}</td><td>{m['state'].replace('_', ' ')}{', ' + m['cross'] if m['cross'] else ''}</td></tr>"
    if ind.get("bollinger"):
        b = ind["bollinger"]
        ind_rows += f"<tr><td>Bollinger (20,2)</td><td>%B {_f(b['pct_b'], 2)}</td><td>{b['position'].replace('_', ' ')}{' · squeeze' if b['squeeze'] else ''}</td></tr>"
    if ind.get("atr"):
        ind_rows += f"<tr><td>ATR (14)</td><td>{_f(ind['atr']['value'])}</td><td>{ind['atr']['pct_of_price']}% of price</td></tr>"
    if ind.get("volume"):
        v = ind["volume"]
        ind_rows += f"<tr><td>Volume vs 20-bar avg</td><td>{_f(v['relative'])}x</td><td>OBV {v.get('obv_direction') or '—'}</td></tr>"
    if ind.get("adx"):
        ind_rows += f"<tr><td>ADX (14)</td><td>{_f(ind['adx']['adx'])}</td><td>{ind['adx']['strength']}</td></tr>"
    pat_rows = "".join(
        f"<tr><td>{escape(p.name)}{' <em>(experimental)</em>' if p.experimental else ''}</td><td>{escape(p.timeframe or '')}</td>"
        f"<td>{escape(p.stage.value.replace('_', ' '))}</td><td>{_f(p.breakout_level)}</td><td>{_f(p.invalidation_level)}</td>"
        f"<td>{_f(p.target)}</td><td>{p.quality_score:.0f}</td></tr>" for p in r.patterns[:10]) or "<tr><td colspan=7>No current pattern meets detection criteria.</td></tr>"
    k = r.key_levels

    def setup_row(s):
        if not s.valid:
            return f"<tr><td>{s.side.title()}</td><td colspan=6>{escape(' '.join(s.notes) or 'No valid setup')}</td></tr>"
        t = s.targets + [None, None]
        return (f"<tr><td>{'Long' if s.side == 'long' else 'Short / Bearish'}</td><td>{escape(s.entry_trigger)}</td>"
                f"<td>{_f(s.stop_loss)}</td><td>{_f(t[0].price if t[0] else None)}</td><td>{_f(t[1].price if t[1] else None)}</td>"
                f"<td>{_f(s.primary_rr)}{'' if s.meets_min_rr else ' ⚠'}</td><td>{s.confidence:.0f}</td></tr>")
    c = r.commentary
    li = lambda xs: "".join(f"<li>{escape(x)}</li>" for x in xs) or "<li>—</li>"  # noqa: E731
    comps = "".join(f"<tr><td>{escape(x.name.replace('_', ' '))}</td><td>{x.weight:g}</td><td>{'n/a' if x.points is None else f'{x.points:.1f}'}</td><td>{escape('; '.join(x.reasons))}</td></tr>" for x in r.score.components)
    warn = "".join(f"<div class='warn'>{escape(w)}</div>" for w in r.warnings)
    src = r.overview.data_source
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(r.overview.symbol)} Technical Report</title>
<style>
:root{{--brand:#0092DF;--ink:#1b2430;--muted:#5b6675;--surface:#f4f6f8;--line:#dfe4ea}}
body{{font-family:Inter,Segoe UI,Arial,sans-serif;color:var(--ink);background:#fff;margin:0;padding:24px;max-width:1100px;margin:auto}}
header{{border-bottom:3px solid var(--brand);padding-bottom:10px;margin-bottom:16px;display:flex;justify-content:space-between;align-items:end;flex-wrap:wrap;gap:8px}}
h1{{margin:0;font-size:24px}} h2{{font-size:16px;color:var(--brand);border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:26px}}
.brand{{color:var(--brand);font-weight:700;letter-spacing:.04em}} .muted{{color:var(--muted);font-size:12px}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:var(--surface);font-weight:600}} .bullish{{color:#13865a}} .bearish{{color:#c0392b}}
.kpi{{display:flex;gap:12px;flex-wrap:wrap}} .kpi div{{background:var(--surface);padding:10px 14px;border-radius:6px;min-width:140px}}
.kpi b{{display:block;font-size:18px}} .warn{{background:#fff4e5;border-left:4px solid #f0a020;padding:6px 10px;margin:4px 0;font-size:12px}}
.chart{{width:100%;border:1px solid var(--line);border-radius:6px}} .conclusion{{background:#eaf6fd;border-left:4px solid var(--brand);padding:10px 14px;font-weight:600}}
@media print{{body{{padding:0}} h2{{break-after:avoid}} table,img{{break-inside:avoid}} @page{{size:A4;margin:14mm}}}}
</style></head><body>
<header><div><div class="brand">SHERMAN SECURITIES · TECHNICAL RESEARCH</div><h1>{escape(r.overview.symbol)} — {escape(r.overview.company_name or '')}</h1>
<div class="muted">Report {r.report_id} · generated {r.generated_at:%Y-%m-%d %H:%M} · engine v{r.engine_version}</div></div>
<div class="muted">Data: {escape(src.provider)} · {escape(src.attribution)}<br>As of {r.overview.data_timestamp:%Y-%m-%d %H:%M} (Asia/Karachi)</div></header>
{warn}
<h2>A. Stock Overview</h2><div class="kpi"><div>Price<b>{_f(r.overview.price)}</b></div><div>Change<b>{_f(r.overview.change_pct)}%</b></div>
<div>Timeframe<b>{escape(r.overview.timeframe)}</b></div><div>Confluence<b>{r.score.score:.0f}/100</b><span class="muted">{escape(r.score.rating)} · coverage {r.score.coverage:.0%}</span></div>
<div>Data quality<b>{escape(r.data_quality.rating if r.data_quality else '—')}</b></div></div>
{img}
<h2>B. Trend Analysis</h2><table><tr><th>Timeframe</th><th>Trend</th><th>Strength</th><th>Structure</th><th>MA structure</th></tr>{trend_rows}</table>
<p class="muted">Alignment: <b>{escape(r.trend.alignment.replace('_', ' ') if r.trend else '—')}</b>. {escape(' '.join(r.trend.commentary) if r.trend else '')}</p>
<h2>C. Technical Indicators</h2><table><tr><th>Indicator</th><th>Value</th><th>State</th></tr>{ind_rows}</table>
<h2>D. Detected Chart Patterns</h2><table><tr><th>Pattern</th><th>TF</th><th>Stage</th><th>Confirmation level</th><th>Invalidation</th><th>Target</th><th>Quality</th></tr>{pat_rows}</table>
<h2>E. Important Technical Levels</h2><table>
<tr><th>Immediate support</th><td>{_f(k.immediate_support)}</td><th>Immediate resistance</th><td>{_f(k.immediate_resistance)}</td></tr>
<tr><th>Major support</th><td>{_f(k.major_support)}</td><th>Major resistance</th><td>{_f(k.major_resistance)}</td></tr>
<tr><th>Breakout trigger</th><td>{_f(k.breakout_trigger)}</td><th>Breakdown trigger</th><td>{_f(k.breakdown_trigger)}</td></tr></table>
<h2>F. Trading Report Card</h2><table><tr><th>Scenario</th><th>Entry / Trigger</th><th>Stop-loss</th><th>Target 1</th><th>Target 2</th><th>R:R</th><th>Confidence</th></tr>
{setup_row(r.long_setup)}{setup_row(r.short_setup)}</table>
<p class="muted">⚠ = below the preferred 1:2 reward/risk (shown, not hidden; levels not adjusted). {escape(r.short_setup.execution or '')}</p>
<h3 style="font-size:14px">Confluence components</h3><table><tr><th>Component</th><th>Weight</th><th>Points</th><th>Reasons</th></tr>{comps}</table>
<p class="muted">{escape(r.score.disclaimer)}</p>
<h2>G. Analyst Commentary <span class="muted">({escape(c.engine if c else '—')})</span></h2>
<p>{escape(c.summary if c else '')}</p>
<b>Strongest evidence</b><ul>{li(c.strongest_evidence if c else [])}</ul>
<b>Conflicting / missing evidence</b><ul>{li(c.conflicting_signals if c else [])}</ul>
<b>What would change the assessment</b><ul>{li(c.what_changes_view if c else [])}</ul>
<b>Confirmation needed</b><ul>{li(c.confirmation_needed if c else [])}</ul>
<div class="conclusion">{escape(c.conclusion if c else '')}</div>
<p class="muted" style="margin-top:24px">{escape(r.disclaimer)}</p>
</body></html>"""
