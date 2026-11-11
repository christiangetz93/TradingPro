"""HTML-/Markdown-Report für Backtests (Equity-Kurve als SVG, Kennzahlen, Benchmarks)."""
from __future__ import annotations

import html
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from .backtest import BacktestResult, trade_stats
from .report import CSS

PCT = {"Rendite gesamt", "CAGR", "Max. Drawdown", "Kostenquote", "Investitionsgrad", "Trefferquote", "TP1 erreicht"}
EUR = {"Endkapital", "Kosten gesamt", "Brutto-Gewinn", "Netto-Gewinn"}


def _fmt(k: str, v) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", ".")
    if not isinstance(v, (float, np.floating)) or not np.isfinite(v):
        return "–"
    if k == "Trades":
        return f"{v:,.0f}".replace(",", ".")
    if k in PCT:
        return f"{v:+.1%}" if k in ("CAGR", "Rendite gesamt") else f"{v:.1%}"
    if k in EUR:
        return f"{v:,.0f} €".replace(",", ".")
    return f"{v:.2f}"


def _equity_svg(curves: dict[str, pd.Series], width: int = 860, height: int = 300) -> str:
    colors = ["var(--entry)", "var(--muted)", "var(--target)", "var(--invalid)"]
    allv = pd.concat([c / c.iloc[0] for c in curves.values()])
    lo, hi = float(np.log(allv.min())), float(np.log(allv.max()))
    hi = hi if hi > lo else lo + 1e-6
    t0 = min(c.index[0] for c in curves.values())
    t1 = max(c.index[-1] for c in curves.values())
    span = max((t1 - t0).days, 1)
    L, R, T, B = 8, 70, 12, 22
    W, H = width - L - R, height - T - B
    X = lambda d: L + W * (d - t0).days / span  # noqa: E731
    Y = lambda v: T + H * (1 - (np.log(v) - lo) / (hi - lo))  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Kapitalkurve">']
    for k in range(5):
        v = float(np.exp(lo + (hi - lo) * k / 4))
        parts.append(f'<line x1="{L}" x2="{L + W}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                     f'<text x="{L + W + 4}" y="{Y(v) + 3:.1f}" class="axis">{v:.2f}x</text>')
    for i, (name, c) in enumerate(curves.items()):
        c = c / c.iloc[0]
        step = max(1, len(c) // 900)
        pts = " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in list(c.items())[::step])
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i % 4]}" stroke-width="1.6"/>'
                     f'<text x="{L + 6}" y="{T + 12 + 13 * i}" class="lbl" fill="{colors[i % 4]}">{html.escape(name)}</text>')
    parts.append(f'<text x="{L}" y="{height - 6}" class="axis">{t0:%Y}</text>'
                 f'<text x="{L + W}" y="{height - 6}" text-anchor="end" class="axis">{t1:%Y}</text></svg>')
    return "".join(parts)


def _by_key(trades) -> list[tuple[str, dict]]:
    g = defaultdict(list)
    for t in trades:
        g[t.key].append(t)
    return sorted(((k, trade_stats(v, "r_net")) for k, v in g.items()), key=lambda kv: -kv[1]["Trades"])


def build_backtest_report(segments: list[dict], cfg_text: list[str], notes: list[str]) -> tuple[str, str]:
    """segments: [{"name", "result": BacktestResult, "equity_stats", "random": [stats...], "bh": (cagr, dd, sharpe), "bh_curve"}]"""
    e = html.escape
    h = [f"<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Elliott-Wave-Backtest</title><style>{CSS}</style></head>"
         f"<body><main><h1>Elliott-Wave-Backtest</h1><p class='sub'>{datetime.now():%d.%m.%Y %H:%M} · "
         + " · ".join(e(x) for x in cfg_text) + "</p>"]
    md = [f"# Elliott-Wave-Backtest – {datetime.now():%d.%m.%Y %H:%M}", "", *[f"- {x}" for x in cfg_text], ""]
    keys = ["CAGR", "Max. Drawdown", "Sharpe", "Trades", "Trades/Jahr", "Endkapital", "Netto-Gewinn",
            "Kosten gesamt", "Kostenquote", "Investitionsgrad"]
    for seg in segments:
        res: BacktestResult = seg["result"]
        es = seg["equity_stats"]
        ts = trade_stats(res.trades, "r_net")
        rnd = seg.get("random") or []
        bh = seg.get("bh")
        h.append(f"<section class='card'><h2>{e(seg['name'])}</h2>")
        md += [f"## {seg['name']}", ""]
        cols = ["Strategie"] + (["Zufall (Median)", "Zufall (Spanne)"] if rnd else []) + (["Buy & Hold*"] if bh else [])
        h.append("<div class='tbl'><table><tr><th>Kennzahl</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
        md += ["| Kennzahl | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for k in keys:
            row = [_fmt(k, es.get(k, float("nan")))]
            if rnd:
                vals = [r.get(k, float("nan")) for r in rnd]
                row += [_fmt(k, float(np.median(vals))), f"{_fmt(k, float(np.min(vals)))} … {_fmt(k, float(np.max(vals)))}"]
            if bh:
                row += [{"CAGR": _fmt("CAGR", bh[0]), "Max. Drawdown": _fmt("Max. Drawdown", bh[1]),
                         "Sharpe": _fmt("Sharpe", bh[2])}.get(k, "–")]
            h.append(f"<tr><th>{k}</th>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>")
            md.append(f"| {k} | " + " | ".join(row) + " |")
        h.append("</table></div>")
        if ts.get("Trades"):
            line = (f"Trades: {ts['Trades']} · Trefferquote {ts['Trefferquote']:.0%} · Ø {ts['Ø R']:+.3f} R netto "
                    f"(t-Wert {ts['t-Wert']:.1f}) · Profitfaktor {ts['Profitfaktor']:.2f} · Ø Gewinn {ts['Ø Gewinn R']:+.2f} R · "
                    f"Ø Verlust {ts['Ø Verlust R']:+.2f} R · Ø Haltedauer {ts['Ø Haltedauer']:.0f} Tage")
            h.append(f"<p>{e(line)}</p>")
            md += ["", line]
        if rnd:
            better = sum(es.get("CAGR", -1) > r.get("CAGR", 0) for r in rnd)
            line = (f"Die Strategie war besser als {better} von {len(rnd)} Läufen mit Zufalls-Einstiegen (gleiches Regelwerk, "
                    "gleiche Stop-Abstände und Haltedauern). Liegt sie nicht deutlich darüber, stammt der Ertrag aus dem "
                    "Regelwerk (Trend, Risiko, Ausstieg) und nicht aus der Wellenprognose.")
            h.append(f"<p class='foot'>{e(line)}</p>")
            md += ["", f"_{line}_"]
        curves = {"Strategie": res.equity}
        if seg.get("bh_curve") is not None:
            curves["Buy & Hold (gleichgewichtet)*"] = seg["bh_curve"]
        h.append(_equity_svg(curves))
        yr = res.equity.resample("YE").last()
        rets = yr.pct_change()
        rets.iloc[0] = yr.iloc[0] / res.start_capital - 1
        h.append("<h3>Jahresrenditen</h3><div class='tbl'><table><tr>" + "".join(f"<th>{d.year}</th>" for d in rets.index)
                 + "</tr><tr>" + "".join(f"<td>{v:+.1%}</td>" for v in rets.values) + "</tr></table></div>")
        md += ["", "Jahresrenditen: " + " · ".join(f"{d.year}: {v:+.1%}" for d, v in rets.items())]
        bk = _by_key(res.trades)
        if bk:
            h.append("<h3>Ergebnis je Setup-Typ (nach allen Kosten)</h3><div class='tbl'><table><tr><th>Setup</th><th>Trades</th>"
                     "<th>Treffer</th><th>Ø R</th><th>t-Wert</th><th>Profitfaktor</th></tr>" + "".join(
                         f"<tr><td>{e(k)}</td><td>{s['Trades']}</td><td>{s['Trefferquote']:.0%}</td><td>{s['Ø R']:+.3f}</td>"
                         f"<td>{s['t-Wert']:.1f}</td><td>{s['Profitfaktor']:.2f}</td></tr>" for k, s in bk) + "</table></div>")
            md += ["", "| Setup | Trades | Treffer | Ø R | t | PF |", "|---|---|---|---|---|---|"]
            md += [f"| {k} | {s['Trades']} | {s['Trefferquote']:.0%} | {s['Ø R']:+.3f} | {s['t-Wert']:.1f} | {s['Profitfaktor']:.2f} |"
                   for k, s in bk]
        reasons = Counter(t.exit_reason.split(" (")[0] for t in res.trades)
        if reasons:
            line = "Ausstiegsgründe: " + " · ".join(f"{k} {v}" for k, v in reasons.most_common())
            h.append(f"<p class='foot'>{e(line)}</p>")
            md += ["", line]
        last = res.trades[-25:]
        if last:
            h.append("<h3>Letzte Trades</h3><div class='tbl'><table><tr><th>Titel</th><th>Setup</th><th>Kauf</th><th>Verkauf</th>"
                     "<th>Einstieg</th><th>Netto €</th><th>R</th><th>Grund</th></tr>" + "".join(
                         f"<tr><td>{e(t.ticker)}</td><td>{e(t.key)}</td><td>{t.entry_date}</td><td>{t.exit_date}</td>"
                         f"<td>{t.entry:.2f}</td><td>{t.net:+.0f}</td><td>{t.r_net:+.2f}</td><td>{e(t.exit_reason)}</td></tr>"
                         for t in last) + "</table></div>")
        h.append("</section>")
        md.append("")
    h.append("<section class='card'><h3>Hinweise</h3><ul>" + "".join(f"<li>{e(n)}</li>" for n in notes) + "</ul></section>")
    h.append("</main></body></html>")
    md += ["## Hinweise", ""] + [f"- {n}" for n in notes]
    return "".join(h), "\n".join(md)
