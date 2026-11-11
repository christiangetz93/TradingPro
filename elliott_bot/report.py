"""Erzeugt Markdown- und HTML-Reports (HTML mit SVG-Charts, ohne externe Abhängigkeiten)."""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np

from .analyzer import TitleReport, fmt
from .config import BotConfig

SIGNAL_TEXT = {
    "KAUFEN": "Kaufsignal – alle Kriterien erfüllt",
    "SHORT": "Short-Signal – alle Kriterien erfüllt",
    "WARTEN": "Warten – Einstieg erst in der Zielzone",
    "BEOBACHTEN": "Beobachten – Kriterien (noch) nicht erfüllt",
    "MEIDEN": "Meiden – kein vorteilhaftes Setup",
}

DISCLAIMER = (
    "Dieser Report wurde automatisiert erstellt und ist keine Anlageberatung. Die Elliott-Wellen-"
    "Theorie ist wissenschaftlich umstritten; Zählungen sind interpretationsabhängig. Alle "
    "Wahrscheinlichkeiten sind Modellschätzungen auf Basis historischer Kurse und können sich als "
    "falsch erweisen. Vergangene Entwicklungen sind kein verlässlicher Indikator für künftige "
    "Ergebnisse. Der Bot sendet keine Orders; Paper-Trading dient nur der Erprobung."
)

METHOD = [
    "Pivots: ZigZag auf drei Wellengraden (Schwelle = Vielfaches der ATR).",
    "Zählungen: Alle Muster W3/W4/W5/A/B/C auf- und abwärts, die auf dem letzten bestätigten Pivot enden.",
    "Harte Regeln (Verstoß = verworfen): W2 < 100 % von W1, W3 nie die kürzeste, keine W4/W1-Überlappung, B ≤ 138,2 % von A.",
    "Richtlinien (Fibonacci-Verhältnisse, Alternation, RSI-Divergenz, Volumen) ergeben einen Qualitätsscore; "
    "wenige Prüfungen = schwache Evidenz.",
    "Szenario-Gewichte = geschärfte Scores, inkl. Pseudo-Szenario 'keine klare Zählung'.",
    "Basis-Wahrscheinlichkeit: Block-Bootstrap-Simulation OHNE Drift (Zufallspfad) des kompletten Ausstiegsplans.",
    "Wellen-Edge: Verschiebung der Basis proportional zum Szenario-Konsens (konservativer Faktor κ).",
    "Kalibrierung: Walk-Forward-Test derselben Methode auf der Historie des Titels; der gemessene Vorsprung "
    "ggü. Zufall wird per Bayes-Mittelung eingerechnet.",
    "Positionsgröße: Minimum aus fixem Risiko, Bruchteil-Kelly, Portfolio-/Regionen-Heat, Positions- und Cash-Limit.",
]


def _pct(x: float | None) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.0%}"


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def to_markdown(reports: list[TitleReport], cfg: BotConfig, header_lines: list[str]) -> str:
    cur = cfg.budget.currency
    out = [f"# Elliott-Wave-Report – {datetime.now():%d.%m.%Y %H:%M}", ""]
    out += header_lines + [""]
    out += ["| Titel | Region | Signal | Zählung | CRV | P(Ziel) | E[R] | Stück | Risiko |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        p = r.decision.plan
        out.append(
            f"| {r.fund.ticker} | {r.fund.region} | **{r.signal}** | "
            f"{r.analysis.primary.state if r.analysis.primary else '–'} | "
            f"{f'{p.crv:.1f}' if p else '–'} | {_pct(r.p_final)} | "
            f"{f'{r.mc_model.mean_r:+.2f}' if r.mc_model else '–'} | "
            f"{f'{r.size.units:g}' if r.size and r.tradeable else '–'} | "
            f"{f'{fmt(r.size.risk_amount, 0)} {cur}' if r.size and r.tradeable else '–'} |")
    for r in reports:
        out += ["", "---", "", f"## {r.fund.ticker} – {r.fund.name}", ""]
        out.append(f"**Signal: {r.signal}** – {SIGNAL_TEXT[r.signal]}. {r.decision.reason}.")
        for why in r.reasons:
            out.append(f"- ⚠ {why}")
        st = r.stats
        out += ["", f"Kurs {fmt(r.close)} ({r.series.currency}, {r.series.source}) · ATR {fmt(r.atr)} "
                f"({st['atr_pct']:.1%}) · Vola 1J {st['vola_1j']:.0%} · Perf. 1J {st['perf_1j']:+.1%} · "
                f"Max. Drawdown 1J {st['mdd_1j']:.1%}", ""]
        wa = r.analysis
        out += ["### Wellenanalyse", "",
                f"P(steigend) {wa.p_up:.0%} · P(fallend) {wa.p_down:.0%} · P(unklar) {wa.p_unclear:.0%}", "",
                "| # | Szenario | Gewicht | Ziele (nah → fern) | Stop-Niveau | Invalidierung |",
                "|---|---|---|---|---|---|"]
        for i, c in enumerate(wa.counts[:5], 1):
            tg = " / ".join(fmt(p) for _, p in c.targets)
            out.append(f"| {i} | {c.description} | {c.weight:.0%} | {tg} | {fmt(c.stop_ref)} | {fmt(c.invalidation)} |")
        if wa.primary:
            out += ["", "Regel- und Richtlinienprüfung der Hauptzählung:", ""]
            out += [f"- {'✔' if ch.hard else ('●' if ch.score >= 0.6 else '○')} {ch.name}: {ch.value}"
                    + ("" if ch.hard else f" (Score {ch.score:.2f})") for ch in wa.primary.checks]
        p = r.decision.plan
        if p:
            out += ["", "### Chance / Risiko", "",
                    f"- Einstieg ({p.entry_type}): {fmt(p.entry)}",
                    f"- Stop: {fmt(p.stop)} (Risiko je Anteil inkl. Kosten {fmt(p.risk_per_unit)})"]
            out += [f"- Ziel {i}: {fmt(v)} ({n})" for i, (n, v) in enumerate(p.levels, 1)]
            out += [f"- CRV bis Hauptziel ({p.main_target[0]}): **{p.crv:.2f}**"]
            if r.mc_model:
                c = r.calib
                out += ["", "### Wahrscheinlichkeiten", "",
                        f"- Zufallspfad (kein Wellen-Vorteil): P(Hauptziel vor Stop) {r.p_random:.0%}, "
                        f"E[R] {r.mc_random.mean_r:+.2f}",
                        f"- Szenario-Konsens (Edge) {r.edge:+.0%} → Modell {r.p_model:.0%}",
                        (f"- Walk-Forward-Kalibrierung: {c.signals} Signale, Trefferquote {_pct(c.hit_rate)} "
                         f"vs. Basis {_pct(c.baseline_mean)}, Ø {c.avg_r:+.2f} R" if c else "- Keine Kalibrierung"),
                        f"- **Final: P(Hauptziel) {r.p_final:.0%} · P(Teilziel 1) {r.mc_model.p_tp1:.0%} · "
                        f"P(Stop vor TP1) {r.mc_model.p_stop_first:.0%} · P(Zeitstopp) {r.mc_model.p_time_exit:.0%}**",
                        f"- Erwartungswert {r.mc_model.mean_r:+.2f} R · 5-%-Quantil {r.mc_model.quantile(0.05):+.2f} R · "
                        f"95-%-Quantil {r.mc_model.quantile(0.95):+.2f} R"]
            if r.size:
                s = r.size
                out += ["", "### Positionsgröße & Budget", "",
                        f"- {s.units:g} Anteile ≈ {fmt(s.invest, 0)} {cur} "
                        f"({s.invest / cfg.budget.capital:.1%} des Budgets), Risiko {fmt(s.risk_amount, 0)} {cur} "
                        f"({s.risk_pct:.2%}) – begrenzt durch **{s.binding}**",
                        f"- Volles Kelly {s.kelly_full:.1%} des Kapitals als Risiko – genutzt wird höchstens "
                        f"{cfg.budget.kelly_fraction:.0%} davon"]
                out += [f"- {n}" for n in s.notes]
        if r.exit_rules:
            out += ["", "### Ausstiegsstrategie", ""] + [f"{i}. {t}" for i, t in enumerate(r.exit_rules, 1)]
        if r.chances:
            out += ["", "### Chancen", ""] + [f"- {t}" for t in r.chances]
        out += ["", "### Risiken", ""] + [f"- {t}" for t in r.risks]
    out += ["", "---", "", "## Methodik", ""] + [f"- {m}" for m in METHOD]
    out += ["", f"_{DISCLAIMER}_", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _svg_chart(r: TitleReport, width: int = 860, height: int = 320) -> str:
    df = r.series.df
    wa = r.analysis
    prim = wa.primary
    first = len(df) - 260
    if prim:
        first = min(first, wa.offset + prim.points[0].idx - 15)
    first = max(0, max(first, len(df) - 600))
    d = df.iloc[first:]
    n = len(d)
    extra = max(10, int(n * 0.12))            # Platz rechts für Projektionen
    levels: list[tuple[float, str, str, str]] = []
    plan = r.decision.plan
    if plan:
        levels.append((plan.entry, "Einstieg", "var(--entry)", ""))
        levels.append((plan.stop, "Stop", "var(--stop)", ""))
        for i, (_, v) in enumerate(plan.levels, 1):
            levels.append((v, f"Ziel {i}", "var(--target)", "5 4"))
    elif r.decision.setup:
        su = r.decision.setup
        levels.append((su.target, "Setup-Ziel", "var(--target)", "5 4"))
        levels.append((su.stop, "Setup-Stop", "var(--stop)", ""))
    elif prim:
        for i, (_, v) in enumerate(prim.targets, 1):
            levels.append((v, f"Proj. {i}", "var(--target)", "5 4"))
    if prim:
        levels.append((prim.invalidation, "Invalidierung", "var(--invalid)", "2 3"))
    setup = r.decision.setup
    lo_v = [float(d["low"].min())] + [lv[0] for lv in levels]
    hi_v = [float(d["high"].max())] + [lv[0] for lv in levels]
    if setup:
        lo_v += [setup.stop, *setup.zone]
        hi_v += [setup.target, *setup.zone]
    ymin, ymax = min(lo_v), max(hi_v)
    pad = (ymax - ymin) * 0.06 or 1
    ymin, ymax = ymin - pad, ymax + pad
    L, R, T, B = 8, 92, 12, 22
    W, H = width - L - R, height - T - B

    def X(i: float) -> float:
        return L + W * i / (n + extra - 1)

    def Y(v: float) -> float:
        return T + H * (1 - (v - ymin) / (ymax - ymin))

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Kursverlauf {html.escape(r.fund.ticker)}">']
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        parts.append(f'<line x1="{L}" x2="{L + W}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                     f'<text x="{L + W + 4}" y="{Y(v) + 3:.1f}" class="axis">{fmt(v)}</text>')
    if setup:
        z0, z1 = sorted(setup.zone)
        parts.append(f'<rect x="{X(n - 1):.1f}" width="{X(n + extra - 1) - X(n - 1):.1f}" y="{Y(z1):.1f}" '
                     f'height="{max(2, Y(z0) - Y(z1)):.1f}" class="zone"/>'
                     f'<text x="{X(n - 1) - 4:.1f}" y="{(Y(z0) + Y(z1)) / 2 + 3:.1f}" text-anchor="end" '
                     f'class="lbl" fill="var(--entry)">Kaufzone</text>')
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(d["close"].to_numpy(float)))
    parts.append(f'<polyline points="{pts}" class="price"/>')
    # Beschriftungen vertikal entzerren (mind. 11 px Abstand)
    order = sorted(range(len(levels)), key=lambda k: Y(levels[k][0]))
    label_y: dict[int, float] = {}
    prev = -1e9
    for k in order:
        y = Y(levels[k][0]) - 3                  # bevorzugt über der Linie
        if y < prev + 11:
            y = max(Y(levels[k][0]) + 11, prev + 11)   # sonst darunter
        label_y[k] = y
        prev = y
    for k, (v, name, color, dash) in enumerate(levels):
        parts.append(f'<line x1="{X(n - 1):.1f}" x2="{L + W}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                     f'stroke="{color}" stroke-width="1.6" stroke-dasharray="{dash}"/>'
                     f'<text x="{L + W - 2}" y="{label_y[k]:.1f}" text-anchor="end" class="lbl" '
                     f'fill="{color}">{name}</text>')
    if prim:
        coords = []
        for p, lab in zip(prim.points, prim.labels):
            i = wa.offset + p.idx - first
            if 0 <= i < n:
                coords.append((X(i), Y(p.price), lab, p.kind))
        if len(coords) > 1:
            parts.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in coords)
                         + '" class="wave"/>')
        for x, y, lab, kind in coords:
            dy = -8 if kind == "H" else 16
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" class="pt"/>'
                         f'<text x="{x:.1f}" y="{y + dy:.1f}" text-anchor="middle" class="wlab">{lab}</text>')
    d0, d1 = d.index[0], d.index[-1]
    parts.append(f'<text x="{L}" y="{height - 6}" class="axis">{d0:%d.%m.%Y}</text>'
                 f'<text x="{X(n - 1):.1f}" y="{height - 6}" text-anchor="middle" class="axis">{d1:%d.%m.%Y}</text>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#5d6678;--line:#e3e6ec;--grid:#eceef2;
--price:#2b3445;--wave:#7a4fd1;--entry:#2f6fdf;--stop:#d23c3c;--target:#1e9a5a;--invalid:#d98a1c;
--zone:rgba(47,111,223,.12);--KAUFEN:#1e9a5a;--SHORT:#7a4fd1;--WARTEN:#c98a12;--BEOBACHTEN:#4c6a92;--MEIDEN:#c23b3b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11141a;--card:#1a1f28;--ink:#e6e9ef;
--muted:#9aa3b5;--line:#2a313d;--grid:#232a35;--price:#d5dbe6;--wave:#a98bf0;--entry:#6b9cff;--stop:#ff6b6b;
--target:#4cd18a;--invalid:#f0b052;--zone:rgba(107,156,255,.15)}}
:root[data-theme="dark"]{--bg:#11141a;--card:#1a1f28;--ink:#e6e9ef;--muted:#9aa3b5;--line:#2a313d;--grid:#232a35;
--price:#d5dbe6;--wave:#a98bf0;--entry:#6b9cff;--stop:#ff6b6b;--target:#4cd18a;--invalid:#f0b052;--zone:rgba(107,156,255,.15)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 60px}h1{font-size:1.6rem;margin:0 0 4px}
h2{font-size:1.25rem;margin:0}h3{font-size:1rem;margin:22px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--muted);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:600;font-size:.82rem;color:#fff}
.tag{display:inline-block;padding:2px 8px;border-radius:6px;border:1px solid var(--line);color:var(--muted);font-size:.8rem;margin:2px 4px 2px 0}
.head{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:14px 0}
.kpi{border:1px solid var(--line);border-radius:10px;padding:8px 12px}.kpi b{display:block;font-size:1.1rem}
.kpi span{color:var(--muted);font-size:.8rem}table{width:100%;border-collapse:collapse;font-size:.9rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}
.tbl{overflow-x:auto}.bar{display:flex;height:12px;border-radius:6px;overflow:hidden;margin:6px 0 2px}
.bar i{display:block}.warn{color:var(--stop)}ol,ul{padding-left:20px;margin:6px 0}li{margin:3px 0}
svg{width:100%;height:auto;display:block}svg .grid{stroke:var(--grid)}svg .axis{fill:var(--muted);font-size:10px}
svg .price{fill:none;stroke:var(--price);stroke-width:1.3}svg .wave{fill:none;stroke:var(--wave);stroke-width:2;opacity:.85}
svg .pt{fill:var(--wave)}svg .wlab{fill:var(--wave);font-size:12px;font-weight:700}svg .lbl{font-size:10px;font-weight:600}
svg .zone{fill:var(--zone)}.foot{color:var(--muted);font-size:.85rem}.ok{color:var(--target)}
"""


def _badge(signal: str) -> str:
    return f'<span class="badge" style="background:var(--{signal})">{signal}</span>'


def to_html(reports: list[TitleReport], cfg: BotConfig, header_lines: list[str]) -> str:
    e = html.escape
    cur = cfg.budget.currency
    h = [f"<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Elliott-Wave-Report</title><style>{CSS}</style></head>"
         f"<body><main><h1>Elliott-Wave-Report</h1><p class='sub'>{datetime.now():%d.%m.%Y %H:%M} · "
         + " · ".join(e(x) for x in header_lines) + "</p>"]
    h.append("<div class='card tbl'><table><tr><th>Titel</th><th>Region</th><th>Signal</th><th>Zählung</th>"
             "<th>CRV</th><th>P(Ziel)</th><th>E[R]</th><th>Stück</th><th>Risiko</th></tr>")
    for r in reports:
        p = r.decision.plan
        h.append(f"<tr><td><a href='#{e(r.fund.ticker)}'>{e(r.fund.ticker)}</a></td><td>{e(r.fund.region)}</td>"
                 f"<td>{_badge(r.signal)}</td><td>{r.analysis.primary.state if r.analysis.primary else '–'}</td>"
                 f"<td>{f'{p.crv:.1f}' if p else '–'}</td><td>{_pct(r.p_final)}</td>"
                 f"<td>{f'{r.mc_model.mean_r:+.2f} R' if r.mc_model else '–'}</td>"
                 f"<td>{f'{r.size.units:g}' if r.tradeable else '–'}</td>"
                 f"<td>{f'{fmt(r.size.risk_amount, 0)} {cur}' if r.tradeable else '–'}</td></tr>")
    h.append("</table></div>")

    for r in reports:
        f, st, wa, p = r.fund, r.stats, r.analysis, r.decision.plan
        tags = [f.region, f.asset_class, f.segment] + (["Wachstumsmarkt"] if f.em else []) + \
               (["Wachstumsthema"] if f.growth_theme else [])
        h.append(f"<section class='card' id='{e(f.ticker)}'><div class='head'><div><h2>{e(f.ticker)} – {e(f.name)}</h2>"
                 + "".join(f"<span class='tag'>{e(t)}</span>" for t in tags) + f"</div>{_badge(r.signal)}</div>")
        h.append(f"<p><b>{e(SIGNAL_TEXT[r.signal])}.</b> {e(r.decision.reason)}.</p>")
        if r.reasons:
            h.append("<ul>" + "".join(f"<li class='warn'>{e(x)}</li>" for x in r.reasons) + "</ul>")
        h.append("<div class='kpis'>" + "".join(
            f"<div class='kpi'><span>{k}</span><b>{v}</b></div>" for k, v in [
                ("Kurs", f"{fmt(r.close)} {e(r.series.currency)}"), ("ATR (14)", f"{fmt(r.atr)} · {st['atr_pct']:.1%}"),
                ("Volatilität 1J", f"{st['vola_1j']:.0%}"), ("Performance 1J", f"{st['perf_1j']:+.1%}"),
                ("Max. Drawdown 1J", f"{st['mdd_1j']:.1%}"), ("Datenquelle", e(r.series.source))]) + "</div>")
        h.append(_svg_chart(r))

        h.append("<h3>Wellenanalyse</h3>")
        h.append(f"<div class='bar'><i style='width:{wa.p_up:.1%};background:var(--target)'></i>"
                 f"<i style='width:{wa.p_down:.1%};background:var(--stop)'></i>"
                 f"<i style='width:{wa.p_unclear:.1%};background:var(--line)'></i></div>"
                 f"<div class='foot'>steigend {wa.p_up:.0%} · fallend {wa.p_down:.0%} · keine klare Zählung {wa.p_unclear:.0%}</div>")
        h.append("<div class='tbl'><table><tr><th>#</th><th>Szenario</th><th>Gewicht</th><th>Ziele (nah → fern)</th>"
                 "<th>Stop-Niveau</th><th>Invalidierung</th></tr>")
        for i, c in enumerate(wa.counts[:5], 1):
            h.append(f"<tr><td>{i}</td><td>{e(c.description)}</td><td>{c.weight:.0%}</td>"
                     f"<td>{' / '.join(fmt(v) for _, v in c.targets)}</td><td>{fmt(c.stop_ref)}</td>"
                     f"<td>{fmt(c.invalidation)}</td></tr>")
        h.append("</table></div>")
        if wa.primary:
            h.append("<h3>Regelprüfung der Hauptzählung</h3><div class='tbl'><table><tr><th>Prüfung</th><th>Wert</th><th>Score</th></tr>")
            for ch in wa.primary.checks:
                sc = "<span class='ok'>Regel ✔</span>" if ch.hard else f"{ch.score:.2f}"
                h.append(f"<tr><td>{e(ch.name)}</td><td>{e(ch.value)}</td><td>{sc}</td></tr>")
            h.append(f"</table></div><p class='foot'>Qualitätsscore der Zählung: {wa.primary.score:.2f}</p>")

        if p:
            rows = [("Einstieg", fmt(p.entry), p.entry_type), ("Stop", fmt(p.stop),
                    f"{abs(p.entry - p.stop) / p.entry:.1%} · {p.stop_atr:.1f} ATR")]
            rows += [(f"Ziel {i}", fmt(v), f"{n} · {p.side * (v - p.entry) / p.entry:+.1%} · "
                      f"{p.side * (v - p.entry) / p.risk_per_unit:.1f} R") for i, (n, v) in enumerate(p.levels, 1)]
            rows += [("CRV", f"{p.crv:.2f}", f"bis Hauptziel {p.main_target[0]}")]
            h.append("<h3>Chance / Risiko</h3><div class='tbl'><table>" + "".join(
                f"<tr><th>{a}</th><td>{b}</td><td>{e(c)}</td></tr>" for a, b, c in rows) + "</table></div>")
            if r.mc_model:
                c, m = r.calib, r.mc_model
                rows = [("Zufallspfad (kein Wellen-Vorteil)", f"{r.p_random:.0%}", f"E[R] {r.mc_random.mean_r:+.2f}"),
                        ("Szenario-Konsens → Modell", f"{r.p_model:.0%}", f"Edge {r.edge:+.0%}"),
                        ("Walk-Forward-Historie", _pct(c.hit_rate) if c else "–",
                         f"{c.signals} Signale, Basis {_pct(c.baseline_mean)}, Ø {c.avg_r:+.2f} R" if c and c.signals else "keine Signale"),
                        ("<b>Final: Hauptziel vor Stop</b>", f"<b>{r.p_final:.0%}</b>", ""),
                        ("Teilziel 1 vor Stop", f"{m.p_tp1:.0%}", "danach Stop auf Einstand"),
                        ("Stop vor Teilziel 1", f"{m.p_stop_first:.0%}", ""),
                        ("Zeitstopp", f"{m.p_time_exit:.0%}", f"nach {p.time_stop_bars} Tagen"),
                        ("Erwartungswert", f"{m.mean_r:+.2f} R", f"Median {m.median_r:+.2f} R"),
                        ("Spannweite 5 % / 95 %", f"{m.quantile(0.05):+.2f} / {m.quantile(0.95):+.2f} R",
                         f"CVaR 5 %: {m.cvar5:+.2f} R")]
                h.append("<h3>Wahrscheinlichkeiten</h3><div class='tbl'><table>" + "".join(
                    f"<tr><th>{a}</th><td>{b}</td><td>{e(cc)}</td></tr>" for a, b, cc in rows) + "</table></div>")
            if r.size:
                s = r.size
                h.append("<h3>Positionsgröße &amp; Budget</h3>"
                         f"<p><b>{s.units:g} Anteile</b> ≈ {fmt(s.invest, 0)} {cur} ({s.invest / cfg.budget.capital:.1%} des Budgets) · "
                         f"Risiko bis Stop {fmt(s.risk_amount, 0)} {cur} ({s.risk_pct:.2%}) · begrenzt durch <b>{e(s.binding)}</b></p>"
                         "<div class='tbl'><table><tr><th>Grenze</th><th>max. Anteile</th></tr>" + "".join(
                             f"<tr><td>{e(k)}</td><td>{v:,.1f}</td></tr>" for k, v in s.limits.items()) + "</table></div>")
                if s.notes:
                    h.append("<ul>" + "".join(f"<li>{e(x)}</li>" for x in s.notes) + "</ul>")
        if r.exit_rules:
            h.append("<h3>Ausstiegsstrategie</h3><ol>" + "".join(f"<li>{e(x)}</li>" for x in r.exit_rules) + "</ol>")
        if r.chances:
            h.append("<h3>Chancen</h3><ul>" + "".join(f"<li>{e(x)}</li>" for x in r.chances) + "</ul>")
        h.append("<h3>Risiken</h3><ul>" + "".join(f"<li>{e(x)}</li>" for x in r.risks) + "</ul></section>")

    h.append("<section class='card'><h3>Methodik</h3><ul>" + "".join(f"<li>{e(m)}</li>" for m in METHOD)
             + f"</ul><p class='foot'>{e(DISCLAIMER)}</p></section></main></body></html>")
    return "".join(h)
