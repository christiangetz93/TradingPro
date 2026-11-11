"""Erzeugt Markdown- und HTML-Reports (HTML mit SVG-Charts, ohne externe Abhängigkeiten)."""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np

from .analyzer import TREND_TEXT, TitleReport, fmt
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
    "Pivots: ZigZag auf drei Wellengraden (Schwelle = 2,5 / 4 / 6 x ATR), kausal – jeder Pivot wird erst ab "
    "seiner Bestätigung verwendet.",
    "Zählungen: Muster W3/W4/W5/A/B/C auf- und abwärts, die auf dem letzten bestätigten Pivot enden. Harte Regeln "
    "(W2 < 100 % von W1, W3 nie die kürzeste, keine W4/W1-Überlappung – auch in der laufenden Welle, B ≤ 138,2 % von A) "
    "führen zum Verwerfen; Fibonacci-Richtlinien, Alternation, RSI-Divergenz und Volumen ergeben den Qualitätsscore.",
    "Einstiege (v2): Rücksetzer-Käufe in Korrekturen (W4, A, C) einer Aufwärtsstruktur sowie Impuls-Einstiege (W3/W5) – "
    "beides nur bei intaktem Aufwärtstrend (Kurs über steigender 200-Tage-Linie). Ausführung zum nächsten Eröffnungskurs.",
    "Ausstieg (v2): Stop an der Elliott-Invalidierung, Chandelier-Trailing-Stop 4 ATR ab Einstieg, Struktur-Exit, "
    "maximale Haltedauer.",
    "Kosten: feste und prozentuale Ordergebühren, Spread und Slippage für jede Order; Trades mit Kosten über 15 % des "
    "Risikos werden nicht eröffnet.",
    "Wahrscheinlichkeiten: Bayes-Mischung aus Zufallspfad-Simulation (Block-Bootstrap ohne Drift) und der gemessenen "
    "Trefferquote des Setup-Typs im Backtest auf echten Daten.",
    "Positionsgröße: Minimum aus fixem Risiko (inkl. Gebühren), Bruchteil-Kelly, Portfolio-Heat, Regionen-/"
    "Korrelations-Heat, Positionslimit und Cash-Reserve; Drawdown-Bremse.",
    "Validierung (17 echte Instrumente mit Tageshochs/-tiefs): Regeln auf 1990–2010 gewählt (CAGR +2,5 %, max. "
    "Drawdown −8,4 %), Out-of-Sample 2011–2021 nur ≈ +0,5 % pro Jahr nach Kosten – gleichauf mit zufälligen "
    "Einstiegen und weit unter Kaufen & Halten (S&P 500: +14,1 %). Die Elliott-Richtungsprognose traf in ≈ 50 % "
    "der Fälle (Münzwurf-Niveau). Belegt ist der Schutz (Kosten-/Risikofilter, geringere Drawdowns), nicht ein Ertragsvorteil.",
]


def _pct(x: float | None) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.0%}"


def probability_rows(r: TitleReport) -> list[tuple[str, str, str]]:
    p, m0, m = r.decision.plan, r.mc_random, r.mc_model
    if p is None or m0 is None:
        return []
    rows = [("Zufallspfad (Benchmark ohne Prognose)", f"{m0.p_tp1:.0%} / {m0.p_main:.0%}",
             f"P(1. Ziel) / P(Hauptziel) vor Stop · E[R] {m0.mean_r:+.2f}")]
    s = r.setup_stat
    if s:
        rows.append((f"Backtest Setup-Typ „{p.setup_key}“", f"{s['P(Teilziel 1)']:.0%} / {s['P(Hauptziel)']:.0%}",
                     f"{s['Trades']} Trades auf echten Daten · Treffer {s['Trefferquote']:.0%} · Ø {s['Ø R']:+.2f} R"))
    else:
        rows.append(("Backtest Setup-Typ", "–", "keine Statistik vorhanden"))
    bt = r.ticker_bt
    if bt and bt.get("Trades"):
        rows.append((f"Backtest dieses Titels ({bt['Zeitraum']})", f"{bt['TP1 erreicht']:.0%} erreicht 1. Ziel",
                     f"{bt['Trades']} Trades · Treffer {bt['Trefferquote']:.0%} · Ø {bt['Ø R']:+.2f} R · PF {bt['Profitfaktor']:.2f}"))
    rows.append(("Kombiniert (gewichtete Mischung)", f"{_pct(r.p_tp1)} / {_pct(r.p_main)}",
                 "geschätzte Wahrscheinlichkeit, 1. Ziel bzw. Hauptziel vor dem Stop zu erreichen"))
    if r.exp_r is not None:
        rows.append(("Erwartungswert nach allen Kosten", f"{r.exp_r:+.2f} R",
                     "geschrumpfter Mittelwert des Setup-Typs (Backtest)" if s else
                     "nur Zufallspfad-Simulation – keine Backtest-Statistik für diesen Setup-Typ"))
    if m is not None:
        rows.append(("Simulierte Spannweite 5 % / 95 %", f"{m.quantile(0.05):+.2f} / {m.quantile(0.95):+.2f} R",
                     f"P(Verlust) {m.p_loss:.0%} · CVaR 5 % {m.cvar5:+.2f} R"))
    return rows


def cost_rows(r: TitleReport, cfg: BotConfig) -> list[tuple[str, str, str]]:
    c = r.costs
    if not c:
        return []
    cur = cfg.budget.currency
    k = cfg.costs
    return [("Kauf", f"{fmt(c['Kauf'])} {cur}", f"Gebühr {fmt(k.broker_fee(r.size.invest))} {cur} + Spread/Slippage"),
            ("Verkäufe", f"{fmt(c['Verkäufe'])} {cur}", f"{c['Orders'] - 1:.0f} Order(s)"),
            ("Gesamt", f"{fmt(c['Gesamt'])} {cur}", f"{c['in % der Position']:.2%} der Position · {c['in R']:.0%} des Risikos")]


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
            f"{f'{r.exp_r:+.2f}' if r.exp_r is not None else '–'} | "
            f"{f'{r.size.units:g}' if r.size and r.tradeable else '–'} | "
            f"{f'{fmt(r.size.risk_amount, 0)} {cur}' if r.size and r.tradeable else '–'} |")
    for r in reports:
        out += ["", "---", "", f"## {r.fund.ticker} – {r.fund.name}", ""]
        out.append(f"**Signal: {r.signal}** – {SIGNAL_TEXT[r.signal]}. {r.decision.reason}.")
        for why in r.reasons:
            out.append(f"- ⚠ {why}")
        st = r.stats
        out += ["", f"Trend: {TREND_TEXT[r.trend]} · Kurs {fmt(r.close)} ({r.series.currency}, {r.series.source}) · ATR {fmt(r.atr)} "
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
            pr = probability_rows(r)
            if pr:
                out += ["", "### Wahrscheinlichkeiten & Erwartung", "", "| | Wert | Erläuterung |", "|---|---|---|"]
                out += [f"| {a} | {b} | {c} |" for a, b, c in pr]
            if r.size:
                s = r.size
                out += ["", "### Positionsgröße, Kosten & Budget", "",
                        f"- {s.units:g} Anteile ≈ {fmt(s.invest, 0)} {cur} "
                        f"({s.invest / cfg.budget.capital:.1%} des Budgets), Risiko inkl. Gebühren {fmt(s.risk_amount, 0)} {cur} "
                        f"({s.risk_pct:.2%}) – begrenzt durch **{s.binding}**"]
                out += [f"- Kosten {a}: {b} ({c})" for a, b, c in cost_rows(r, cfg)]
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
             "<th>CRV</th><th>P(Ziel)</th><th>E[R] netto</th><th>Stück</th><th>Risiko</th></tr>")
    for r in reports:
        p = r.decision.plan
        h.append(f"<tr><td><a href='#{e(r.fund.ticker)}'>{e(r.fund.ticker)}</a></td><td>{e(r.fund.region)}</td>"
                 f"<td>{_badge(r.signal)}</td><td>{r.analysis.primary.state if r.analysis.primary else '–'}</td>"
                 f"<td>{f'{p.crv:.1f}' if p else '–'}</td><td>{_pct(r.p_final)}</td>"
                 f"<td>{f'{r.exp_r:+.2f} R' if r.exp_r is not None else '–'}</td>"
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
                ("Trend (200 Tage)", TREND_TEXT[r.trend]), ("Kurs", f"{fmt(r.close)} {e(r.series.currency)}"),
                ("ATR (14)", f"{fmt(r.atr)} · {st['atr_pct']:.1%}"),
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
            pr = probability_rows(r)
            if pr:
                h.append("<h3>Wahrscheinlichkeiten &amp; Erwartung</h3><div class='tbl'><table>" + "".join(
                    f"<tr><th>{e(a)}</th><td>{e(b)}</td><td>{e(cc)}</td></tr>" for a, b, cc in pr) + "</table></div>")
            if r.size:
                s = r.size
                h.append("<h3>Positionsgröße, Kosten &amp; Budget</h3>"
                         f"<p><b>{s.units:g} Anteile</b> ≈ {fmt(s.invest, 0)} {cur} ({s.invest / cfg.budget.capital:.1%} des Budgets) · "
                         f"Risiko bis Stop inkl. Gebühren {fmt(s.risk_amount, 0)} {cur} ({s.risk_pct:.2%}) · begrenzt durch <b>{e(s.binding)}</b></p>")
                cr = cost_rows(r, cfg)
                if cr:
                    h.append("<div class='tbl'><table>" + "".join(
                        f"<tr><th>Kosten {e(a)}</th><td>{e(b)}</td><td>{e(cc)}</td></tr>" for a, b, cc in cr) + "</table></div>")
                h.append("<div class='tbl'><table><tr><th>Grenze</th><th>max. Anteile</th></tr>" + "".join(
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
