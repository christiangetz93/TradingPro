"""Komplette Analyse eines Titels: Wellen -> Plan -> Kosten -> Wahrscheinlichkeiten -> Sizing.

Die Kaufentscheidung folgt exakt dem Regelwerk, das im Backtest geprüft wurde
(strategy.decide + Positionsgröße + Kostenfilter). Wahrscheinlichkeiten und
Monte-Carlo-Verteilungen sind Zusatzinformationen und keine Kaufkriterien.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BotConfig
from .data_sources import PriceSeries, is_close_only
from .execution import EXIT_MODE_TEXT
from .indicators import annual_volatility, atr, max_drawdown, rsi, trend_state
from .montecarlo import BootstrapSampler, MCResult, simulate_plan, solve_drift
from .risk import BookState, PositionSize, size_position
from .setup_stats import SetupStats
from .strategy import Decision, decide
from .universe import Fund
from .waves import WaveAnalysis, WaveContext

TREND_TEXT = {1: "Aufwärtstrend", -1: "Abwärtstrend", 0: "kein klarer Trend"}


@dataclass
class TitleReport:
    fund: Fund
    series: PriceSeries
    analysis: WaveAnalysis
    decision: Decision
    close: float
    atr: float
    trend: int
    stats: dict[str, float]
    signal: str
    reasons: list[str] = field(default_factory=list)
    mc_random: MCResult | None = None
    mc_model: MCResult | None = None
    setup_stat: dict | None = None
    p_tp1: float | None = None
    p_main: float | None = None
    exp_r: float | None = None            # Erwartung in R nach ALLEN Kosten (inkl. Fixgebühren)
    size: PositionSize | None = None
    costs: dict[str, float] = field(default_factory=dict)
    ticker_bt: dict | None = None
    risks: list[str] = field(default_factory=list)
    chances: list[str] = field(default_factory=list)
    exit_rules: list[str] = field(default_factory=list)

    @property
    def p_final(self) -> float | None:
        return self.p_main

    @property
    def tradeable(self) -> bool:
        return self.signal in ("KAUFEN", "SHORT") and self.size is not None and self.size.ok


def _stats(df: pd.DataFrame, atr_now: float) -> dict[str, float]:
    c = df["close"]
    last_year = c.iloc[-252:]
    return {
        "vola_1j": annual_volatility(last_year),
        "mdd_1j": max_drawdown(last_year),
        "perf_1j": float(last_year.iloc[-1] / last_year.iloc[0] - 1),
        "perf_3m": float(c.iloc[-1] / c.iloc[-63] - 1) if len(c) > 63 else float("nan"),
        "atr_pct": atr_now / float(c.iloc[-1]),
    }


def cost_breakdown(plan, size: PositionSize, cfg: BotConfig) -> dict[str, float]:
    """Erwartete Kosten eines Trades (Kauf + alle geplanten Verkäufe)."""
    c = cfg.costs
    value = size.units * plan.entry
    n_exit = max(1, len(plan.legs))
    entry_cost = c.order_cost(value)
    exit_cost = sum(c.order_cost(value / n_exit) for _ in range(n_exit))
    total = entry_cost + exit_cost
    return {"Kauf": entry_cost, "Verkäufe": exit_cost, "Gesamt": total, "Orders": 1 + n_exit,
            "in R": total / size.risk_amount if size.risk_amount > 0 else float("nan"),
            "in % der Position": total / value if value > 0 else float("nan")}


def ticker_backtest(fund: Fund, df: pd.DataFrame, cfg: BotConfig) -> dict | None:
    """Regelwerk auf der eigenen Historie des Titels (Signal-Studie, nach variablen Kosten)."""
    from .backtest import elliott_candidates, signal_study, trade_stats
    if len(df) < 400:
        return None
    cands = {fund.ticker: elliott_candidates(df, cfg, 0, len(df))}
    trades = signal_study({fund.ticker: df}, cands, cfg)
    st = trade_stats(trades, "r_var")
    st["Zeitraum"] = f"{df.index[0]:%Y}–{df.index[-1]:%Y}"
    return st


def analyze_title(fund: Fund, series: PriceSeries, cfg: BotConfig, book: BookState,
                  stats: SetupStats | None = None) -> TitleReport:
    df = series.df
    atr_s = atr(df, cfg.analysis.atr_period)
    rsi_s = rsi(df["close"])
    trend = int(trend_state(df["close"]).iloc[-1])
    close = float(df["close"].iloc[-1])
    atr_now = float(atr_s.iloc[-1])
    wa = WaveContext(df, atr_s, rsi_s, cfg.analysis).analyze()
    dec = decide(wa, close, atr_now, cfg, trend)
    rep = TitleReport(fund=fund, series=series, analysis=wa, decision=dec, close=close, atr=atr_now,
                      trend=trend, stats=_stats(df, atr_now), signal=dec.signal, reasons=list(dec.rejections))

    plan = dec.plan
    if plan is not None:
        a, f = cfg.analysis, cfg.filters
        sampler = BootstrapSampler(df, a.mc_return_window, a.mc_block, a.mc_paths,
                                   horizon=plan.time_stop_bars * 2 + 5, seed=a.seed)
        rep.mc_random = simulate_plan(plan, sampler, cfg, 0.0)
        n0 = cfg.strategy.stats_prior_n
        rep.setup_stat = stats.describe(plan.setup_key, n0) if stats else None
        if rep.setup_stat:
            n = rep.setup_stat["Trades"]
            rep.p_tp1 = (n0 * rep.mc_random.p_tp1 + n * rep.setup_stat["P(Teilziel 1)"]) / (n0 + n)
            rep.p_main = (n0 * rep.mc_random.p_main + n * rep.setup_stat["P(Hauptziel)"]) / (n0 + n)
            mean_r, m2, _ = stats.expectancy(plan.setup_key, n0)
        else:
            rep.p_tp1, rep.p_main = rep.mc_random.p_tp1, rep.mc_random.p_main
            mean_r, m2 = rep.mc_random.mean_r, rep.mc_random.second_moment
        rep.mc_model = solve_drift(plan, sampler, cfg, rep.p_main)
        kelly_mean = mean_r if (stats is not None and rep.setup_stat) else 1.0
        kelly_m2 = m2 if (stats is not None and rep.setup_stat) else 1.0
        rep.size = size_position(plan.entry, plan.risk_per_unit, kelly_mean, kelly_m2, fund.region, book,
                                 cfg.budget, costs=cfg.costs, n_orders=1 + max(1, len(plan.legs)))
        if rep.size.ok:
            rep.costs = cost_breakdown(plan, rep.size, cfg)
            fixed_r = rep.size.fixed_fees / rep.size.risk_amount if rep.size.risk_amount else 0.0
            rep.exp_r = mean_r - fixed_r
        if a.ticker_backtest:
            rep.ticker_bt = ticker_backtest(fund, df, cfg)

        if rep.signal in ("KAUFEN", "SHORT"):
            if not rep.size.ok:
                rep.reasons.append(f"Positionsgröße 0 – begrenzt durch: {rep.size.binding}")
                rep.reasons.extend(rep.size.notes)
            elif f.max_cost_r > 0 and rep.costs.get("in R", 0) > f.max_cost_r:
                rep.reasons.append(f"Kosten {rep.costs['in R']:.0%} des Risikos > Maximum {f.max_cost_r:.0%} – "
                                   "Stop zu eng bzw. Position zu klein für die Gebühren")
            if cfg.strategy.use_setup_stats and rep.size.ok:
                if rep.exp_r is None or rep.exp_r < f.min_expected_r:
                    rep.reasons.append(f"Erwartungswert dieses Setup-Typs zu gering ({(rep.exp_r or 0):+.2f} R)")
                exp_net = (rep.exp_r or 0) * rep.size.risk_amount
                if exp_net < max(f.min_net_profit, f.fee_multiple * rep.costs.get("Gesamt", 0)):
                    rep.reasons.append(f"Erwarteter Nettogewinn {exp_net:.0f} € zu klein im Verhältnis zu den Gebühren")
            if rep.reasons:
                rep.signal = "BEOBACHTEN"

    rep.risks = build_risks(rep, cfg)
    rep.chances = build_chances(rep, cfg)
    rep.exit_rules = build_exit_rules(rep, cfg)
    return rep


# ---------------------------------------------------------------------------
# Textbausteine
# ---------------------------------------------------------------------------

def fmt(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def build_risks(rep: TitleReport, cfg: BotConfig) -> list[str]:
    r: list[str] = []
    f, wa = rep.fund, rep.analysis
    prim = wa.primary
    r.append("Methodik-Risiko: Im Backtest auf echten Kursdaten (17 Instrumente mit echten Tageskursen, 1990–2021) "
             "lag die Trefferquote der Elliott-Richtungsprognose bei ≈ 50 % (Münzwurf-Niveau, 48–52 % je nach Zeitraum). Out-of-Sample "
             "(2011–2021) lag das Regelwerk nach Kosten nur bei ≈ +0,5 % pro Jahr – auf dem Niveau zufälliger "
             "Einstiege und weit unter Kaufen & Halten. Ein positives Ergebnis ist damit NICHT belegt.")
    if is_close_only(rep.series.df):
        r.append("Datenwarnung: Diese Kursreihe enthält nur Schlusskurse (keine echten Tageshochs/-tiefs). Stops, die "
                 "im Tagesverlauf gerissen würden, werden nicht erkannt – Simulation und Backtest sind zu optimistisch.")
    r.append(f"Zählungen sind mehrdeutig; das Szenario 'keine klare Zählung' erhält {wa.p_unclear:.0%}.")
    if prim:
        opp = [c for c in wa.counts if c.next_dir != prim.next_dir]
        if opp:
            r.append(f"Gegen-Szenario ({opp[0].weight:.0%}): {opp[0].description}.")
        if prim.state in ("W4", "A", "C") and prim.next_dir != prim.direction:
            r.append("Rücksetzer-Kauf: Die Korrektur kann tiefer laufen als erwartet; der Stop liegt deshalb an der "
                     "Elliott-Invalidierung und nicht direkt unter dem Kurs.")
        if prim.state == "W5" and prim.next_dir == prim.direction:
            r.append("Spätphase des Impulses: Welle 5 endet oft abrupt, danach folgt eine Korrektur.")
    if rep.trend != 1:
        r.append(f"Übergeordneter Trend: {TREND_TEXT[rep.trend]} – Long-Einstiege sind nur bei Aufwärtstrend vorgesehen.")
    if f.em:
        r.append("Schwellenland-Risiken: Währungsschwankungen, politische/regulatorische Eingriffe, "
                 "Kapitalverkehrskontrollen, geringere Liquidität und breitere Spreads in Stressphasen.")
    if f.region == "China":
        r.append("China-spezifisch: Regulierung, geopolitische Spannungen, VIE-Strukturen bei Offshore-Listings.")
    if f.growth_theme:
        r.append("Wachstums-/Themenfonds: hohe Zins- und Bewertungssensitivität, Konzentration auf wenige Werte.")
    if f.asset_class == "Anleihen":
        r.append("Anleihen-ETF: dominiert von Zins- und Kreditrisiko; Wellenmuster weniger ausgeprägt.")
    if f.asset_class == "Rohstoffe":
        r.append("Rohstoff-ETP: kein laufender Ertrag; bei Futures-basierten Produkten Rollverluste (Contango).")
    if not rep.fund.ticker.endswith(".DE"):
        r.append("US-gelistete ETFs sind für EU-Privatanleger oft nicht kaufbar (PRIIPs) – ggf. UCITS-Pendant "
                 "nutzen und die Analyse auf dessen Kurs wiederholen.")
    cur = rep.series.currency
    if cur not in ("n/a", "SYN") and cur != cfg.budget.currency:
        r.append(f"Wechselkursrisiko: Kurse in {cur}, Budget in {cfg.budget.currency}.")
    st = rep.stats
    if st["vola_1j"] > 0.30:
        r.append(f"Hohe Volatilität ({st['vola_1j']:.0%} p. a.) – Stops werden häufiger ausgelöst.")
    r.append(f"Maximaler Rückgang der letzten 12 Monate: {st['mdd_1j']:.1%}.")
    r.append("Kurslücken (Gaps) können über den Stop springen – der Verlust kann größer sein als geplant "
             "(im Backtest berücksichtigt: Ausführung dann zum Eröffnungskurs).")
    if rep.mc_model is not None:
        r.append(f"Simulierte Verlustseite des Plans: P(Verlust) {rep.mc_model.p_loss:.0%}, 5-%-Quantil "
                 f"{rep.mc_model.quantile(0.05):+.2f} R, CVaR(5 %) {rep.mc_model.cvar5:+.2f} R.")
    bt = rep.ticker_bt
    if bt and bt.get("Trades", 0) and bt["Ø R"] < 0:
        r.append(f"Achtung: Auf der eigenen Historie dieses Titels war das Regelwerk negativ "
                 f"({bt['Trades']} Trades, Ø {bt['Ø R']:+.2f} R).")
    if rep.series.synthetic:
        r.insert(0, "SYNTHETISCHE TESTDATEN – diese Analyse beschreibt keinen realen Markt.")
    return r


def build_chances(rep: TitleReport, cfg: BotConfig) -> list[str]:
    ch: list[str] = []
    plan = rep.decision.plan
    if plan is not None:
        units = rep.size.units if rep.size else 0
        for name, price in plan.levels:
            move = plan.side * (price - plan.entry)
            line = f"Kursziel {name}: {fmt(price)} ({move / plan.entry:+.1%}, {move / plan.risk_per_unit:.1f} R)"
            if units:
                line += f" ≈ {fmt(units * move, 0)} {cfg.budget.currency} brutto auf die Gesamtposition"
            ch.append(line)
        if rep.setup_stat:
            s = rep.setup_stat
            ch.append(f"Historische Bilanz des Setup-Typs „{plan.setup_key}“: {s['Trades']} Trades, Trefferquote "
                      f"{s['Trefferquote']:.0%}, Ø {s['Ø R']:+.2f} R nach Spread/Slippage (t-Wert {s['t-Wert']:.1f}).")
        if rep.exp_r is not None and rep.size and rep.size.ok:
            ch.append(f"Erwartung nach allen Kosten: {rep.exp_r:+.2f} R ≈ {fmt(rep.exp_r * rep.size.risk_amount, 0)} "
                      f"{cfg.budget.currency} je Trade (geschrumpfter Mittelwert, keine Garantie).")
    s = rep.decision.setup
    if s:
        ch.append(f"{s.title}: Zone {fmt(s.zone[0])}–{fmt(s.zone[1])}, Stop {fmt(s.stop)}, Ziel {fmt(s.target)} "
                  f"(CRV ≈ {s.crv:.1f}). {s.text}")
    return ch


def build_exit_rules(rep: TitleReport, cfg: BotConfig) -> list[str]:
    plan = rep.decision.plan
    if plan is None:
        return []
    ex, bud = cfg.exits, cfg.budget
    cur = bud.currency
    units = rep.size.units if rep.size else 0
    side_word = "unter" if plan.side == 1 else "über"
    loss = rep.size.risk_amount if rep.size else 0.0
    rules = [f"Ausstiegsmodus: {EXIT_MODE_TEXT.get(plan.exit_mode, plan.exit_mode)}.",
             f"Initial-Stop {fmt(plan.stop)} ({abs(plan.entry - plan.stop) / plan.entry:.1%} bzw. {plan.stop_atr:.1f} ATR "
             f"{side_word} Einstieg) an der Elliott-Invalidierung ({fmt(plan.invalidation)}) ± {ex.stop_atr_buffer:g} ATR. "
             f"Max. geplanter Verlust inkl. aller Kosten: {fmt(loss, 0)} {cur} ({loss / bud.capital:.2%} des Budgets)."]
    if plan.exit_mode == "trail":
        rules.append(f"Chandelier-Trailing-Stop ab Einstieg: höchstes Hoch seit Kauf minus {ex.trail_atr_mult:g} ATR "
                     f"(aktuell ≈ {fmt(ex.trail_atr_mult * plan.atr)}). Der Stop wird nur nachgezogen, nie gesenkt; "
                     "Gewinne laufen, bis der Trend dreht. Die Kursziele dienen nur zur Orientierung.")
        rules.append(f"Maximale Haltedauer: {plan.time_stop_bars} Handelstage (≈ 1,5 x erwartete Wellendauer), "
                     "danach Verkauf zum Schlusskurs.")
    else:
        for i, (name, price, frac) in enumerate(plan.legs, 1):
            n = units * frac if bud.allow_fractional_shares else int(units * frac)
            rules.append(f"Teilverkauf {i} bei {fmt(price)} ({name}): {frac:.0%} der Position (≈ {n:g} Anteile).")
        if plan.exit_mode == "scale_be":
            rules.append("Nach Teilverkauf 1: Stop auf Einstand + Kosten – ab dann ist der Trade risikofrei.")
        if plan.exit_mode in ("scale", "scale_be", "half"):
            rules.append(f"Restposition mit Trailing-Stop {ex.trail_atr_mult:g} ATR vom höchsten Hoch.")
        rules.append(f"Zeitstopp: Wird Teilziel 1 nicht innerhalb von {plan.time_stop_bars} Handelstagen erreicht, "
                     "wird verkauft.")
    rules.append(f"Struktur-Exit: Schlusskurs {side_word} {fmt(plan.invalidation)} = Wellenzählung ungültig -> sofort raus.")
    if rep.costs:
        c = rep.costs
        rules.append(f"Gebühren eingeplant: {c['Orders']:.0f} Orders, ≈ {fmt(c['Gesamt'])} {cur} gesamt "
                     f"({c['in % der Position']:.2%} der Position, {c['in R']:.0%} des Risikos). Trades, bei denen die "
                     f"Kosten mehr als {cfg.filters.max_cost_r:.0%} des Risikos ausmachen, werden nicht eröffnet.")
    rules.append(f"Budgetschutz: max. {bud.risk_per_trade:.1%} Risiko je Trade inkl. Gebühren, max. "
                 f"{bud.max_portfolio_heat:.0%} offenes Gesamtrisiko, max. {bud.max_cluster_heat:.1%} je Region bzw. "
                 f"korreliertem Cluster, max. {bud.max_positions} Positionen, ≥ {bud.min_cash_reserve_pct:.0%} Cash; "
                 f"ab {bud.drawdown_soft:.0%} Depot-Drawdown halbes Risiko, ab {bud.drawdown_hard:.0%} keine neuen Trades.")
    return rules
