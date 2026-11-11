"""Komplette Analyse eines Titels: Wellen -> Plan -> Wahrscheinlichkeiten -> Sizing."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibration import CalibResult, blend_probability, walk_forward
from .config import BotConfig
from .data_sources import PriceSeries
from .indicators import annual_volatility, atr, max_drawdown, rsi
from .montecarlo import BootstrapSampler, MCResult, simulate_plan, solve_drift
from .risk import BookState, PositionSize, size_position
from .strategy import Decision, decide
from .universe import Fund
from .waves import WaveAnalysis, analyze_waves


@dataclass
class TitleReport:
    fund: Fund
    series: PriceSeries
    analysis: WaveAnalysis
    decision: Decision
    close: float
    atr: float
    stats: dict[str, float]
    signal: str
    reasons: list[str] = field(default_factory=list)
    edge: float = 0.0
    p_random: float | None = None
    p_model: float | None = None
    p_final: float | None = None
    mc_random: MCResult | None = None
    mc_model: MCResult | None = None
    calib: CalibResult | None = None
    size: PositionSize | None = None
    risks: list[str] = field(default_factory=list)
    chances: list[str] = field(default_factory=list)
    exit_rules: list[str] = field(default_factory=list)

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


def analyze_title(fund: Fund, series: PriceSeries, cfg: BotConfig, book: BookState) -> TitleReport:
    df = series.df
    atr_s = atr(df, cfg.analysis.atr_period)
    rsi_s = rsi(df["close"])
    close = float(df["close"].iloc[-1])
    atr_now = float(atr_s.iloc[-1])
    wa = analyze_waves(df, atr_s, rsi_s, cfg.analysis)
    dec = decide(wa, close, atr_now, cfg)
    rep = TitleReport(fund=fund, series=series, analysis=wa, decision=dec, close=close,
                      atr=atr_now, stats=_stats(df, atr_now), signal=dec.signal,
                      reasons=list(dec.rejections))

    plan = dec.plan
    if plan is not None:
        a = cfg.analysis
        sampler = BootstrapSampler(df, a.mc_return_window, a.mc_block, a.mc_paths,
                                   horizon=plan.time_stop_bars * 2 + 5, seed=a.seed)
        rep.mc_random = simulate_plan(plan, sampler, cfg, 0.0)
        p_dir = wa.p_up if plan.side > 0 else wa.p_down
        p_opp = wa.p_down if plan.side > 0 else wa.p_up
        rep.edge = p_dir - p_opp
        p0 = rep.mc_random.p_main
        k = a.edge_kappa
        rep.p_random = p0
        rep.p_model = p0 + k * rep.edge * (1 - p0) if rep.edge > 0 else p0 * (1 + k * rep.edge)
        if a.calibrate and len(df) > a.calib_min_history + 60:
            rep.calib = walk_forward(df, atr_s, rsi_s, cfg)
        rep.p_final = blend_probability(p0, rep.p_model, rep.calib, a.calib_prior_strength)
        rep.mc_model = solve_drift(plan, sampler, cfg, rep.p_final)
        rep.size = size_position(plan.entry, plan.risk_per_unit, rep.mc_model.mean_r,
                                 rep.mc_model.second_moment, fund.region, book, cfg.budget)

        if rep.signal in ("KAUFEN", "SHORT"):
            f = cfg.filters
            if rep.p_final < f.min_probability:
                rep.reasons.append(f"Trefferwahrscheinlichkeit {rep.p_final:.0%} < Minimum {f.min_probability:.0%}")
            if rep.mc_model.p_tp1 < f.min_tp1_probability:
                rep.reasons.append(f"P(Teilziel 1 vor Stop) {rep.mc_model.p_tp1:.0%} < Minimum {f.min_tp1_probability:.0%}")
            if rep.mc_model.mean_r < f.min_expected_r:
                rep.reasons.append(f"Erwartungswert {rep.mc_model.mean_r:+.2f} R < Minimum {f.min_expected_r:+.2f} R")
            if not rep.size.ok:
                rep.reasons.append(f"Positionsgröße 0 – begrenzt durch: {rep.size.binding}")
                rep.reasons.extend(rep.size.notes)
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
    r.append("Elliott-Wellen-Zählungen sind mehrdeutig. Die Wahrscheinlichkeiten sind Modellgewichte, "
             "keine Garantie; das Szenario 'keine klare Zählung' erhält "
             f"{wa.p_unclear:.0%}.")
    if prim:
        opp = [c for c in wa.counts if c.next_dir != prim.next_dir]
        if opp:
            o = opp[0]
            r.append(f"Gegen-Szenario ({o.weight:.0%}): {o.description}. Gesamtgewicht aller "
                     f"Gegen-Szenarien: {sum(c.weight for c in opp):.0%}.")
        if prim.state in ("W4", "B") or (rep.decision.plan and prim.direction != rep.decision.plan.side):
            r.append("Gegentrend-Trade: Die laufende Welle ist eine Korrektur – diese verlaufen "
                     "unregelmäßiger und sind schwerer zu prognostizieren.")
        if prim.state == "W5":
            r.append("Spätphase des Trends: Welle 5 endet oft abrupt, danach folgt eine ABC-Korrektur "
                     "gegen den gesamten Impuls.")
        if prim.state == "W3":
            r.append("Frühe Welle-3-Zählungen entpuppen sich häufig als Welle C oder als Teil einer "
                     "verlängerten Welle 2 – daher enger Stop unter dem Welle-2-Tief.")
    if f.em:
        r.append("Schwellenland-Risiken: Währungsschwankungen der Lokalwährungen, politische und "
                 "regulatorische Eingriffe, Kapitalverkehrskontrollen, geringere Liquidität in Stressphasen.")
    if f.region == "China":
        r.append("China-spezifisch: Regulierung (z. B. Tech-Sektor), geopolitische Spannungen, "
                 "VIE-Strukturen bei Offshore-Listings.")
    if f.growth_theme:
        r.append("Wachstums-/Themenfonds: hohe Zins- und Bewertungssensitivität, Konzentration auf "
                 "wenige Schwergewichte, Themen-Hypes können lange und tief korrigieren.")
    if f.asset_class == "Anleihen":
        r.append("Anleihen-ETF: dominiert vom Zins- und Kreditrisiko; Elliott-Muster sind bei "
                 "Rentenprodukten oft weniger ausgeprägt.")
    if f.asset_class == "Rohstoffe":
        r.append("Rohstoff-ETP: kein laufender Ertrag; bei Futures-basierten Produkten Rollverluste (Contango).")
    if not rep.fund.ticker.endswith(".DE"):
        r.append("US-gelistete ETFs sind für Privatanleger in der EU wegen fehlendem Basisinformationsblatt "
                 "(PRIIPs) oft nicht kaufbar – ggf. vergleichbaren UCITS-ETF nutzen (Wellenanalyse dann "
                 "auf dem UCITS-Kurs wiederholen).")
    cur = rep.series.currency
    if cur not in ("n/a", "SYN") and cur != cfg.budget.currency:
        r.append(f"Wechselkursrisiko: Kurse in {cur}, Budget in {cfg.budget.currency}.")
    st = rep.stats
    if st["vola_1j"] > 0.30:
        r.append(f"Hohe Volatilität ({st['vola_1j']:.0%} p. a.) – Stops werden häufiger ausgelöst.")
    r.append(f"Maximaler Rückgang der letzten 12 Monate: {st['mdd_1j']:.1%}.")
    r.append("Kurslücken (Gaps) können über den Stop hinweg springen – der tatsächliche Verlust kann "
             "größer sein als geplant.")
    if rep.mc_model is not None:
        r.append(f"Simuliertes Verlustrisiko des Plans: P(Verlust) {rep.mc_model.p_loss:.0%}, "
                 f"5-%-Quantil {rep.mc_model.quantile(0.05):+.2f} R, CVaR(5 %) {rep.mc_model.cvar5:+.2f} R.")
    if rep.calib is not None:
        c = rep.calib
        if c.signals < 5:
            r.append(f"Nur {c.signals} historische Signale – die Wahrscheinlichkeit beruht überwiegend "
                     "auf der Modellannahme, nicht auf Erfahrung mit diesem Titel.")
        elif c.empirical_lift < 0:
            r.append(f"Achtung: Bei diesem Titel lag die Methode historisch UNTER Zufall "
                     f"(Trefferquote {c.hit_rate:.0%} vs. Basis {c.baseline_mean:.0%}).")
    if rep.series.synthetic:
        r.insert(0, "SYNTHETISCHE TESTDATEN – diese Analyse beschreibt keinen realen Markt.")
    return r


def build_chances(rep: TitleReport, cfg: BotConfig) -> list[str]:
    ch: list[str] = []
    plan = rep.decision.plan
    prim = rep.analysis.primary
    if prim and prim.state == "W3" and prim.next_dir == prim.direction:
        ch.append("Welle 3 ist meist die längste und dynamischste Welle – bestes Chance/Risiko im Zyklus.")
    if prim and prim.state == "C":
        ch.append("Mit dem Ende von Welle C ist die Korrektur abgeschlossen – Chance auf einen neuen Impuls.")
    if plan is not None:
        units = rep.size.units if rep.size else 0
        for name, price in plan.levels:
            move = plan.side * (price - plan.entry)
            line = f"Ziel {name}: {fmt(price)} ({move / plan.entry:+.1%}, {move / plan.risk_per_unit:.1f} R)"
            if units:
                line += f" ≈ {fmt(units * move, 0)} {cfg.budget.currency} brutto auf die Gesamtposition"
            ch.append(line)
        if rep.mc_model is not None:
            m = rep.mc_model
            ch.append(f"Erwartungswert des Ausstiegsplans (kalibriert): {m.mean_r:+.2f} R je Trade, "
                      f"Median {m.median_r:+.2f} R, 95-%-Quantil {m.quantile(0.95):+.2f} R.")
            if rep.size and rep.size.ok:
                ch.append(f"In {cfg.budget.currency}: Erwartung {fmt(m.mean_r * rep.size.risk_amount, 0)}, "
                          f"gutes Szenario (95 %) {fmt(m.quantile(0.95) * rep.size.risk_amount, 0)}.")
    s = rep.decision.setup
    if s:
        ch.append(f"{s.title}: Zone {fmt(s.zone[0])}–{fmt(s.zone[1])}, Stop {fmt(s.stop)}, "
                  f"Ziel {fmt(s.target)} (CRV ≈ {s.crv:.1f}). {s.text}")
    return ch


def build_exit_rules(rep: TitleReport, cfg: BotConfig) -> list[str]:
    plan = rep.decision.plan
    if plan is None:
        return []
    ex, bud = cfg.exits, cfg.budget
    cur = bud.currency
    units = rep.size.units if rep.size else 0
    side_word = "unter" if plan.side == 1 else "über"
    c = plan.count
    struct = {"W3": "Ende Welle 2", "W5": "Ende Welle 4", "B": "Ende Welle A",
              "W4": "Ende Welle 3", "A": "Ende Welle 5", "C": "Ende Welle B"}[c.state]
    loss = units * plan.risk_per_unit
    fees = bud.fee_pct + bud.slippage_pct
    be = plan.entry * (1 + plan.side * 2 * fees)
    t1, t2, t3 = plan.levels
    n1 = round(units * ex.tp1_fraction, 4) if bud.allow_fractional_shares else int(units * ex.tp1_fraction)
    n2 = round(units * ex.tp2_fraction, 4) if bud.allow_fractional_shares else int(units * ex.tp2_fraction)
    rules = [
        f"Initial-Stop {fmt(plan.stop)} ({abs(plan.entry - plan.stop) / plan.entry:.1%} bzw. "
        f"{plan.stop_atr:.1f} ATR {side_word} Einstieg): {struct} ({fmt(c.stop_ref)}) ± "
        f"{ex.stop_atr_buffer:g} ATR Puffer. Max. geplanter Verlust: {fmt(loss, 0)} {cur} "
        f"({loss / bud.capital:.2%} des Budgets).",
        f"Teilverkauf 1 bei {fmt(t1[1])} ({t1[0]}): {n1} Anteile ({ex.tp1_fraction:.0%}) verkaufen und "
        f"Stop auf Einstand + Kosten ({fmt(be)}) nachziehen – ab hier ist der Trade risikofrei.",
        f"Teilverkauf 2 bei {fmt(t2[1])} ({t2[0]}): weitere {n2} Anteile ({ex.tp2_fraction:.0%}); "
        f"Stop auf Teilziel 1 ({fmt(t1[1])}) anheben.",
        f"Restposition: Chandelier-Trailing-Stop {ex.trail_atr_mult:g} ATR "
        f"(≈ {fmt(ex.trail_atr_mult * plan.atr)}) vom höchsten Hoch seit Einstieg; Endziel {fmt(t3[1])} ({t3[0]}).",
        f"Zeitstopp: Wird Teilziel 1 nicht innerhalb von {plan.time_stop_bars} Handelstagen erreicht, "
        "wird die Position geschlossen (Welle entwickelt sich nicht wie gezählt).",
        f"Struktur-Exit: Schlusskurs {side_word} {fmt(c.invalidation)} = Zählung ungültig -> sofort raus, "
        "unabhängig von anderen Regeln.",
        f"Warnsignal-Regel: Kippt die tägliche Neuanalyse auf ein Gegen-Szenario (≥ 60 %), wird der "
        f"Trailing-Stop auf {ex.tight_trail_atr_mult:g} ATR verengt.",
        f"Budgetschutz: max. {bud.risk_per_trade:.1%} Risiko je Trade, max. {bud.max_portfolio_heat:.0%} "
        f"offenes Gesamtrisiko, max. {bud.max_cluster_heat:.1%} je Region, ≥ {bud.min_cash_reserve_pct:.0%} "
        f"Cash-Reserve; ab {bud.drawdown_soft:.0%} Depot-Drawdown halbes Risiko, ab "
        f"{bud.drawdown_hard:.0%} keine neuen Trades.",
    ]
    return rules
