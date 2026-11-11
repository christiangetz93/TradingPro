"""Übersetzt eine Wellenanalyse in einen konkreten Trade-Plan."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import BotConfig
from .waves import WaveAnalysis, WaveCount


@dataclass
class TradePlan:
    side: int                              # +1 Long, -1 Short
    entry: float
    stop: float
    levels: list[tuple[str, float]]        # TP1, TP2, TP3 (Runner-Ziel)
    main_target: tuple[str, float]
    risk_per_unit: float                   # inkl. Kosten
    reward_per_unit: float                 # bis Hauptziel, nach Kosten
    time_stop_bars: int
    atr: float
    count: WaveCount
    entry_type: str = "Markt"
    notes: list[str] = field(default_factory=list)

    @property
    def crv(self) -> float:
        return self.reward_per_unit / self.risk_per_unit

    @property
    def stop_atr(self) -> float:
        return abs(self.entry - self.stop) / self.atr


@dataclass
class ConditionalSetup:
    """Ein Setup, das erst bei Erreichen einer Zone handelbar wird (Limit-Order)."""
    title: str
    zone: tuple[float, float]
    stop: float
    target: float
    text: str

    @property
    def crv(self) -> float:
        mid = sum(self.zone) / 2
        return abs(self.target - mid) / max(abs(mid - self.stop), 1e-9)


@dataclass
class Decision:
    signal: str                            # KAUFEN / SHORT / WARTEN / BEOBACHTEN / MEIDEN
    reason: str
    plan: TradePlan | None = None
    setup: ConditionalSetup | None = None
    rejections: list[str] = field(default_factory=list)


def build_plan(count: WaveCount, close: float, atr_now: float, cfg: BotConfig,
               side_override: int | None = None) -> tuple[TradePlan | None, str]:
    ex, bud = cfg.exits, cfg.budget
    side = side_override or count.next_dir
    cost = bud.fee_pct + bud.slippage_pct
    entry = close
    stop = count.stop_ref - side * ex.stop_atr_buffer * atr_now
    dist = side * (entry - stop)
    if dist <= 0:
        return None, "Kurs liegt bereits jenseits des Stop-Niveaus der Zählung"
    if dist > ex.max_stop_atr * atr_now:
        return None, (f"Stop wäre {dist / atr_now:.1f} ATR entfernt (> {ex.max_stop_atr:g}) – "
                      "Risiko nicht sinnvoll begrenzbar")
    if dist < ex.min_stop_atr * atr_now:
        stop = entry - side * ex.min_stop_atr * atr_now
        dist = ex.min_stop_atr * atr_now

    risk = dist + cost * (entry + stop)
    remaining = [(n, p) for n, p in count.targets if side * (p - entry) > 0.5 * risk]
    if not remaining:
        return None, "Alle Kursziele der laufenden Welle sind (fast) erreicht – Einstieg zu spät"

    base_name, base_price = count.targets[1]
    if any(p == base_price for _, p in remaining):
        main = (base_name, base_price)
    else:
        main = remaining[-1]
    max_reward = cfg.filters.max_crv_for_probability * risk
    if side * (main[1] - entry) - cost * (entry + main[1]) > max_reward:
        capped = entry + side * max_reward
        main = (f"{main[0]} (gekappt auf {cfg.filters.max_crv_for_probability:g} R)", capped)
        remaining = [(n, p) for n, p in remaining if side * (p - capped) < 0] + [main]

    if len(remaining) == 1:
        t = remaining[0]
        levels = [("Zwischenziel 50 %", entry + 0.5 * (t[1] - entry)), t, t]
    elif len(remaining) == 2:
        levels = [remaining[0], remaining[1], remaining[1]]
    else:
        levels = remaining[:3]
    if main not in levels:
        main = levels[1]

    reward = side * (main[1] - entry) - cost * (entry + main[1])
    if reward <= 0:
        return None, "Kein Gewinnpotenzial nach Kosten"
    remaining_bars = max(count.expected_bars - count.elapsed_bars, count.expected_bars // 2)
    time_stop = max(ex.min_time_stop_bars, int(ex.time_stop_factor * remaining_bars))
    plan = TradePlan(side=side, entry=entry, stop=stop, levels=levels, main_target=main,
                     risk_per_unit=risk, reward_per_unit=reward, time_stop_bars=time_stop,
                     atr=atr_now, count=count)
    return plan, "ok"


def _wave1_len(count: WaveCount) -> float:
    return abs(count.points[1].price - count.points[0].price)


def conditional_setup(count: WaveCount, atr_now: float, cfg: BotConfig) -> ConditionalSetup | None:
    """Kauf- bzw. Verkaufs-Setup für die NÄCHSTE Welle nach der laufenden Korrektur."""
    buf = cfg.exits.stop_atr_buffer * atr_now
    d = count.direction
    if count.state == "W4":
        z = sorted([count.targets[1][1], count.targets[2][1]])
        stop = count.invalidation - d * buf
        entry = sum(z) / 2
        target = entry + d * _wave1_len(count)
        kind = "Kauf" if d == 1 else "Short"
        return ConditionalSetup(
            title=f"{kind}-Setup für Welle 5",
            zone=(z[0], z[1]), stop=stop, target=target,
            text=(f"Welle 4 korrigiert typischerweise 38,2–50 % von Welle 3. Limit-Zone "
                  f"{z[0]:.2f}–{z[1]:.2f}; Stop jenseits des Endes von Welle 1 ({count.invalidation:.2f}), "
                  f"da dort die Zählung ungültig wird. Ziel Welle 5 ≈ Länge von Welle 1."))
    if count.state == "C":
        a_len = abs(count.points[5].price - count.points[6].price)
        z = sorted([count.targets[1][1], count.targets[0][1]])
        stop = count.targets[2][1] - d * buf
        target = count.points[7].price
        kind = "Kauf" if d == 1 else "Short"
        return ConditionalSetup(
            title=f"{kind}-Setup nach Abschluss der ABC-Korrektur",
            zone=(z[0], z[1]), stop=stop, target=target,
            text=(f"Welle C endet häufig bei 0,618–1,0 x A (A = {a_len:.2f}). Danach wird ein neuer "
                  f"Impuls in Trendrichtung erwartet. Erstes Ziel: Ende von Welle B ({target:.2f}). "
                  "Erst nach einer Umkehrbestätigung (z. B. Bruch der C-Abwärtstrendlinie) einsteigen."))
    return None


def decide(wa: WaveAnalysis, close: float, atr_now: float, cfg: BotConfig) -> Decision:
    """Wählt das Szenario und baut den Trade-Plan (ohne Wahrscheinlichkeits-Filter)."""
    f = cfg.filters
    primary = wa.primary
    if primary is None:
        return Decision("MEIDEN", "Keine regelkonforme Elliott-Wellen-Zählung gefunden")
    p_dir = wa.p_up if primary.next_dir > 0 else wa.p_down
    rej: list[str] = []
    if primary.score < f.min_count_score:
        rej.append(f"Qualität der führenden Zählung zu gering ({primary.score:.3f} < {f.min_count_score:.2f})")
    if p_dir < f.min_consensus:
        rej.append(f"Szenarien uneinig: nur {p_dir:.0%} zeigen in Richtung der Hauptzählung "
                   f"(Mindestens {f.min_consensus:.0%})")

    if primary.next_dir < 0 and not cfg.budget.allow_short:
        setup = conditional_setup(primary, atr_now, cfg) if primary.direction == 1 else None
        if setup and not rej:
            return Decision("WARTEN", "Korrektur läuft – Long-Einstieg erst in der Zielzone",
                            setup=setup, rejections=rej)
        return Decision("MEIDEN", "Hauptszenario erwartet fallende Kurse (Short deaktiviert) – "
                        "keine neuen Käufe, bestehende Positionen absichern", rejections=rej)

    plan, why = build_plan(primary, close, atr_now, cfg)
    if plan is None:
        rej.append(why)
    elif plan.crv < f.min_crv:
        rej.append(f"CRV {plan.crv:.2f} unter Minimum {f.min_crv:.1f}")
    signal = "KAUFEN" if primary.next_dir > 0 else "SHORT"
    if rej:
        setup = conditional_setup(primary, atr_now, cfg)
        return Decision("BEOBACHTEN", "Setup erfüllt die Einstiegskriterien (noch) nicht",
                        plan=plan, setup=setup, rejections=rej)
    return Decision(signal, primary.description, plan=plan)
