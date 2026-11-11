"""Übersetzt eine Wellenanalyse in einen konkreten Trade-Plan.

Zwei Signalarten:

* **Markt-Signal** – die laufende Welle zeigt in Trade-Richtung (z. B. Welle 3
  oder 5 aufwärts): Einstieg zum nächsten Eröffnungskurs.
* **Rücksetzer-Setup** – eine Korrektur läuft (Welle 4 oder C) und wird
  voraussichtlich in einer Fibonacci-Zone enden. Einstieg erst in dieser Zone
  (Limit) bzw. nach einer Umkehrbestätigung. Ziele: Welle 5 bzw. neuer Impuls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import BotConfig
from .execution import make_legs
from .waves import WaveAnalysis, WaveCount


def trend_tag(trend: int, side: int) -> str:
    if trend == 0:
        return "neutral"
    return "mit Trend" if trend == side else "gegen Trend"


@dataclass
class TradePlan:
    side: int                              # +1 Long, -1 Short
    entry: float
    stop: float
    levels: list[tuple[str, float]]        # TP1, TP2, TP3
    main_target: tuple[str, float]
    risk_per_unit: float                   # inkl. variabler Kosten
    reward_per_unit: float                 # bis Hauptziel, nach variablen Kosten
    time_stop_bars: int
    atr: float
    count: WaveCount
    invalidation: float
    kind: str = "market"                   # "market" | "setup"
    setup_key: str = ""
    entry_type: str = "Markt (nächste Eröffnung)"
    exit_mode: str = "scale_be"
    legs: list[list] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def crv(self) -> float:
        return self.reward_per_unit / self.risk_per_unit

    @property
    def stop_atr(self) -> float:
        return abs(self.entry - self.stop) / self.atr

    def to_dict(self) -> dict:
        return {"stop": self.stop, "legs": [list(x) for x in self.legs], "invalidation": self.invalidation,
                "atr": self.atr, "time_stop_bars": self.time_stop_bars, "exit_mode": self.exit_mode,
                "meta": {"setup_key": self.setup_key, "state": self.count.state,
                         "direction": self.count.direction, "degree": self.count.degree,
                         "crv": self.crv, "kind": self.kind}}


@dataclass
class ConditionalSetup:
    """Rücksetzer-Setup: wird erst bei Erreichen einer Zone handelbar."""
    title: str
    side: int
    zone: tuple[float, float]              # (nah, fern) aus Sicht des aktuellen Kurses
    stop: float
    cancel_level: float                    # vor Zonen-Test überschritten -> Setup hinfällig
    expiry_bars: int
    geometry: dict                         # Wellenpunkte für die Zielberechnung nach Ausführung
    text: str
    count: WaveCount
    setup_key: str = ""

    @property
    def limit(self) -> float:
        return sum(self.zone) / 2

    @property
    def target(self) -> float:
        return setup_levels(self, self.limit)[1][1]

    @property
    def crv(self) -> float:
        return abs(self.target - self.limit) / max(abs(self.limit - self.stop), 1e-9)


def setup_levels(setup: ConditionalSetup, fill: float) -> list[tuple[str, float]]:
    """Kursziele eines Setups relativ zum tatsächlichen Einstiegskurs."""
    g, s = setup.geometry, setup.side
    if g["type"] == "W4":
        w1, w3 = g["W1"], g["W3"]
        cap = fill + s * w3 if w3 < w1 else None        # Welle 3 darf nicht die kürzeste werden
        cands = [("Hoch von Welle 3", g["P3"]), ("W5 = 0,618 x W1", fill + s * 0.618 * w1),
                 ("W5 = 1,0 x W1", fill + s * 1.0 * w1), ("W5 = 0,618 x (0–3)", fill + s * 0.618 * g["L03"])]
        if cap is not None:
            cands = [(n, p if s * p <= s * cap else cap) for n, p in cands]
    else:  # C
        cands = [("Ende Welle B", g["P7"]), ("Start Welle A", g["P5"]),
                 ("neuer Impuls 1,618 x (0–5)", fill + s * 1.618 * g["L05"])]
    cands = sorted(cands, key=lambda x: s * x[1])
    out: list[tuple[str, float]] = []
    for n, p in cands:
        if s * (p - fill) > 0 and (not out or abs(p - out[-1][1]) > 1e-9):
            out.append((n, p))
    while len(out) < 3:
        out.append(out[-1] if out else ("Ziel", fill + s * abs(fill - setup.stop) * 2))
    return out[:3]


@dataclass
class Decision:
    signal: str                            # KAUFEN / SHORT / WARTEN / BEOBACHTEN / MEIDEN
    reason: str
    plan: TradePlan | None = None
    setup: ConditionalSetup | None = None
    rejections: list[str] = field(default_factory=list)
    trend: int = 0


def build_plan(count: WaveCount, close: float, atr_now: float, cfg: BotConfig,
               side_override: int | None = None) -> tuple[TradePlan | None, str]:
    ex = cfg.exits
    side = side_override or count.next_dir
    v = cfg.costs.variable_rate()
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

    risk = dist + v * (entry + stop)
    remaining = [(n, p) for n, p in count.targets if side * (p - entry) > 0.5 * risk]
    if not remaining:
        return None, "Alle Kursziele der laufenden Welle sind (fast) erreicht – Einstieg zu spät"

    base_name, base_price = count.targets[1]
    main = (base_name, base_price) if any(p == base_price for _, p in remaining) else remaining[-1]
    max_reward = cfg.filters.max_crv_for_probability * risk
    if side * (main[1] - entry) - v * (entry + main[1]) > max_reward:
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

    reward = side * (main[1] - entry) - v * (entry + main[1])
    if reward <= 0:
        return None, "Kein Gewinnpotenzial nach Kosten"
    remaining_bars = max(count.expected_bars - count.elapsed_bars, count.expected_bars // 2)
    time_stop = max(ex.min_time_stop_bars, int(ex.time_stop_factor * remaining_bars))
    plan = TradePlan(side=side, entry=entry, stop=stop, levels=levels, main_target=main,
                     risk_per_unit=risk, reward_per_unit=reward, time_stop_bars=time_stop,
                     atr=atr_now, count=count, invalidation=count.invalidation, exit_mode=ex.mode)
    plan.legs, plan.exit_mode = make_legs(levels, main, ex.mode, cfg, None)
    return plan, "ok"


def plan_from_setup(setup: ConditionalSetup, fill: float, atr_now: float, cfg: BotConfig) -> TradePlan:
    """Erzeugt den Trade-Plan, wenn ein Setup ausgeführt wird (Ziele relativ zum Einstieg)."""
    v = cfg.costs.variable_rate()
    s = setup.side
    levels = setup_levels(setup, fill)
    main = levels[1]
    risk = abs(fill - setup.stop) + v * (fill + setup.stop)
    reward = s * (main[1] - fill) - v * (fill + main[1])
    plan = TradePlan(side=s, entry=fill, stop=setup.stop, levels=levels, main_target=main,
                     risk_per_unit=risk, reward_per_unit=reward,
                     time_stop_bars=max(cfg.exits.min_time_stop_bars, setup.geometry["time_bars"]),
                     atr=atr_now, count=setup.count, invalidation=setup.stop, kind="setup",
                     setup_key=setup.setup_key, entry_type="Setup", exit_mode=cfg.exits.mode)
    plan.legs, plan.exit_mode = make_legs(levels, main, cfg.exits.mode, cfg, None)
    return plan


def conditional_setup(count: WaveCount, atr_now: float, cfg: BotConfig) -> ConditionalSetup | None:
    """Rücksetzer-Setup für die Welle NACH der laufenden Korrektur (Ende W4 bzw. Ende C)."""
    buf = cfg.exits.stop_atr_buffer * atr_now
    d = count.direction
    pts = count.points
    p = [x.price for x in pts]
    ix = [x.idx for x in pts]
    kind = "Kauf" if d == 1 else "Short"
    if count.state == "W4":
        w1, w3 = abs(p[1] - p[0]), abs(p[3] - p[2])
        zone = (p[3] - d * 0.382 * w3, p[3] - d * 0.5 * w3)
        if d * zone[1] <= d * p[1]:
            zone = (zone[0], p[1] + d * 0.1 * w1)
        geo = {"type": "W4", "P0": p[0], "P1": p[1], "P3": p[3], "W1": w1, "W3": w3,
               "L03": abs(p[3] - p[0]), "time_bars": int(cfg.exits.time_stop_factor * max(5, ix[1] - ix[0]))}
        expiry = int(cfg.strategy.setup_expiry_factor * max(count.expected_bars, 5))
        return ConditionalSetup(
            title=f"{kind}-Setup für Welle 5", side=d, zone=zone, stop=p[1] - d * buf,
            cancel_level=p[3], expiry_bars=expiry, geometry=geo, count=count,
            text=(f"Welle 4 korrigiert typischerweise 38,2–50 % von Welle 3: Zone {zone[0]:.2f}–{zone[1]:.2f}. "
                  f"Stop jenseits des Endes von Welle 1 ({p[1]:.2f}), dort wird die Zählung ungültig. "
                  f"Abbruch, falls der Kurs vorher über das Hoch von Welle 3 ({p[3]:.2f}) läuft."))
    if count.state == "C":
        a_len = abs(p[5] - p[6])
        zone = (p[7] - d * 0.618 * a_len, p[7] - d * 1.0 * a_len)
        stop_lvl = p[7] - d * 1.618 * a_len
        if d * stop_lvl < d * p[0]:
            stop_lvl = p[0]                        # nie tiefer als der Start des Impulses
        if d * zone[1] <= d * stop_lvl:
            return None
        geo = {"type": "C", "P5": p[5], "P7": p[7], "L05": abs(p[5] - p[0]),
               "time_bars": int(cfg.exits.time_stop_factor * max(5, ix[6] - ix[5]))}
        expiry = int(cfg.strategy.setup_expiry_factor * max(count.expected_bars, 5))
        return ConditionalSetup(
            title=f"{kind}-Setup nach Abschluss der ABC-Korrektur", side=d, zone=zone,
            stop=stop_lvl - d * buf, cancel_level=p[7], expiry_bars=expiry, geometry=geo, count=count,
            text=(f"Welle C endet häufig bei 0,618–1,0 x A (A = {a_len:.2f}): Zone {zone[0]:.2f}–{zone[1]:.2f}. "
                  f"Danach wird ein neuer Impuls erwartet; Ziele: Ende B ({p[7]:.2f}), Start A ({p[5]:.2f}). "
                  "Einstieg erst nach Umkehrbestätigung."))
    return None


PULLBACK_STATES = ("W4", "A", "C")


def build_pullback_plan(count: WaveCount, close: float, atr_now: float,
                        cfg: BotConfig) -> tuple[TradePlan | None, str]:
    """Kauf (bzw. Short) WÄHREND einer Korrektur in Richtung der übergeordneten Struktur.

    Stop an der Elliott-Invalidierung: Welle 4 darf das Ende von Welle 1 nicht
    unterschreiten; A/B/C-Korrekturen eines Impulses gehen selten tiefer als
    78,6 %. Erstes Ziel ist der Beginn der Korrektur (Test des Hochs).
    """
    ex = cfg.exits
    d = count.direction
    p = [x.price for x in count.points]
    ix = [x.idx for x in count.points]
    v = cfg.costs.variable_rate()
    entry = close
    if count.state == "W4":
        w1, w3 = abs(p[1] - p[0]), abs(p[3] - p[2])
        stop_lvl = p[1]
        cap = entry + d * w3 if w3 < w1 else None
        cands = [("Hoch von Welle 3", p[3]), ("W5 = 1,0 x W1", entry + d * w1),
                 ("W5 = 0,618 x (0–3)", entry + d * 0.618 * abs(p[3] - p[0])), ("W5 = 1,618 x W1", entry + d * 1.618 * w1)]
        if cap is not None:
            cands = [(n, x if d * x <= d * cap else cap) for n, x in cands]
        time_bars = int(ex.time_stop_factor * max(5, ix[3] - ix[2]))
    else:
        L = abs(p[5] - p[0])
        stop_lvl = p[5] - d * 0.786 * L
        cands = [("Start der Korrektur", p[5]), ("Hoch + 0,382 x Impuls", p[5] + d * 0.382 * L),
                 ("Hoch + 0,618 x Impuls", p[5] + d * 0.618 * L)]
        if count.state == "C":
            cands.append(("Ende Welle B", p[7]))
        time_bars = int(ex.time_stop_factor * max(5, (ix[5] - ix[0]) // 2))
    stop = stop_lvl - d * ex.stop_atr_buffer * atr_now
    dist = d * (entry - stop)
    if dist <= 0:
        return None, "Kurs liegt bereits unter der Elliott-Invalidierung"
    if dist > ex.max_stop_atr * atr_now:
        return None, (f"Stop wäre {dist / atr_now:.1f} ATR entfernt (> {ex.max_stop_atr:g}) – Korrektur noch zu flach")
    if dist < ex.min_stop_atr * atr_now:
        stop = entry - d * ex.min_stop_atr * atr_now
        dist = ex.min_stop_atr * atr_now
    risk = dist + v * (entry + stop)
    cands = sorted(cands, key=lambda x: d * x[1])
    levels: list[tuple[str, float]] = []
    for n, x in cands:
        if d * (x - entry) > 0.25 * risk and (not levels or abs(x - levels[-1][1]) > 1e-9):
            levels.append((n, x))
    if not levels:
        return None, "Kurs bereits am Korrekturbeginn – kein Rücksetzer mehr"
    while len(levels) < 3:
        levels.append(levels[-1])
    levels = levels[:3]
    main = levels[0]
    reward = d * (main[1] - entry) - v * (entry + main[1])
    if reward <= 0:
        return None, "Kein Gewinnpotenzial nach Kosten"
    plan = TradePlan(side=d, entry=entry, stop=stop, levels=levels, main_target=main, risk_per_unit=risk,
                     reward_per_unit=reward, time_stop_bars=max(ex.min_time_stop_bars, time_bars), atr=atr_now,
                     count=count, invalidation=stop_lvl, kind="market", exit_mode=ex.mode,
                     entry_type="Markt (Rücksetzer-Kauf)")
    plan.legs, plan.exit_mode = make_legs(levels, main, ex.mode, cfg, None)
    return plan, "ok"


def _trend_ok(side: int, trend: int, cfg: BotConfig) -> bool:
    st = cfg.strategy
    if st.trend_filter == "none":
        return True
    if st.trend_mode == "strict":
        return trend == side
    return trend != -side


def decide(wa: WaveAnalysis, close: float, atr_now: float, cfg: BotConfig, trend: int = 0) -> Decision:
    """Wählt das Szenario und baut den Plan (Struktur-, Trend- und CRV-Filter)."""
    f, st = cfg.filters, cfg.strategy
    primary = wa.primary
    if primary is None:
        return Decision("MEIDEN", "Keine regelkonforme Elliott-Wellen-Zählung gefunden", trend=trend)
    if st.entry_style == "pullback":
        return _decide_pullback(wa, primary, close, atr_now, cfg, trend)
    if st.entry_style == "both" and primary.state in PULLBACK_STATES and primary.next_dir != primary.direction:
        dec = _decide_pullback(wa, primary, close, atr_now, cfg, trend)
        if dec.signal in ("KAUFEN", "SHORT") or primary.direction != 1 or cfg.budget.allow_short:
            return dec
    p_dir = wa.p_up if primary.next_dir > 0 else wa.p_down
    rej: list[str] = []
    if primary.score < f.min_count_score:
        rej.append(f"Qualität der führenden Zählung zu gering ({primary.score:.3f} < {f.min_count_score:.2f})")
    if p_dir < f.min_consensus:
        rej.append(f"Szenarien uneinig: nur {p_dir:.0%} zeigen in Richtung der Hauptzählung "
                   f"(mindestens {f.min_consensus:.0%})")

    # Korrektur gegen die Hauptrichtung -> Rücksetzer-Setup statt Gegentrend-Trade
    is_correction = primary.state in ("W4", "C") and primary.next_dir != primary.direction
    if is_correction and (primary.next_dir < 0 and not cfg.budget.allow_short or st.trade_setups):
        setup = conditional_setup(primary, atr_now, cfg)
        side = primary.direction
        if setup is not None:
            setup.setup_key = f"S-{primary.state}-{trend_tag(trend, side)}"
        if setup and side < 0 and not cfg.budget.allow_short:
            setup = None
        if setup and not _trend_ok(side, trend, cfg):
            rej.append("Setup gegen den übergeordneten Trend (200-Tage-Linie)")
        if setup and setup.crv < f.min_crv:
            rej.append(f"Setup-CRV {setup.crv:.2f} unter Minimum {f.min_crv:.1f}")
        if setup and not rej:
            return Decision("WARTEN", "Korrektur läuft – Einstieg erst in der Zielzone",
                            setup=setup, rejections=rej, trend=trend)
        if primary.next_dir < 0 and not cfg.budget.allow_short:
            return Decision("MEIDEN" if setup is None else "BEOBACHTEN",
                            "Korrektur läuft – kein Einstieg" if setup is None else
                            "Rücksetzer-Setup erfüllt die Kriterien nicht", setup=setup,
                            rejections=rej, trend=trend)

    if primary.next_dir < 0 and not cfg.budget.allow_short:
        return Decision("MEIDEN", "Hauptszenario erwartet fallende Kurse (Short deaktiviert) – "
                        "keine neuen Käufe, bestehende Positionen absichern", rejections=rej, trend=trend)

    plan, why = build_plan(primary, close, atr_now, cfg)
    if plan is None:
        rej.append(why)
    else:
        plan.setup_key = f"M-{primary.state}-{trend_tag(trend, plan.side)}"
        if plan.crv < f.min_crv:
            rej.append(f"CRV {plan.crv:.2f} unter Minimum {f.min_crv:.1f}")
        if primary.state not in st.allowed_states:
            rej.append(f"Zählungstyp {primary.state} ist für Markt-Einstiege deaktiviert")
        if not _trend_ok(plan.side, trend, cfg):
            rej.append("Signal gegen den übergeordneten Trend (200-Tage-Linie)")
        if not st.trade_market_signals:
            rej.append("Markt-Einstiege deaktiviert (nur Setups)")
        if st.w3_breakout and primary.state == "W3" and plan.side * (close - primary.points[1].price) <= 0:
            rej.append("Welle 3 noch nicht bestätigt (Ende von Welle 1 nicht überschritten)")
    signal = "KAUFEN" if primary.next_dir > 0 else "SHORT"
    if rej:
        setup = conditional_setup(primary, atr_now, cfg)
        return Decision("BEOBACHTEN", "Setup erfüllt die Einstiegskriterien (noch) nicht",
                        plan=plan, setup=setup, rejections=rej, trend=trend)
    return Decision(signal, primary.description, plan=plan, trend=trend)


def _decide_pullback(wa: WaveAnalysis, primary: WaveCount, close: float, atr_now: float,
                     cfg: BotConfig, trend: int) -> Decision:
    """Einstiegsstil "pullback": Korrekturen (W4, A, C) innerhalb der übergeordneten
    Struktur kaufen – in Richtung des 200-Tage-Trends."""
    f = cfg.filters
    side = primary.direction
    is_corr = primary.state in PULLBACK_STATES and primary.next_dir == -side
    if not is_corr:
        if primary.next_dir > 0:
            return Decision("BEOBACHTEN", f"{primary.description} – Impulsphase, kein Rücksetzer-Einstieg "
                            "(historisch kein Vorteil beim Kauf laufender Impulse)", trend=trend)
        return Decision("MEIDEN", f"{primary.description} – keine Kaufgelegenheit", trend=trend)
    if side < 0 and not cfg.budget.allow_short:
        return Decision("MEIDEN", "Korrektur innerhalb einer Abwärtsstruktur – Long-Rücksetzer nicht sinnvoll, "
                        "Short deaktiviert", trend=trend)
    rej: list[str] = []
    if primary.score < f.min_count_score:
        rej.append(f"Qualität der führenden Zählung zu gering ({primary.score:.3f} < {f.min_count_score:.2f})")
    if not _trend_ok(side, trend, cfg):
        rej.append("Kein intakter übergeordneter Trend in Kaufrichtung (200-Tage-Linie)")
    plan, why = build_pullback_plan(primary, close, atr_now, cfg)
    if plan is None:
        rej.append(why)
    else:
        plan.setup_key = f"P-{primary.state}-{trend_tag(trend, side)}"
        if plan.crv < f.min_crv_pullback:
            rej.append(f"CRV bis Teilziel 1 {plan.crv:.2f} unter Minimum {f.min_crv_pullback:.1f} – "
                       "Korrektur noch nicht tief genug")
        if primary.state not in cfg.strategy.allowed_states:
            rej.append(f"Zählungstyp {primary.state} ist deaktiviert")
    if rej:
        return Decision("BEOBACHTEN", "Rücksetzer erkannt, Einstiegskriterien (noch) nicht erfüllt",
                        plan=plan, rejections=rej, trend=trend)
    return Decision("KAUFEN" if side > 0 else "SHORT",
                    f"{primary.description} – Kauf des Rücksetzers im übergeordneten Trend", plan=plan, trend=trend)
