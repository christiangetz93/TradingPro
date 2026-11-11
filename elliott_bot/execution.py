"""Order-Ausführung und Positionsverwaltung – gemeinsam für Backtest und Paper-Depot.

Regeln je Handelstag (Tagesbalken Open/High/Low/Close):

1. Eröffnet der Kurs bereits jenseits des Stops (Kurslücke), wird zum
   Eröffnungskurs verkauft – nicht zum Stop-Kurs. Das ist das reale Gap-Risiko.
2. Teilziele, die schon zur Eröffnung erreicht sind, werden zum (besseren)
   Eröffnungskurs ausgeführt.
3. Werden im selben Balken Stop UND Ziel berührt, wird pessimistisch der Stop
   angenommen (die Reihenfolge innerhalb des Tages ist unbekannt).
4. Trailing-Stops werden mit dem Tageshoch nachgezogen und gelten ab dem
   nächsten Tag (kein Blick in die Zukunft innerhalb des Balkens).
5. Struktur-Exit: Schlusskurs jenseits der Elliott-Invalidierung.
6. Zeitstopp: kein Teilziel innerhalb der erwarteten Wellendauer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import BotConfig, CostConfig

EXIT_MODES = {
    # name: (Anteile je Ziel, Stop auf Einstand nach Ziel 1, Stop auf Vorziel, Trailing aktiv ab)
    "scale_be": ("scale", True, True, "first_leg"),
    "scale": ("scale", False, True, "first_leg"),
    "half": ("half", False, False, "first_leg"),
    "target": ("target", False, False, None),
    "trail": ("none", False, False, "entry"),
}

EXIT_MODE_TEXT = {
    "scale_be": "Drittel-Verkäufe an Ziel 1/2/3, nach Ziel 1 Stop auf Einstand, Rest mit Trailing-Stop",
    "scale": "Drittel-Verkäufe an Ziel 1/2/3, Rest mit Trailing-Stop (kein vorzeitiger Einstands-Stop)",
    "half": "Hälfte an Ziel 1, Rest mit Trailing-Stop ohne festes Endziel",
    "target": "komplette Position am Hauptziel, fester Stop",
    "trail": "keine festen Ziele, Chandelier-Trailing-Stop ab Einstieg",
}


@dataclass
class Fill:
    units: float
    price: float
    reason: str


@dataclass
class Position:
    ticker: str
    side: int
    units: float
    units_initial: float
    entry: float
    entry_date: str
    stop: float
    initial_stop: float
    legs: list[list]                 # [[name, preis, anteil_der_anfangsstückzahl], ...]
    invalidation: float
    atr: float
    risk_per_unit: float             # Einstieg -> Stop inkl. variabler Kosten
    time_stop_bars: int
    be_after_first: bool = False
    stop_to_prev_leg: bool = False
    trail_mult: float = 0.0
    trail_from_entry: bool = False
    var_rate: float = 0.0
    allow_fractional: bool = False
    next_leg: int = 0
    bars_held: int = 0
    highest: float = 0.0             # bestes Extrem in Trade-Richtung (orientiert: side * Preis)
    realized: float = 0.0            # realisierter Kursgewinn (ohne Kosten)
    fees: float = 0.0                # bezahlte Kosten (Gebühren, Spread, Slippage)
    region: str = ""
    name: str = ""
    last_date: str = ""
    last_price: float = 0.0
    meta: dict = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.units > 1e-12

    def open_risk(self) -> float:
        """Was bei Auslösung des aktuellen Stops (vom letzten Kurs aus) verloren ginge."""
        return max(0.0, self.side * (self.last_price - self.stop)) * self.units


def make_legs(levels: list[tuple[str, float]], main: tuple[str, float], mode: str,
              cfg: BotConfig, position_value: float | None) -> tuple[list[list], str]:
    """Teilt die Kursziele gemäß Exit-Modus auf und berücksichtigt Fixgebühren.

    Ist eine Teilorder so klein, dass die Fixgebühr mehr als ``max_fee_ratio``
    ausmacht, werden Teilverkäufe zusammengelegt (scale -> half -> target).
    """
    ex, c = cfg.exits, cfg.costs
    kind = EXIT_MODES[mode][0]

    def too_small(frac: float) -> bool:
        if position_value is None or position_value <= 0:
            return False
        return c.fixed_part(frac * position_value) > c.max_fee_ratio * frac * position_value

    if kind == "scale" and too_small(min(ex.tp1_fraction, ex.tp2_fraction)):
        kind, mode = ("half", "half") if not too_small(0.5) else ("target", "target")
    if kind == "half" and too_small(0.5):
        kind, mode = "target", "target"
    if kind == "scale":
        f1, f2 = ex.tp1_fraction, ex.tp2_fraction
        legs = [[levels[0][0], levels[0][1], f1], [levels[1][0], levels[1][1], f2],
                [levels[2][0], levels[2][1], 1.0 - f1 - f2]]
    elif kind == "half":
        legs = [[levels[0][0], levels[0][1], 0.5]]
    elif kind == "target":
        legs = [[main[0], main[1], 1.0]]
    else:
        legs = []
    # Mehrere Teilverkäufe auf identischem Kurs zusammenfassen
    merged: list[list] = []
    for leg in legs:
        if merged and abs(merged[-1][1] - leg[1]) < 1e-12:
            merged[-1][2] += leg[2]
        else:
            merged.append(list(leg))
    return merged, mode


def open_position(ticker: str, side: int, units: float, fill_price: float, date: str, plan: dict,
                  cfg: BotConfig) -> Position:
    """Erzeugt eine Position aus einem Plan-Dictionary (siehe strategy.TradePlan.to_dict)."""
    mode = plan.get("exit_mode", cfg.exits.mode)
    _, be, prev, trail_start = EXIT_MODES[mode]
    v = cfg.costs.variable_rate()
    risk = abs(fill_price - plan["stop"]) + v * (fill_price + plan["stop"])
    return Position(
        ticker=ticker, side=side, units=units, units_initial=units, entry=fill_price, entry_date=date,
        stop=plan["stop"], initial_stop=plan["stop"], legs=[list(x) for x in plan["legs"]],
        invalidation=plan["invalidation"], atr=plan["atr"], risk_per_unit=risk,
        time_stop_bars=int(plan["time_stop_bars"]), be_after_first=be, stop_to_prev_leg=prev,
        trail_mult=cfg.exits.trail_atr_mult if trail_start else 0.0,
        trail_from_entry=trail_start == "entry", var_rate=v,
        allow_fractional=cfg.budget.allow_fractional_shares, highest=side * fill_price,
        region=plan.get("region", ""), name=plan.get("name", ""), last_date=date,
        last_price=fill_price, meta=dict(plan.get("meta", {})))


def _qty(pos: Position, fraction: float, last: bool) -> float:
    if last:
        return pos.units
    q = pos.units_initial * fraction
    if not pos.allow_fractional:
        q = math.floor(q + 1e-9)
        if q < 1:
            q = 1 if pos.units >= 1 else pos.units
    return min(q, pos.units)


def _take_legs(pos: Position, fills: list[Fill], reach_q: float, open_q: float | None) -> None:
    s = pos.side
    while pos.is_open and pos.next_leg < len(pos.legs):
        name, price, frac = pos.legs[pos.next_leg]
        lvl_q = s * price
        if reach_q < lvl_q:
            break
        px_q = max(lvl_q, open_q) if open_q is not None else lvl_q
        last = pos.next_leg == len(pos.legs) - 1 and sum(l[2] for l in pos.legs) >= 0.999
        qty = _qty(pos, frac, last)
        fills.append(Fill(qty, s * px_q, f"Teilziel {pos.next_leg + 1} ({name})" if not last else f"Endziel ({name})"))
        pos.units -= qty
        if pos.next_leg == 0 and pos.be_after_first:
            be_q = s * pos.entry * (1 + s * 2 * pos.var_rate)
            pos.stop = s * max(s * pos.stop, s * be_q)
        if pos.stop_to_prev_leg and pos.next_leg >= 1:
            prev_q = s * pos.legs[pos.next_leg - 1][1]
            pos.stop = s * max(s * pos.stop, prev_q)
        pos.next_leg += 1


def process_bar(pos: Position, o: float, h: float, lo: float, c: float, date: str,
                atr_now: float | None = None) -> list[Fill]:
    """Wendet die Ausstiegsregeln auf einen Tagesbalken an und liefert die Ausführungen."""
    s = pos.side
    oq, hq, lq, cq = (o, h, lo, c) if s == 1 else (-o, -lo, -h, -c)
    fills: list[Fill] = []
    pos.bars_held += 1
    pos.last_date, pos.last_price = date, c

    def close_all(price: float, reason: str) -> None:
        if pos.is_open:
            fills.append(Fill(pos.units, price, reason))
            pos.units = 0.0

    stop_reason = "Stop" if pos.next_leg == 0 and s * pos.stop <= s * pos.initial_stop + 1e-12 else "Nachgezogener Stop"
    if oq <= s * pos.stop:                                   # 1) Kurslücke über den Stop hinweg
        close_all(o, f"{stop_reason} (Kurslücke)")
        return fills
    _take_legs(pos, fills, oq, oq)                           # 2) Ziele schon zur Eröffnung
    if not pos.is_open:
        return fills
    if lq <= s * pos.stop:                                   # 3) Stop im Tagesverlauf
        close_all(pos.stop, stop_reason)
        return fills
    _take_legs(pos, fills, hq, None)                         # 4) Ziele im Tagesverlauf
    if not pos.is_open:
        return fills
    if atr_now is not None and atr_now > 0:
        pos.atr = atr_now
    pos.highest = max(pos.highest, hq)                       # 5) Trailing für morgen
    if pos.trail_mult > 0 and (pos.trail_from_entry or pos.next_leg > 0):
        trail_q = pos.highest - pos.trail_mult * pos.atr
        if trail_q > s * pos.stop:
            pos.stop = s * trail_q
    if cq < s * pos.invalidation:                            # 6) Struktur-Exit
        close_all(c, "Elliott-Zählung invalidiert")
    elif pos.next_leg == 0 and pos.bars_held >= pos.time_stop_bars:
        close_all(c, "Zeitstopp")                            # 7) Zeitstopp
    return fills


@dataclass
class PendingOrder:
    """Noch nicht ausgeführte Einstiegsorder."""
    ticker: str
    side: int
    kind: str                      # "market" | "limit" | "confirm"
    units: float
    plan: dict
    created_idx: int
    expiry_idx: int
    limit: float = 0.0
    zone: tuple[float, float] = (0.0, 0.0)
    cancel_level: float = 0.0      # Kurs jenseits davon vor Ausführung -> Setup hinfällig
    armed: bool = False
    trigger_next_open: bool = False


def process_pending(order: PendingOrder, idx: int, o: float, h: float, lo: float, c: float,
                    prev_high: float, prev_low: float) -> tuple[str, float | str | None]:
    """Prüft eine Pending-Order an einem Tagesbalken.

    Rückgabe: ("fill", Preis) | ("cancel", Grund) | ("wait", None)
    """
    s = order.side
    oq, hq, lq, cq = (o, h, lo, c) if s == 1 else (-o, -lo, -h, -c)
    stop_q = s * order.plan["stop"]
    if order.kind == "market" or order.trigger_next_open:
        if oq <= stop_q:
            return "cancel", "Kurslücke unter den Stop vor Einstieg"
        return "fill", o
    if idx > order.expiry_idx:
        return "cancel", "Setup abgelaufen"
    zone_hi_q = max(s * order.zone[0], s * order.zone[1])   # orientiert: obere Zonengrenze
    if order.kind == "limit":
        lim_q = s * order.limit
        if lq <= lim_q:
            if oq <= stop_q:
                return "cancel", "Kurslücke unter den Stop"
            return "fill", s * min(oq, lim_q)
        if hq > s * order.cancel_level:
            return "cancel", "Korrektur ohne Zonen-Test beendet"
        return "wait", None
    # confirm: Zone muss erreicht werden, dann Umkehrbestätigung (Schluss über Vortageshoch)
    if lq <= stop_q:
        return "cancel", "Stop-Niveau vor Einstieg unterschritten"
    if not order.armed:
        if lq <= zone_hi_q:
            order.armed = True          # Zone erreicht – Umkehr kann schon heute bestätigt werden
        elif hq > s * order.cancel_level:
            return "cancel", "Korrektur ohne Zonen-Test beendet"
        else:
            return "wait", None
    prev_hq = prev_high if s == 1 else -prev_low
    if cq > prev_hq:
        order.trigger_next_open = True
    return "wait", None


def fill_costs(costs: CostConfig, units: float, price: float) -> float:
    return costs.order_cost(units * price)
