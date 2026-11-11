"""Paper-Trading-Depot: eröffnet Positionen und verwaltet die Ausstiegsregeln.

Es werden keine echten Orders gesendet. Der Zustand liegt als JSON-Datei vor
und wird bei jedem ``manage``-Lauf mit den neuen Tageskursen fortgeschrieben.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .analyzer import TitleReport
from .config import BotConfig
from .data_sources import DataError
from .indicators import atr, rsi
from .risk import BookState
from .waves import analyze_waves


@dataclass
class Position:
    ticker: str
    name: str
    region: str
    side: int
    units: float
    units_initial: float
    entry: float
    entry_date: str
    stop: float
    initial_stop: float
    levels: list[list]              # [[name, preis], ...] TP1..TP3
    invalidation: float
    atr: float
    risk_per_unit: float
    time_stop_bars: int
    stage: int = 0
    highest: float = 0.0            # bestes Extrem in Trade-Richtung (orientiert)
    bars_held: int = 0
    last_date: str = ""
    last_price: float = 0.0
    trail_mult: float = 3.0
    log: list[str] = field(default_factory=list)


@dataclass
class ClosedTrade:
    ticker: str
    side: int
    units: float
    entry: float
    exit: float
    entry_date: str
    exit_date: str
    pnl: float
    r_multiple: float
    reason: str


class PaperPortfolio:
    def __init__(self, path: str | Path, cfg: BotConfig):
        self.path = Path(path)
        self.cfg = cfg
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.cash = raw["cash"]
            self.start_capital = raw["start_capital"]
            self.peak_equity = raw["peak_equity"]
            self.positions = [Position(**p) for p in raw["positions"]]
            self.closed = [ClosedTrade(**c) for c in raw["closed"]]
            self.equity_history = raw.get("equity_history", [])
        else:
            self.cash = cfg.budget.capital
            self.start_capital = cfg.budget.capital
            self.peak_equity = cfg.budget.capital
            self.positions: list[Position] = []
            self.closed: list[ClosedTrade] = []
            self.equity_history: list[list] = []

    # ------------------------------------------------------------------ Zustand
    @property
    def equity(self) -> float:
        return self.cash + sum(p.side * p.units * p.last_price for p in self.positions)

    def open_risk(self, p: Position) -> float:
        return max(0.0, p.side * (p.last_price - p.stop)) * p.units

    def book_state(self) -> BookState:
        clusters: dict[str, float] = {}
        for p in self.positions:
            clusters[p.region] = clusters.get(p.region, 0.0) + self.open_risk(p)
        return BookState(equity=self.equity, peak_equity=max(self.peak_equity, self.equity),
                         cash=self.cash, open_risk=sum(self.open_risk(p) for p in self.positions),
                         cluster_risk=clusters)

    def holds(self, ticker: str) -> bool:
        return any(p.ticker == ticker for p in self.positions)

    def save(self) -> None:
        self.path.write_text(json.dumps({
            "start_capital": self.start_capital, "cash": self.cash, "peak_equity": self.peak_equity,
            "positions": [asdict(p) for p in self.positions],
            "closed": [asdict(c) for c in self.closed],
            "equity_history": self.equity_history,
        }, indent=2, ensure_ascii=False))

    # ---------------------------------------------------------------- Aktionen
    def _fee(self, units: float, price: float) -> float:
        b = self.cfg.budget
        return abs(units * price) * (b.fee_pct + b.slippage_pct)

    def open_from_report(self, rep: TitleReport) -> Position | None:
        if not rep.tradeable or self.holds(rep.fund.ticker):
            return None
        plan, size = rep.decision.plan, rep.size
        date = str(rep.series.df.index[-1].date())
        self.cash -= plan.side * size.units * plan.entry + self._fee(size.units, plan.entry)
        pos = Position(
            ticker=rep.fund.ticker, name=rep.fund.name, region=rep.fund.region, side=plan.side,
            units=size.units, units_initial=size.units, entry=plan.entry, entry_date=date,
            stop=plan.stop, initial_stop=plan.stop, levels=[[n, p] for n, p in plan.levels],
            invalidation=plan.count.invalidation, atr=plan.atr, risk_per_unit=plan.risk_per_unit,
            time_stop_bars=plan.time_stop_bars, highest=plan.side * plan.entry, last_date=date,
            last_price=plan.entry, trail_mult=self.cfg.exits.trail_atr_mult,
            log=[f"{date}: Eröffnet {size.units:g} @ {plan.entry:.2f}, Stop {plan.stop:.2f}"])
        self.positions.append(pos)
        return pos

    def _sell(self, pos: Position, units: float, price: float, date: str, reason: str) -> None:
        units = min(units, pos.units)
        if units <= 0:
            return
        self.cash += pos.side * units * price - self._fee(units, price)
        pnl = pos.side * units * (price - pos.entry) - self._fee(units, price) - self._fee(units, pos.entry)
        self.closed.append(ClosedTrade(pos.ticker, pos.side, units, pos.entry, price, pos.entry_date,
                                       date, pnl, pnl / (units * pos.risk_per_unit), reason))
        pos.units -= units
        pos.log.append(f"{date}: {reason}: {units:g} @ {price:.2f} (PnL {pnl:+.2f})")

    def _process_bar(self, pos: Position, d: str, o: float, h: float, lo: float, c: float) -> None:
        ex = self.cfg.exits
        s = pos.side
        oq, hq, lq, cq = (s * o, s * h, s * lo, s * c) if s == 1 else (-o, -lo, -h, -c)
        stop_q = s * pos.stop
        pos.bars_held += 1
        # 1) Stop (Gap-Risiko: Ausführung zum Eröffnungskurs, falls darunter eröffnet)
        if lq <= stop_q:
            fill = s * min(oq, stop_q)
            self._sell(pos, pos.units, fill, d, "Stop ausgelöst" if pos.stage == 0 else "Trailing-Stop")
            return
        # 2) Teilziele
        tq = [s * p for _, p in pos.levels]
        fractions = [ex.tp1_fraction, ex.tp2_fraction]
        while pos.stage < 3 and pos.units > 0 and hq >= tq[pos.stage]:
            name, price = pos.levels[pos.stage]
            if pos.stage < 2:
                qty = pos.units_initial * fractions[pos.stage]
                if not self.cfg.budget.allow_fractional_shares:
                    qty = max(1, int(qty)) if pos.units > 1 else pos.units
                self._sell(pos, qty, price, d, f"Teilziel {pos.stage + 1} ({name})")
                fees = self.cfg.budget.fee_pct + self.cfg.budget.slippage_pct
                new_stop = (pos.entry * (1 + s * 2 * fees)) if pos.stage == 0 else pos.levels[0][1]
                pos.stop = new_stop if s * new_stop > s * pos.stop else pos.stop
            else:
                self._sell(pos, pos.units, price, d, f"Endziel ({name})")
            pos.stage += 1
        if pos.units <= 0:
            return
        # 3) Trailing-Stop ab Stufe 1
        pos.highest = max(pos.highest, hq)
        if pos.stage >= 1:
            trail = s * (pos.highest - pos.trail_mult * pos.atr)
            if s * trail > s * pos.stop:
                pos.stop = trail
        # 4) Struktur-Exit (Schlusskurs jenseits der Invalidierung)
        if s * c < s * pos.invalidation:
            self._sell(pos, pos.units, c, d, "Elliott-Zählung invalidiert")
            return
        # 5) Zeitstopp
        if pos.stage == 0 and pos.bars_held >= pos.time_stop_bars:
            self._sell(pos, pos.units, c, d, "Zeitstopp")

    def update(self, provider) -> list[str]:
        events: list[str] = []
        for pos in list(self.positions):
            try:
                series = provider.get(pos.ticker)
            except DataError as exc:
                events.append(f"{pos.ticker}: keine Kursdaten ({exc}) – Position unverändert")
                continue
            df = series.df
            new = df[df.index > pd.Timestamp(pos.last_date)]
            n_log = len(pos.log)
            for ts, row in new.iterrows():
                self._process_bar(pos, str(ts.date()), row.open, row.high, row.low, row.close)
                pos.last_date, pos.last_price = str(ts.date()), float(row.close)
                if pos.units <= 0:
                    break
            if pos.units > 0 and len(new):
                # Warnsignal-Regel: tägliche Neuanalyse
                a_s = atr(df, self.cfg.analysis.atr_period)
                wa = analyze_waves(df, a_s, rsi(df["close"]), self.cfg.analysis)
                p_against = wa.p_down if pos.side == 1 else wa.p_up
                if p_against >= 0.6 and pos.trail_mult > self.cfg.exits.tight_trail_atr_mult:
                    pos.trail_mult = self.cfg.exits.tight_trail_atr_mult
                    pos.atr = float(a_s.iloc[-1])
                    tight = pos.side * (pos.highest - pos.trail_mult * pos.atr)
                    if pos.side * tight > pos.side * pos.stop and pos.side * tight < pos.side * pos.last_price:
                        pos.stop = tight
                    pos.log.append(f"{pos.last_date}: Warnsignal – Gegen-Szenario {p_against:.0%}, "
                                   f"Stop eng auf {pos.stop:.2f}")
            events.extend(f"{pos.ticker}: {line}" for line in pos.log[n_log:])
            if pos.units <= 0:
                self.positions.remove(pos)
        eq = self.equity
        self.peak_equity = max(self.peak_equity, eq)
        today = str(pd.Timestamp.today().date())
        if not self.equity_history or self.equity_history[-1][0] != today:
            self.equity_history.append([today, eq])
        else:
            self.equity_history[-1][1] = eq
        return events

    def summary_lines(self) -> list[str]:
        eq = self.equity
        bs = self.book_state()
        lines = [
            f"Depotwert {eq:,.2f} | Cash {self.cash:,.2f} | Start {self.start_capital:,.2f} | "
            f"Performance {eq / self.start_capital - 1:+.2%} | Drawdown {bs.drawdown:.1%}",
            f"Offenes Risiko (Heat) {bs.open_risk:,.2f} = {bs.open_risk / eq:.2%} des Depots",
        ]
        for p in self.positions:
            pnl = p.side * p.units * (p.last_price - p.entry)
            lines.append(f"  {p.ticker:8s} {p.units:g} Stk. Einstieg {p.entry:.2f} Kurs {p.last_price:.2f} "
                         f"Stop {p.stop:.2f} Stufe {p.stage} PnL {pnl:+.2f}")
        if self.closed:
            wins = sum(c.pnl > 0 for c in self.closed)
            lines.append(f"Abgeschlossene Teil-Trades: {len(self.closed)}, davon Gewinner {wins}, "
                         f"Summe PnL {sum(c.pnl for c in self.closed):+,.2f}")
        return lines
