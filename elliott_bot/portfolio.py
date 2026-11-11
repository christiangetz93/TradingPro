"""Paper-Trading-Depot: platziert Orders und verwaltet Positionen mit denselben
Regeln wie der Backtest (execution.py).

* Ein Signal wird als Order gespeichert und beim nächsten ``manage``-Lauf zum
  Eröffnungskurs des Folgetags ausgeführt (nie zum Schlusskurs des Signaltags).
* Danach gelten Stop, Trailing-Stop, Teilziele, Struktur-Exit und Zeitstopp
  Tag für Tag – inklusive Kurslücken und aller Gebühren.

Es werden keine echten Orders gesendet.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .analyzer import TitleReport
from .config import BotConfig
from .data_sources import DataError
from .execution import Position, make_legs, open_position, process_bar
from .indicators import atr
from .risk import BookState


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
    fees: float
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
            self.pending: list[dict] = raw.get("pending", [])
            self.closed = [ClosedTrade(**c) for c in raw["closed"]]
            self.equity_history = raw.get("equity_history", [])
        else:
            self.cash = cfg.budget.capital
            self.start_capital = cfg.budget.capital
            self.peak_equity = cfg.budget.capital
            self.positions: list[Position] = []
            self.pending = []
            self.closed: list[ClosedTrade] = []
            self.equity_history: list[list] = []

    # ------------------------------------------------------------------ Zustand
    @property
    def equity(self) -> float:
        return self.cash + sum(p.side * p.units * p.last_price for p in self.positions)

    def book_state(self) -> BookState:
        clusters: dict[str, float] = {}
        for p in self.positions:
            clusters[p.region] = clusters.get(p.region, 0.0) + p.open_risk()
        for o in self.pending:
            clusters[o["region"]] = clusters.get(o["region"], 0.0) + o["risk_amount"]
        open_risk = sum(p.open_risk() for p in self.positions) + sum(o["risk_amount"] for o in self.pending)
        return BookState(equity=self.equity, peak_equity=max(self.peak_equity, self.equity), cash=self.cash,
                         open_risk=open_risk, cluster_risk=clusters)

    def holds(self, ticker: str) -> bool:
        return any(p.ticker == ticker for p in self.positions) or any(o["ticker"] == ticker for o in self.pending)

    def save(self) -> None:
        self.path.write_text(json.dumps({
            "start_capital": self.start_capital, "cash": self.cash, "peak_equity": self.peak_equity,
            "positions": [asdict(p) for p in self.positions], "pending": self.pending,
            "closed": [asdict(c) for c in self.closed], "equity_history": self.equity_history,
        }, indent=2, ensure_ascii=False, default=float))

    # ---------------------------------------------------------------- Aktionen
    def open_from_report(self, rep: TitleReport) -> dict | None:
        """Speichert eine Kauf-Order für den nächsten Eröffnungskurs."""
        if not rep.tradeable or self.holds(rep.fund.ticker):
            return None
        plan, size = rep.decision.plan, rep.size
        order = {"ticker": rep.fund.ticker, "name": rep.fund.name, "region": rep.fund.region,
                 "side": plan.side, "units": size.units, "risk_amount": size.risk_amount,
                 "signal_date": str(rep.series.df.index[-1].date()), "plan": plan.to_dict(),
                 "levels": [list(x) for x in plan.levels], "main": list(plan.main_target),
                 "setup_key": plan.setup_key}
        self.pending.append(order)
        return order

    def _fill(self, order: dict, date: str, o: float, atr_now: float) -> Position | None:
        cfg = self.cfg
        plan = dict(order["plan"])
        side = order["side"]
        if side * (o - plan["stop"]) <= 0:
            return None                                       # Kurslücke unter den Stop
        v = cfg.costs.variable_rate()
        risk_unit = abs(o - plan["stop"]) + v * (o + plan["stop"])
        units = min(order["units"], order["risk_amount"] / risk_unit)
        if not cfg.budget.allow_fractional_shares:
            units = math.floor(units)
        if units <= 0:
            return None
        levels = [tuple(x) for x in order["levels"]]
        plan["legs"], plan["exit_mode"] = make_legs(levels, tuple(order["main"]), cfg.exits.mode, cfg, units * o)
        pos = open_position(order["ticker"], side, units, o, date, plan, cfg)
        pos.region, pos.name = order["region"], order["name"]
        cost = cfg.costs.order_cost(units * o)
        pos.fees += cost
        self.cash -= side * units * o + cost
        pos.bars_held = -1
        pos.log.append(f"{date}: Gekauft {units:g} @ {o:.2f} (Kosten {cost:.2f}), Stop {pos.stop:.2f}")
        return pos

    def _settle(self, pos: Position, fills, date: str) -> None:
        for f in fills:
            cost = self.cfg.costs.order_cost(f.units * f.price)
            pos.fees += cost
            self.cash += pos.side * f.units * f.price - cost
            gross = pos.side * f.units * (f.price - pos.entry)
            pos.realized += gross
            pos.log.append(f"{date}: {f.reason}: {f.units:g} @ {f.price:.2f} (brutto {gross:+.2f})")
            if not pos.is_open:
                net = pos.realized - pos.fees
                self.closed.append(ClosedTrade(pos.ticker, pos.side, pos.units_initial, pos.entry, f.price,
                                               pos.entry_date, date, net, pos.fees,
                                               net / (pos.units_initial * pos.risk_per_unit), f.reason))

    def update(self, provider) -> list[str]:
        events: list[str] = []
        tickers = sorted({p.ticker for p in self.positions} | {o["ticker"] for o in self.pending})
        for tk in tickers:
            try:
                df = provider.get(tk).df
            except DataError as exc:
                events.append(f"{tk}: keine Kursdaten ({exc}) – unverändert")
                continue
            atr_v = atr(df, self.cfg.analysis.atr_period)
            pos = next((p for p in self.positions if p.ticker == tk), None)
            order = next((o for o in self.pending if o["ticker"] == tk), None)
            start = pos.last_date if pos else order["signal_date"]
            n_log = len(pos.log) if pos else 0
            for ts, row in df[df.index > pd.Timestamp(start)].iterrows():
                d = str(ts.date())
                if order is not None:
                    self.pending.remove(order)
                    pos = self._fill(order, d, float(row.open), float(atr_v.loc[ts]))
                    order = None
                    if pos is None:
                        events.append(f"{tk}: Order verworfen (Kurslücke unter den Stop bzw. Größe 0)")
                        break
                    self.positions.append(pos)
                    n_log = 0
                fills = process_bar(pos, float(row.open), float(row.high), float(row.low), float(row.close),
                                    d, float(atr_v.loc[ts]))
                self._settle(pos, fills, d)
                if not pos.is_open:
                    break
            if pos is not None:
                events.extend(f"{tk}: {line}" for line in pos.log[n_log:])
                if not pos.is_open and pos in self.positions:
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
        for o in self.pending:
            lines.append(f"  {o['ticker']:8s} Order: {o['units']:g} Stk. zum nächsten Eröffnungskurs "
                         f"(Signal {o['signal_date']}, Stop {o['plan']['stop']:.2f})")
        for p in self.positions:
            pnl = p.side * p.units * (p.last_price - p.entry) + p.realized - p.fees
            lines.append(f"  {p.ticker:8s} {p.units:g} Stk. Einstieg {p.entry:.2f} Kurs {p.last_price:.2f} "
                         f"Stop {p.stop:.2f} PnL inkl. Kosten {pnl:+.2f}")
        if self.closed:
            wins = sum(c.pnl > 0 for c in self.closed)
            lines.append(f"Abgeschlossene Trades: {len(self.closed)}, Gewinner {wins}, Summe netto "
                         f"{sum(c.pnl for c in self.closed):+,.2f}, Kosten {sum(c.fees for c in self.closed):,.2f}")
        return lines
