"""Positionsgröße und Budgetschutz.

Die Positionsgröße ist das Minimum aus mehreren unabhängigen Grenzen:

1. Fixes Risiko je Trade (z. B. 1 % des Kapitals bis zum Stop)
2. Bruchteil-Kelly aus der simulierten R-Verteilung (f* = E[R] / E[R²])
3. Verbleibende Portfolio-Heat (Summe aller offenen Risiken)
4. Verbleibende Cluster-Heat (offenes Risiko je Region)
5. Maximaler Positionsanteil am Kapital
6. Verfügbares Cash abzüglich Mindestreserve

Zusätzlich halbiert eine Drawdown-Bremse das Risiko ab ``drawdown_soft`` und
stoppt neue Trades ab ``drawdown_hard``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import BudgetConfig


@dataclass
class BookState:
    """Zustand des (Paper-)Depots, der für das Sizing relevant ist."""
    equity: float
    peak_equity: float
    cash: float
    open_risk: float = 0.0
    cluster_risk: dict[str, float] = field(default_factory=dict)

    @property
    def drawdown(self) -> float:
        return max(0.0, 1 - self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0


@dataclass
class PositionSize:
    units: float
    invest: float
    risk_amount: float
    risk_pct: float
    binding: str
    kelly_full: float
    limits: dict[str, float]
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.units > 0


def drawdown_factor(book: BookState, bud: BudgetConfig) -> tuple[float, str | None]:
    dd = book.drawdown
    if dd >= bud.drawdown_hard:
        return 0.0, f"Drawdown-Notbremse aktiv ({dd:.1%} ≥ {bud.drawdown_hard:.0%}): keine neuen Trades"
    if dd >= bud.drawdown_soft:
        return 0.5, f"Drawdown {dd:.1%} ≥ {bud.drawdown_soft:.0%}: Risiko je Trade halbiert"
    return 1.0, None


def size_position(entry: float, risk_per_unit: float, mean_r: float, second_moment_r: float,
                  region: str, book: BookState, bud: BudgetConfig) -> PositionSize:
    notes: list[str] = []
    dd_f, dd_note = drawdown_factor(book, bud)
    if dd_note:
        notes.append(dd_note)
    eq = book.equity
    kelly = mean_r / second_moment_r if second_moment_r > 0 else 0.0
    risk_caps = {
        "Fixes Risiko je Trade": bud.risk_per_trade * dd_f * eq,
        "Bruchteil-Kelly": max(0.0, bud.kelly_fraction * kelly) * eq,
        "Portfolio-Heat": max(0.0, bud.max_portfolio_heat * eq - book.open_risk),
        "Regionen-Heat": max(0.0, bud.max_cluster_heat * eq - book.cluster_risk.get(region, 0.0)),
    }
    unit_caps = {k: v / risk_per_unit for k, v in risk_caps.items()}
    cost = bud.fee_pct + bud.slippage_pct
    unit_caps["Max. Positionsanteil"] = bud.max_position_pct * eq / entry
    unit_caps["Verfügbares Cash"] = max(0.0, book.cash - bud.min_cash_reserve_pct * eq) / (entry * (1 + cost))
    binding = min(unit_caps, key=unit_caps.get)
    units = unit_caps[binding]
    if not bud.allow_fractional_shares:
        units = math.floor(units)
        if units == 0 and unit_caps[binding] > 0:
            notes.append("Budget reicht nicht für 1 ganzen Anteil innerhalb der Risikogrenzen")
    units = max(0.0, units)
    if kelly <= 0:
        notes.append("Kelly ≤ 0: negativer Erwartungswert – kein Einsatz")
    risk_amount = units * risk_per_unit
    return PositionSize(units=units, invest=units * entry, risk_amount=risk_amount,
                        risk_pct=risk_amount / eq if eq else 0.0, binding=binding,
                        kelly_full=kelly, limits=unit_caps, notes=notes)
