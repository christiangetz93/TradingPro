"""Zentrale Konfiguration des Elliott-Wave-Bots.

Alle Werte lassen sich über eine TOML-Datei (siehe ``config.example.toml``)
überschreiben. Die Defaults ("v2") wurden per Backtest auf echten Kursdaten
1990–2010 ausgewählt und auf 2011–2022 out-of-sample bestätigt (siehe README).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class BudgetConfig:
    capital: float = 10_000.0            # verfügbares Budget
    currency: str = "EUR"
    risk_per_trade: float = 0.01          # max. 1 % des Kapitals Verlust je Trade (inkl. Kosten)
    max_portfolio_heat: float = 0.05      # max. 5 % offenes Gesamtrisiko
    max_cluster_heat: float = 0.025       # max. 2,5 % offenes Risiko je Region / korreliertem Cluster
    max_position_pct: float = 0.20        # max. 20 % des Kapitals in einer Position
    max_positions: int = 8                # max. gleichzeitig offene Positionen
    min_cash_reserve_pct: float = 0.10    # 10 % Cash werden nie investiert
    kelly_fraction: float = 0.25          # Viertel-Kelly als Obergrenze
    allow_fractional_shares: bool = False
    allow_short: bool = False             # Fonds/ETFs: standardmäßig nur Long
    correlation_heat: bool = True         # korrelierte Positionen teilen sich das Cluster-Risiko
    corr_window: int = 120                # Tage für die Korrelationsschätzung
    drawdown_soft: float = 0.10           # ab 10 % Drawdown: Risiko halbieren
    drawdown_hard: float = 0.20           # ab 20 % Drawdown: keine neuen Trades


@dataclass
class CostConfig:
    """Handelskosten. Standard: typischer Neobroker mit 1 € je Order."""
    fee_fixed: float = 1.0                # feste Gebühr je Order
    fee_pct: float = 0.0                  # variable Gebühr in % des Ordervolumens
    fee_min: float = 0.0                  # Mindestgebühr je Order
    spread_pct: float = 0.0015            # volle Geld-Brief-Spanne (je Order zahlt man die Hälfte)
    slippage_pct: float = 0.0005          # Ausführungsabweichung je Order
    max_fee_ratio: float = 0.005          # Gebühr darf max. 0,5 % einer (Teil-)Order ausmachen

    def variable_rate(self) -> float:
        """Kosten je Seite, die proportional zum Ordervolumen anfallen."""
        return self.fee_pct + self.spread_pct / 2 + self.slippage_pct

    def broker_fee(self, value: float) -> float:
        return max(self.fee_min, self.fee_fixed + self.fee_pct * abs(value))

    def order_cost(self, value: float) -> float:
        """Gesamtkosten einer Order: Brokergebühr + halber Spread + Slippage."""
        return self.broker_fee(value) + (self.spread_pct / 2 + self.slippage_pct) * abs(value)

    def fixed_part(self, value: float) -> float:
        """Nicht volumenproportionaler Teil der Kosten einer Order (Fix- bzw. Mindestgebühr)."""
        return self.broker_fee(value) - self.fee_pct * abs(value)


@dataclass
class SignalFilterConfig:
    min_crv: float = 2.0                  # Mindest-Chance/Risiko-Verhältnis (Impuls-Einstiege)
    min_crv_pullback: float = 1.0         # Mindest-CRV bis Teilziel 1 bei Rücksetzer-Käufen
    min_expected_r: float = 0.10          # nur mit use_setup_stats: Mindest-Erwartungswert in R
    min_consensus: float = 0.55           # Mindestanteil aller Szenarien, die in Trade-Richtung zeigen
    min_count_score: float = 0.40         # Mindestqualität der führenden Zählung
    max_crv_for_probability: float = 8.0  # CRV darüber gilt als unrealistisch -> gekappt
    max_cost_r: float = 0.15              # Kosten (Spread, Slippage, Gebühren) max. 15 % des Risikos (1 R)
    min_net_profit: float = 10.0          # erwarteter Nettogewinn je Trade mind. 10 €
    fee_multiple: float = 3.0             # ... und mind. 3x die erwarteten Gebühren


@dataclass
class StrategyConfig:
    entry_style: str = "both"             # "impulse": mit der laufenden Welle | "pullback": Korrekturen kaufen | "both"
    trend_filter: str = "sma200"          # "none" | "sma200" (Kurs vs. steigender/fallender 200-Tage-Schnitt)
    trend_mode: str = "strict"            # "strict": nur mit dem Trend | "not_against": nicht gegen den Trend
    trade_market_signals: bool = True     # Impuls-Signale (W3/W5/C ...) direkt zum nächsten Eröffnungskurs
    trade_setups: bool = False            # Rücksetzer-Setups (Ende W4 / Ende C) als Order platzieren
    setup_entry: str = "confirm"          # "limit" (Limit in der Zone) | "confirm" (Umkehrbestätigung)
    setup_expiry_factor: float = 1.5      # Gültigkeit des Setups = Faktor x erwartete Korrekturdauer
    allowed_states: tuple[str, ...] = ("W3", "W4", "W5", "A", "B", "C")
    w3_breakout: bool = False             # Welle 3 erst handeln, wenn das Hoch von Welle 1 überschritten ist
    use_setup_stats: bool = False         # Signale nur, wenn die gelernte Setup-Statistik positiv ist
    stats_path: str = ""                  # leer = mitgelieferte Statistik
    stats_prior_n: float = 30.0           # Schrumpfung zur Null (Skepsis) in "Pseudo-Trades"


@dataclass
class ExitConfig:
    mode: str = "trail"                   # trail | scale_be | scale | half | target  (siehe execution.EXIT_MODES)
    stop_atr_buffer: float = 0.5          # Stop = Invalidierung ± 0,5 ATR
    max_stop_atr: float = 6.0             # Stop weiter als 6 ATR -> kein Trade
    min_stop_atr: float = 1.0             # Stop nie enger als 1 ATR (Rauschen)
    tp1_fraction: float = 1 / 3           # Teilverkauf am 1. Ziel (scale-Modi)
    tp2_fraction: float = 1 / 3           # Teilverkauf am 2. Ziel (scale-Modi)
    trail_atr_mult: float = 4.0           # Chandelier-Trailing-Stop (Abstand in ATR vom höchsten Hoch)
    time_stop_factor: float = 1.5         # Zeitstopp/max. Haltedauer = 1,5 x erwartete Wellendauer
    min_time_stop_bars: int = 10


@dataclass
class AnalysisConfig:
    atr_period: int = 14
    zigzag_atr_mults: tuple[float, ...] = (2.5, 4.0, 6.0)   # drei Wellengrade
    unclear_weight: float = 0.45          # Gewicht des Szenarios "keine klare Zählung"
    sharpness: float = 3.0                # Schärfung der Scores -> Gewichte
    mc_paths: int = 3000
    mc_block: int = 5                     # Block-Bootstrap-Länge (Volatilitäts-Cluster)
    mc_return_window: int = 500
    ticker_backtest: bool = True          # Regelwerk auf der Historie des Titels testen (Info im Report)
    seed: int | None = None


@dataclass
class SelectionConfig:
    n_titles: int = 5
    em_weight: float = 3.0                # Schwellenländer/Wachstumsmärkte 3x bevorzugt
    growth_theme_weight: float = 2.0      # Wachstumsthemen 2x bevorzugt
    base_weight: float = 1.0
    max_per_region: int = 2               # Diversifikation
    seed: int | None = None               # None = wirklich zufällig


@dataclass
class BotConfig:
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    filters: SignalFilterConfig = field(default_factory=SignalFilterConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)


def _apply(obj: Any, values: dict[str, Any]) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unbekannter Konfigurationsschlüssel: {key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _apply(current, value)
        elif isinstance(current, tuple):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None) -> BotConfig:
    cfg = BotConfig()
    if path:
        with open(path, "rb") as fh:
            _apply(cfg, tomllib.load(fh))
    return cfg


def apply_overrides(cfg: BotConfig, overrides: dict[str, Any]) -> BotConfig:
    """Setzt Werte per Punkt-Notation, z. B. {"strategy.trend_filter": "sma200"}."""
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        _apply(getattr(cfg, section), {key: value})
    return cfg
