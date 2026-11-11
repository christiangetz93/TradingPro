"""Zentrale Konfiguration des Elliott-Wave-Bots.

Alle Werte lassen sich über eine TOML-Datei (siehe ``config.example.toml``)
überschreiben. Die Defaults sind bewusst konservativ gewählt.
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
    risk_per_trade: float = 0.01          # max. 1 % des Kapitals Verlust je Trade
    max_portfolio_heat: float = 0.05      # max. 5 % offenes Gesamtrisiko
    max_cluster_heat: float = 0.025       # max. 2,5 % offenes Risiko je Region
    max_position_pct: float = 0.20        # max. 20 % des Kapitals in einer Position
    min_cash_reserve_pct: float = 0.10    # 10 % Cash werden nie investiert
    kelly_fraction: float = 0.25          # Viertel-Kelly als Obergrenze
    allow_fractional_shares: bool = False
    allow_short: bool = False             # Fonds/ETFs: standardmäßig nur Long
    fee_pct: float = 0.001                # Gebühren je Seite (0,1 %)
    slippage_pct: float = 0.0005          # Slippage je Seite
    drawdown_soft: float = 0.10           # ab 10 % Drawdown: Risiko halbieren
    drawdown_hard: float = 0.20           # ab 20 % Drawdown: keine neuen Trades


@dataclass
class SignalFilterConfig:
    min_crv: float = 2.0                  # Mindest-Chance/Risiko-Verhältnis
    min_probability: float = 0.20         # Mindest-Wahrscheinlichkeit: Hauptziel vor Stop
    min_tp1_probability: float = 0.45     # Mindest-Wahrscheinlichkeit: Teilziel 1 vor Stop (-> risikofrei)
    min_expected_r: float = 0.15          # Mindest-Erwartungswert in R (nach Kosten)
    min_consensus: float = 0.55           # Mindestanteil aller Szenarien, die in Trade-Richtung zeigen
    min_count_score: float = 0.40         # Mindestqualität der führenden Zählung
    max_crv_for_probability: float = 8.0  # CRV darüber gilt als unrealistisch -> gekappt


@dataclass
class ExitConfig:
    stop_atr_buffer: float = 0.5          # Stop = Invalidierung ± 0,5 ATR
    max_stop_atr: float = 6.0             # Stop weiter als 6 ATR -> kein Trade
    min_stop_atr: float = 1.0             # Stop nie enger als 1 ATR (Rauschen)
    tp1_fraction: float = 1 / 3           # Teilverkauf am 1. Ziel
    tp2_fraction: float = 1 / 3           # Teilverkauf am 2. Ziel
    trail_atr_mult: float = 3.0           # Chandelier-Trailing-Stop für den Rest
    tight_trail_atr_mult: float = 2.0     # engerer Trail bei Warnsignalen
    time_stop_factor: float = 1.5         # Zeitstopp = 1,5 x erwartete Wellendauer
    min_time_stop_bars: int = 10


@dataclass
class AnalysisConfig:
    lookback_bars: int = 750              # ca. 3 Jahre Tagesdaten
    atr_period: int = 14
    zigzag_atr_mults: tuple[float, ...] = (2.5, 4.0, 6.0)   # drei Wellengrade
    require_confirmed_pivot: bool = True  # Zählung muss auf einem bestätigten Pivot enden
    unclear_weight: float = 0.45          # Gewicht des Szenarios "keine klare Zählung"
    sharpness: float = 3.0                # Schärfung der Scores -> Gewichte
    edge_kappa: float = 0.25              # wie stark die Wellenanalyse die Basis-Wahrscheinlichkeit verschiebt
    mc_paths: int = 3000
    mc_block: int = 5                     # Block-Bootstrap-Länge (Volatilitäts-Cluster)
    mc_return_window: int = 500
    calibrate: bool = True                # Walk-Forward-Kalibrierung
    calib_step: int = 5
    calib_min_history: int = 250
    calib_prior_strength: float = 20.0    # Gewicht des Modells ggü. der Historie (Beta-Prior)
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
    filters: SignalFilterConfig = field(default_factory=SignalFilterConfig)
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
