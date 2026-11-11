"""Fonds-Universum laden und gewichtet zufällig Titel auswählen."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SelectionConfig

DEFAULT_UNIVERSE = Path(__file__).parent / "data" / "universe_top200.csv"


@dataclass(frozen=True)
class Fund:
    ticker: str
    name: str
    region: str
    asset_class: str
    segment: str
    em: bool
    growth_theme: bool

    @property
    def is_growth_market(self) -> bool:
        return self.em or self.growth_theme


def load_universe(path: str | Path | None = None) -> list[Fund]:
    df = pd.read_csv(path or DEFAULT_UNIVERSE)
    required = {"ticker", "name", "region", "asset_class", "segment", "em", "growth_theme"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Universum-Datei: Spalten fehlen: {sorted(missing)}")
    return [
        Fund(
            ticker=str(r.ticker).strip(),
            name=str(r.name).strip(),
            region=str(r.region).strip(),
            asset_class=str(r.asset_class).strip(),
            segment=str(r.segment).strip(),
            em=bool(int(r.em)),
            growth_theme=bool(int(r.growth_theme)),
        )
        for r in df.itertuples(index=False)
    ]


def selection_weight(fund: Fund, cfg: SelectionConfig) -> float:
    w = cfg.base_weight
    if fund.em:
        w = max(w, cfg.em_weight)
    if fund.growth_theme:
        w = max(w, cfg.growth_theme_weight)
    if fund.em and fund.growth_theme:
        w = cfg.em_weight + 0.5 * cfg.growth_theme_weight
    return w


class RandomSelector:
    """Zieht Titel zufällig (gewichtet) ohne Zurücklegen.

    Wachstumsmärkte (Schwellenländer) und Wachstumsthemen erhalten ein höheres
    Gewicht. ``max_per_region`` verhindert Klumpenrisiken.
    """

    def __init__(self, universe: list[Fund], cfg: SelectionConfig):
        self.universe = list(universe)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._weights = np.array([selection_weight(f, cfg) for f in self.universe], dtype=float)
        self._remaining = np.ones(len(self.universe), dtype=bool)
        self._per_region: dict[str, int] = {}

    def draw(self) -> Fund | None:
        mask = self._remaining.copy()
        for i, f in enumerate(self.universe):
            if mask[i] and self._per_region.get(f.region, 0) >= self.cfg.max_per_region:
                mask[i] = False
        if not mask.any():
            mask = self._remaining.copy()          # Regionen-Limit erschöpft -> Rest freigeben
        if not mask.any():
            return None
        w = np.where(mask, self._weights, 0.0)
        idx = int(self.rng.choice(len(self.universe), p=w / w.sum()))
        self._remaining[idx] = False
        fund = self.universe[idx]
        self._per_region[fund.region] = self._per_region.get(fund.region, 0) + 1
        return fund

    def release_region(self, fund: Fund) -> None:
        """Gibt den Regions-Slot wieder frei (z. B. wenn keine Daten ladbar waren)."""
        self._per_region[fund.region] = max(0, self._per_region.get(fund.region, 0) - 1)

    def growth_share(self) -> float:
        g = sum(w for w, f in zip(self._weights, self.universe) if f.is_growth_market)
        return g / self._weights.sum()
