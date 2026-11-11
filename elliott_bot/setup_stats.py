"""Gelernte Statistik je Setup-Typ (z. B. "M-W3-mit Trend", "S-W4-mit Trend").

Aus Backtest-Trades wird je Setup-Typ die Verteilung des Ergebnisses in R
(nach variablen Kosten) gespeichert. Für Entscheidungen wird der Mittelwert
zur Null hin geschrumpft (Bayes mit ``prior_n`` Pseudo-Trades à 0 R):
Wenige gute Trades reichen also nicht, um ein Setup freizuschalten.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).parent / "data" / "setup_stats.json"


@dataclass
class StatRow:
    n: int
    sum_r: float
    sum_r2: float
    wins: int
    tp1: int
    main: int

    @property
    def mean(self) -> float:
        return self.sum_r / self.n if self.n else 0.0


class SetupStats:
    def __init__(self, rows: dict[str, StatRow] | None = None, meta: dict | None = None):
        self.rows = rows or {}
        self.meta = meta or {}

    @classmethod
    def from_trades(cls, trades, meta: dict | None = None) -> "SetupStats":
        rows: dict[str, StatRow] = {}
        for t in trades:
            r = float(t.r_var)
            row = rows.setdefault(t.key, StatRow(0, 0.0, 0.0, 0, 0, 0))
            row.n += 1
            row.sum_r += r
            row.sum_r2 += r * r
            row.wins += int(r > 0)
            row.tp1 += int(bool(t.hit_tp1))
            row.main += int(bool(t.hit_main))
        return cls(rows, meta)

    def expectancy(self, key: str, prior_n: float = 30.0) -> tuple[float, float, int]:
        """(geschrumpfter Mittelwert in R, zweites Moment, Anzahl Trades)."""
        row = self.rows.get(key)
        if row is None or row.n == 0:
            return 0.0, 1.0, 0
        mean = row.sum_r / (row.n + prior_n)
        m2 = max((row.sum_r2 + prior_n * 1.0) / (row.n + prior_n), 1e-6)
        return float(mean), float(m2), row.n

    def describe(self, key: str, prior_n: float = 30.0) -> dict | None:
        row = self.rows.get(key)
        if row is None or row.n == 0:
            return None
        sd = np.sqrt(max(row.sum_r2 / row.n - row.mean ** 2, 0.0))
        return {"Trades": row.n, "Trefferquote": row.wins / row.n, "Ø R": row.mean,
                "Ø R (geschrumpft)": self.expectancy(key, prior_n)[0],
                "P(Teilziel 1)": row.tp1 / row.n, "P(Hauptziel)": row.main / row.n,
                "t-Wert": row.mean / (sd / np.sqrt(row.n)) if sd > 0 and row.n > 1 else float("nan")}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"meta": self.meta, "rows": {k: vars(v) for k, v in self.rows.items()}},
                                         indent=1, ensure_ascii=False, default=float))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SetupStats | None":
        p = Path(path) if path else DEFAULT_PATH
        if not p.exists():
            return None
        raw = json.loads(p.read_text())
        return cls({k: StatRow(**v) for k, v in raw["rows"].items()}, raw.get("meta", {}))
