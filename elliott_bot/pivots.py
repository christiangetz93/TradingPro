"""ZigZag-Pivoterkennung mit ATR-basierter Schwelle.

Ein Hoch/Tief gilt als bestätigt, sobald sich der Kurs um ``mult * ATR`` davon
entfernt hat. Der letzte Pivot ist immer *vorläufig* (laufendes Extrem).
Verschiedene Multiplikatoren entsprechen verschiedenen Wellengraden.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Pivot:
    idx: int
    date: pd.Timestamp
    price: float
    kind: str          # "H" (Hoch) oder "L" (Tief)
    confirmed: bool = True


def zigzag(df: pd.DataFrame, atr_series: pd.Series, mult: float) -> list[Pivot]:
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    thr = (atr_series.to_numpy(float) * mult)
    dates = df.index
    n = len(df)
    if n < 3:
        return []

    pivots: list[Pivot] = []
    trend = 0
    hi_i, lo_i = 0, 0
    for i in range(1, n):
        if trend == 0:
            if high[i] > high[hi_i]:
                hi_i = i
            if low[i] < low[lo_i]:
                lo_i = i
            if high[hi_i] - low[lo_i] >= thr[i]:
                if lo_i < hi_i:
                    pivots.append(Pivot(lo_i, dates[lo_i], low[lo_i], "L"))
                    trend, ext = 1, hi_i
                else:
                    pivots.append(Pivot(hi_i, dates[hi_i], high[hi_i], "H"))
                    trend, ext = -1, lo_i
            continue
        if trend == 1:
            if high[i] >= high[ext]:
                ext = i
            elif high[ext] - low[i] >= thr[i]:
                pivots.append(Pivot(ext, dates[ext], high[ext], "H"))
                trend, ext = -1, i
        else:
            if low[i] <= low[ext]:
                ext = i
            elif high[i] - low[ext] >= thr[i]:
                pivots.append(Pivot(ext, dates[ext], low[ext], "L"))
                trend, ext = 1, i

    if trend != 0:
        kind = "H" if trend == 1 else "L"
        price = high[ext] if trend == 1 else low[ext]
        pivots.append(Pivot(ext, dates[ext], price, kind, confirmed=False))
    return pivots


def pivot_arrays(pivots: list[Pivot]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([p.idx for p in pivots], dtype=int),
            np.array([p.price for p in pivots], dtype=float))
