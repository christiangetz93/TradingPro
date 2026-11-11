"""Technische Hilfsindikatoren."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    # kausal: kein Auffüllen mit späteren Werten (frühere Version nutzte bfill)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def trend_state(close: pd.Series, length: int = 200, slope_bars: int = 20) -> pd.Series:
    """+1 = Aufwärtstrend (Kurs > steigender SMA), -1 = Abwärtstrend, 0 = neutral/unbekannt."""
    sma = close.rolling(length, min_periods=length).mean()
    rising = sma > sma.shift(slope_bars)
    falling = sma < sma.shift(slope_bars)
    up = (close > sma) & rising
    down = (close < sma) & falling
    return pd.Series(np.where(up, 1, np.where(down, -1, 0)), index=close.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def max_drawdown(close: pd.Series) -> float:
    peak = close.cummax()
    return float((close / peak - 1).min())


def annual_volatility(close: pd.Series) -> float:
    r = np.log(close).diff().dropna()
    return float(r.std() * np.sqrt(252))
