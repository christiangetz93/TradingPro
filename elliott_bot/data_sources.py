"""Kursdatenquellen.

* ``YahooProvider``     – Tageskurse über die öffentliche Yahoo-Finance-Chart-API
                          (benötigt Internetzugang, mit lokalem Cache).
* ``CSVProvider``       – eigene CSV-Dateien ``<TICKER>.csv`` (Date,Open,High,Low,Close,Volume).
* ``SyntheticProvider`` – künstliche, Elliott-ähnliche Kursverläufe für Tests/Demo.
                          Ergebnisse daraus sind KEINE Marktanalyse.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

COLUMNS = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    pass


@dataclass
class PriceSeries:
    ticker: str
    df: pd.DataFrame          # Spalten: open, high, low, close, volume; DatetimeIndex
    currency: str
    source: str
    synthetic: bool = False


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df


class YahooProvider:
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    def __init__(self, cache_dir: str | Path = ".cache/prices", max_age_hours: float = 12.0,
                 timeout: float = 15.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age_hours * 3600
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (compatible; research-client)"

    def get(self, ticker: str, years: int = 5) -> PriceSeries:
        cache = self.cache_dir / f"{ticker.replace('/', '_')}.json"
        if cache.exists() and time.time() - cache.stat().st_mtime < self.max_age:
            payload = json.loads(cache.read_text())
        else:
            try:
                r = self.session.get(self.URL.format(ticker=ticker),
                                     params={"range": f"{years}y", "interval": "1d",
                                             "includeAdjustedClose": "true"},
                                     timeout=self.timeout)
                r.raise_for_status()
                payload = r.json()
            except (requests.RequestException, ValueError) as exc:
                raise DataError(f"{ticker}: Download fehlgeschlagen ({exc})") from exc
            cache.write_text(json.dumps(payload))
        try:
            res = payload["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
            df = pd.DataFrame({k: q[k] for k in COLUMNS}, index=idx)
            adj = res["indicators"].get("adjclose")
            if adj and adj[0].get("adjclose"):
                # Ausschüttungen/Splits bereinigen: OHLC proportional anpassen
                factor = np.array(adj[0]["adjclose"], dtype=float) / df["close"].to_numpy(dtype=float)
                factor = np.where(np.isfinite(factor), factor, 1.0)
                for col in ("open", "high", "low", "close"):
                    df[col] = df[col] * factor
            currency = res.get("meta", {}).get("currency", "USD")
        except (KeyError, IndexError, TypeError) as exc:
            raise DataError(f"{ticker}: unerwartetes Datenformat") from exc
        df = _clean(df)
        if len(df) < 200:
            raise DataError(f"{ticker}: zu wenig Historie ({len(df)} Tage)")
        return PriceSeries(ticker, df, currency, "Yahoo Finance")


class CSVProvider:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def get(self, ticker: str, years: int = 5) -> PriceSeries:
        path = self.directory / f"{ticker}.csv"
        if not path.exists():
            raise DataError(f"{ticker}: Datei {path} nicht gefunden")
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "adj close" in df.columns:
            factor = df["adj close"] / df["close"]
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] * factor
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df.index = pd.to_datetime(df.pop("date"))
        df = _clean(df)
        if len(df) < 200:
            raise DataError(f"{ticker}: zu wenig Historie ({len(df)} Tage)")
        return PriceSeries(ticker, df, "n/a", f"CSV ({path.name})")


class SyntheticProvider:
    """Erzeugt reproduzierbare Kursreihen aus verschachtelten Elliott-Mustern.

    Die Reihe beginnt immer am selben Datum; der Verlauf früherer Tage bleibt
    gleich, wenn später weitere Tage hinzukommen (wichtig für den Paper-Trading-Modus).
    """

    START = pd.Timestamp("2019-01-02")

    def __init__(self, seed: int = 7, end: pd.Timestamp | None = None):
        self.seed = seed
        self.end = (end or pd.Timestamp.today()).normalize()

    def _rng(self, ticker: str) -> np.random.Generator:
        h = int(hashlib.sha256(f"{ticker}:{self.seed}".encode()).hexdigest()[:12], 16)
        return np.random.default_rng(h)

    @staticmethod
    def _impulse_ratios(rng: np.random.Generator) -> list[float]:
        w1 = 1.0
        w2 = w1 * rng.uniform(0.38, 0.79)
        w3 = w1 * rng.choice([1.0, 1.618, 2.0, 2.618], p=[0.15, 0.45, 0.2, 0.2]) * rng.uniform(0.9, 1.1)
        w4 = min(w3 * rng.uniform(0.22, 0.5), w1 + w3 - w2 - 0.05)   # keine Überlappung mit W1
        w5 = w1 * rng.uniform(0.618, 1.2)
        return [w1, -w2, w3, -w4, w5]

    @staticmethod
    def _zigzag_ratios(rng: np.random.Generator) -> list[float]:
        a = 1.0
        return [a, -a * rng.uniform(0.38, 0.8), a * rng.uniform(0.8, 1.65)]

    def _split(self, rng: np.random.Generator, move: float, motive: bool, depth: int) -> list[float]:
        """Teilt eine Bewegung rekursiv in Unterwellen (5 bei Impuls-, 3 bei Korrekturwellen)."""
        if depth == 0:
            return [move]
        ratios = self._impulse_ratios(rng) if motive else self._zigzag_ratios(rng)
        scale = move / sum(ratios)
        out: list[float] = []
        for k, r in enumerate(ratios):
            sub_motive = (k % 2 == 0)       # 1,3,5 bzw. A,C sind Antriebswellen
            out.extend(self._split(rng, r * scale, sub_motive, depth - 1))
        return out

    def _moves(self, rng: np.random.Generator, n_needed: int) -> list[float]:
        moves: list[float] = []
        trend = 1.0 if rng.random() < 0.7 else -1.0
        while len(moves) < n_needed:
            size = rng.uniform(0.25, 0.6)                   # Impuls auf hohem Grad (log)
            impulse = [r * size / sum(self._impulse_ratios(rng)) for r in self._impulse_ratios(rng)]
            total = sum(impulse)
            a = total * rng.uniform(0.3, 0.62)
            correction = [-a, a * rng.uniform(0.38, 0.8), -a * rng.uniform(0.8, 1.65)]
            for k, m in enumerate(impulse + correction):
                motive = k in (0, 2, 4, 5, 7)
                moves.extend(self._split(rng, trend * m, motive, depth=1))
            if rng.random() < 0.25:
                trend = -trend
        return moves

    def get(self, ticker: str, years: int = 5) -> PriceSeries:
        rng = self._rng(ticker)
        dates = pd.bdate_range(self.START, self.end)
        n = len(dates)
        start_price = float(rng.uniform(20, 300))
        daily_vol = float(rng.uniform(0.006, 0.012))      # Rauschen innerhalb der kleinsten Wellen
        speed = float(rng.uniform(250, 400))              # Handelstage je 100 % log-Bewegung
        logp = [np.log(start_price)]
        vols: list[float] = []
        for move in self._moves(rng, 3000):
            dur = int(max(3, abs(move) * speed * rng.uniform(0.7, 1.3)))
            steps = rng.normal(0, daily_vol, dur)
            path = np.cumsum(steps)
            bridge = path - np.linspace(0, 1, dur) * path[-1]
            seg = logp[-1] + np.linspace(move / dur, move, dur) + bridge
            logp.extend(seg.tolist())
            vols.extend([abs(move) / dur] * dur)
            if len(logp) > n + 1:
                break
        close = np.exp(np.array(logp[1:n + 1]))
        noise = np.abs(rng.normal(0, daily_vol * 0.6, (2, n)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) * (1 + noise[0])
        low = np.minimum(open_, close) * (1 - noise[1])
        trend_strength = np.array(vols[:n]) / (np.mean(vols[:n]) + 1e-12)
        volume = rng.lognormal(13, 0.3, n) * (0.6 + 0.4 * trend_strength)
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                           "volume": volume}, index=dates)
        df = df.iloc[-int(years * 252):]
        return PriceSeries(ticker, _clean(df), "SYN", "Synthetische Testdaten", synthetic=True)


def make_provider(kind: str, **kwargs):
    kind = kind.lower()
    if kind == "yahoo":
        return YahooProvider(**kwargs)
    if kind == "csv":
        return CSVProvider(kwargs.get("directory", "data"))
    if kind == "synthetic":
        return SyntheticProvider(**kwargs)
    raise ValueError(f"Unbekannte Datenquelle: {kind}")
