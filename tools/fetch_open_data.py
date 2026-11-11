"""Lädt die offenen Kursdatensätze, mit denen die Strategie validiert wurde, und bereitet sie auf.

Quellen (öffentliche Beispiel-/Testdaten von Open-Source-Projekten):
  * QuantConnect LEAN (GitHub): SPY, QQQ, IWM, EEM, USO, AAPL, AIG, BAC, IBM, GOOG, FB, Gold (Oanda-CFD)
  * backtrader (GitHub, Yahoo-Exporte): NVDA, ORCL, YHOO
  * arch (PyPI, Yahoo-Daten): S&P-500- und NASDAQ-Composite-Index
  * skfolio (PyPI): Faktor-ETFs MTUM/QUAL/SIZE/USMV/VLUE und 20 S&P-500-Aktien (nur Schlusskurse)

Die Daten werden NICHT mit diesem Repository verteilt, sondern bei dir heruntergeladen.

    python tools/fetch_open_data.py data/open
    python -m elliott_bot backtest --provider csv --csv-dir data/open --tickers ALL --split 2011-01-01
"""
from __future__ import annotations

import gzip
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW = "https://raw.githubusercontent.com"
LEAN = f"{RAW}/QuantConnect/Lean/master/Data"
INDEX_FUNDS = {"SPY", "QQQ", "IWM", "SP500", "NASDAQ"}
FACTORS = {"MTUM", "QUAL", "SIZE", "USMV", "VLUE"}


def get(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def wheel_member(pkg: str, member_suffix: str) -> bytes:
    info = requests.get(f"https://pypi.org/pypi/{pkg}/json", timeout=30).json()
    url = next(f["url"] for f in info["urls"] if f["filename"].endswith(".whl"))
    zf = zipfile.ZipFile(io.BytesIO(get(url)))
    name = next(n for n in zf.namelist() if n.endswith(member_suffix))
    return zf.read(name)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Close"]).sort_index()
    df = df[~df.index.duplicated()]
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)].copy()
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    vol = df["Volume"].astype(float)
    med = vol.rolling(60, min_periods=5).median()
    df["Volume"] = np.where((med > 0) & (vol > 50 * med), med, vol)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def lean_equity(t: str) -> pd.DataFrame:
    raw = zipfile.ZipFile(io.BytesIO(get(f"{LEAN}/equity/usa/daily/{t}.zip"))).read(f"{t}.csv").decode()
    df = pd.read_csv(io.StringIO(raw), header=None, names=["d", "Open", "High", "Low", "Close", "Volume"])
    df.index = pd.to_datetime(df.pop("d").str[:8], format="%Y%m%d")
    df[["Open", "High", "Low", "Close"]] /= 10000.0
    ff = pd.read_csv(io.StringIO(get(f"{LEAN}/equity/usa/factor_files/{t}.csv").decode()),
                     header=None, names=["d", "pf", "sf", "ref"])
    ff["d"] = pd.to_datetime(ff["d"].astype(str), format="%Y%m%d")
    pos = np.clip(np.searchsorted(ff["d"].to_numpy(), df.index.to_numpy(), side="left"), 0, len(ff) - 1)
    factor = (ff["pf"] * ff["sf"]).to_numpy()[pos]
    for c in ("Open", "High", "Low", "Close"):
        df[c] *= factor
    df["Volume"] = df["Volume"] / ff["sf"].to_numpy()[pos]
    if t == "goog":                     # Split 03.04.2014 fehlt in der Faktordatei
        d0 = pd.Timestamp("2014-04-03")
        prev = df.index[df.index < d0][-1]
        ratio = df.loc[d0, "Open"] / df.loc[prev, "Close"]
        for c in ("Open", "High", "Low", "Close"):
            df.loc[df.index < d0, c] *= ratio
    return df[df.index >= ff["d"].iloc[0]]


def close_only(s: pd.Series) -> pd.DataFrame:
    s = s.dropna()
    s = s[s > 0]
    df = pd.DataFrame({"Close": s})
    df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    df["High"] = df[["Open", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Close"]].min(axis=1)
    df["Volume"] = 0.0
    return df


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = []

    def save(t: str, df: pd.DataFrame, group: str, source: str) -> None:
        df = clean(df)
        df.to_csv(out / f"{t}.csv", float_format="%.6f")
        region = "USA-Index" if t in INDEX_FUNDS else ("USA-Faktor" if t in FACTORS else t)
        meta.append((t, group, region, source, str(df.index[0].date()), str(df.index[-1].date()), len(df)))
        print(f"  {t:6s} {df.index[0].date()} – {df.index[-1].date()} ({len(df)} Tage)")

    print("QuantConnect LEAN ...")
    for t in ["spy", "qqq", "iwm", "eem", "uso"]:
        save(t.upper(), lean_equity(t), "fonds", "QuantConnect LEAN")
    for t in ["aapl", "aig", "bac", "ibm", "goog", "fb"]:
        save(t.upper(), lean_equity(t), "aktie", "QuantConnect LEAN")
    zf = zipfile.ZipFile(io.BytesIO(get(f"{LEAN}/cfd/oanda/daily/xauusd.zip")))
    g = pd.read_csv(io.StringIO(zf.read(zf.namelist()[0]).decode()), header=None)
    g.index = pd.to_datetime(g[0].str[:8], format="%Y%m%d")
    gold = pd.DataFrame({"Open": g[1], "High": g[2], "Low": g[3], "Close": g[4], "Volume": 0.0}, index=g.index)
    save("GOLD", gold[gold.index.dayofweek < 5], "fonds", "QuantConnect LEAN (Oanda CFD)")

    print("backtrader (Yahoo-Exporte) ...")
    for f, t in [("nvda-1999-2014", "NVDA"), ("orcl-1995-2014", "ORCL"), ("yhoo-1996-2014", "YHOO")]:
        df = pd.read_csv(io.BytesIO(get(f"{RAW}/mementum/backtrader/master/datas/{f}.txt")),
                         parse_dates=["Date"], index_col="Date")
        fac = df["Adj Close"] / df["Close"]
        for c in ("Open", "High", "Low", "Close"):
            df[c] = df[c] * fac
        save(t, df, "aktie", "backtrader (Yahoo)")

    print("arch (PyPI) ...")
    for member, t in [("sp500.csv.gz", "SP500"), ("nasdaq.csv.gz", "NASDAQ")]:
        df = pd.read_csv(io.BytesIO(gzip.decompress(wheel_member("arch", f"data/{t.lower()}/{member}"))))
        df.index = pd.to_datetime(df.pop("Date"), format="%m/%d/%Y")
        save(t, df, "index", "arch (Yahoo)")

    print("skfolio (PyPI) ...")
    fac = pd.read_csv(io.BytesIO(gzip.decompress(wheel_member("skfolio", "factors_dataset.csv.gz"))),
                      parse_dates=["Date"], index_col="Date")
    for c in fac.columns:
        save(c, close_only(fac[c]), "fonds_close", "skfolio (Faktor-ETFs, nur Schlusskurse)")
    stk = pd.read_csv(io.BytesIO(gzip.decompress(wheel_member("skfolio", "sp500_dataset.csv.gz"))),
                      parse_dates=["Date"], index_col="Date")
    have = {m[0] for m in meta}
    for c in stk.columns:
        if c not in have:
            save(c, close_only(stk[c]), "aktie_close", "skfolio (S&P-500-Aktien, nur Schlusskurse)")

    pd.DataFrame(meta, columns=["ticker", "gruppe", "region", "quelle", "von", "bis", "tage"]).to_csv(
        out / "meta.csv", index=False)
    print(f"\n{len(meta)} Instrumente in {out}/ (Übersicht: meta.csv)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/open")
