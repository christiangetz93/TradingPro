"""Portfolio-Backtest der Elliott-Wave-Strategie.

Phase 1 (je Titel, parallel): Für jeden Handelstag t wird die Analyse NUR mit
Daten bis t berechnet (WaveContext ist kausal) -> Signal-Kandidaten.

Phase 2 (Portfolio, Tag für Tag):
  * Einstieg frühestens zum Eröffnungskurs des Folgetags (kein Handel zum
    Schlusskurs des Signaltags), Kurslücken werden realistisch behandelt.
  * Alle Kosten: feste und prozentuale Gebühren, Mindestgebühr, Spread, Slippage.
  * Positionsgröße, Portfolio-/Cluster-Heat (inkl. Korrelation), Positionslimit,
    Cash-Reserve, Drawdown-Bremse – identisch zur Live-Logik.

Benchmarks:
  * "random": Zufalls-Einstiege mit denselben Ausstiegsregeln und Stop-Abständen
    -> zeigt, ob die Wellenanalyse überhaupt einen Mehrwert hat.
  * Buy & Hold der gleichen Titel.
  * Null-Daten: Kursreihen ohne Struktur (Block-Bootstrap ohne Drift) -> eine
    korrekte Implementierung darf dort nach Kosten KEINEN Gewinn zeigen.
"""
from __future__ import annotations

import copy
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BotConfig
from .execution import (PendingOrder, Position, make_legs, open_position, process_bar,
                        process_pending)
from .indicators import atr, rsi, trend_state
from .risk import BookState, size_position
from .setup_stats import SetupStats
from .strategy import ConditionalSetup, decide, plan_from_setup
from .waves import WaveContext

WARMUP = 250


@dataclass
class Candidate:
    t: int
    kind: str                      # "market" | "setup"
    side: int
    key: str
    entry_ref: float
    risk_per_unit: float
    crv: float
    plan: dict | None = None       # Markt-Signal: fertiger Plan (Stop, Teilziele, ...)
    levels: list | None = None
    main: tuple | None = None
    setup: ConditionalSetup | None = None
    score: float = 0.0
    p_dir: float = 0.0
    trend: int = 0


@dataclass
class Trade:
    ticker: str
    key: str
    kind: str
    side: int
    entry_date: str
    exit_date: str
    entry: float
    units: float
    risk_amount: float             # geplanter Verlust bis Stop inkl. variabler Kosten
    gross: float                   # Kursgewinn ohne Kosten
    fees: float                    # alle Kosten
    fees_fixed: float              # davon Fix-/Mindestgebühren
    bars: int
    exit_reason: str
    hit_tp1: bool
    hit_main: bool
    mfe_r: float

    @property
    def net(self) -> float:
        return self.gross - self.fees

    @property
    def r_net(self) -> float:
        return self.net / self.risk_amount if self.risk_amount > 0 else 0.0

    @property
    def r_var(self) -> float:
        """R nach variablen Kosten (ohne Fixgebühren) – größenunabhängig."""
        return (self.gross - (self.fees - self.fees_fixed)) / self.risk_amount if self.risk_amount > 0 else 0.0


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    start_capital: float
    fees_total: float
    cancelled: dict[str, int] = field(default_factory=dict)
    exposure: float = 0.0          # mittlerer investierter Anteil
    label: str = ""


# ---------------------------------------------------------------- Phase 1

def _prep(df: pd.DataFrame, cfg: BotConfig):
    a = atr(df, cfg.analysis.atr_period)
    r = rsi(df["close"])
    tr = trend_state(df["close"]).to_numpy(int)
    return a, r, tr


def elliott_candidates(df: pd.DataFrame, cfg: BotConfig, t_from: int, t_to: int) -> list[Candidate]:
    a, r, tr = _prep(df, cfg)
    ctx = WaveContext(df, a, r, cfg.analysis)
    close = df["close"].to_numpy(float)
    atr_v = a.to_numpy(float)
    out: list[Candidate] = []
    last_setup_key = None
    last_setup_t = -10**9
    for t in range(max(t_from, WARMUP), min(t_to, len(df) - 1)):
        wa = ctx.analyze(t)
        dec = decide(wa, close[t], atr_v[t], cfg, int(tr[t]))
        prim = wa.primary
        if dec.signal in ("KAUFEN", "SHORT") and dec.plan is not None:
            p = dec.plan
            out.append(Candidate(t, "market", p.side, p.setup_key, close[t], p.risk_per_unit, p.crv,
                                 plan=p.to_dict(), levels=list(p.levels), main=p.main_target,
                                 score=prim.score, p_dir=wa.p_up if p.side > 0 else wa.p_down,
                                 trend=int(tr[t])))
        elif dec.signal == "WARTEN" and dec.setup is not None and cfg.strategy.trade_setups:
            s = dec.setup
            sk = (s.setup_key, tuple(x.idx for x in s.count.points))
            if sk != last_setup_key or t - last_setup_t >= 5:
                v = cfg.costs.variable_rate()
                risk = abs(s.limit - s.stop) + v * (s.limit + s.stop)
                out.append(Candidate(t, "setup", s.side, s.setup_key, s.limit, risk, s.crv, setup=s,
                                     score=prim.score, p_dir=wa.p_up if prim.next_dir > 0 else wa.p_down,
                                     trend=int(tr[t])))
                last_setup_key, last_setup_t = sk, t
    return out


def random_candidates(df: pd.DataFrame, cfg: BotConfig, t_from: int, t_to: int, template: list[Candidate],
                      rate: float, seed: int, use_trend: bool) -> list[Candidate]:
    """Zufalls-Einstiege mit der Verteilung von Stop-Abstand, Zielen und Haltedauer der Strategie."""
    rng = np.random.default_rng(seed)
    a, _r, tr = _prep(df, cfg)
    close = df["close"].to_numpy(float)
    atr_v = a.to_numpy(float)
    v = cfg.costs.variable_rate()
    shapes = []
    for c in template:
        if c.kind != "market" or c.plan is None:
            continue
        risk_px = abs(c.entry_ref - c.plan["stop"])
        if risk_px <= 0:
            continue
        shapes.append((risk_px / max(c.plan["atr"], 1e-12),
                       [(lv[1] - c.entry_ref) * c.side / risk_px for lv in c.levels],
                       (c.main[1] - c.entry_ref) * c.side / risk_px, c.plan["time_stop_bars"]))
    if not shapes:
        shapes = [(3.0, [1.0, 2.0, 3.0], 2.0, 30)]
    out = []
    for t in range(max(t_from, WARMUP), min(t_to, len(df) - 1)):
        if rng.random() >= rate:
            continue
        if use_trend and tr[t] != 1:
            continue
        k, lv_r, main_r, ts = shapes[rng.integers(len(shapes))]
        e = close[t]
        d = k * atr_v[t]
        stop = e - d
        if stop <= 0:
            continue
        levels = [(f"Ziel {i + 1}", e + x * d) for i, x in enumerate(lv_r)]
        main = ("Hauptziel", e + main_r * d)
        legs, mode = make_legs(levels, main, cfg.exits.mode, cfg, None)
        risk = d + v * (e + stop)
        plan = {"stop": stop, "legs": legs, "invalidation": -1e18, "atr": atr_v[t], "time_stop_bars": ts,
                "exit_mode": mode, "meta": {"setup_key": "Zufall", "state": "-", "kind": "market"}}
        out.append(Candidate(t, "market", 1, "Zufall", e, risk, main_r, plan=plan, levels=levels,
                             main=main, trend=int(tr[t])))
    return out


def _phase1(args):
    ticker, df, cfg, t_from, t_to = args
    return ticker, elliott_candidates(df, cfg, t_from, t_to)


def generate_all(data: dict[str, pd.DataFrame], cfg: BotConfig, start: str | None, end: str | None,
                 workers: int = 4) -> dict[str, list[Candidate]]:
    jobs = []
    for tk, df in data.items():
        t_from = int(np.searchsorted(df.index, pd.Timestamp(start))) if start else 0
        t_to = int(np.searchsorted(df.index, pd.Timestamp(end), side="right")) if end else len(df)
        jobs.append((tk, df, cfg, t_from, t_to))
    if workers <= 1 or len(jobs) == 1:
        return dict(_phase1(j) for j in jobs)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(_phase1, jobs))


# ---------------------------------------------------------------- Phase 2

@dataclass
class _Inst:
    ticker: str
    df: pd.DataFrame
    region: str
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr: np.ndarray
    pos_of: dict


def run_portfolio(data: dict[str, pd.DataFrame], candidates: dict[str, list[Candidate]], cfg: BotConfig,
                  start: str | None = None, end: str | None = None, regions: dict[str, str] | None = None,
                  stats: SetupStats | None = None, label: str = "") -> BacktestResult:
    bud, costs, filt = cfg.budget, cfg.costs, cfg.filters
    regions = regions or {}
    insts: dict[str, _Inst] = {}
    for tk, df in data.items():
        a = atr(df, cfg.analysis.atr_period).to_numpy(float)
        insts[tk] = _Inst(tk, df, regions.get(tk, tk), df["open"].to_numpy(float), df["high"].to_numpy(float),
                          df["low"].to_numpy(float), df["close"].to_numpy(float), a,
                          {d: i for i, d in enumerate(df.index)})
    cand_at: dict[str, dict[int, list[Candidate]]] = {}
    for tk, lst in candidates.items():
        m: dict[int, list[Candidate]] = {}
        for cnd in lst:
            m.setdefault(cnd.t, []).append(cnd)
        cand_at[tk] = m

    cal = sorted(set().union(*[set(d.index) for d in data.values()]))
    if start:
        cal = [d for d in cal if d >= pd.Timestamp(start)]
    if end:
        cal = [d for d in cal if d <= pd.Timestamp(end)]
    rets = pd.DataFrame({tk: np.log(df["close"]).diff() for tk, df in data.items()}).reindex(cal)

    cash = bud.capital
    peak = cash
    positions: dict[str, Position] = {}
    pending: dict[str, tuple[PendingOrder, Candidate, float]] = {}
    trades: list[Trade] = []
    fees_total = 0.0
    cancelled: dict[str, int] = {}
    eq_dates, eq_vals, invested = [], [], []
    last_px: dict[str, float] = {}

    def pay(pos: Position, units: float, price: float) -> float:
        cst = costs.order_cost(units * price)
        pos.fees += cst
        pos.meta["fees_fixed"] = pos.meta.get("fees_fixed", 0.0) + costs.fixed_part(units * price)
        return cst

    def settle(pos: Position, fills, date) -> None:
        nonlocal cash, fees_total
        for f in fills:
            cst = pay(pos, f.units, f.price)
            fees_total += cst
            cash += pos.side * f.units * f.price - cst
            pos.realized += pos.side * f.units * (f.price - pos.entry)
            pos.meta["last_reason"] = f.reason
        if not pos.is_open:
            lv = pos.meta.get("levels", [])
            s = pos.side
            trades.append(Trade(
                ticker=pos.ticker, key=pos.meta.get("setup_key", ""), kind=pos.meta.get("kind", "market"),
                side=s, entry_date=pos.entry_date, exit_date=date, entry=pos.entry, units=pos.units_initial,
                risk_amount=pos.units_initial * pos.risk_per_unit, gross=pos.realized, fees=pos.fees,
                fees_fixed=pos.meta.get("fees_fixed", 0.0), bars=pos.bars_held,
                exit_reason=pos.meta.get("last_reason", ""),
                hit_tp1=bool(lv) and pos.highest >= s * lv[0][1],
                hit_main=bool(pos.meta.get("main")) and pos.highest >= s * pos.meta["main"],
                mfe_r=(pos.highest - s * pos.entry) / pos.risk_per_unit))
            del positions[pos.ticker]

    def book(equity: float) -> BookState:
        return BookState(equity=equity, peak_equity=max(peak, equity), cash=cash,
                         open_risk=sum(p.open_risk() for p in positions.values())
                         + sum(r for _, _, r in pending.values()))

    def cluster_risk(tk: str, di: int) -> float:
        tot = 0.0
        reg = insts[tk].region
        others = list(positions.values()) + [o for o, _, _ in pending.values()]
        for p in others:
            r = p.open_risk() if isinstance(p, Position) else pending[p.ticker][2]
            w = 1.0 if insts[p.ticker].region == reg else 0.0
            if bud.correlation_heat and w < 1.0:
                win = rets.iloc[max(0, di - bud.corr_window):di + 1][[tk, p.ticker]].dropna()
                if len(win) > 30:
                    w = max(w, float(win[tk].corr(win[p.ticker])))
            tot += max(0.0, w) * r
        return tot

    for di, d in enumerate(cal):
        ds = str(d.date())
        # 1) Pending-Orders und Positionen mit dem heutigen Balken
        for tk, inst in insts.items():
            i = inst.pos_of.get(d)
            if i is None:
                continue
            o, h, l, c = inst.o[i], inst.h[i], inst.l[i], inst.c[i]
            last_px[tk] = c
            if tk in pending:
                order, cnd, reserved = pending[tk]
                status, val = process_pending(order, i, o, h, l, c, inst.h[i - 1], inst.l[i - 1])
                if status == "cancel":
                    cancelled[val] = cancelled.get(val, 0) + 1
                    del pending[tk]
                elif status == "fill":
                    del pending[tk]
                    fill = float(val)
                    if cnd.kind == "setup":
                        plan = plan_from_setup(cnd.setup, fill, inst.atr[i - 1], cfg)
                        if plan.crv < 0.75 * filt.min_crv:
                            cancelled["CRV nach Ausführung zu gering"] = cancelled.get("CRV nach Ausführung zu gering", 0) + 1
                            continue
                        pdict, levels, main = plan.to_dict(), plan.levels, plan.main_target
                    else:
                        pdict, levels, main = cnd.plan, cnd.levels, cnd.main
                    v = costs.variable_rate()
                    risk_unit = abs(fill - pdict["stop"]) + v * (fill + pdict["stop"])
                    units = min(order.units, reserved / risk_unit) if risk_unit > 0 else 0.0
                    if not bud.allow_fractional_shares:
                        units = math.floor(units)
                    if units <= 0 or (fill - pdict["stop"]) * cnd.side <= 0:
                        cancelled["Einstieg nicht mehr sinnvoll"] = cancelled.get("Einstieg nicht mehr sinnvoll", 0) + 1
                        continue
                    pdict = dict(pdict)
                    legs, mode = make_legs(levels, main, cfg.exits.mode, cfg, units * fill)
                    pdict["legs"], pdict["exit_mode"] = legs, mode
                    pos = open_position(tk, cnd.side, units, fill, ds, pdict, cfg)
                    pos.region = inst.region
                    pos.meta.update({"levels": [list(x) for x in levels], "main": main[1],
                                     "setup_key": cnd.key, "kind": cnd.kind})
                    cst = pay(pos, units, fill)
                    fees_total += cst
                    cash -= cnd.side * units * fill + cst
                    positions[tk] = pos
                    pos.bars_held = -1          # Einstiegstag zählt nicht als Haltetag
                    if order.kind == "limit" and cnd.kind == "setup":
                        fills = process_bar(pos, fill, max(fill, c), l, c, ds, inst.atr[i])
                    else:
                        fills = process_bar(pos, o, h, l, c, ds, inst.atr[i])
                    settle(pos, fills, ds)
                    continue
            if tk in positions:
                pos = positions[tk]
                fills = process_bar(pos, o, h, l, c, ds, inst.atr[i])
                settle(pos, fills, ds)

        # 2) Bewertung zum Schlusskurs
        pos_val = sum(p.side * p.units * last_px.get(p.ticker, p.entry) for p in positions.values())
        equity = cash + pos_val
        peak = max(peak, equity)
        eq_dates.append(d)
        eq_vals.append(equity)
        invested.append(sum(p.units * last_px.get(p.ticker, p.entry) for p in positions.values()) / max(equity, 1e-9))

        # 3) Neue Signale des Tages -> Orders für morgen
        todays: list[tuple[str, Candidate]] = []
        for tk, inst in insts.items():
            i = inst.pos_of.get(d)
            if i is None or tk in positions or tk in pending:
                continue
            for cnd in cand_at.get(tk, {}).get(i, []):
                todays.append((tk, cnd))
        if not todays:
            continue
        exp_cache = {}

        def expectancy(cnd: Candidate):
            if stats is None:
                return None
            if cnd.key not in exp_cache:
                exp_cache[cnd.key] = stats.expectancy(cnd.key, cfg.strategy.stats_prior_n)
            return exp_cache[cnd.key]

        def prio(item):
            e = expectancy(item[1])
            return (e[0] if e else 0.0, item[1].score)

        todays.sort(key=prio, reverse=True)
        for tk, cnd in todays:
            if len(positions) + len(pending) >= bud.max_positions:
                break
            if tk in pending or tk in positions:
                continue
            e = expectancy(cnd)
            if cfg.strategy.use_setup_stats:
                if e is None or e[0] < filt.min_expected_r:
                    continue
                mean_r, m2 = e[0], e[1]
            else:
                mean_r, m2 = 1.0, 1.0           # ohne Statistik: nur fixes Risiko je Trade begrenzt
            bk = book(equity)
            bk.cluster_risk = {insts[tk].region: cluster_risk(tk, di)}
            size = size_position(cnd.entry_ref, cnd.risk_per_unit, mean_r, m2, insts[tk].region, bk, bud,
                                 costs=costs, n_orders=1 + max(1, len((cnd.plan or {}).get("legs", [1, 1, 1]))))
            if not size.ok:
                continue
            if filt.max_cost_r > 0:
                cost_r = (2 * costs.variable_rate() * cnd.entry_ref * size.units + size.fixed_fees) / \
                    max(size.risk_amount, 1e-9)
                if cost_r > filt.max_cost_r:
                    cancelled["Kosten zu hoch im Verhältnis zum Risiko"] = \
                        cancelled.get("Kosten zu hoch im Verhältnis zum Risiko", 0) + 1
                    continue
            if cfg.strategy.use_setup_stats and filt.min_net_profit > 0:
                fees_est = size.fixed_fees + 2 * costs.variable_rate() * size.invest
                exp_net = mean_r * size.units * cnd.risk_per_unit - size.fixed_fees
                if exp_net < max(filt.min_net_profit, filt.fee_multiple * fees_est):
                    cancelled["Erwarteter Nettogewinn zu klein ggü. Gebühren"] = \
                        cancelled.get("Erwarteter Nettogewinn zu klein ggü. Gebühren", 0) + 1
                    continue
            i = insts[tk].pos_of[d]
            if cnd.kind == "market":
                order = PendingOrder(tk, cnd.side, "market", size.units, cnd.plan, i, i + 1)
            else:
                s = cnd.setup
                kind = "limit" if cfg.strategy.setup_entry == "limit" else "confirm"
                order = PendingOrder(tk, cnd.side, kind, size.units, {"stop": s.stop}, i, i + s.expiry_bars,
                                     limit=s.limit, zone=s.zone, cancel_level=s.cancel_level)
            pending[tk] = (order, cnd, size.risk_amount)

    # offene Positionen zum letzten Kurs bewerten (nicht schließen)
    equity_s = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates), name=label or "Strategie")
    return BacktestResult(equity=equity_s, trades=trades, start_capital=bud.capital, fees_total=fees_total,
                          cancelled=cancelled, exposure=float(np.mean(invested)) if invested else 0.0, label=label)


# ---------------------------------------------------------------- Kennzahlen

def trade_stats(trades: list[Trade], use: str = "r_net") -> dict:
    if not trades:
        return {"Trades": 0}
    r = np.array([getattr(t, use) for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    se = r.std(ddof=1) / math.sqrt(len(r)) if len(r) > 1 else float("nan")
    return {
        "Trades": len(r), "Trefferquote": float((r > 0).mean()), "Ø R": float(r.mean()),
        "Median R": float(np.median(r)), "t-Wert": float(r.mean() / se) if se and se > 0 else float("nan"),
        "Profitfaktor": float(pf), "Ø Gewinn R": float(wins.mean()) if len(wins) else 0.0,
        "Ø Verlust R": float(losses.mean()) if len(losses) else 0.0,
        "TP1 erreicht": float(np.mean([t.hit_tp1 for t in trades])),
        "Ø Haltedauer": float(np.mean([t.bars for t in trades])),
    }


def equity_stats(res: BacktestResult) -> dict:
    eq = res.equity
    if len(eq) < 2:
        return {}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / res.start_capital - 1
    cagr = (eq.iloc[-1] / res.start_capital) ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    r = eq.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0
    gross = sum(t.gross for t in res.trades)
    net = sum(t.net for t in res.trades)
    return {"Endkapital": float(eq.iloc[-1]), "Rendite gesamt": float(total), "CAGR": float(cagr),
            "Max. Drawdown": dd, "Sharpe": sharpe, "Trades": len(res.trades),
            "Trades/Jahr": len(res.trades) / years if years > 0 else 0.0,
            "Kosten gesamt": res.fees_total, "Brutto-Gewinn": gross, "Netto-Gewinn": net,
            "Kostenquote": res.fees_total / gross if gross > 0 else float("nan"),
            "Investitionsgrad": res.exposure, "Jahre": years}


def buy_and_hold(data: dict[str, pd.DataFrame], start: str | None, end: str | None) -> dict:
    out = {}
    for tk, df in data.items():
        c = df["close"]
        if start:
            c = c[c.index >= pd.Timestamp(start)]
        if end:
            c = c[c.index <= pd.Timestamp(end)]
        if len(c) < 30:
            continue
        yrs = (c.index[-1] - c.index[0]).days / 365.25
        out[tk] = {"CAGR": (c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1, "Max. Drawdown": float((c / c.cummax() - 1).min())}
    return out


def null_data(df: pd.DataFrame, seed: int, block: int = 10) -> pd.DataFrame:
    """Strukturlose Vergleichsreihe: Block-Bootstrap der Tagesbalken, Drift entfernt."""
    rng = np.random.default_rng(seed)
    c = df["close"].to_numpy(float)
    prev = np.concatenate([[c[0]], c[:-1]])
    r_oc = np.log(df["open"].to_numpy(float) / prev)
    r_co = np.log(c / df["open"].to_numpy(float))
    r_h = np.log(df["high"].to_numpy(float) / c)
    r_l = np.log(df["low"].to_numpy(float) / c)
    total = r_oc + r_co
    # Drift so entfernen, dass der ARITHMETISCHE Erwartungswert 0 ist: E[exp(r)] = 1.
    # (Nur den log-Mittelwert abzuziehen ließe einen Aufwärtsdrift von ~σ²/2 je Tag übrig.)
    mu = float(np.log(np.mean(np.exp(total[1:]))))
    n = len(df)
    starts = rng.integers(1, n - block, size=n // block + 2)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    ro, rc = r_oc[idx] - mu / 2, r_co[idx] - mu / 2
    logc = np.log(c[0]) + np.cumsum(ro + rc)
    close = np.exp(logc)
    open_ = np.exp(logc - rc)
    high = np.maximum(close * np.exp(r_h[idx]), np.maximum(open_, close))
    low = np.minimum(close * np.exp(r_l[idx]), np.minimum(open_, close))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": df["volume"].to_numpy(float)}, index=df.index)


def clone_cfg(cfg: BotConfig) -> BotConfig:
    return copy.deepcopy(cfg)


def study_config(cfg: BotConfig, gate: bool = False) -> BotConfig:
    """Konfiguration für die Signal-Studie: ein Titel, keine Kapital-/Heat-Grenzen,
    sehr großes Konto (Fixgebühren vernachlässigbar) -> reine Signalqualität in R."""
    c = clone_cfg(cfg)
    b = c.budget
    b.capital, b.risk_per_trade = 1e8, 0.0001
    b.max_portfolio_heat = b.max_cluster_heat = b.max_position_pct = 1.0
    b.min_cash_reserve_pct, b.max_positions = 0.0, 1
    b.drawdown_soft = b.drawdown_hard = 10.0
    b.correlation_heat = False
    b.allow_fractional_shares = True
    if not gate:
        c.strategy.use_setup_stats = False
    c.filters.min_net_profit = 0.0
    return c


def signal_study(data: dict[str, pd.DataFrame], candidates: dict[str, list[Candidate]], cfg: BotConfig,
                 start: str | None = None, end: str | None = None, stats: SetupStats | None = None,
                 gate: bool = False) -> list[Trade]:
    c = study_config(cfg, gate)
    trades: list[Trade] = []
    for tk, df in data.items():
        res = run_portfolio({tk: df}, {tk: candidates.get(tk, [])}, c, start, end,
                            stats=stats if gate else None)
        trades.extend(res.trades)
    return trades
