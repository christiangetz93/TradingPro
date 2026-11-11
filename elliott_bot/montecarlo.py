"""Monte-Carlo-Simulation eines Trade-Plans inkl. Ausstiegsregeln.

Pfade entstehen per Block-Bootstrap aus den historischen Tagesrenditen des
Titels (Volatilitäts-Cluster und Fat Tails bleiben erhalten). Die Renditen
werden um ihren Mittelwert bereinigt; der Drift ist ein expliziter Parameter:

* drift = 0  -> "Zufallspfad": Was bringt der Plan, wenn die Wellenanalyse
               KEINEN Vorhersagewert hat? (kritischer Benchmark)
* drift > 0  -> kalibrierter Drift, bei dem die Treffer-Wahrscheinlichkeit
               der des Modells entspricht.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BotConfig
from .strategy import TradePlan


@dataclass
class MCResult:
    drift: float
    p_tp1: float           # TP1 vor Initial-Stop
    p_main: float          # Hauptziel vor Initial-Stop (reine Barriere)
    p_stop_first: float    # Initial-Stop vor TP1
    p_time_exit: float     # Zeitstopp ohne Zielerreichung
    r: np.ndarray          # realisierte R-Vielfache je Pfad (mit Exit-Management)
    avg_bars: float

    @property
    def mean_r(self) -> float:
        return float(self.r.mean())

    @property
    def median_r(self) -> float:
        return float(np.median(self.r))

    @property
    def p_loss(self) -> float:
        return float((self.r < 0).mean())

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.r, q))

    @property
    def cvar5(self) -> float:
        cut = self.quantile(0.05)
        tail = self.r[self.r <= cut]
        return float(tail.mean()) if len(tail) else cut

    @property
    def second_moment(self) -> float:
        return float((self.r ** 2).mean())


class BootstrapSampler:
    def __init__(self, df: pd.DataFrame, window: int, block: int, n_paths: int,
                 horizon: int, seed: int | None):
        d = df.iloc[-(window + 1):]
        c = d["close"].to_numpy(float)
        r = np.diff(np.log(c))
        up = np.log(d["high"].to_numpy(float)[1:] / c[1:])
        dn = np.log(d["low"].to_numpy(float)[1:] / c[1:])
        self.r = r - r.mean()
        self.up, self.dn = up, dn
        self.sigma = float(r.std())
        n = len(r)
        block = max(1, min(block, n // 4))
        n_blocks = -(-horizon // block)
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, n - block + 1, size=(n_paths, n_blocks))
        self.idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :horizon]
        self.horizon = horizon

    def paths(self, entry: float, drift: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logc = np.cumsum(self.r[self.idx] + drift, axis=1)
        close = entry * np.exp(logc)
        return close, close * np.exp(self.up[self.idx]), close * np.exp(self.dn[self.idx])


def simulate_plan(plan: TradePlan, sampler: BootstrapSampler, cfg: BotConfig,
                  drift: float = 0.0) -> MCResult:
    ex = cfg.exits
    cost = cfg.budget.fee_pct + cfg.budget.slippage_pct
    s = plan.side
    close, high, low = sampler.paths(plan.entry, s * drift)
    if s == 1:
        cq, hq, lq = close, high, low
    else:
        cq, hq, lq = -close, -low, -high
    e, st0 = s * plan.entry, s * plan.stop
    t1, t2, t3 = (s * p for _, p in plan.levels)
    main = s * plan.main_target[1]
    P, H = cq.shape
    H = min(H, max(plan.time_stop_bars * 2, plan.time_stop_bars + 5))

    stop = np.full(P, st0)
    stage = np.zeros(P, dtype=int)
    rem = np.ones(P)
    pnl = np.zeros(P)
    done = np.zeros(P, dtype=bool)
    highest = np.full(P, e)
    exit_bar = np.full(P, H, dtype=float)
    time_exit = np.zeros(P, dtype=bool)
    b_done = np.zeros(P, dtype=bool)
    b_tp1 = np.zeros(P, dtype=bool)
    b_main = np.zeros(P, dtype=bool)
    b_stop_first = np.zeros(P, dtype=bool)
    be_level = e + cost * 2 * abs(e)
    f1, f2 = ex.tp1_fraction, ex.tp2_fraction

    for t in range(H):
        h, lo, c = hq[:, t], lq[:, t], cq[:, t]
        # Barriere-Statistik (unabhängig vom Management; pessimistisch: Stop zuerst)
        bd = ~b_done
        sl = bd & (lo <= st0)
        b_stop_first |= sl & ~b_tp1
        ok = bd & ~sl
        b_tp1 |= ok & (h >= t1)
        hit_main = ok & (h >= main)
        b_main |= hit_main
        b_done |= sl | hit_main

        # Exit-Management
        a = ~done
        hit = a & (lo <= stop)
        pnl += np.where(hit, rem * (stop - e), 0.0)
        rem = np.where(hit, 0.0, rem)
        exit_bar[hit] = t
        done |= hit
        a = ~done
        s0 = a & (stage == 0) & (h >= t1)
        pnl += np.where(s0, f1 * (t1 - e), 0.0)
        rem = np.where(s0, rem - f1, rem)
        stage[s0] = 1
        stop = np.where(s0, np.maximum(stop, be_level), stop)
        s1 = a & (stage == 1) & (h >= t2)
        pnl += np.where(s1, f2 * (t2 - e), 0.0)
        rem = np.where(s1, rem - f2, rem)
        stage[s1] = 2
        stop = np.where(s1, np.maximum(stop, t1), stop)
        s2 = a & (stage == 2) & (h >= t3)
        pnl += np.where(s2, rem * (t3 - e), 0.0)
        rem = np.where(s2, 0.0, rem)
        exit_bar[s2] = t
        done |= s2
        highest = np.maximum(highest, h)
        trail = highest - ex.trail_atr_mult * plan.atr
        stop = np.where((stage >= 1) & ~done, np.maximum(stop, trail), stop)
        if t == plan.time_stop_bars - 1:
            ts = ~done & (stage == 0)
            pnl += np.where(ts, rem * (c - e), 0.0)
            rem = np.where(ts, 0.0, rem)
            exit_bar[ts] = t
            time_exit |= ts
            done |= ts
        if done.all():
            break

    open_ = ~done
    pnl += np.where(open_, rem * (cq[:, H - 1] - e), 0.0)
    pnl -= cost * 2 * abs(e)
    r = pnl / plan.risk_per_unit
    return MCResult(drift=drift, p_tp1=float(b_tp1.mean()), p_main=float(b_main.mean()),
                    p_stop_first=float(b_stop_first.mean()), p_time_exit=float(time_exit.mean()),
                    r=r, avg_bars=float(exit_bar.mean()))


def solve_drift(plan: TradePlan, sampler: BootstrapSampler, cfg: BotConfig,
                target_p: float, iters: int = 14) -> MCResult:
    """Sucht per Bisektion den Drift, bei dem P(Hauptziel vor Stop) = target_p."""
    lo, hi = -sampler.sigma, sampler.sigma
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        res = simulate_plan(plan, sampler, cfg, mid)
        best = res
        if abs(res.p_main - target_p) < 0.004:
            break
        if res.p_main < target_p:
            lo = mid
        else:
            hi = mid
    return best
