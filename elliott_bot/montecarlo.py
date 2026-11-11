"""Monte-Carlo-Simulation eines Trade-Plans inkl. der echten Ausstiegsregeln.

Pfade entstehen per Block-Bootstrap aus den historischen Tagesrenditen des
Titels (Volatilitäts-Cluster und Fat Tails bleiben erhalten). Die Renditen
werden so um ihren Mittelwert bereinigt, dass der Kurs ohne Drift ein faires
Spiel ist; der Drift ist ein expliziter Parameter:

* drift = 0  -> "Zufallspfad": Was bringt der Plan, wenn die Analyse KEINEN
               Vorhersagewert hat? (kritischer Benchmark)
* drift ≠ 0  -> Drift, bei dem die Treffer-Wahrscheinlichkeit der
               empirischen Setup-Statistik entspricht.

Die Ausstiegslogik ist dieselbe wie in execution.process_bar (vektorisiert).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BotConfig
from .execution import EXIT_MODES
from .strategy import TradePlan


@dataclass
class MCResult:
    drift: float
    p_tp1: float           # erstes Kursziel vor Initial-Stop erreicht
    p_main: float          # Hauptziel vor Initial-Stop erreicht (reine Barriere)
    p_stop_first: float    # Initial-Stop vor dem ersten Kursziel
    p_time_exit: float     # Zeitstopp / maximale Haltedauer
    r: np.ndarray          # realisierte R-Vielfache je Pfad (nach variablen Kosten)
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
        self.up = np.log(d["high"].to_numpy(float)[1:] / c[1:])
        self.dn = np.log(d["low"].to_numpy(float)[1:] / c[1:])
        self.r = r - np.log(np.mean(np.exp(r)))          # fairer Kurs: E[exp(r)] = 1
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
    v = cfg.costs.variable_rate()
    s = plan.side
    close, high, low = sampler.paths(plan.entry, s * drift)
    if s == 1:
        cq, hq, lq = close, high, low
    else:
        cq, hq, lq = -close, -low, -high
    e, st0 = s * plan.entry, s * plan.stop
    inv = s * plan.invalidation
    tp1 = s * plan.levels[0][1]
    main = s * plan.main_target[1]
    _, be_after, prev_leg, trail_start = EXIT_MODES.get(plan.exit_mode, EXIT_MODES["trail"])
    legs_q = [s * lg[1] for lg in plan.legs]
    legs_f = [lg[2] for lg in plan.legs]
    full = sum(legs_f) >= 0.999
    trail_mult = ex.trail_atr_mult if trail_start else 0.0
    P, H = cq.shape
    H = min(H, max(plan.time_stop_bars * 2, plan.time_stop_bars + 5))

    stop = np.full(P, st0)
    nxt = np.zeros(P, dtype=int)
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
    be_level = e + v * 2 * abs(e)

    def close_out(mask, price, bar):
        nonlocal pnl, rem, done
        pnl += np.where(mask, rem * (price - e), 0.0)
        rem = np.where(mask, 0.0, rem)
        exit_bar[mask] = bar
        done |= mask

    for t in range(H):
        h, lo, c = hq[:, t], lq[:, t], cq[:, t]
        bd = ~b_done                                     # Barriere-Statistik
        sl = bd & (lo <= st0)
        b_stop_first |= sl & ~b_tp1
        ok = bd & ~sl
        b_tp1 |= ok & (h >= tp1)
        hit_main = ok & (h >= main)
        b_main |= hit_main
        b_done |= sl | hit_main

        a = ~done                                        # Ausstiegsregeln
        close_out(a & (lo <= stop), stop, t)
        for k, (lvl, frac) in enumerate(zip(legs_q, legs_f)):
            m = ~done & (nxt == k) & (h >= lvl)
            last = full and k == len(legs_q) - 1
            sell = rem if last else np.minimum(rem, frac)
            pnl += np.where(m, sell * (lvl - e), 0.0)
            rem = np.where(m, rem - sell, rem)
            nxt = np.where(m, k + 1, nxt)
            if k == 0 and be_after:
                stop = np.where(m, np.maximum(stop, be_level), stop)
            if k >= 1 and prev_leg:
                stop = np.where(m, np.maximum(stop, legs_q[k - 1]), stop)
            fin = m & (rem <= 1e-12)
            exit_bar[fin] = t
            done |= fin
        highest = np.maximum(highest, h)
        if trail_mult > 0:
            active = ~done & ((nxt > 0) | (trail_start == "entry"))
            stop = np.where(active, np.maximum(stop, highest - trail_mult * plan.atr), stop)
        close_out(~done & (c < inv), c, t)
        if t == plan.time_stop_bars - 1:
            ts = ~done & (nxt == 0)
            time_exit |= ts
            close_out(ts, c, t)
        if done.all():
            break

    close_out(~done, cq[:, H - 1], H - 1)
    pnl -= v * 2 * abs(e)
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
