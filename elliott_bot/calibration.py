"""Walk-Forward-Kalibrierung der Treffer-Wahrscheinlichkeit.

Für jeden historischen Zeitpunkt t (Schrittweite ``calib_step``) wird die
Analyse NUR mit Daten bis t ausgeführt (kein Blick in die Zukunft). Ergibt sich
ein Signal, wird geprüft, was danach tatsächlich passiert ist:
Hauptziel zuerst (Treffer), Stop zuerst (Fehlschlag) oder Zeitstopp.

Der gemessene Vorsprung gegenüber einem Zufallspfad (Basisquote b/(a+b))
fließt per Bayes-Mittelung in die aktuelle Wahrscheinlichkeit ein.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BotConfig
from .strategy import decide
from .waves import analyze_waves


@dataclass
class CalibResult:
    signals: int = 0
    hits: int = 0
    stops: int = 0
    timeouts: int = 0
    baseline_mean: float = 0.0
    realized_r: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.signals if self.signals else float("nan")

    @property
    def empirical_lift(self) -> float:
        return self.hit_rate - self.baseline_mean if self.signals else 0.0

    @property
    def avg_r(self) -> float:
        return float(np.mean(self.realized_r)) if self.realized_r else float("nan")


def walk_forward(df: pd.DataFrame, atr_s: pd.Series, rsi_s: pd.Series, cfg: BotConfig) -> CalibResult:
    a = cfg.analysis
    res = CalibResult()
    baselines: list[float] = []
    n = len(df)
    hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    atr_v = atr_s.to_numpy(float)
    i = max(a.calib_min_history, 60)
    while i < n - 5:
        wa = analyze_waves(df.iloc[:i], atr_s.iloc[:i], rsi_s.iloc[:i], a)
        dec = decide(wa, cl[i - 1], atr_v[i - 1], cfg)
        plan = dec.plan
        if dec.signal not in ("KAUFEN", "SHORT") or plan is None:
            i += a.calib_step
            continue
        s = plan.side
        main = plan.main_target[1]
        end = min(n, i + plan.time_stop_bars)
        outcome, used, r = None, end - i, 0.0
        for j in range(i, end):
            if (s == 1 and lo[j] <= plan.stop) or (s == -1 and hi[j] >= plan.stop):
                outcome, used, r = "stop", j - i + 1, -1.0
                break
            if (s == 1 and hi[j] >= main) or (s == -1 and lo[j] <= main):
                outcome, used, r = "hit", j - i + 1, plan.crv
                break
        if outcome is None:
            if i + plan.time_stop_bars > n:     # Zukunft noch nicht bekannt -> verwerfen
                break
            outcome = "timeout"
            r = s * (cl[end - 1] - plan.entry) / plan.risk_per_unit
        a_dist = abs(main - plan.entry)
        b_dist = abs(plan.entry - plan.stop)
        baselines.append(b_dist / (a_dist + b_dist))
        res.signals += 1
        res.hits += outcome == "hit"
        res.stops += outcome == "stop"
        res.timeouts += outcome == "timeout"
        res.realized_r.append(r)
        i += max(used, a.calib_step)           # keine überlappenden Trades
    res.baseline_mean = float(np.mean(baselines)) if baselines else 0.0
    return res


def blend_probability(p_random: float, p_model: float, calib: CalibResult | None,
                      prior_strength: float) -> float:
    """Bayes-Mittelung des Vorsprungs (Lift) ggü. Zufall: Modellannahme vs. Historie."""
    model_lift = p_model - p_random
    if calib is None or calib.signals == 0:
        lift = model_lift
    else:
        n = calib.signals
        lift = (prior_strength * model_lift + n * calib.empirical_lift) / (prior_strength + n)
    return float(np.clip(p_random + lift, 0.02, 0.95))
