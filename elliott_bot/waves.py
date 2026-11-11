"""Elliott-Wellen-Erkennung.

Vorgehen
--------
1. Pivots auf mehreren Wellengraden (ZigZag mit 2,5 / 4 / 6 ATR).
2. Für die jüngsten Pivots werden alle plausiblen Zählungen aufgestellt, die bis
   zur Gegenwart reichen – auf- und abwärts:
      W3  : 0-1-2 fertig       -> Welle 3 läuft
      W4  : 0-1-2-3 fertig     -> Welle 4 läuft
      W5  : 0-1-2-3-4 fertig   -> Welle 5 läuft
      A   : Impuls 0-5 fertig  -> Korrektur A läuft
      B   : 0-5 + A fertig     -> Welle B läuft
      C   : 0-5 + A + B fertig -> Welle C läuft
3. Harte Regeln (Verstoß = Zählung verworfen):
      * Welle 2 retraced nie mehr als 100 % von Welle 1
      * Welle 3 ist nie die kürzeste der Wellen 1, 3, 5
      * Welle 4 dringt nicht in das Kursgebiet von Welle 1 ein
      * Welle 3 übersteigt das Ende von Welle 1
      * B-Welle max. 138,2 % von A (expandierte Flat)
4. Richtlinien (Fibonacci-Verhältnisse, Alternation, RSI-Divergenz, Volumen)
   ergeben einen Qualitätsscore 0..1.
5. Aus den Scores werden Szenario-Gewichte. Ein Pseudo-Szenario "keine klare
   Zählung" verhindert Scheingenauigkeit, wenn alle Zählungen schwach sind.

Alle Berechnungen laufen im "orientierten" Raum q = Richtung * Preis, damit
Abwärtsimpulse mit derselben Logik behandelt werden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .pivots import Pivot, zigzag

STATES = {
    # state: (Anzahl Punkte, Prior, Wellenrichtung im q-Raum, Beschreibung)
    "W3": (3, 0.72, +1, "Wellen 1 und 2 abgeschlossen – Welle 3 (meist die stärkste) läuft"),
    "W4": (4, 0.85, -1, "Wellen 1–3 abgeschlossen – Korrekturwelle 4 läuft"),
    "W5": (5, 0.90, +1, "Wellen 1–4 abgeschlossen – finale Welle 5 läuft"),
    "A": (6, 0.88, -1, "Impuls 1–5 abgeschlossen – Korrekturwelle A läuft"),
    "B": (7, 0.55, +1, "Impuls und Welle A abgeschlossen – Gegenbewegung B läuft"),
    "C": (8, 0.85, -1, "Impuls, A und B abgeschlossen – Welle C läuft"),
}
LABELS = ["0", "1", "2", "3", "4", "5", "A", "B"]


@dataclass
class Check:
    name: str
    value: str
    score: float
    hard: bool = False


@dataclass
class WaveCount:
    state: str
    direction: int                 # +1 übergeordneter Aufwärtsimpuls, -1 Abwärtsimpuls
    degree: float                  # ATR-Multiplikator (Wellengrad)
    points: list[Pivot]
    labels: list[str]
    score: float
    checks: list[Check]
    next_dir: int                  # Richtung der laufenden Welle im Preis (+1 steigt)
    targets: list[tuple[str, float]]   # von nah nach fern
    stop_ref: float                # strukturelles Stop-Niveau (Preis)
    invalidation: float            # Niveau, ab dem die Zählung sicher falsch ist
    expected_bars: int
    elapsed_bars: int
    follow_up: str | None = None
    weight: float = 0.0

    @property
    def description(self) -> str:
        trend = "Aufwärtsimpuls" if self.direction == 1 else "Abwärtsimpuls"
        return f"{STATES[self.state][3]} ({trend}, Grad {self.degree:g} ATR)"

    @property
    def key(self) -> tuple:
        return (self.state, self.direction, tuple(p.idx for p in self.points))


@dataclass
class WaveAnalysis:
    counts: list[WaveCount]
    p_up: float
    p_down: float
    p_unclear: float
    pivots: dict[float, list[Pivot]] = field(default_factory=dict)
    offset: int = 0                 # Index-Offset des analysierten Fensters im Gesamt-DataFrame

    @property
    def primary(self) -> WaveCount | None:
        return self.counts[0] if self.counts else None


def fib_score(x: float, ideals: list[float], tol: float = 0.18) -> float:
    """Nähe eines Verhältnisses zu idealen Fibonacci-Werten (log-Abstand)."""
    if not np.isfinite(x) or x <= 0:
        return 0.0
    d = min(abs(np.log(x / i)) for i in ideals)
    return float(np.exp(-0.5 * (d / tol) ** 2))


def evidence_score(scores: list[float], pseudo: float = 3.0, neutral: float = 0.45) -> float:
    """Mittelwert der Richtlinien-Scores, geschrumpft zu einem neutralen Wert.

    Wenige erfüllte Prüfungen (z. B. nur ein Welle-2-Retracement) sind schwache
    Evidenz; erst viele erfüllte Richtlinien heben den Score deutlich an.
    """
    s = np.clip(np.array(scores, dtype=float), 0.0, 1.0)
    return float((s.sum() + pseudo * neutral) / (len(s) + pseudo))


@dataclass
class _Arrays:
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    rsi: np.ndarray
    volume: np.ndarray
    has_volume: bool


def evaluate_count(points: list[Pivot], state: str, direction: int, degree: float,
                   arr: _Arrays) -> WaveCount | None:
    npts, prior, wave_dir_q, _ = STATES[state]
    if len(points) != npts:
        return None
    if points[0].kind != ("L" if direction == 1 else "H"):
        return None

    q = np.array([p.price for p in points], dtype=float) * direction
    t = [p.idx for p in points]
    dur = lambda a, b: max(1, t[b] - t[a])  # noqa: E731
    rsi_q = (arr.rsi if direction == 1 else 100 - arr.rsi)
    k = npts - 1
    checks: list[Check] = []

    W = {}
    W[1] = q[1] - q[0]
    W[2] = q[1] - q[2]
    if W[1] <= 0 or W[2] <= 0:
        return None

    # --- harte Regeln --------------------------------------------------------
    if q[2] <= q[0]:
        return None
    checks.append(Check("Regel: Welle 2 < 100 % von Welle 1", f"{W[2] / W[1]:.0%}", 1.0, True))
    if k >= 3:
        W[3] = q[3] - q[2]
        if q[3] <= q[1]:
            return None
        checks.append(Check("Regel: Welle 3 übersteigt Ende von Welle 1", "erfüllt", 1.0, True))
    if k >= 4:
        W[4] = q[3] - q[4]
        if q[4] <= q[1]:
            return None
        checks.append(Check("Regel: Welle 4 ohne Überlappung mit Welle 1", "erfüllt", 1.0, True))
    if k >= 5:
        W[5] = q[5] - q[4]
        if W[5] <= 0 or (W[3] < W[1] and W[3] < W[5]):
            return None
        checks.append(Check("Regel: Welle 3 nicht die kürzeste", "erfüllt", 1.0, True))

    # --- Richtlinien ---------------------------------------------------------
    soft: list[float] = []

    def add(name: str, value: str, score: float) -> None:
        checks.append(Check(name, value, score))
        soft.append(score)

    r2 = W[2] / W[1]
    add("Welle 2 Retracement (ideal 50–61,8 %)", f"{r2:.1%}", fib_score(r2, [0.5, 0.618], 0.22))
    if k >= 3:
        e3 = W[3] / W[1]
        add("Welle 3 / Welle 1 (ideal 1,618 / 2,618)", f"{e3:.2f}",
            max(fib_score(e3, [1.618, 2.618], 0.2), 0.6 * fib_score(e3, [1.0], 0.15)))
        mom = rsi_q[t[3]] >= rsi_q[t[1]]
        add("Momentum: RSI-Hoch in Welle 3 ≥ Welle 1", f"{rsi_q[t[3]]:.0f} vs {rsi_q[t[1]]:.0f}",
            1.0 if mom else 0.55)
    if k >= 4:
        r4 = W[4] / W[3]
        add("Welle 4 Retracement (ideal 23,6–38,2 %)", f"{r4:.1%}",
            fib_score(r4, [0.236, 0.382], 0.3))
        depth_alt = (r2 >= 0.5) != (r4 >= 0.5)
        time_ratio = dur(3, 4) / dur(1, 2)
        time_alt = time_ratio < 0.66 or time_ratio > 1.5
        add("Alternation Welle 2 / Welle 4", f"Tiefe {'ja' if depth_alt else 'nein'}, "
            f"Zeit {'ja' if time_alt else 'nein'}", 1.0 if (depth_alt or time_alt) else 0.5)
    if k >= 5:
        e5 = max(fib_score(W[5] / W[1], [0.618, 1.0, 1.618], 0.18),
                 fib_score(W[5] / (q[3] - q[0]), [0.618], 0.18))
        add("Welle 5 Länge (= W1, 0,618·W1 oder 0,618·(0–3))", f"{W[5] / W[1]:.2f}·W1", e5)
        if q[5] < q[3]:
            add("Welle 5 gestutzt (Truncation)", "ja", 0.35)
        div = rsi_q[t[5]] < rsi_q[t[3]]
        add("RSI-Divergenz am Ende von Welle 5", f"{rsi_q[t[5]]:.0f} vs {rsi_q[t[3]]:.0f}",
            1.0 if div else 0.5)
        if arr.has_volume:
            v3 = arr.volume[t[2]:t[3] + 1].mean()
            v5 = arr.volume[t[4]:t[5] + 1].mean()
            add("Volumen Welle 3 > Welle 5", f"{v3 / max(v5, 1e-9):.2f}x", 1.0 if v3 > v5 else 0.6)
    if k >= 6:
        A = q[5] - q[6]
        if A <= 0:
            return None
        ra = A / (q[5] - q[0])
        add("Welle A Retracement des Impulses (38,2–61,8 %)", f"{ra:.1%}",
            fib_score(ra, [0.382, 0.5, 0.618], 0.3) * (0.4 if q[6] < q[0] else 1.0))
    if k >= 7:
        B = q[7] - q[6]
        rb = B / A
        if rb > 1.382:
            return None
        add("Welle B / Welle A (Zigzag 50–78,6 %, Flat ~100 %)", f"{rb:.1%}",
            fib_score(rb, [0.5, 0.618, 0.786], 0.25) if rb <= 1.0 else 0.5)

    # --- Projektionen für die laufende Welle ---------------------------------
    last = q[-1]
    follow_up = None
    if state == "W3":
        tg = [("1,0 x W1", last + 1.0 * W[1]), ("1,618 x W1", last + 1.618 * W[1]),
              ("2,618 x W1", last + 2.618 * W[1])]
        stop_ref, invalid = q[2], q[0]
        exp_bars = int(1.618 * dur(0, 1))
    elif state == "W4":
        floor = q[1] + 0.02 * W[1]
        tg = [("23,6 % von W3", last - 0.236 * W[3]), ("38,2 % von W3", last - 0.382 * W[3]),
              ("50 % von W3", max(last - 0.5 * W[3], floor))]
        stop_ref, invalid = q[3], q[1]
        exp_bars = int(max(dur(1, 2), 1) * 1.618)
        follow_up = "W5"
    elif state == "W5":
        cap = last + W[3] if W[3] < W[1] else np.inf   # W3 darf nicht die kürzeste werden
        tg = sorted([("0,618 x W1", min(last + 0.618 * W[1], cap)),
                     ("1,0 x W1", min(last + 1.0 * W[1], cap)),
                     ("0,618 x (0–3)", min(last + 0.618 * (q[3] - q[0]), cap))],
                    key=lambda x: x[1])
        stop_ref, invalid = q[4], q[1]
        exp_bars = int(dur(0, 1))
    elif state == "A":
        L = q[5] - q[0]
        tg = [("38,2 % Retracement", last - 0.382 * L), ("50 % Retracement", last - 0.5 * L),
              ("61,8 % Retracement", last - 0.618 * L)]
        stop_ref, invalid = q[5], q[5]
        exp_bars = int(0.382 * dur(0, 5))
    elif state == "B":
        A = q[5] - q[6]
        tg = [("50 % von A", last + 0.5 * A), ("61,8 % von A", last + 0.618 * A),
              ("78,6 % von A", last + 0.786 * A)]
        stop_ref, invalid = q[6], q[6]
        exp_bars = int(dur(5, 6))
    else:  # C
        A = q[5] - q[6]
        tg = [("C = 0,618 x A", last - 0.618 * A), ("C = 1,0 x A", last - 1.0 * A),
              ("C = 1,618 x A", last - 1.618 * A)]
        stop_ref, invalid = q[7], q[5]
        exp_bars = int(dur(5, 6))
        follow_up = "NEW"
    exp_bars = max(exp_bars, 5)

    # --- laufende Welle seit dem letzten Punkt prüfen -------------------------
    last_idx = len(arr.close) - 1
    seg = slice(t[-1] + 1, last_idx + 1)
    if direction == 1:
        hq, lq = arr.high[seg], arr.low[seg]
    else:
        hq, lq = -arr.low[seg], -arr.high[seg]
    elapsed = last_idx - t[-1]
    if len(hq):
        if wave_dir_q > 0 and lq.min() < stop_ref:
            return None
        if wave_dir_q < 0 and hq.max() > stop_ref:
            return None
        # Regeln gelten auch für die noch laufende Welle
        if state == "W4" and lq.min() <= q[1]:
            return None                      # Welle 4 überlappt Welle 1
        if state == "B" and hq.max() > q[6] + 1.382 * (q[5] - q[6]):
            return None                      # B > 138,2 % von A
        fav = hq.max() if wave_dir_q > 0 else lq.min()
        far = tg[-1][1]
        if (wave_dir_q > 0 and fav > far) or (wave_dir_q < 0 and fav < far):
            add("Laufende Welle hat das weiteste Ziel bereits überschritten", "ja", 0.4)
    if elapsed > 2.5 * exp_bars:
        add("Laufende Welle deutlich länger als erwartet", f"{elapsed} vs ~{exp_bars} Tage", 0.6)

    score = prior * evidence_score(soft)
    to_price = lambda x: float(x * direction)  # noqa: E731
    return WaveCount(
        state=state, direction=direction, degree=degree, points=list(points),
        labels=LABELS[:npts], score=score, checks=checks,
        next_dir=wave_dir_q * direction,
        targets=[(name, to_price(v)) for name, v in tg],
        stop_ref=to_price(stop_ref), invalidation=to_price(invalid),
        expected_bars=exp_bars, elapsed_bars=elapsed, follow_up=follow_up,
    )


def analyze_waves(df: pd.DataFrame, atr_s: pd.Series, rsi_s: pd.Series,
                  cfg: AnalysisConfig) -> WaveAnalysis:
    offset = max(0, len(df) - cfg.lookback_bars)
    win = df.iloc[offset:]
    atr_w = atr_s.iloc[offset:]
    vol = win["volume"].to_numpy(float)
    arr = _Arrays(win["high"].to_numpy(float), win["low"].to_numpy(float),
                  win["close"].to_numpy(float), rsi_s.iloc[offset:].to_numpy(float),
                  vol, bool(np.nansum(vol) > 0))

    found: dict[tuple, WaveCount] = {}
    pivots_by_degree: dict[float, list[Pivot]] = {}
    for mult in cfg.zigzag_atr_mults:
        piv = zigzag(win, atr_w, mult)
        pivots_by_degree[mult] = piv
        n = len(piv)
        anchors = (n - 2,) if cfg.require_confirmed_pivot else (n - 1, n - 2)
        for anchor in anchors:
            for state, (npts, *_rest) in STATES.items():
                start = anchor - (npts - 1)
                if start < 0:
                    continue
                pts = piv[start:anchor + 1]
                for direction in (1, -1):
                    wc = evaluate_count(pts, state, direction, mult, arr)
                    if wc is None:
                        continue
                    if wc.key not in found or found[wc.key].score < wc.score:
                        found[wc.key] = wc

    counts = sorted(found.values(), key=lambda c: c.score, reverse=True)
    raw = np.array([c.score ** cfg.sharpness for c in counts])
    unclear = cfg.unclear_weight ** cfg.sharpness
    total = raw.sum() + unclear
    for c, w in zip(counts, raw):
        c.weight = float(w / total)
    p_up = sum(c.weight for c in counts if c.next_dir > 0)
    p_down = sum(c.weight for c in counts if c.next_dir < 0)
    return WaveAnalysis(counts=counts, p_up=float(p_up), p_down=float(p_down),
                        p_unclear=float(unclear / total), pivots=pivots_by_degree, offset=offset)
