import math

import numpy as np
import pandas as pd
import pytest

from elliott_bot.analyzer import analyze_title
from elliott_bot.calibration import CalibResult, blend_probability
from elliott_bot.cli import main
from elliott_bot.config import BotConfig, BudgetConfig, SelectionConfig
from elliott_bot.data_sources import SyntheticProvider
from elliott_bot.indicators import atr
from elliott_bot.montecarlo import BootstrapSampler, simulate_plan
from elliott_bot.pivots import Pivot, zigzag
from elliott_bot.portfolio import PaperPortfolio
from elliott_bot.risk import BookState, size_position
from elliott_bot.strategy import build_plan
from elliott_bot.universe import RandomSelector, load_universe
from elliott_bot.waves import WaveCount, _Arrays, evaluate_count, fib_score


# --------------------------------------------------------------------- Helfer

def make_arrays(n: int, close: float = 100.0) -> _Arrays:
    c = np.full(n, close)
    return _Arrays(high=c + 0.5, low=c - 0.5, close=c, rsi=np.full(n, 50.0),
                   volume=np.zeros(n), has_volume=False)


def pivots(prices, spacing=10, start_kind="L"):
    kinds = ["L", "H"] if start_kind == "L" else ["H", "L"]
    return [Pivot(i * spacing, pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * spacing), p,
                  kinds[i % 2]) for i, p in enumerate(prices)]


IDEAL_IMPULSE = [100, 110, 104, 120.2, 114.0, 124.0]   # W1=10, W2=60 %, W3=1,62xW1, W4=38 %, W5=1xW1


# ---------------------------------------------------------------- Fibonacci

def test_fib_score_peaks_at_ideal():
    assert fib_score(0.618, [0.618]) == pytest.approx(1.0)
    assert fib_score(0.9, [0.618]) < 0.5
    assert fib_score(-1, [0.618]) == 0.0


# ------------------------------------------------------------------- Pivots

def test_zigzag_alternates_and_finds_swings():
    prices = np.concatenate([np.linspace(100, 120, 30), np.linspace(120, 105, 20), np.linspace(105, 130, 30)])
    df = pd.DataFrame({"open": prices, "high": prices + 0.2, "low": prices - 0.2, "close": prices,
                       "volume": 0.0}, index=pd.bdate_range("2024-01-01", periods=len(prices)))
    piv = zigzag(df, atr(df), 3.0)
    kinds = [p.kind for p in piv]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))
    assert piv[-1].confirmed is False
    assert any(p.kind == "H" and abs(p.price - 120.2) < 0.5 for p in piv)
    assert any(p.kind == "L" and abs(p.price - 104.8) < 0.5 for p in piv)


# -------------------------------------------------------------- Wellenregeln

def test_ideal_impulse_is_accepted_with_good_score():
    pts = pivots(IDEAL_IMPULSE)
    arr = make_arrays(pts[-1].idx + 3, close=123.5)
    wc = evaluate_count(pts, "A", 1, 4.0, arr)
    assert wc is not None
    assert wc.next_dir == -1                               # nach Impuls folgt Korrektur
    assert all(c.score == 1.0 for c in wc.checks if c.hard)
    assert wc.score > 0.5


def test_wave2_beyond_start_is_rejected():
    pts = pivots([100, 110, 99])
    assert evaluate_count(pts, "W3", 1, 4.0, make_arrays(25, 105)) is None


def test_wave4_overlap_is_rejected():
    pts = pivots([100, 110, 104, 120, 109])               # W4-Tief 109 < W1-Hoch 110
    assert evaluate_count(pts, "W5", 1, 4.0, make_arrays(45, 112)) is None


def test_wave3_shortest_is_rejected():
    pts = pivots([100, 110, 105, 113, 111, 125])          # W3=8 < W1=10 und < W5=14
    assert evaluate_count(pts, "A", 1, 4.0, make_arrays(55, 124)) is None


def test_running_wave4_overlap_invalidates_count():
    pts = pivots([100, 110, 104, 120.2])
    arr = make_arrays(40, 115)
    arr.low[35] = 109.0                                    # laufende W4 fällt unter W1-Hoch
    assert evaluate_count(pts, "W4", 1, 4.0, arr) is None


def test_downtrend_is_mirrored():
    mirrored = [200 - p for p in IDEAL_IMPULSE]
    pts = pivots(mirrored, start_kind="H")
    wc = evaluate_count(pts, "A", -1, 4.0, make_arrays(pts[-1].idx + 3, close=200 - 123.5))
    assert wc is not None and wc.next_dir == 1
    assert all(t > 200 - 124 for _, t in wc.targets)


def test_wave3_targets_are_fibonacci_extensions():
    pts = pivots([100, 110, 104])
    wc = evaluate_count(pts, "W3", 1, 4.0, make_arrays(25, 106))
    vals = [v for _, v in wc.targets]
    assert vals == pytest.approx([114.0, 120.18, 130.18])
    assert wc.stop_ref == 104 and wc.invalidation == 100


# --------------------------------------------------------------- Strategie

def _count(stop_ref=95.0, targets=(105.0, 110.0, 120.0)):
    return WaveCount(state="W3", direction=1, degree=4.0, points=pivots([90, 100, 95]),
                     labels=["0", "1", "2"], score=0.6, checks=[], next_dir=1,
                     targets=[("t1", targets[0]), ("t2", targets[1]), ("t3", targets[2])],
                     stop_ref=stop_ref, invalidation=90.0, expected_bars=30, elapsed_bars=5)


def test_plan_rejects_too_wide_stop():
    plan, why = build_plan(_count(stop_ref=80.0), close=100.0, atr_now=2.0, cfg=BotConfig())
    assert plan is None and "ATR" in why


def test_plan_crv_and_levels():
    plan, _ = build_plan(_count(), close=100.0, atr_now=2.0, cfg=BotConfig())
    assert plan.stop == pytest.approx(94.0)                # 95 - 0,5 ATR
    assert plan.main_target[1] == 110.0
    assert plan.crv == pytest.approx((10 - 0.0015 * 210) / (6 + 0.0015 * 194), rel=1e-6)


# ------------------------------------------------------------ Monte Carlo

def test_random_walk_barrier_probability_matches_theory():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 3000)
    close = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 0.0},
                      index=pd.bdate_range("2010-01-01", periods=len(close)))
    cfg = BotConfig()
    cfg.budget.fee_pct = cfg.budget.slippage_pct = 0.0
    plan, _ = build_plan(_count(stop_ref=95.5, targets=(105, 110, 120)), 100.0, 1.0, cfg)
    plan.time_stop_bars = 2000
    sampler = BootstrapSampler(df, 2500, 1, 3000, 4005, seed=1)
    res = simulate_plan(plan, sampler, cfg, 0.0)
    a, b = 10.0, 100.0 - plan.stop
    # Log-Preis-Barrieren: Theorie über log-Abstände
    la, lb = math.log(110 / 100), math.log(100 / plan.stop)
    assert res.p_main == pytest.approx(lb / (la + lb), abs=0.04)
    assert abs(res.mean_r) < 0.15                          # ohne Vorteil kein systematischer Gewinn


# ------------------------------------------------------------------- Risiko

def test_position_size_respects_all_limits():
    bud = BudgetConfig(capital=10_000)
    book = BookState(equity=10_000, peak_equity=10_000, cash=10_000)
    s = size_position(entry=50.0, risk_per_unit=2.0, mean_r=0.5, second_moment_r=2.0,
                      region="Indien", book=book, bud=bud)
    assert s.risk_amount <= 100.0 + 1e-9                    # 1 % Risiko
    assert s.invest <= 2_000.0 + 1e-9                       # 20 % Positionslimit
    assert s.units == int(s.units)


def test_drawdown_brake_and_negative_ev_block_trades():
    bud = BudgetConfig(capital=10_000)
    dd_book = BookState(equity=7_500, peak_equity=10_000, cash=7_500)
    assert size_position(50, 2, 0.5, 2, "X", dd_book, bud).units == 0
    ok_book = BookState(equity=10_000, peak_equity=10_000, cash=10_000)
    assert size_position(50, 2, -0.1, 2, "X", ok_book, bud).units == 0


def test_portfolio_heat_cap():
    bud = BudgetConfig(capital=10_000)
    book = BookState(equity=10_000, peak_equity=10_000, cash=10_000, open_risk=480.0)
    s = size_position(10.0, 1.0, 1.0, 2.0, "X", book, bud)
    assert s.risk_amount <= 20.0 + 1e-9 and s.binding == "Portfolio-Heat"


# --------------------------------------------------------------- Kalibrierung

def test_blend_probability_shrinks_to_history():
    assert blend_probability(0.3, 0.4, None, 20) == pytest.approx(0.4)
    bad = CalibResult(signals=80, hits=8, baseline_mean=0.3)
    assert blend_probability(0.3, 0.4, bad, 20) < 0.3      # Historie schlechter als Zufall


# ------------------------------------------------------------------ Universum

def test_universe_has_200_unique_funds_and_growth_bias():
    u = load_universe()
    assert len(u) == 200 and len({f.ticker for f in u}) == 200
    sel = RandomSelector(u, SelectionConfig(seed=1, max_per_region=2))
    share_in_universe = sum(f.is_growth_market for f in u) / len(u)
    assert sel.growth_share() > share_in_universe
    drawn = [sel.draw() for _ in range(15)]
    regions = pd.Series([f.region for f in drawn]).value_counts()
    assert regions.max() <= 2


# --------------------------------------------------------------- End-to-End

def test_scan_writes_reports(tmp_path):
    main(["scan", "--provider", "synthetic", "-n", "3", "--seed", "5", "--no-calibration",
          "--out", str(tmp_path)])
    html = list(tmp_path.glob("*.html"))
    md = list(tmp_path.glob("*.md"))
    assert html and md
    text = md[0].read_text()
    assert "SYNTHETISCHE TESTDATEN" in text and "Methodik" in text


def test_paper_portfolio_opens_and_manages_positions(tmp_path):
    cfg = BotConfig()
    cfg.analysis.calibrate = False
    cfg.analysis.mc_paths = 600
    cfg.filters.min_consensus = 0.0
    cfg.filters.min_count_score = 0.0
    cfg.filters.min_crv = 1.0
    cfg.filters.min_probability = 0.0
    cfg.filters.min_tp1_probability = 0.0
    cfg.filters.min_expected_r = -1.0
    cfg.budget.risk_per_trade = 0.01
    t0 = pd.Timestamp("2025-06-02")
    pf = PaperPortfolio(tmp_path / "pf.json", cfg)
    u = {f.ticker: f for f in load_universe()}
    opened = 0
    for tk in list(u)[:60]:
        series = SyntheticProvider(end=t0).get(tk)
        rep = analyze_title(u[tk], series, cfg, pf.book_state())
        if pf.open_from_report(rep):
            opened += 1
        if opened == 3:
            break
    assert opened >= 1
    eq0 = pf.equity
    assert eq0 == pytest.approx(cfg.budget.capital, rel=0.01)
    pf.save()

    pf2 = PaperPortfolio(tmp_path / "pf.json", cfg)
    pf2.update(SyntheticProvider(end=t0 + pd.Timedelta(days=200)))
    # Jede Position wurde verwaltet: geschlossen oder fortgeschrieben
    assert pf2.closed or all(p.last_date > str(t0.date()) for p in pf2.positions)
    for c in pf2.closed:
        assert c.r_multiple >= -1.6                         # Verlust je Trade nahe am geplanten 1 R
    assert pf2.equity > 0.9 * cfg.budget.capital
