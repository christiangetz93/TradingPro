import math

import numpy as np
import pandas as pd
import pytest

from elliott_bot import backtest as bt
from elliott_bot.cli import main
from elliott_bot.config import BotConfig, BudgetConfig, CostConfig, SelectionConfig, apply_overrides
from elliott_bot.data_sources import PriceSeries, SyntheticProvider
from elliott_bot.execution import PendingOrder, make_legs, open_position, process_bar, process_pending
from elliott_bot.indicators import atr, rsi
from elliott_bot.montecarlo import BootstrapSampler, simulate_plan
from elliott_bot.pivots import Pivot, zigzag
from elliott_bot.portfolio import PaperPortfolio
from elliott_bot.risk import BookState, size_position
from elliott_bot.setup_stats import SetupStats
from elliott_bot.strategy import build_plan
from elliott_bot.universe import RandomSelector, load_universe
from elliott_bot.waves import WaveContext, WaveCount, _Arrays, evaluate_count, fib_score


# --------------------------------------------------------------------- Helfer

def make_arrays(n: int, close: float = 100.0) -> _Arrays:
    c = np.full(n, close)
    return _Arrays(high=c + 0.5, low=c - 0.5, close=c, rsi=np.full(n, 50.0),
                   volume=np.zeros(n), has_volume=False)


def pivots(prices, spacing=10, start_kind="L"):
    kinds = ["L", "H"] if start_kind == "L" else ["H", "L"]
    return [Pivot(i * spacing, pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * spacing), p,
                  kinds[i % 2], True, i * spacing + 2) for i, p in enumerate(prices)]


def bars(rows, start="2020-01-01"):
    o, h, lo, c = zip(*rows)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 0.0},
                        index=pd.bdate_range(start, periods=len(rows)))


def no_cost_cfg(**over) -> BotConfig:
    return apply_overrides(BotConfig(), {"costs.fee_fixed": 0.0, "costs.spread_pct": 0.0,
                                         "costs.slippage_pct": 0.0, **over})


IDEAL_IMPULSE = [100, 110, 104, 120.2, 114.0, 124.0]   # W1=10, W2=60 %, W3=1,62xW1, W4=38 %, W5=1xW1


class FakeProvider:
    def __init__(self, data):
        self.data = data

    def get(self, ticker, years=5):
        return PriceSeries(ticker, self.data[ticker], "EUR", "Test")


# ---------------------------------------------------------------- Fibonacci

def test_fib_score_peaks_at_ideal():
    assert fib_score(0.618, [0.618]) == pytest.approx(1.0)
    assert fib_score(0.9, [0.618]) < 0.5
    assert fib_score(-1, [0.618]) == 0.0


# ------------------------------------------------------------------- Pivots

def test_zigzag_alternates_and_is_confirmed_after_the_fact():
    prices = np.concatenate([np.linspace(100, 120, 30), np.linspace(120, 105, 20), np.linspace(105, 130, 30)])
    df = pd.DataFrame({"open": prices, "high": prices + 0.2, "low": prices - 0.2, "close": prices,
                       "volume": 0.0}, index=pd.bdate_range("2024-01-01", periods=len(prices)))
    piv = zigzag(df, atr(df), 3.0)
    kinds = [p.kind for p in piv]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))
    assert piv[-1].confirmed is False
    assert all(p.confirm_idx >= p.idx for p in piv if p.confirmed)   # Bestätigung erst danach
    assert any(p.kind == "H" and abs(p.price - 120.2) < 0.5 for p in piv)


def test_wave_analysis_is_causal():
    """Analyse zum Zeitpunkt t ist identisch, egal ob spätere Kurse vorhanden sind."""
    df = SyntheticProvider().get("VWO").df
    cfg = BotConfig().analysis
    full = WaveContext(df, atr(df), rsi(df["close"]), cfg)
    for t in (300, 700, 1100):
        sub = df.iloc[:t + 1]
        cut = WaveContext(sub, atr(sub), rsi(sub["close"]), cfg).analyze()
        a = full.analyze(t)
        assert [(c.key, round(c.score, 9)) for c in a.counts] == [(c.key, round(c.score, 9)) for c in cut.counts]


def test_backtest_signals_do_not_use_future_data():
    df = SyntheticProvider().get("EWZ").df
    cfg = BotConfig()
    full = {c.t: (c.key, round(c.entry_ref, 9), round(c.plan["stop"], 9))
            for c in bt.elliott_candidates(df, cfg, 0, len(df))}
    for t in sorted(full)[:: max(1, len(full) // 5)]:
        part = bt.elliott_candidates(df.iloc[:t + 2], cfg, t, t + 1)
        assert part and (part[0].key, round(part[0].entry_ref, 9), round(part[0].plan["stop"], 9)) == full[t]


# -------------------------------------------------------------- Wellenregeln

def test_ideal_impulse_is_accepted_with_good_score():
    pts = pivots(IDEAL_IMPULSE)
    wc = evaluate_count(pts, "A", 1, 4.0, make_arrays(pts[-1].idx + 3, close=123.5))
    assert wc is not None and wc.next_dir == -1
    assert all(c.score == 1.0 for c in wc.checks if c.hard)
    assert wc.score > 0.5


def test_wave2_beyond_start_is_rejected():
    assert evaluate_count(pivots([100, 110, 99]), "W3", 1, 4.0, make_arrays(25, 105)) is None


def test_wave4_overlap_is_rejected():
    assert evaluate_count(pivots([100, 110, 104, 120, 109]), "W5", 1, 4.0, make_arrays(45, 112)) is None


def test_wave3_shortest_is_rejected():
    assert evaluate_count(pivots([100, 110, 105, 113, 111, 125]), "A", 1, 4.0, make_arrays(55, 124)) is None


def test_running_wave4_overlap_invalidates_count():
    arr = make_arrays(40, 115)
    arr.low[35] = 109.0
    assert evaluate_count(pivots([100, 110, 104, 120.2]), "W4", 1, 4.0, arr) is None


def test_downtrend_is_mirrored():
    pts = pivots([200 - p for p in IDEAL_IMPULSE], start_kind="H")
    wc = evaluate_count(pts, "A", -1, 4.0, make_arrays(pts[-1].idx + 3, close=200 - 123.5))
    assert wc is not None and wc.next_dir == 1
    assert all(t > 200 - 124 for _, t in wc.targets)


def test_wave3_targets_are_fibonacci_extensions():
    wc = evaluate_count(pivots([100, 110, 104]), "W3", 1, 4.0, make_arrays(25, 106))
    assert [v for _, v in wc.targets] == pytest.approx([114.0, 120.18, 130.18])
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


def test_plan_crv_includes_variable_costs():
    cfg = BotConfig()
    v = cfg.costs.variable_rate()
    plan, _ = build_plan(_count(), close=100.0, atr_now=2.0, cfg=cfg)
    assert plan.stop == pytest.approx(94.0)
    assert plan.main_target[1] == 110.0
    assert plan.crv == pytest.approx((10 - v * 210) / (6 + v * 194))


# ------------------------------------------------------------------ Kosten

def test_cost_model():
    c = CostConfig(fee_fixed=4.9, fee_pct=0.0025, fee_min=9.9, spread_pct=0.002, slippage_pct=0.0)
    assert c.broker_fee(1000) == pytest.approx(9.9)                 # Mindestgebühr greift
    assert c.broker_fee(10000) == pytest.approx(4.9 + 25)
    assert c.order_cost(1000) == pytest.approx(9.9 + 1.0)
    assert c.fixed_part(10000) == pytest.approx(4.9)
    assert c.variable_rate() == pytest.approx(0.0025 + 0.001)


def test_small_positions_merge_partial_exits_to_save_fees():
    cfg = apply_overrides(BotConfig(), {"exits.mode": "scale_be", "costs.fee_fixed": 5.0})
    levels = [("a", 103.0), ("b", 106.0), ("c", 109.0)]
    legs_big, mode_big = make_legs(levels, levels[1], "scale_be", cfg, 20_000)
    legs_small, mode_small = make_legs(levels, levels[1], "scale_be", cfg, 1_500)
    assert mode_big == "scale_be" and len(legs_big) == 3
    assert mode_small == "target" and len(legs_small) == 1


# ---------------------------------------------------------------- Ausführung

def _pos(cfg, mode="target", stop=95.0, legs=None, time_stop=50):
    legs = legs if legs is not None else [["Z", 110.0, 1.0]]
    plan = {"stop": stop, "legs": legs, "invalidation": 0.0, "atr": 2.0, "time_stop_bars": time_stop,
            "exit_mode": mode}
    return open_position("X", 1, 10, 100.0, "d0", plan, cfg)


def test_gap_below_stop_fills_at_open():
    pos = _pos(no_cost_cfg())
    fills = process_bar(pos, 90.0, 92.0, 89.0, 91.0, "d1")
    assert fills[0].price == 90.0 and not pos.is_open             # nicht zum Stop-Kurs 95


def test_stop_and_target_same_day_is_pessimistic():
    pos = _pos(no_cost_cfg())
    fills = process_bar(pos, 100.0, 111.0, 94.0, 105.0, "d1")
    assert fills[0].price == 95.0


def test_trailing_stop_only_moves_up_and_applies_next_day():
    cfg = no_cost_cfg(**{"exits.trail_atr_mult": 3.0})
    pos = _pos(cfg, mode="trail", legs=[])
    process_bar(pos, 100, 110, 99, 109, "d1")
    assert pos.stop == pytest.approx(110 - 3 * 2.0)              # 104 ab morgen
    process_bar(pos, 108, 108.5, 105, 106, "d2")
    assert pos.stop == pytest.approx(104.0) and pos.is_open       # gesunkener Kurs senkt den Stop nicht
    fills = process_bar(pos, 105, 105.5, 103, 104, "d3")
    assert fills and fills[0].price == pytest.approx(104.0)


def test_time_stop_closes_at_close():
    pos = _pos(no_cost_cfg(), time_stop=2)
    process_bar(pos, 100, 101, 99, 100, "d1")
    fills = process_bar(pos, 100, 101, 99, 100.5, "d2")
    assert fills[0].reason == "Zeitstopp" and fills[0].price == 100.5


def test_pending_orders():
    mk = PendingOrder("X", 1, "market", 10, {"stop": 95.0}, 0, 1)
    assert process_pending(mk, 1, 94.0, 96, 93, 95, 100, 98)[0] == "cancel"     # Gap unter Stop
    assert process_pending(mk, 1, 99.0, 100, 98, 99, 100, 98) == ("fill", 99.0)
    lim = PendingOrder("X", 1, "limit", 10, {"stop": 90.0}, 0, 20, limit=97.0, zone=(98, 96), cancel_level=105)
    assert process_pending(lim, 1, 99, 100, 98, 99, 100, 98)[0] == "wait"
    assert process_pending(lim, 2, 98, 98.5, 96.5, 97.5, 100, 98) == ("fill", 97.0)
    cf = PendingOrder("X", 1, "confirm", 10, {"stop": 90.0}, 0, 20, zone=(98, 96), cancel_level=105)
    assert process_pending(cf, 1, 99, 99.5, 97.5, 98.0, 99.0, 97.0)[0] == "wait" and cf.armed
    process_pending(cf, 2, 98, 99.8, 97.8, 99.6, 99.5, 97.5)             # Schluss über Vortageshoch
    assert cf.trigger_next_open and process_pending(cf, 3, 99.7, 100, 99, 99.9, 99.8, 97.8) == ("fill", 99.7)


# ------------------------------------------------------------------- Risiko

def test_position_size_respects_all_limits_including_fees():
    bud = BudgetConfig(capital=10_000)
    book = BookState(equity=10_000, peak_equity=10_000, cash=10_000)
    costs = CostConfig(fee_fixed=5.0)
    s = size_position(entry=50.0, risk_per_unit=2.0, mean_r=0.5, second_moment_r=2.0,
                      region="Indien", book=book, bud=bud, costs=costs, n_orders=2)
    assert s.risk_amount <= 100.0 + 1e-9                    # 1 % Risiko INKL. Fixgebühren
    assert s.fixed_fees == pytest.approx(10.0)
    assert s.invest <= 2_000.0 + 1e-9 and s.units == int(s.units)


def test_drawdown_brake_and_negative_ev_block_trades():
    bud = BudgetConfig(capital=10_000)
    assert size_position(50, 2, 0.5, 2, "X", BookState(7_500, 10_000, 7_500), bud).units == 0
    assert size_position(50, 2, -0.1, 2, "X", BookState(10_000, 10_000, 10_000), bud).units == 0


def test_portfolio_heat_cap():
    bud = BudgetConfig(capital=10_000)
    s = size_position(10.0, 1.0, 1.0, 2.0, "X", BookState(10_000, 10_000, 10_000, open_risk=480.0), bud)
    assert s.risk_amount <= 20.0 + 1e-9 and s.binding == "Portfolio-Heat"


# ---------------------------------------------------------------- Backtest

def test_backtest_accounting_is_exact():
    rows = [(100, 100, 100, 100)] * 261 + [(100, 101, 99.5, 100.5)] * 4 + [(101, 111, 100.5, 109)] + [(109, 109, 109, 109)] * 5
    df = bars(rows)
    cfg = apply_overrides(BotConfig(), {"exits.mode": "target", "costs.fee_fixed": 1.0, "costs.spread_pct": 0.001,
                                        "costs.slippage_pct": 0.0})
    levels = [("Z", 110.0)] * 3
    cand = bt.Candidate(260, "market", 1, "Test", 100.0, 5.0 + 0.0005 * 195, 2.0,
                        plan={"stop": 95.0, "legs": [["Z", 110.0, 1.0]], "invalidation": -1e9, "atr": 1.0,
                              "time_stop_bars": 50, "exit_mode": "target", "meta": {}},
                        levels=levels, main=("Z", 110.0))
    res = bt.run_portfolio({"X": df}, {"X": [cand]}, cfg)
    t = res.trades[0]
    assert t.units == 19 and t.entry == 100.0
    entry_cost, exit_cost = 1 + 0.0005 * 1900, 1 + 0.0005 * 2090
    assert t.gross == pytest.approx(190.0)
    assert t.net == pytest.approx(190.0 - entry_cost - exit_cost)
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t.net)
    assert res.fees_total == pytest.approx(entry_cost + exit_cost)


def test_rules_have_no_edge_on_a_fair_game():
    """Exaktes Martingal: zufällige Einstiege + Ausstiegsregeln dürfen im Mittel nichts verdienen."""
    rng = np.random.default_rng(5)
    data = {}
    for i in range(12):
        s = 0.015 / np.sqrt(8)
        r = rng.normal(-0.5 * s * s, s, (2500, 8))
        path = 100 * np.exp(np.cumsum(r.ravel()).reshape(2500, 8))
        close = path[:, -1]
        open_ = np.concatenate([[100.0], close[:-1]])
        data[f"M{i}"] = pd.DataFrame({"open": open_, "high": np.maximum(path.max(1), open_),
                                      "low": np.minimum(path.min(1), open_), "close": close, "volume": 0.0},
                                     index=pd.bdate_range("2000-01-03", periods=2500))
    tmpl = [bt.Candidate(0, "market", 1, "x", 100.0, 3.0, 2.0, plan={"stop": 97.0, "atr": 1.0, "time_stop_bars": 30},
                         levels=[("a", 103.0), ("b", 106.0), ("c", 109.0)], main=("b", 106.0))]
    cfg = no_cost_cfg(**{"strategy.trend_filter": "none"})
    cands = {tk: bt.random_candidates(df, cfg, 0, len(df), tmpl, 0.03, seed=i, use_trend=False)
             for i, (tk, df) in enumerate(data.items())}
    st = bt.trade_stats(bt.signal_study(data, cands, cfg), "r_var")
    assert st["Trades"] > 300
    assert abs(st["t-Wert"]) < 3.0 and abs(st["Ø R"]) < 0.1


def test_close_only_series_are_detected():
    from elliott_bot.data_sources import is_close_only
    df = SyntheticProvider().get("VWO").df
    assert not is_close_only(df)
    c = df["close"]
    fake = pd.DataFrame({"open": c.shift(1).fillna(c), "close": c, "volume": 0.0})
    fake["high"] = fake[["open", "close"]].max(axis=1)
    fake["low"] = fake[["open", "close"]].min(axis=1)
    assert is_close_only(fake)


def test_null_data_has_no_arithmetic_drift():
    df = SyntheticProvider().get("SMH").df
    ratios = [bt.null_data(df, seed=s)["close"].pct_change().dropna().mean() for s in range(20)]
    assert abs(np.mean(ratios)) < 2e-4


def test_setup_stats_shrink_towards_zero(tmp_path):
    trades = [bt.Trade("X", "K", "market", 1, "a", "b", 100, 1, 1.0, g, 0.0, 0.0, 5, "x", True, g > 0, 1.0)
              for g in (2.0, 2.0, -1.0)]
    st = SetupStats.from_trades(trades)
    mean, _, n = st.expectancy("K", prior_n=30)
    assert n == 3 and mean == pytest.approx(3.0 / 33)          # (2 + 2 - 1) / (3 + 30)
    st.save(tmp_path / "s.json")
    assert SetupStats.load(tmp_path / "s.json").rows["K"].n == 3


def test_shipped_setup_stats_are_loadable():
    st = SetupStats.load()
    assert st is not None and sum(r.n for r in st.rows.values()) > 200


# ------------------------------------------------------------ Monte Carlo

def test_random_walk_barrier_probability_matches_theory():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 0.0},
                      index=pd.bdate_range("2010-01-01", periods=len(close)))
    cfg = no_cost_cfg(**{"exits.mode": "target"})
    plan, _ = build_plan(_count(stop_ref=95.5, targets=(105, 110, 120)), 100.0, 1.0, cfg)
    plan.time_stop_bars = 2000
    res = simulate_plan(plan, BootstrapSampler(df, 2500, 1, 3000, 4005, seed=1), cfg, 0.0)
    la, lb = math.log(110 / 100), math.log(100 / plan.stop)
    assert res.p_main == pytest.approx(lb / (la + lb), abs=0.04)
    assert abs(res.mean_r) < 0.15


# ------------------------------------------------------------------ Universum

def test_universe_has_200_unique_funds_and_growth_bias():
    u = load_universe()
    assert len(u) == 200 and len({f.ticker for f in u}) == 200
    sel = RandomSelector(u, SelectionConfig(seed=1, max_per_region=2))
    assert sel.growth_share() > sum(f.is_growth_market for f in u) / len(u)
    regions = pd.Series([sel.draw().region for _ in range(15)]).value_counts()
    assert regions.max() <= 2


# --------------------------------------------------------------- End-to-End

def test_scan_writes_reports(tmp_path):
    main(["scan", "--provider", "synthetic", "-n", "3", "--seed", "5", "--fast", "--out", str(tmp_path)])
    md = list(tmp_path.glob("*.md"))
    assert md and list(tmp_path.glob("*.html"))
    text = md[0].read_text()
    assert "SYNTHETISCHE TESTDATEN" in text and "Methodik" in text


def test_backtest_command_writes_report(tmp_path):
    main(["backtest", "--provider", "synthetic", "--tickers", "VWO,EWZ,SMH", "--random-runs", "1",
          "--workers", "1", "--split", "2023-01-01", "--out", str(tmp_path)])
    html = next(tmp_path.glob("backtest_*.html")).read_text()
    assert "Out-of-Sample" in html and "Zufall" in html


def test_paper_portfolio_fills_at_next_open_and_exits(tmp_path):
    cfg = apply_overrides(BotConfig(), {"exits.mode": "target"})
    rows = [(100, 100.5, 99.5, 100)] * 30 + [(100.4, 101, 100, 100.8)] + [(101, 111, 100.5, 110.5)] + [(110, 110, 110, 110)]
    df = bars(rows)
    signal_day = str(df.index[29].date())
    pf = PaperPortfolio(tmp_path / "pf.json", cfg)
    pf.pending.append({"ticker": "X", "name": "Test", "region": "R", "side": 1, "units": 10, "risk_amount": 60.0,
                       "signal_date": signal_day, "levels": [["Z", 110.0]] * 3, "main": ["Z", 110.0],
                       "setup_key": "T", "plan": {"stop": 95.0, "legs": [["Z", 110.0, 1.0]], "invalidation": 90.0,
                                                  "atr": 1.0, "time_stop_bars": 40, "exit_mode": "target", "meta": {}}})
    pf.save()
    pf2 = PaperPortfolio(tmp_path / "pf.json", cfg)
    pf2.update(FakeProvider({"X": df}))
    assert not pf2.pending and not pf2.positions and len(pf2.closed) == 1
    c = pf2.closed[0]
    assert c.entry == 100.4 and c.exit == 110.0                  # Kauf zum Eröffnungskurs nach dem Signaltag
    assert pf2.cash == pytest.approx(10_000 + c.pnl)
    assert c.fees > 2.0                                         # 2 Orders à 1 € + Spread/Slippage
