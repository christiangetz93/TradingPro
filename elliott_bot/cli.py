"""Kommandozeile des Elliott-Wave-Bots.

Beispiele::

    python -m elliott_bot scan -n 5 --capital 10000            # Live-Daten (Yahoo)
    python -m elliott_bot scan --provider synthetic --seed 42   # Offline-Demo
    python -m elliott_bot scan --tickers VWO,INDA,EWZ
    python -m elliott_bot run                                   # Depot verwalten + neue Titel scannen
    python -m elliott_bot manage                                # nur Ausstiegsregeln anwenden
    python -m elliott_bot backtest --tickers SPY,EEM,QQQ --years 20 --split 2015-01-01
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from .analyzer import TitleReport, analyze_title
from .bt_report import build_backtest_report
from .config import BotConfig, load_config
from .setup_stats import SetupStats
from .data_sources import DataError, is_close_only, make_provider
from .portfolio import PaperPortfolio
from .report import to_html, to_markdown
from .risk import BookState
from .universe import Fund, RandomSelector, load_universe


def _provider(args):
    if args.provider == "csv":
        return make_provider("csv", directory=args.csv_dir)
    if args.provider == "synthetic":
        return make_provider("synthetic", seed=args.data_seed)
    return make_provider("yahoo", cache_dir=args.cache_dir)


def _config(args) -> BotConfig:
    cfg = load_config(args.config)
    if args.capital is not None:
        cfg.budget.capital = args.capital
    if args.risk is not None:
        cfg.budget.risk_per_trade = args.risk / 100
    if args.allow_short:
        cfg.budget.allow_short = True
    if getattr(args, "n", None) is not None:
        cfg.selection.n_titles = args.n
    if getattr(args, "seed", None) is not None:
        cfg.selection.seed = args.seed
        cfg.analysis.seed = args.seed
    if getattr(args, "fast", False):
        cfg.analysis.ticker_backtest = False
    return cfg


def _stats(cfg: BotConfig) -> SetupStats | None:
    return SetupStats.load(cfg.strategy.stats_path or None)


def _signal_line(r: TitleReport, cur: str) -> str:
    p = r.decision.plan
    parts = [f"{r.fund.ticker:9s} {r.signal:10s}",
             f"{r.analysis.primary.state if r.analysis.primary else '-':3s}"]
    if p:
        parts.append(f"CRV {p.crv:4.1f}")
    if r.p_main is not None:
        parts.append(f"P(1. Ziel) {r.p_tp1:4.0%}  E[R] {r.exp_r if r.exp_r is not None else float('nan'):+.2f}")
    if r.tradeable:
        parts.append(f"-> {r.size.units:g} Stk., Risiko {r.size.risk_amount:,.0f} {cur}")
    return "  ".join(parts)


def cmd_scan(args, portfolio: PaperPortfolio | None = None) -> list[TitleReport]:
    cfg = _config(args)
    provider = _provider(args)
    universe = load_universe(args.universe)
    by_ticker = {f.ticker: f for f in universe}
    if args.tickers:
        queue = [by_ticker.get(t.strip(), Fund(t.strip(), t.strip(), "Unbekannt", "Unbekannt", "-", False, False))
                 for t in args.tickers.split(",") if t.strip()]
        selector = None
    else:
        selector = RandomSelector(universe, cfg.selection)
        queue = []
    if args.paper and portfolio is None:
        portfolio = PaperPortfolio(args.portfolio, cfg)
    cap = cfg.budget.capital
    book = portfolio.book_state() if portfolio else BookState(equity=cap, peak_equity=cap, cash=cap)

    if args.provider == "synthetic":
        print("!! Synthetische Testdaten – Ergebnisse beschreiben KEINEN realen Markt !!\n")
    stats = _stats(cfg)
    target_n = len(queue) if queue else cfg.selection.n_titles
    min_signals = getattr(args, "min_signals", 0) or 0
    max_titles = getattr(args, "max_titles", 60)
    reports: list[TitleReport] = []
    failures = 0

    def need_more() -> bool:
        if len(reports) < target_n:
            return True
        found = sum(r.tradeable for r in reports)
        return selector is not None and found < min_signals and len(reports) < max_titles

    while need_more() and failures < 3 * target_n + 10:
        fund = queue.pop(0) if queue else (selector.draw() if selector else None)
        if fund is None:
            break
        if portfolio and portfolio.holds(fund.ticker):
            continue
        try:
            series = provider.get(fund.ticker)
        except DataError as exc:
            print(f"  übersprungen: {exc}")
            failures += 1
            if selector:
                selector.release_region(fund)
            continue
        rep = analyze_title(fund, series, cfg, book, stats)
        if rep.tradeable:
            book.open_risk += rep.size.risk_amount
            book.cluster_risk[fund.region] = book.cluster_risk.get(fund.region, 0.0) + rep.size.risk_amount
            book.cash -= rep.size.invest
            if portfolio:
                portfolio.open_from_report(rep)
        reports.append(rep)
        print(_signal_line(rep, cfg.budget.currency))

    if not reports:
        print("Keine Titel analysiert (Datenquelle erreichbar?).")
        return reports
    trades = [r for r in reports if r.tradeable]
    heat = sum(r.size.risk_amount for r in trades)
    invest = sum(r.size.invest for r in trades)
    cur = cfg.budget.currency
    header = [
        f"Budget {cap:,.0f} {cur}",
        f"{len(reports)} Titel ({sum(r.fund.is_growth_market for r in reports)} aus Wachstumsmärkten/-themen)",
        f"{len(trades)} Trade(s): investiert {invest:,.0f} {cur}, Risiko gesamt {heat:,.0f} {cur} "
        f"({heat / cap:.2%} von max. {cfg.budget.max_portfolio_heat:.0%})",
        f"Datenquelle: {reports[0].series.source}",
    ]
    if selector:
        header.append(f"Zufallsauswahl, Wachstumsmärkte gewichtet (Anteil am Ziehungsgewicht "
                      f"{selector.growth_share():.0%})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md, ht = out / f"elliott_report_{stamp}.md", out / f"elliott_report_{stamp}.html"
    md.write_text(to_markdown(reports, cfg, header), encoding="utf-8")
    ht.write_text(to_html(reports, cfg, header), encoding="utf-8")
    print("\n" + "\n".join(header))
    print(f"\nReport: {ht}\n        {md}")
    if portfolio:
        portfolio.save()
        print(f"Paper-Depot gespeichert: {portfolio.path}")
    return reports


def cmd_manage(args) -> PaperPortfolio:
    cfg = _config(args)
    pf = PaperPortfolio(args.portfolio, cfg)
    events = pf.update(_provider(args))
    print("Depot-Update:")
    for ev in events or ["keine Veränderungen"]:
        print(f"  {ev}")
    print("\n".join(pf.summary_lines()))
    pf.save()
    return pf


def cmd_backtest(args) -> None:
    cfg = _config(args)
    provider = _provider(args)
    universe = load_universe(args.universe)
    by_ticker = {f.ticker: f for f in universe}
    meta_regions: dict[str, str] = {}
    meta_path = Path(args.csv_dir) / "meta.csv"
    if args.provider == "csv" and meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "region" in meta.columns:
            meta_regions = dict(zip(meta["ticker"], meta["region"]))
    if args.tickers and args.tickers.upper() == "ALL" and meta_regions:
        tickers = list(meta_regions)
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        sel = RandomSelector(universe, cfg.selection)
        tickers = [f.ticker for f in (sel.draw() for _ in range(args.n or 20)) if f]
    data: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        try:
            data[tk] = provider.get(tk, years=args.years).df
        except DataError as exc:
            print(f"  übersprungen: {exc}")
    data = {tk: df for tk, df in data.items() if len(df) > bt.WARMUP + 100}
    co = [tk for tk, df in data.items() if is_close_only(df)]
    if co and not args.allow_close_only:
        print(f"  ausgeschlossen (nur Schlusskurse, Stops würden unterschätzt): {', '.join(co)}")
        data = {tk: df for tk, df in data.items() if tk not in co}
    if not data:
        print("Keine verwertbaren Kursdaten.")
        return
    regions = {tk: meta_regions.get(tk) or (by_ticker[tk].region if tk in by_ticker else tk) for tk in data}
    print(f"Backtest über {len(data)} Titel – Phase 1 (Signale, kausal) ...")
    cands = bt.generate_all(data, cfg, None, None, workers=args.workers)
    print(f"  {sum(len(v) for v in cands.values())} Signal-Tage")
    start = args.start
    segments_def = [("Gesamter Zeitraum", start, args.end)]
    if args.split:
        segments_def = [("In-Sample bis " + args.split, start, str((pd.Timestamp(args.split) - pd.Timedelta(days=1)).date())),
                        ("Out-of-Sample ab " + args.split, args.split, args.end)]
    pool = [c for v in cands.values() for c in v]
    days = sum(max(0, len(df) - bt.WARMUP) for df in data.values())
    rets = pd.DataFrame({tk: df["close"].pct_change() for tk, df in data.items()})
    segments = []
    for name, s, e in segments_def:
        res = bt.run_portfolio(data, cands, cfg, s, e, regions=regions, label=name)
        es = bt.equity_stats(res)
        rnd_stats = []
        for rep in range(args.random_runs):
            rc = {tk: bt.random_candidates(df, cfg, 0, len(df), pool, len(pool) / max(days, 1), seed=1000 * rep + i,
                                           use_trend=cfg.strategy.trend_filter != "none")
                  for i, (tk, df) in enumerate(data.items())}
            rnd_stats.append(bt.equity_stats(bt.run_portfolio(data, rc, cfg, s, e, regions=regions)))
        r = rets
        if s:
            r = r[r.index >= pd.Timestamp(s)]
        if e:
            r = r[r.index <= pd.Timestamp(e)]
        port = r.mean(axis=1, skipna=True).fillna(0)
        curve = (1 + port).cumprod() * cfg.budget.capital
        yrs = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
        bh = ((curve.iloc[-1] / cfg.budget.capital) ** (1 / yrs) - 1, float((curve / curve.cummax() - 1).min()),
              float(port.mean() / port.std() * np.sqrt(252)) if port.std() > 0 else 0.0)
        segments.append({"name": name, "result": res, "equity_stats": es, "random": rnd_stats, "bh": bh, "bh_curve": curve})
        print(f"  {name}: CAGR {es.get('CAGR', float('nan')):+.1%} | Max. DD {es.get('Max. Drawdown', 0):.1%} | "
              f"Sharpe {es.get('Sharpe', 0):.2f} | Trades {es.get('Trades', 0)} | Kosten {es.get('Kosten gesamt', 0):,.0f} € | "
              f"Zufall-Median CAGR {np.median([x.get('CAGR', 0) for x in rnd_stats]) if rnd_stats else float('nan'):+.1%} | "
              f"Buy&Hold CAGR {bh[0]:+.1%} (DD {bh[1]:.1%})")
    if args.learn_stats:
        trades = bt.signal_study(data, cands, cfg)
        SetupStats.from_trades(trades, meta={"quelle": f"eigener Backtest über {len(data)} Titel",
                                             "trades": len(trades)}).save(args.learn_stats)
        print(f"Setup-Statistik gespeichert: {args.learn_stats} ({len(trades)} Trades)")
    st, c = cfg.strategy, cfg.costs
    cfg_text = [f"{len(data)} Titel: {', '.join(list(data)[:12])}{' …' if len(data) > 12 else ''}",
                f"Budget {cfg.budget.capital:,.0f} € · Risiko {cfg.budget.risk_per_trade:.1%} je Trade",
                f"Einstieg {st.entry_style}, Trendfilter {st.trend_filter}, Ausstieg {cfg.exits.mode}",
                f"Kosten: {c.fee_fixed:g} € + {c.fee_pct:.2%} je Order (min. {c.fee_min:g} €), Spread {c.spread_pct:.2%}, "
                f"Slippage {c.slippage_pct:.2%}"]
    notes = ["Signale werden nur mit Daten bis zum jeweiligen Tag berechnet; Ausführung zum Eröffnungskurs des Folgetags.",
             "Kurslücken über den Stop werden zum Eröffnungskurs ausgeführt; berühren Stop und Ziel denselben Tag, "
             "wird pessimistisch der Stop angenommen.",
             "Zufall: Einstiege an zufälligen Tagen (gleiche Häufigkeit, Stop-Abstände, Ziele, Haltedauer und Trendfilter).",
             "* Buy & Hold: gleichgewichtetes Portfolio der Titel, täglich neu gewichtet, OHNE Kosten und zu 100 % investiert.",
             "Ergebnisse der Vergangenheit sind keine Garantie für die Zukunft. Yahoo-Kurse sind dividenden- und "
             "splitbereinigt; Überlebende-Verzerrung (heute existierende Fonds) ist möglich."]
    html_txt, md_txt = build_backtest_report(segments, cfg_text, notes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"backtest_{stamp}.html").write_text(html_txt, encoding="utf-8")
    (out / f"backtest_{stamp}.md").write_text(md_txt, encoding="utf-8")
    print(f"Report: {out / f'backtest_{stamp}.html'}")


def cmd_universe(args) -> None:
    cfg = _config(args)
    u = load_universe(args.universe)
    sel = RandomSelector(u, cfg.selection)
    print(f"{len(u)} Fonds im Universum, davon {sum(f.em for f in u)} Schwellenländer und "
          f"{sum(f.growth_theme for f in u)} Wachstumsthemen.")
    print(f"Anteil der Wachstumsmärkte am Ziehungsgewicht: {sel.growth_share():.0%}")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="TOML-Konfigurationsdatei")
    common.add_argument("--provider", choices=["yahoo", "csv", "synthetic"], default="yahoo")
    common.add_argument("--csv-dir", default="data", help="Ordner mit <TICKER>.csv (für --provider csv)")
    common.add_argument("--cache-dir", default=".cache/prices")
    common.add_argument("--data-seed", type=int, default=7, help="Seed der synthetischen Daten")
    common.add_argument("--universe", help="eigene Universum-CSV")
    common.add_argument("--capital", type=float, help="Budget (überschreibt Konfiguration)")
    common.add_argument("--risk", type=float, help="Risiko je Trade in Prozent, z. B. 1.0")
    common.add_argument("--allow-short", action="store_true")
    common.add_argument("--portfolio", default="paper_portfolio.json")

    scan_opts = argparse.ArgumentParser(add_help=False)
    scan_opts.add_argument("-n", type=int, help="Anzahl zufälliger Titel")
    scan_opts.add_argument("--seed", type=int, help="Zufalls-Seed (reproduzierbare Auswahl)")
    scan_opts.add_argument("--tickers", help="feste Liste statt Zufallsauswahl, z. B. VWO,INDA")
    scan_opts.add_argument("--min-signals", type=int, default=0,
                           help="weiter ziehen, bis so viele handelbare Signale gefunden sind")
    scan_opts.add_argument("--max-titles", type=int, default=60,
                           help="Obergrenze analysierter Titel bei --min-signals")
    scan_opts.add_argument("--out", default="reports")
    scan_opts.add_argument("--paper", action="store_true", help="Signale im Paper-Depot eröffnen")
    scan_opts.add_argument("--fast", action="store_true", help="ohne Backtest des einzelnen Titels (schneller)")

    bt_opts = argparse.ArgumentParser(add_help=False)
    bt_opts.add_argument("-n", type=int, help="Anzahl zufälliger Titel aus dem Universum (Standard 20)")
    bt_opts.add_argument("--seed", type=int, help="Zufalls-Seed für die Titelauswahl")
    bt_opts.add_argument("--tickers", help="feste Titelliste, z. B. SPY,EEM,QQQ (ALL = alle aus meta.csv)")
    bt_opts.add_argument("--years", type=int, default=20, help="Jahre Historie (Yahoo)")
    bt_opts.add_argument("--start", help="Beginn der Auswertung (JJJJ-MM-TT)")
    bt_opts.add_argument("--end", help="Ende der Auswertung")
    bt_opts.add_argument("--split", help="Datum für Trennung in In-/Out-of-Sample")
    bt_opts.add_argument("--random-runs", type=int, default=5, help="Läufe des Zufalls-Benchmarks")
    bt_opts.add_argument("--learn-stats", help="Setup-Statistik aus dem Backtest in diese Datei schreiben")
    bt_opts.add_argument("--workers", type=int, default=4)
    bt_opts.add_argument("--out", default="reports")
    bt_opts.add_argument("--allow-close-only", action="store_true",
                         help="auch Reihen ohne echte Hoch/Tief-Kurse verwenden (Ergebnisse zu optimistisch!)")
    bt_opts.add_argument("--fast", action="store_true", help=argparse.SUPPRESS)

    p = argparse.ArgumentParser(prog="elliott_bot", description="Elliott-Wave-Analyse- und Paper-Trading-Bot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan", parents=[common, scan_opts], help="Titel zufällig wählen und analysieren")
    sub.add_parser("manage", parents=[common], help="Paper-Depot mit neuen Kursen fortschreiben")
    sub.add_parser("run", parents=[common, scan_opts], help="manage + scan --paper")
    sub.add_parser("backtest", parents=[common, bt_opts], help="Strategie historisch testen (mit Benchmarks)")
    sub.add_parser("universe", parents=[common, scan_opts], help="Universum anzeigen")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "scan":
        cmd_scan(args)
    elif args.cmd == "manage":
        cmd_manage(args)
    elif args.cmd == "run":
        pf = cmd_manage(args)
        print()
        args.paper = True
        cmd_scan(args, portfolio=pf)
    elif args.cmd == "backtest":
        cmd_backtest(args)
    elif args.cmd == "universe":
        cmd_universe(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
