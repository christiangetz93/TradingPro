"""Kommandozeile des Elliott-Wave-Bots.

Beispiele::

    python -m elliott_bot scan -n 5 --capital 10000            # Live-Daten (Yahoo)
    python -m elliott_bot scan --provider synthetic --seed 42   # Offline-Demo
    python -m elliott_bot scan --tickers VWO,INDA,EWZ
    python -m elliott_bot run                                   # Depot verwalten + neue Titel scannen
    python -m elliott_bot manage                                # nur Ausstiegsregeln anwenden
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .analyzer import TitleReport, analyze_title
from .config import BotConfig, load_config
from .data_sources import DataError, make_provider
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
    if getattr(args, "no_calibration", False):
        cfg.analysis.calibrate = False
    return cfg


def _signal_line(r: TitleReport, cur: str) -> str:
    p = r.decision.plan
    parts = [f"{r.fund.ticker:9s} {r.signal:10s}",
             f"{r.analysis.primary.state if r.analysis.primary else '-':3s}"]
    if p:
        parts.append(f"CRV {p.crv:4.1f}")
    if r.p_final is not None:
        parts.append(f"P(Ziel) {r.p_final:4.0%}  E[R] {r.mc_model.mean_r:+.2f}")
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
        rep = analyze_title(fund, series, cfg, book)
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
    scan_opts.add_argument("--no-calibration", action="store_true", help="Walk-Forward-Test überspringen (schneller)")

    p = argparse.ArgumentParser(prog="elliott_bot", description="Elliott-Wave-Analyse- und Paper-Trading-Bot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan", parents=[common, scan_opts], help="Titel zufällig wählen und analysieren")
    sub.add_parser("manage", parents=[common], help="Paper-Depot mit neuen Kursen fortschreiben")
    sub.add_parser("run", parents=[common, scan_opts], help="manage + scan --paper")
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
    elif args.cmd == "universe":
        cmd_universe(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
