# TradingPro – Elliott-Wave-Bot für Fonds & ETFs

Ein Analyse- und Paper-Trading-Bot, der nach der **Elliott-Wellen-Theorie** arbeitet:

1. Er zieht **zufällig Titel aus ~200 großen Fonds/ETFs**. Wachstumsmärkte (Schwellenländer) und Wachstumsthemen werden bevorzugt (Gewicht 3x bzw. 2x, max. 2 Titel je Region).
2. Für jeden Titel erstellt er eine **detaillierte Analyse**: Wellenzählung mit Alternativ-Szenarien, Fibonacci-Kursziele, **Chance/Risiko-Verhältnis**, **Wahrscheinlichkeiten** der Wellenbewegungen und daraus folgende **Gewinn- und Verlustmöglichkeiten in Euro**.
3. Er leitet eine **Ausstiegsstrategie** ab, die das Budget schützt: Stop an der Elliott-Invalidierung, Teilverkäufe, Break-even-Stop, Trailing-Stop, Zeitstopp, Portfolio-Heat-Grenzen und Drawdown-Notbremse.

Ergebnis ist ein HTML-Report mit Charts (inkl. Dunkelmodus) und ein Markdown-Report. Optional führt der Bot ein **Paper-Depot** und wendet die Ausstiegsregeln täglich auf neue Kurse an. **Es werden keine echten Orders gesendet.**

> ⚠️ Keine Anlageberatung. Die Elliott-Wellen-Theorie ist umstritten, Zählungen sind interpretationsabhängig. Der Bot ist bewusst kritisch gebaut: Er vergleicht jede Prognose mit einem Zufallspfad und prüft per Walk-Forward-Test, ob die Methode beim jeweiligen Titel früher überhaupt funktioniert hat.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Python ≥ 3.11 (wegen `tomllib`).

## Benutzung

```bash
# 5 zufällige Titel mit Live-Kursen (Yahoo Finance) analysieren, Budget 10.000 €
python -m elliott_bot scan -n 5 --capital 10000
```

```bash
# Bestimmte Titel analysieren
python -m elliott_bot scan --tickers VWO,INDA,EWZ,IS3N.DE
```

```bash
# Täglicher Bot-Lauf: Paper-Depot fortschreiben (Ausstiege) + neue Titel scannen und Signale eröffnen
python -m elliott_bot run -n 5
```

```bash
# Nur Depot verwalten (Stops, Teilziele, Trailing, Zeitstopp)
python -m elliott_bot manage
```

```bash
# Offline-Demo mit synthetischen Testdaten (keine Internetverbindung nötig)
python -m elliott_bot scan --provider synthetic -n 8 --seed 42
```

```bash
# So lange zufällig weiterziehen, bis 2 handelbare Signale gefunden sind (max. 80 Titel)
python -m elliott_bot scan -n 5 --min-signals 2 --max-titles 80
```

> **Signale sind bewusst selten.** In Tests lieferte nur etwa jeder 30. bis 100. Titel zum jeweiligen Stichtag ein vollständiges Kaufsignal. Gelockerte Filter (z. B. Szenario-Konsens 50 % statt 55 %) brachten im Walk-Forward-Test mehr Signale, aber keinen Vorsprung mehr gegenüber dem Zufall. Deshalb bleiben die strengen Standardwerte. Häufiger sind **WARTEN**-Setups mit Kaufzone.

Weitere Optionen: `--config config.toml`, `--risk 0.5` (Risiko je Trade in %), `--allow-short`, `--seed` (reproduzierbare Auswahl), `--no-calibration` (schneller), `--provider csv --csv-dir data/` (eigene Kursdateien `TICKER.csv` mit `Date,Open,High,Low,Close,Volume`).

Reports landen in `reports/`, das Paper-Depot in `paper_portfolio.json`.

Für einen täglichen Lauf z. B. per cron (werktags 22:30 Uhr):

```bash
30 22 * * 1-5 cd /pfad/zu/TradingPro && .venv/bin/python -m elliott_bot run -n 3 >> bot.log 2>&1
```

## So funktioniert die Analyse

| Schritt | Was passiert |
|---|---|
| Pivots | ZigZag auf drei Wellengraden (2,5 / 4 / 6 × ATR) |
| Zählungen | Alle Muster, die auf dem letzten bestätigten Pivot enden: Welle 3, 4 oder 5 läuft bzw. Korrektur A, B oder C läuft, jeweils auf- und abwärts |
| Harte Regeln | W2 < 100 % von W1 · W3 nie die kürzeste · keine Überlappung W4/W1 (auch in der laufenden Welle) · B ≤ 138,2 % von A. Verstoß = Zählung verworfen |
| Richtlinien | Fibonacci-Verhältnisse (W2 50–61,8 %, W3 1,618 × W1, W4 23,6–38,2 %, W5 = W1 …), Alternation, RSI-Divergenz, Volumen → Qualitätsscore |
| Szenarien | Gewichte aus den Scores inkl. Pseudo-Szenario „keine klare Zählung“ → P(steigend), P(fallend), P(unklar) |
| Ziele & Stops | Fibonacci-Projektionen der laufenden Welle; Stop am strukturellen Niveau (z. B. Ende Welle 2) ± 0,5 ATR |

### Wahrscheinlichkeiten – dreistufig und kritisch

1. **Zufallspfad:** Block-Bootstrap-Simulation (3.000 Pfade aus den echten Tagesrenditen, *ohne* Drift) des **kompletten Ausstiegsplans**. Ergebnis: Was bringt der Trade, wenn Elliott keinen Vorhersagewert hat?
2. **Wellen-Edge:** Die Basis-Wahrscheinlichkeit wird proportional zum Szenario-Konsens verschoben, mit einem konservativen Faktor κ = 0,25.
3. **Walk-Forward-Kalibrierung:** Dieselbe Methode wird auf der Historie des Titels ohne Blick in die Zukunft getestet. Der gemessene Vorsprung gegenüber dem Zufall fließt per Bayes-Mittelung ein. Hat die Methode bei einem Titel historisch schlechter als Zufall abgeschnitten, sinkt die Wahrscheinlichkeit entsprechend.

Aus der kalibrierten Simulation folgen: P(Hauptziel vor Stop), P(Teilziel 1 vor Stop), P(Zeitstopp), Erwartungswert in R und €, 5-%/95-%-Quantile, CVaR.

### Signale

| Signal | Bedeutung |
|---|---|
| **KAUFEN** / SHORT | Alle Filter erfüllt (CRV ≥ 2, Konsens ≥ 55 %, P(TP1) ≥ 45 %, E[R] ≥ +0,15, Positionsgröße > 0) |
| **WARTEN** | Korrektur läuft (Welle 4 oder C) – Kaufzone, Stop und Ziel für die nächste Welle werden angegeben |
| **BEOBACHTEN** | Zählung vorhanden, aber mindestens ein Filter nicht erfüllt (Gründe stehen im Report) |
| **MEIDEN** | Abwärtsszenario oder keine regelkonforme Zählung |

## Budgetschutz & Ausstiegsstrategie

**Positionsgröße** = Minimum aus

- fixem Risiko je Trade (Standard 1 % des Kapitals bis zum Stop),
- Viertel-Kelly aus der simulierten R-Verteilung (f* = E[R] / E[R²]),
- verbleibender Portfolio-Heat (max. 5 % offenes Gesamtrisiko),
- verbleibender Regionen-Heat (max. 2,5 % je Region, gegen Klumpenrisiko),
- max. 20 % des Kapitals je Position,
- verfügbarem Cash abzüglich 10 % Reserve.

**Ausstieg** (wird im Paper-Depot täglich automatisch angewendet):

1. **Initial-Stop** an der Elliott-Invalidierung ± 0,5 ATR (nie enger als 1 ATR; liegt er weiter als 6 ATR entfernt, wird kein Trade eingegangen)
2. **Teilziel 1** (erstes Fibonacci-Ziel): ⅓ verkaufen, Stop auf Einstand + Kosten → Trade wird risikofrei
3. **Teilziel 2**: ⅓ verkaufen, Stop auf Teilziel 1
4. **Rest** mit Chandelier-Trailing-Stop (3 ATR vom Hoch) bis Endziel
5. **Zeitstopp**: Teilziel 1 nicht innerhalb von 1,5 × erwarteter Wellendauer erreicht → raus
6. **Struktur-Exit**: Schlusskurs jenseits der Invalidierung → Zählung falsch → raus
7. **Warnsignal**: Kippt die tägliche Neuanalyse auf ≥ 60 % Gegen-Szenario, wird der Trailing-Stop auf 2 ATR verengt
8. **Drawdown-Notbremse**: ab 10 % Depot-Drawdown halbes Risiko, ab 20 % keine neuen Trades

Kursgaps werden im Paper-Depot realistisch zum Eröffnungskurs abgerechnet.

## Validierung

- 19 automatisierte Tests (`python -m pytest`) prüfen Wellenregeln, Fibonacci-Ziele, Spiegelung für Abwärtstrends, Positionsgröße, Heat-Grenzen, Drawdown-Bremse, die Monte-Carlo-Simulation gegen die theoretische Trefferquote eines Zufallspfads und den kompletten Paper-Trading-Ablauf.
- Auf **synthetischen Daten mit echter fraktaler Elliott-Struktur** liegt die Walk-Forward-Trefferquote über der Zufallsbasis (≈ 27 % gegenüber 25 % beim Hauptziel, Ø +0,8 R je Signal über 30 Titel). Damit ist belegt, dass die Mechanik vorhandene Wellenstrukturen erkennt und nutzt.
- **Ob echte Märkte diese Struktur haben, ist damit nicht belegt.** Genau dafür zeigt jeder Report die Walk-Forward-Statistik des jeweiligen Titels. Diese Zahlen sollten vor jeder echten Verwendung kritisch geprüft werden.

## Bekannte Grenzen

- Die Yahoo-Chart-Schnittstelle ist inoffiziell und kann sich ändern oder drosseln. Alternativ eigene CSV-Dateien nutzen.
- Das Universum (`elliott_bot/data/universe_top200.csv`) ist eine kuratierte Liste großer, liquider ETFs, kein offizielles Live-Ranking. Es lässt sich per `--universe` durch eine eigene Datei ersetzen.
- Viele US-ETFs sind für EU-Privatanleger nicht kaufbar (PRIIPs). Die Liste enthält daher auch einige Xetra-UCITS-ETFs (`.DE`).
- Erkannt werden Impulse und Zigzag-Korrekturen. Diagonalen, Dreiecke und komplexe Kombinationen (W-X-Y) werden nicht gesondert modelliert.

## Projektstruktur

```
elliott_bot/
  universe.py      Fonds-Universum & gewichtete Zufallsauswahl
  data_sources.py  Yahoo / CSV / synthetische Daten
  pivots.py        ZigZag-Pivots
  waves.py         Elliott-Zählung, Regeln, Richtlinien, Projektionen
  strategy.py      Trade-Plan, Setups, Signal-Entscheidung
  montecarlo.py    Bootstrap-Simulation inkl. Exit-Management
  calibration.py   Walk-Forward-Test & Bayes-Kalibrierung
  risk.py          Positionsgröße, Heat, Drawdown-Bremse
  analyzer.py      Gesamtanalyse je Titel, Risiko-/Chancen-Texte
  portfolio.py     Paper-Depot mit Ausstiegsregeln
  report.py        Markdown- & HTML-Report
  cli.py           Kommandozeile
```
