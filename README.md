# TradingPro – Elliott-Wave-Bot für Fonds & ETFs

Analyse-, Backtest- und Paper-Trading-Bot nach der **Elliott-Wellen-Theorie**:

1. Zieht **zufällig Titel aus 200 großen Fonds/ETFs**. Wachstumsmärkte (Schwellenländer) werden 3x, Wachstumsthemen 2x bevorzugt, max. 2 Titel je Region.
2. Erstellt je Titel eine **detaillierte Analyse**: Wellenzählung mit Alternativ-Szenarien, Fibonacci-Kursziele, Chance/Risiko-Verhältnis, Wahrscheinlichkeiten (Zufallspfad-Simulation plus gemessene Backtest-Trefferquoten), Gewinn- und Verlustmöglichkeiten in Euro, Risiken.
3. Plant **Einstieg, Ausstieg und Positionsgröße** so, dass das Budget geschützt bleibt, **inklusive aller Gebühren**.
4. Testet das Regelwerk mit einem **realistischen Portfolio-Backtest** gegen Zufall und Kaufen & Halten.
5. Führt optional ein **Paper-Depot**. Es werden keine echten Orders gesendet.

> ⚠️ **Bitte zuerst lesen – Ergebnis der kritischen Prüfung:** Auf echten Kursdaten (1990–2021) war die Richtungsprognose der Elliott-Zählung **nicht besser als ein Münzwurf**. Das Regelwerk lag **out-of-sample nach Kosten bei ≈ +0,5 % pro Jahr**, also gleichauf mit zufälligen Einstiegen und weit unter Kaufen & Halten. Nachweisbar wirksam sind die **Schutzmechanismen**: Kostenfilter, Risikobegrenzung und geringere Drawdowns. Ein Ertragsvorteil ist es nicht. Details unten. Keine Anlageberatung.

---

## Ergebnis der kritischen Prüfung

### Wie getestet wurde

- **Daten:** 17 echte Instrumente mit echten Tageshochs und -tiefs: SPY, QQQ, IWM, **EEM (Schwellenländer)**, USO, Gold, S&P 500, NASDAQ sowie 9 US-Aktien (AAPL, AIG, BAC, IBM, GOOG, FB, NVDA, ORCL, YHOO), 1990–2021. Quelle sind offene Test- und Beispieldaten von QuantConnect LEAN, backtrader und arch. Die Daten sind split- und dividendenbereinigt; ein fehlender Split (GOOG 2014) wurde repariert. Reproduzierbar mit `tools/fetch_open_data.py`.
- **Ohne Blick in die Zukunft:** Ein Wellen-Pivot wird erst ab dem Tag seiner Bestätigung verwendet. Ein Test stellt sicher, dass Signale identisch sind, egal ob spätere Kurse vorhanden sind. Eingestiegen wird zum **Eröffnungskurs des Folgetags**.
- **Realistische Ausführung:** Kurslücken über den Stop werden zum Eröffnungskurs ausgeführt. Berührt ein Tag Stop und Ziel, wird pessimistisch der Stop angenommen. Jede Order kostet Gebühr, Spread und Slippage.
- **Train/Test:** Regeln ausschließlich auf 1990–2010 ausgewählt, 2011–2021 **einmalig** geprüft.
- **Benchmarks:** Zufalls-Einstiege mit *identischem* Regelwerk (gleiche Häufigkeit, Stop-Abstände, Ziele, Haltedauer, Trendfilter) sowie Kaufen & Halten.
- **Kontrolltests:** Auf einem exakten Martingal (faires Spiel) liefern alle Ausstiegsregeln ≈ 0 R. Die Mechanik erzeugt also keinen künstlichen Gewinn.

### 1. Hat die Wellenzählung Vorhersagekraft?

| Richtung der Hauptzählung trifft (20 Tage, trendbereinigt) | 1990–2010 | 2011–2021 |
|---|---|---|
| alle Signaltage | 51,5 % | 50,1 % |
| nur bei hohem Szenario-Konsens (≥ 70 %) | 51,9 % | 48,3 % |
| Wochenchart statt Tageschart (8 Wochen) | 50,7 % | 49,4 % |

**Münzwurf-Niveau.** Ein schwacher, in beiden Zeiträumen gleichgerichteter Effekt: Im Aufwärtstrend entwickeln sich Kurse nach Elliott-*Korrekturen* (W4, A, C) besser als nach Elliott-*Impulsen* (W3, W5). Die erste Version kaufte genau umgekehrt.

### 2. Portfolio-Backtest (10.000 €, 1 % Risiko je Trade, Neobroker 1 €/Order, Spread 0,15 %)

| | 1990–2010 (Auswahl) | **2011–2021 (Test)** |
|---|---|---|
| **v1** (erste Version) | CAGR −0,5 %, max. DD −12,8 %, PF 1,09 | CAGR +0,6 %, max. DD −11,2 %, PF 0,97 |
| **v2** (jetzt Standard) | CAGR +2,5 %, max. DD −8,4 %, PF 1,56 | **CAGR +0,5 %, max. DD −11,2 %, PF 1,01** |
| Zufall, gleiches Regelwerk (Median, 10 Läufe) | CAGR +2,6 % (+0,3 … +6,2 %) | CAGR +0,8 % (−1,0 … +2,4 %) |
| Kaufen & Halten S&P 500 (ohne Kosten) | CAGR +3,7 %, max. DD −55 % | CAGR +14,1 %, max. DD −34 % |
| Kaufen & Halten EEM (ohne Kosten) | CAGR +22,5 %, max. DD −66 % | CAGR +3,1 %, max. DD −39 % |

v2 ist im Auswahlzeitraum klar besser als v1. **Im Test ist der Vorsprung verschwunden.** v1, v2 und Zufall liegen nach Kosten nahe null. Das Regelwerk ist meist nur zu 10–25 % investiert. Seine Stärke ist der begrenzte Drawdown, nicht die Rendite.

> **Korrektur einer früheren Aussage:** Ein erster Testlauf enthielt zusätzlich 23 Reihen, die nur Schlusskurse hatten, und meldete deshalb +8,5 % CAGR. Ohne echte Tageshochs und -tiefs werden Stops, die im Tagesverlauf ausgelöst würden, nicht erkannt. Das Ergebnis war daher zu optimistisch (nachgewiesen per Simulation: +0,1 bis +0,4 R Scheingewinn je Trade). Der Backtest schließt solche Reihen jetzt automatisch aus (`--allow-close-only` zum Erzwingen).

### 3. Gebühren entscheiden über Gewinn oder Verlust

| v2, 10.000 € Budget | 1990–2010 | 2011–2021 |
|---|---|---|
| Neobroker 0 € (nur Spread) | +2,8 % | +0,8 % |
| Neobroker 1 €/Order (Standard) | +2,5 % | +0,5 % |
| Direktbank 4,90 € + 0,25 % (mind. 9,90 €) | Kostenfilter: **kein Trade** | Kostenfilter: **kein Trade** |
| … dieselbe Direktbank *ohne* Kostenfilter | **−0,9 %** | **−2,2 %** |

Mit typischen Bank-Gebühren wäre die Strategie bei 10.000 € **sicher verlustbringend**. Der Kostenfilter verhindert das: Trades, deren Kosten mehr als 15 % des Risikos ausmachen, werden nicht eröffnet.

### Was das für dich bedeutet

- Eine Gewinnerwartung aus der Elliott-Analyse ist **nicht belegt**. Nutze den Bot zum Lernen, Analysieren und im Paper-Depot, nicht als Renditequelle.
- Wenn du handelst, dann nur mit sehr günstigem Broker, mit eingeschaltetem Kostenfilter und mit höchstens 1 % Risiko je Trade.
- Prüfe es selbst mit aktuellen Yahoo-Daten deiner Wunsch-ETFs (`backtest`-Befehl unten). Die Validierung oben enthält nur *einen* Schwellenländer-ETF (EEM). Für Wachstumsmärkte ist sie daher nur eingeschränkt aussagekräftig.
- Vergleiche jedes Ergebnis mit Kaufen & Halten: Ein ETF-Sparplan war in den Testdaten deutlich ertragreicher, allerdings mit viel tieferen Zwischenverlusten.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Python ≥ 3.11.

## Benutzung

```bash
# 5 zufällige Titel mit Live-Kursen (Yahoo Finance) analysieren, Budget 10.000 €
python -m elliott_bot scan -n 5 --capital 10000
```

```bash
# bestimmte Titel; so lange weiterziehen, bis 2 handelbare Signale gefunden sind
python -m elliott_bot scan --tickers VWO,INDA,EWZ,IS3N.DE
```

```bash
python -m elliott_bot scan -n 5 --min-signals 2 --max-titles 80
```

```bash
# Paper-Depot: Orders vom Vortag zum Eröffnungskurs ausführen, Ausstiege anwenden, neue Titel scannen
python -m elliott_bot run -n 5
```

```bash
# nur Depot verwalten
python -m elliott_bot manage
```

```bash
# Backtest mit Yahoo-Daten (20 Jahre), Trennung in In-/Out-of-Sample, Zufalls-Benchmark
python -m elliott_bot backtest --tickers EEM,VWO,INDA,EWZ,EWT,SPY,QQQ --years 20 --split 2016-01-01
```

```bash
# Ergebnisse dieser README nachrechnen (lädt die offenen Testdaten herunter)
python tools/fetch_open_data.py data/open
```

```bash
python -m elliott_bot backtest --provider csv --csv-dir data/open --tickers ALL --split 2011-01-01
```

```bash
# zum Vergleich das Verhalten der ersten Version
python -m elliott_bot backtest --config configs/v1_baseline.toml --provider csv --csv-dir data/open --tickers ALL --split 2011-01-01
```

Weitere Optionen: `--config config.toml`, `--risk 0.5` (Risiko je Trade in %), `--fast` (ohne Backtest je Titel), `--provider csv --csv-dir …` (eigene Dateien `TICKER.csv` mit `Date,Open,High,Low,Close,Volume`), `--provider synthetic` (Offline-Demo mit Kunstdaten), `backtest --learn-stats datei.json` (eigene Setup-Statistik lernen).

Reports landen in `reports/` (HTML mit Charts, hell und dunkel, sowie Markdown), das Paper-Depot in `paper_portfolio.json`.

## Regelwerk v2

| Baustein | Regel |
|---|---|
| Wellenzählung | ZigZag auf 3 Graden (2,5 / 4 / 6 ATR). Harte Regeln führen zum Verwerfen: W2 < 100 % von W1, W3 nie die kürzeste, keine W4/W1-Überlappung (auch in der laufenden Welle), B ≤ 138,2 % von A. Fibonacci-Richtlinien, Alternation, RSI-Divergenz und Volumen ergeben den Qualitätsscore. |
| Trendfilter | Long nur, wenn der Kurs über der *steigenden* 200-Tage-Linie liegt. |
| Einstieg „both“ | **Rücksetzer-Kauf** während einer Korrektur (W4, A, C) einer Aufwärtsstruktur; zusätzlich Impuls-Einstiege (W3/W5). Ausführung zum nächsten Eröffnungskurs. |
| Stop | an der Elliott-Invalidierung (z. B. Ende Welle 1 für W4) ± 0,5 ATR; nie enger als 1 ATR. Liegt er weiter als 6 ATR entfernt, gibt es keinen Trade. |
| Ausstieg | Chandelier-Trailing-Stop 4 ATR ab Einstieg, Struktur-Exit (Schluss unter der Invalidierung), maximale Haltedauer 1,5 x erwartete Wellendauer. |
| Kosten | Jede Order: feste Gebühr, %-Gebühr bzw. Mindestgebühr, halber Spread und Slippage. Kein Trade, wenn die Kosten > 15 % des Risikos ausmachen. Teilverkäufe werden zusammengelegt, wenn die Gebühr > 0,5 % der Teilorder wäre. |
| Positionsgröße | Minimum aus 1 % Risiko *inklusive Gebühren*, Viertel-Kelly, 5 % Portfolio-Heat, 2,5 % je Region bzw. korreliertem Cluster (Korrelation der letzten 120 Tage), 20 % je Position, max. 8 Positionen, 10 % Cash-Reserve. |
| Drawdown-Bremse | ab 10 % Depot-Drawdown halbes Risiko, ab 20 % keine neuen Trades. |

Weitere Ausstiegsmodi (`exits.mode`): `scale_be` (Drittel-Verkäufe, Stop auf Einstand), `scale`, `half`, `target`. Im Backtest waren sie nicht besser.

## Wahrscheinlichkeiten im Report

- **Zufallspfad:** Monte-Carlo-Simulation (Block-Bootstrap der echten Tagesrenditen, ohne Drift) des kompletten Ausstiegsplans. Sie zeigt, was der Plan ohne jede Prognosekraft bringt.
- **Backtest des Setup-Typs:** gemessene Trefferquoten (1. Ziel, Hauptziel) und Ø R dieses Setup-Typs auf den 17 echten Instrumenten (`elliott_bot/data/setup_stats.json`).
- **Backtest des Titels:** dasselbe Regelwerk auf der eigenen Historie des analysierten Titels.
- **Kombiniert:** gewichtete Mischung aus Zufallspfad und Backtest (30 Pseudo-Trades Skepsis).

Die Kaufentscheidung hängt **nicht** von diesen Zahlen ab, sondern exakt vom backgetesteten Regelwerk.

## Konfiguration

`config.example.toml` enthält alle Parameter, darunter Gebührenprofile (Neobroker, Direktbank). Kopieren nach `config.toml` und mit `--config config.toml` verwenden.

## Tests

```bash
python -m pytest
```

34 Tests, unter anderem zu:
- Elliott-Regeln
- Kausalität der Analyse und der Backtest-Signale
- exakter Buchhaltung eines Trades
- Kurslücken
- Pending-Orders
- Gebührenmodell und Positionsgröße inklusive Fixgebühren
- Martingal-Test (Regelwerk auf fairem Spiel ≈ 0)
- Erkennung von Schlusskurs-Daten
- Paper-Depot (Ausführung zum nächsten Eröffnungskurs)
- CLI-Befehle

## Grenzen

- Die Yahoo-Chart-Schnittstelle ist inoffiziell. Alternativ können eigene CSV-Dateien genutzt werden.
- Das Universum ist eine kuratierte Liste großer ETFs, kein offizielles Ranking. Viele US-ETFs sind für EU-Privatanleger nicht kaufbar (PRIIPs); dann das UCITS-Pendant nutzen.
- Diagonalen, Dreiecke und komplexe Korrekturen (W-X-Y) werden nicht modelliert.
- Überlebende-Verzerrung: Getestet wurde mit heute (oder bis zum Datenende) existierenden Titeln.

## Projektstruktur

```
elliott_bot/
  universe.py      Fonds-Universum & gewichtete Zufallsauswahl
  data_sources.py  Yahoo / CSV / synthetische Daten, Erkennung von Schlusskurs-Reihen
  pivots.py        ZigZag-Pivots mit Bestätigungszeitpunkt
  waves.py         Elliott-Zählung (kausal zu jedem Zeitpunkt), Regeln, Projektionen
  strategy.py      Einstiegslogik (Impuls, Rücksetzer, Setups), Trade-Plan
  execution.py     Orders, Positionen, Ausstiegsregeln (gemeinsam für Backtest & Depot)
  backtest.py      Portfolio-Backtest, Signal-Studie, Zufalls-/Null-Benchmarks, Kennzahlen
  bt_report.py     Backtest-Report (HTML/Markdown)
  setup_stats.py   gelernte Statistik je Setup-Typ
  montecarlo.py    Bootstrap-Simulation mit identischen Ausstiegsregeln
  risk.py          Positionsgröße inkl. Gebühren, Heat, Drawdown-Bremse
  analyzer.py      Gesamtanalyse je Titel, Kosten, Risiken/Chancen
  portfolio.py     Paper-Depot
  report.py        Analyse-Report (HTML/Markdown)
  cli.py           Kommandozeile (scan, run, manage, backtest, universe)
tools/fetch_open_data.py   offene Testdaten für die Validierung laden
configs/v1_baseline.toml   Verhalten der ersten Version (Vergleich)
```
