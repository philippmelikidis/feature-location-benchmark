# Ground-Truth-Validierung & Dataset v2

> > Schwester-Seiten: *Query-Modus-Ablation*, *V12b vs. V16c*, *Precision-Analyse v2*.

## TL;DR

Die GT-Validierung ergab: **Dataset v1 (206 Samples) enthielt nur 149 eindeutige
Issues**, 57 Duplikat-Kopien (pandas-Issue #58504 war 17× enthalten), 26
Meta-Issues ohne Code-Ziel, tote Targets. Nach Pipeline-Härtung + Neuextraktion
über GitHubs autoritative Issue↔PR-Verknüpfung: **Dataset v2 mit 918 validen
Samples** (4,5× mehr echte Queries), Era-Split auf allen 5 Repos möglich.

**Wichtigste Erkenntnis:** Der zentrale BA-Befund **repliziert auf den sauberen
Daten**, V16c schlägt V12b um +5,3 pp Macro-Recall@10 (v1: +5,9 pp). Neu:
V12b (Hierarchie ohne LLM-Expansion) fällt auf v2 **hinter das flache V10b**
zurück, der Mehrwert der Hierarchie entsteht erst durch die Expansion.

---

## 1. Was an v1 kaputt war

| Problem | Umfang | Wirkung |
|---|---|---|
| Duplikate (1 Sample je verlinktem PR, kein Dedup) | 57 Kopien (206→149) | Mehrfachzählung; pandas dominiert von DOC-Tracker-Kopien (#58504 17×, #58539 12×) |
| Meta-Issues (DOC:/TRACKER:/STY:/BUILD:) | 26 eindeutige | kein einzelnes Code-Ziel → verzerrt Auswertung (deckt sich mit Precision-Analyse) |
| Tote Targets (Datei @HEAD umbenannt/gelöscht) | 12 Samples ganz tot | garantierte Misses |
| PR-Link per Text-Pattern + Timeline-Erwähnungen |, | False-Positive-Gefahr, Mini-Ausbeute (requests: 7 aus 400) |

pandas v1 real: **21 echte Code-Issues** von nominal 95 „Samples".

## 2. Die Fixes

1. **GraphQL-Extraktion**: `linked:pr`-Suche + `closedByPullRequestsReferences`
   (die PRs, die das Issue tatsächlich geschlossen haben), 8× Ausbeute,
   präziser, ~90 % weniger API-Calls.
2. **Issue-Level-Dedup** (1 Sample/Issue, frühester Fix-PR).
3. **Target-Validierung** gegen HEAD-Checkout mit **Suffix-Semantik wie
   `metrics._file_matches`** (requests' `src/`-Umzug ≠ tote Datei, die
   naive Existenzprüfung hätte 123 valide Samples gekillt).
4. **Meta-Issue-Filter** (abschaltbar), deterministische `sample_id`s
   (rebuild-stabil), Validierungs-Report beim Build, 6 neue Tests.

## 3. Dataset v2 (v1 bleibt unangetastet)

| Repo | v1 nominal | v1 eindeutig+valide | **v2** | Era pre/post (v2) |
|---|---|---|---|---|
| pandas | 95 | 21 | **401** | 119 / 282 |
| click | 52 | ~40 | **261** | 144 / 117 |
| flask | 28 | 25 | **146** | 89 / 57 |
| requests | 15 | 9 | **67** | 46 / 21 |
| fastapi | 16 | 13 | **43** (Pool erschöpft) | 28 / 15 |
| **Σ** | 206 | ~108 | **918** | **426 / 492** |

→ **Era-Split jetzt auf allen 5 Repos möglich** (v1: nur click, 16/36) —
Datenbasis für die Era-Analyse.

**Statistik-Einordnung:** Bei Recall@10 (binär je Query) beträgt das
95 %-CI ±14 pp bei n=50, ±10 pp bei n=100. v2 hebt Repo-Aussagen von
„Anekdote" (requests n=15: ±25 pp) auf „belastbare Tendenz"; gepaarte
Condition-Vergleiche (gleiche Queries) sind ab ~n=100 für 10-pp-Effekte
aussagekräftig. fastapi bleibt mit n=43 die schwächste Zelle (Pool: 156
verlinkte Issues, ausgeschöpft).

## 4. Kern-Benchmark auf v2 (k=10)

| Condition | Repo | n | R@10 v2 | (v1) | MRR v2 | (v1) |
|---|---|---|---|---|---|---|
| V12b | pandas | 401 | 0.518 | 0.407 | 0.402 | 0.231 |
| V12b | click | 261 | 0.796 | 0.680 | 0.701 | 0.623 |
| V12b | flask | 146 | 0.833 | 0.845 | 0.707 | 0.730 |
| V12b | requests | 67 | 0.853 | 0.767 | 0.639 | 0.428 |
| V12b | fastapi | 43 | 0.488 | 0.375 | 0.363 | 0.309 |
| **V12b** | **Macro-Ø** | | **0.698** | 0.615 | **0.562** | 0.464 |
| V16c | pandas | 401 | **0.644** | 0.533 | 0.497 | 0.287 |
| V16c | click | 261 | 0.800 | 0.680 | 0.698 | 0.620 |
| V16c | flask | 146 | 0.828 | 0.845 | 0.707 | 0.748 |
| V16c | requests | 67 | 0.853 | 0.767 | 0.652 | 0.428 |
| V16c | fastapi | 43 | **0.628** | 0.542 | 0.379 | 0.320 |
| **V16c** | **Macro-Ø** | | **0.751** | 0.673 | **0.587** | 0.481 |
| V10b (flat hybrid) | Macro-Ø | | 0.734 | – | 0.571 | – |

## 5. Interpretation

1. **Der V16c-Effekt repliziert:** +5,3 pp Macro-R@10 auf v2 (v1: +5,9 pp),
   weiterhin getragen von pandas (+12,6 pp) und fastapi (+14,0 pp). Der
   zentrale Befund der Arbeit übersteht die Datenbereinigung, starkes
   Validitätsargument.
2. **Alle Kennzahlen steigen** (Macro V12b 0.615→0.698): Die Bereinigung
   entfernt garantierte Misses (tote Targets) und die pathologischen
   Tracker-Duplikate. v1 hat die Systeme systematisch **unterschätzt** —
   v. a. MRR (pandas V12b 0.231→0.402: Duplikat-Meta-Issues hatten das
   Ranking-Maß dominiert).
3. **Neuer Befund:** V12b (Hierarchie ohne LLM) liegt auf v2 **unter** dem
   flachen V10b (0.698 vs. 0.734). Der Hierarchie-Vorsprung existiert erst
   MIT LLM-Expansion (V16c 0.751 > V10b 0.734), konsistent mit der
   Query-Modus-Ablation (Expansion als eigentlicher Hebel).
4. flask sinkt minimal (0.845→0.833/0.828): v1-flask war klein (25) und
   „leicht"; v2 bringt 121 neue, im Schnitt schwerere Queries, realistischer.

## 6. Einschränkungen & offene Punkte

- v1↔v2-Vergleiche sind Verteilungs-Vergleiche (andere Query-Mengen), keine
  gepaarten Tests.
- fastapi n=43 (Pool erschöpft), bleibt die schwächste Zelle.
- Balanced-Core-Entscheidung (z. B. 50/Repo für Macro=Micro) offen —
  betrifft nur Auswertung/Gewichtung, nicht die Daten.
- Übrige Conditions (V1–V19-Sweeps, Code-Embeddings) noch auf v1 —
  Re-Runs auf v2 nach Team-Entscheid „v2 als Default".

## 7. Reproduktion

```bash
# Extraktion (GraphQL, Token nötig)
GITHUB_TOKEN=… python -m benchmark.ground_truth.github_extractor \
    --repo psf/requests --limit 200 --source-dirs src/requests requests \
    --output benchmark/data/raw_requests_ext.json
# Build v2 (Dedup + Meta-Filter + Target-Validierung)
python -m benchmark.ground_truth.dataset_builder \
    --input "benchmark/data/raw_*.json" \
    --output benchmark/data/benchmark_dataset_v2.json --version 2.0.0 --era-splits
# Kern-Benchmark auf v2
python -m benchmark.runner --conditions V12b V16c V10b \
    --dataset benchmark/data/benchmark_dataset_v2.json \
    --output benchmark/results/v2_baseline
```

Rohdaten: `benchmark/data/raw_*_ext.json`, Ergebnisse: `benchmark/results/v2_baseline/`
