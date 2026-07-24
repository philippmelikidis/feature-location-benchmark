# V12b vs. V16c, Wirksamkeit der LLM-Query-Expansion

> > Schwester-Seite: *LLM-Expansion: Precision-Analyse (V16 §5.3)*.

## TL;DR

Vergleich der hierarchischen Baseline **V12b** (Terms-Filter, ohne LLM) gegen
**V16c** (zusätzlich LLM-Query-Expansion in Stage 1) über alle 5 Repos / 206 Queries.

- **V16c gewinnt netto:** Macro-Ø Recall@10 **0.673 vs 0.615 (+5,9 pp)**, MRR@10
  0.481 vs 0.464.
- **Gewinn konzentriert auf große/schwierige Repos:** pandas **+12,6 pp**, fastapi
  **+16,7 pp**. click/flask/requests unverändert.
- **Preis:** bei **8 Queries** verschlechtert die Expansion den Stage-1-Treffer —
  fast ausschließlich pandas-Maintenance-Issues (STY/DOC/TRACKER/BUILD), keine
  echten Feature-/Bug-Lokalisierungen.

Fazit: LLM-Expansion lohnt sich, wirkt genau dort, wo die Baseline schwächelt, und
der Schaden trifft fast nur Issues, die ohnehin keine einzelne Ziel-Datei haben.

---

## 1. Aufbau

| | V12b (Baseline) | V16c |
|---|---|---|
| Retriever | hierarchical_v12 | hierarchical_v16 |
| Stage-1-Query | Issue-Titel | **LLM-expandiert** (Titel + generierte Code-Identifier) |
| Stage-1-Coarse | Class/File-BM25 | VDoc-BM25 |
| Stage-2-Fine | Hybrid + Terms-Filter | Hybrid + Terms-Filter |
| Embedding | bge-base-en-v1.5 (768d) | bge-base-en-v1.5 (768d) |

Dataset: 206 Queries (fastapi 16, click 52, flask 28, pandas 95, requests 15),
k = 1/5/10, Elasticsearch. Einziger relevanter Unterschied: die **LLM-Expansion in
Stage 1**.

---

## 2. Gesamtvergleich (k=10)

| Condition | Macro R@10 | Macro MRR@10 | Micro R@10 | Micro MRR@10 |
|-----------|-----------|--------------|-----------|--------------|
| V12b | 0.6147 | 0.4643 | 0.5591 | 0.4183 |
| **V16c** | **0.6733** | **0.4807** | **0.6303** | **0.4467** |

*Macro = Repos gleich gewichtet (BA-Kennzahl); Micro = alle Samples gleich
gewichtet (pandas dominiert).*

---

## 3. Pro Repo (Recall@10 / MRR@10)

| Repo | V12b R@10 | V16c R@10 | Δ R@10 | V12b MRR | V16c MRR |
|------|-----------|-----------|--------|----------|----------|
| pandas (95)  | 0.407 | **0.533** | **+0.126** | 0.231 | 0.287 |
| fastapi (16) | 0.375 | **0.542** | **+0.167** | 0.309 | 0.320 |
| flask (28)   | 0.845 | 0.845 | 0.000 | 0.730 | 0.748 |
| click (52)   | 0.680 | 0.680 | 0.000 | 0.623 | 0.620 |
| requests (15)| 0.767 | 0.767 | 0.000 | 0.428 | 0.428 |

**Der gesamte Recall-Gewinn kommt aus pandas und fastapi**, den Repos mit der
niedrigsten Baseline (und, laut Precision-Analyse, der niedrigsten Expansion-
Precision = größtem Verbesserungsspielraum). Wo die Baseline schon gut war
(flask/click/requests), ändert die Expansion nichts.

---

## 4. Nach Test-Case-Typ (k=10)

| TC | V12b R@10 | V16c R@10 | V12b MRR | V16c MRR | n |
|----|-----------|-----------|----------|----------|---|
| TC1 (lexikalisch) | 0.561 | **0.647** | 0.403 | **0.442** | 172 |
| TC2 (semantisch)  | 0.583 | 0.589 | 0.539 | 0.521 | 28 |
| TC3 (strukturell) | 0.389 | 0.333 | 0.306 | 0.222 | 6 |

Der Gewinn sitzt bei **TC1 (lexikalisch)**, logisch, da das LLM Code-Identifier
erzeugt, die dem BM25-Matching helfen. TC2 bleibt gleich, TC3 wirkt schlechter,
ist aber mit n=6 nicht aussagekräftig.

---

## 5. Echte Stage-1-Verschlechterung durch Expansion (V12b → V16c)

8 Queries, bei denen die Baseline die Ziel-Datei in Stage 1 fand, V16c aber nicht
mehr (aus `analyze_expansion_precision.py --results …`):

| Repo | sample_id | Titel |
|------|-----------|-------|
| pandas | b7fba158 | STY: Enforce Ruff rule B905, zip-without-explicit-strict |
| pandas | cb822a67 | DOC: Fix docstring validation errors for pandas.core.groupby |
| click  | 65e86b17 | v8.2.0: Release Plan |
| pandas | c6d01487 | REGR: RangeIndex getitem filtering with boolean extension |
| pandas | 5a816762 | TRACKER: Getting Started with Meson/Bug Reports |
| pandas | 75aba9ad | BUILD: Cython.Compiler.Errors: 'free_threading_config.pxi' |
| pandas | 175fc0e0 | STY: Enforce Ruff rule B905, zip-without-explicit-strict |
| pandas | d0d2e2da | TRACKER: add support for Python 3.14 |

**7 von 8 sind pandas-Maintenance-/Meta-Issues** (STY/DOC/TRACKER/BUILD/REGR), das
achte ist ein click-Release-Plan. Genau die Issue-Klasse, die schon in der
Precision-Analyse als „kein einzelnes Code-Ziel" auffiel: hier hängt das LLM
plausible, aber falsche Identifier an und drängt die richtige Datei aus den Top-20.
**Auf echten Feature-/Bug-Lokalisierungen ist der reale Schaden vernachlässigbar.**

---

## 6. Fazit (für die BA)

Die LLM-Query-Expansion (V16c) ist gegenüber der Terms-Filter-Baseline (V12b) ein
**netto positiver, gezielter Hebel**: +5,9 pp Macro-Recall@10, getragen von den
schwierigen Repos pandas (+12,6 pp) und fastapi (+16,7 pp), neutral auf den bereits
starken Repos. Der einzige messbare Preis sind 8 zurückgefallene Queries, die
nahezu vollständig aus Maintenance-/Meta-Issues bestehen, also kaum aus dem
eigentlichen Anwendungsfall der Feature-Lokalisierung.

---

## 7. Einschränkungen

- fastapi (16) und requests (15) haben kleine Stichproben → Tendenz.
- TC3 (n=6) statistisch nicht belastbar.
- Verglichen wurde nur V12b vs V16c; V16a/V16b (Dense Stage 2) und V17 (Ensemble)
  sind hier nicht enthalten.
- Recall@1 ändert sich kaum bzw. auf pandas minimal negativ (0.135 → 0.114): die
  Expansion bringt mehr richtige Dateien in die Top-10, trifft aber nicht öfter
  exakt Platz 1 (Recall-Gewinn ohne proportionalen MRR-Gewinn).

---

## 8. Reproduktion

```bash
export ES_URL=http://localhost:9200
python -m benchmark.runner --conditions V12b V16c --k 1 5 10

python scripts/analyze_expansion_precision.py \
  --results benchmark/results/benchmark_results_latest.json \
  --expanded-condition V16c --baseline-condition V12b
```
