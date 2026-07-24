# Query-Modus-Default: full_expanded bestätigt, lean bringt in der Kombi nichts

> > Schwester-Seiten: *Query-Modus-Ablation*,
> *Prompt-Varianten der Query-Expansion: Precision lügt*.

## TL;DR

`full_expanded` (LLM-Terme an vollen Issue-Text angehängt statt ihn zu ersetzen) ist auf
pandas bestätigt der beste Default-Modus für sowohl bge als auch Qwen3-Embedding-0.6B
(k=10, Ø über dense+hybrid: bge ΔR=+0.0544/ΔMRR=+0.0212, Qwen3-0.6B ΔR=+0.0588/ΔMRR=+0.0551,
beide jeweils die höchste ΔR unter den geprüften Modi), jetzt dokumentiert als
`benchmark.config.QUERY_MODE_RECOMMENDED_DEFAULT`. Die Kombination mit der `lean`-Prompt-
Variante bringt für diesen Default-Modus **keinen zusätzlichen Gewinn**: bei `full_expanded`
ist lean gegenüber Baseline-Expansion gemischt bis negativ (3 von 4 Zellen schlechter). Und
V16/V17-Stage-1 profitiert nicht von lean, im Gegenteil, V16c verliert **-18,95pp**
Stage-1-Hit-Rate (70→52 von 95 Samples), stärker als der bereits bekannte Downstream-Verlust
aus dem Prompt-Varianten-Test (-10,7pp R@10). Der Ensemble-Retriever V17a ist deutlich robuster (-3,16pp) —
weil die Vereinigung zweier unterschiedlich granularer Coarse-Legs auch bei identisch
verschlechterter lean-Query nicht auf denselben Samples versagt.

## 1. Ausgangslage

Zwei Hebel wurden bisher nur einzeln gemessen:
- **Aus der Ablation:** `full_expanded` (Expansion an vollen Issue-Text angehängt, statt ihn zu
  ersetzen) ist auf pandas der beste Modus für bge UND Qwen3-0.6B.
- **Aus dem Prompt-Test:** Die `lean`-Prompt-Variante hat die beste Term-Precision (pandas functions
  0.43→0.60), kostete aber im hierarchischen V16c (`llm_expanded`-Modus) −10,7pp R@10 —
  weniger Terme = weniger Stage-1-BM25-Coverage.

Diese Seite kombiniert beide: lean-Expansion **kombiniert mit** `full_expanded` statt
`llm_expanded`, und prüft zusätzlich isoliert, ob der Stage-1-Coverage-Verlust auch bei
`full_expanded`-tauglichen Bedingungen auftritt.

## 2. Kombi-Lauf: lean × full_expanded vs. Baseline × full_expanded

Quelle: `benchmark/results/query_mode_ablation/lean/variant_vs_baseline_lean.md`
(erzeugt durch `scripts/run_query_mode_ablation.py --expansion-variant lean`, bge +
Qwen3-0.6B, pandas, k=10).

| Modell | Retriever | Modus | R@10 Baseline-Exp | R@10 lean-Exp | ΔR | MRR Baseline-Exp | MRR lean-Exp | ΔMRR |
|---|---|---|---|---|---|---|---|---|
| bge | dense | llm_expanded | 0.4368 | 0.3667 | -0.0702 | 0.2937 | 0.2689 | -0.0248 |
| bge | dense | **full_expanded** | 0.4509 | 0.4649 | **+0.0140** | 0.2476 | 0.2528 | +0.0052 |
| bge | hybrid | llm_expanded | 0.4544 | 0.5070 | +0.0526 | 0.2710 | 0.4287 | +0.1577 |
| bge | hybrid | **full_expanded** | 0.4825 | 0.4193 | **-0.0632** | 0.2840 | 0.2808 | -0.0032 |
| qwen3 | dense | llm_expanded | 0.3211 | 0.3772 | +0.0561 | 0.2800 | 0.2485 | -0.0315 |
| qwen3 | dense | **full_expanded** | 0.4000 | 0.3825 | **-0.0175** | 0.3155 | 0.3101 | -0.0054 |
| qwen3 | hybrid | llm_expanded | 0.3491 | 0.4772 | +0.1281 | 0.2930 | 0.4127 | +0.1197 |
| qwen3 | hybrid | **full_expanded** | 0.5070 | 0.4596 | **-0.0474** | 0.3354 | 0.3496 | +0.0142 |

Für den empfohlenen Default-Modus `full_expanded` (fett) ist lean 3-von-4 Zellen schlechter —
nur bge/dense verbessert sich marginal (+0,0140). Interessant: bei `llm_expanded` + hybrid
hilft lean stark (bge +0,0526 R@10/+0,1577 MRR, qwen3 +0,1281 R@10/+0,1197 MRR), das ist der
isolierte MRR-Ausreißer, den `docs/prompt_variants_ergebnisse.md` bereits als "Kandidat für
offen" markiert hatte. Er bestätigt sich, betrifft aber `llm_expanded`, nicht den
empfohlenen `full_expanded`-Default.

## 3. V16/V17 Stage-1: profitiert das kompaktere BM25-Signal?

Quelle: `benchmark/results/stage1_check/{baseline,lean}/` (bge, pandas, k=10).

| Condition | Baseline stage1_hit_rate | Lean stage1_hit_rate | Δ |
|---|---|---|---|
| V16c (VDoc-BM25 sparse Grobstufe) | 0.7368 (70/95) | 0.5474 (52/95) | **-0.1895** |
| V17a (Ensemble: Class/File-BM25 ∪ VDoc-BM25) | 0.7368 (70/95) | 0.7053 (67/95) | -0.0316 |

**Nein, V16/V17-Stage-1 profitiert nicht von lean, sie leidet darunter.** V16c verliert
-18,95pp Stage-1-Hit-Rate, deutlich mehr als der bereits gemessene Downstream-Verlust
(-10,7pp R@10), der Verlust entsteht also überwiegend schon in Stage 1, nicht
erst in Stage 2.

V17a (Dual-Coarse-Ensemble) ist deutlich robuster (-3,16pp). Beide Coarse-Legs erhalten
denselben lean-Query (kein Per-Leg-Modus, siehe
`benchmark/retrievers/hierarchical_ensemble_retriever.py:206-227`), der Grund ist also nicht,
dass ein Leg "immun" gegen die Expansion ist, sondern dass die Vereinigung zweier
unterschiedlich granularer Rankings (Class/File-Level vs. Virtual-Document) auch bei
identisch verschlechterter Query nicht auf denselben Samples versagt (`_merge_file_scores`
bildet eine echte Mengenvereinigung).

## 4. Empfehlung (AC 1 + finale Default-Empfehlung)

1. **`full_expanded` ist Default für die flachen llm-Conditions** von bge und Qwen3-0.6B —
   dokumentiert in `benchmark.config.QUERY_MODE_RECOMMENDED_DEFAULT`.
2. **lean bringt in Kombination mit `full_expanded` keinen zusätzlichen Gewinn** und wird
   NICHT als zusätzlicher Default-Baustein empfohlen. Der einzige robuste lean-Vorteil bleibt
   der zuvor als offen markierte MRR-Ausreißer bei `llm_expanded` + Hybrid-Retriever —
   das ist ein Nebenpfad, kein Ersatz für `full_expanded`.
3. **V16/V17-Stage-1 sollte NICHT auf lean-Expansion umgestellt werden**, der Coverage-Verlust
   ist real und für V16c erheblich (-18,95pp). Falls die kompaktere Stage-1-Query dennoch
   interessant wird (z. B. für Latenz), ist ein Dual-Coarse-Ensemble wie V17a die deutlich
   bessere Absicherung gegen den Coverage-Verlust als ein Single-Coarse-Retriever.
4. qwen3-4b: nicht erneut isoliert bestätigt, bewusst nicht in
   `QUERY_MODE_RECOMMENDED_DEFAULT` gelistet.

## 5. Reproduktion

```bash
python scripts/run_query_mode_ablation.py --models bge qwen3 --repos pandas
python scripts/run_query_mode_ablation.py --models bge qwen3 --repos pandas \
    --expansion-variant lean
python -m benchmark.runner --conditions V16c V17a --repos pandas --k 10 \
    --output benchmark/results/stage1_check/baseline
EXPANSION_VARIANT=lean python -m benchmark.runner --conditions V16c V17a \
    --repos pandas --k 10 --output benchmark/results/stage1_check/lean
```

Code: `scripts/run_query_mode_ablation.py` (`--expansion-variant`) ·
`benchmark/results/stage1_check/` (beide lokal, per `.gitignore` nicht
versioniert, Zahlen oben sind die maßgebliche Quelle)
