# Prompt-Varianten der Query-Expansion: Precision lügt

> > Schwester-Seiten: *LLM-Expansion: Precision-Analyse (V16 §5.3) v2*,
> *Query-Modus-Ablation*.

## TL;DR

Drei Prompt-Varianten gegen die Baseline getestet (`strict-functions`, `lean`,
`fewshot`; qwen2.5-coder:7b). Auf **Term-Precision-Ebene** gewinnt `lean`
(flask 0.883 vs. 0.843; pandas functions 0.43→0.60, halbe Termzahl, repariert
Worst-Queries bis +0.62). **Downstream kehrt sich das Bild um:** `lean` kostet
im hierarchischen V16c auf pandas **−10,7 pp Recall@10**, weniger Terme =
weniger BM25-Abdeckung in Stage 1. **Coverage schlägt Precision.**

**Empfehlung: Baseline-Prompt bleibt Default.** `fewshot` und
`strict-functions` schon auf Precision-Ebene raus, `lean` an der
Downstream-Hürde gescheitert. Einzige Ausnahme (isoliert, weiter beobachten):
Flat Hybrid + lean auf pandas mit **MRR +19,3 pp** (0.28→0.47), Kandidat für
die Kombi mit `full_expanded`.

---

## 1. Varianten (datengetrieben aus der Precision-Analyse)

| Variante | Idee | Zielt auf |
|---|---|---|
| `strict-functions` | Nur Funktionsnamen, die sicher existieren; im Zweifel weglassen | functions-Precision 0.50 |
| `lean` | Nur treffsichere Kategorien (imports/classes/files), harte Caps, keine keywords | BM25-Noise |
| `fewshot` | Geerdetes Beispiel Issue → echte Identifier | Spezifität |

Erzeugung: `precompute_llm_expansions.py --prompt-variant <v>` → eigene Files
`llm_expansions_<repo>__<v>.json`. Downstream via Env `EXPANSION_VARIANT=<v>`.

## 2. Stufe 1, Term-Precision

| | flask (25) | pandas (94) | functions (pandas) | Ø Terme |
|---|---|---|---|---|
| baseline | 0.843 | 0.667 | 0.431 | 8.9 |
| **lean** | **0.883** | **0.679** | **0.600** | **5.3** |
| strict-functions | 0.839 | 0.636 | 0.456 | 7.8 |
| fewshot | 0.787 | – | – | 8.7 |

Pro Query (pandas): lean 41 besser / 31 schlechter; strict-functions 32/41 ❌.
`lean` repariert Baseline-Worst-Queries (SQLAlchemy 0.21→0.83, TRACKER
Py3.14 0.27→0.83). Meta-Issues (BUILD/DOC/TRACKER ohne Ziel-Datei) rettet
keine Variante, konsistent mit der Precision-Analyse.

→ `fewshot` + `strict-functions` eliminiert; `lean` in die Downstream-Runde.

## 3. Stufe 2, Downstream (lean vs. Baseline-Expansion, k=10)

| Condition | Repo | R@10 base | R@10 lean | ΔR | ΔMRR |
|---|---|---|---|---|---|
| V16c (hier., Stage-1-Expansion) | pandas | 0.533 | 0.426 | **−0.107** | −0.007 |
| V16c | flask | 0.845 | 0.827 | −0.018 | −0.042 |
| V18a (flat dense, ersetzt) | pandas | 0.437 | 0.351 | **−0.086** | −0.022 |
| V18a | flask | 0.762 | 0.780 | +0.018 | −0.005 |
| V18b (flat hybrid, ersetzt) | pandas | 0.475 | 0.507 | +0.032 | **+0.193** |
| V18b | flask | 0.845 | 0.833 | −0.012 | −0.063 |

## 4. Interpretation

- **Precision ≠ Retrieval-Erfolg.** Die präziseren, aber halbierten Terme
  reißen genau dort ein Loch, wo die Expansion ihren Wert hat: BM25-Coverage
  in V16-Stage-1. Ein falscher Term kostet fast nichts (BM25 ignoriert ihn),
  ein fehlender richtiger Term kostet den Kandidaten-File-Hit.
- Das erklärt auch rückwirkend, warum die Baseline-Expansion trotz
  functions-Precision 0.43 funktioniert: Sie ist ein Streunetz.
- **Ausreißer V18b/pandas (MRR +19,3 pp):** präzise Terme + α-Fusion ranken
  den Top-Treffer deutlich besser, wenn die Kandidaten ohnehin gefunden
  werden. Isoliert, aber der stärkste MRR-Sprung des gesamten Projekts —
  wurde in einem Folgelauf (lean × full_expanded) erneut geprüft.

## 5. Empfehlung (AC 4)

1. **Baseline-Prompt bleibt Default** für alle Expansion-Nutzungen (V16/V17
   Stage-1, flat llm_expanded/full_expanded).
2. Prompt-Tuning ist als Hebel für Stage-1-Recall **ausgereizt**, der Weg zu
   mehr Recall führt über Modus (`full_expanded`), Top-N und
   Embeddings, nicht über weniger/präzisere Terme.
3. `lean` einzig als MRR-Booster für Flat Hybrid weiterverfolgen.
4. Meta-Issues (DOC/TRACKER/BUILD) bleiben ein Datenqualitäts-, kein
   Prompt-Problem.

## 6. Einschränkungen

- Downstream nur für `lean` gemessen (übrige Varianten schon auf
  Precision-Ebene eliminiert, Downstream-Läufe wären ~1 h/Variante).
- pandas + flask; click/requests/fastapi nur Baseline.
- Eine LLM-Quelle (qwen2.5-coder:7b), Temperatur 0.3, ein Seed.

## 7. Reproduktion

```bash
python scripts/precompute_llm_expansions.py --repo <repo> --prompt-variant lean \
    --llm-url http://localhost:11434/v1 --model qwen2.5-coder:7b
python scripts/analyze_expansion_precision.py --repo <repo> --variant lean
EXPANSION_VARIANT=lean python -m benchmark.runner --conditions V16c V18a V18b \
    --repos pandas flask --output benchmark/results/variant_lean
```

