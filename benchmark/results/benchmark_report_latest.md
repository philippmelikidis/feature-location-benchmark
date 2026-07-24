# PSAIBench – Feature Location Benchmark v1 – Ergebnisse

> Generiert: 2026-07-09 15:47:54
> Dataset Version: 2.0.0
> Runs: 45

> [!IMPORTANT]
> **Aggregationshinweis:**
> Query-Verteilung: fastapi/fastapi=43, pallets/click=261, pallets/flask=146, pandas-dev/pandas=401, psf/requests=67
> Macro-Ø = Repos gleich gewichtet | Micro-Ø = alle Samples gleich gewichtet

## 1. Gesamtübersicht

| Condition | k | Retriever | Chunking | Embedding | Recall@k | MRR@k | n |
|-----------|---|-----------|----------|-----------|----------|-------|---|
| V10b | 1 | hybrid | ast_based | bge-base-en-v1.5 | 0.4776 | 0.5224 | 67 |
| V10b | 1 | hybrid | ast_based | bge-base-en-v1.5 | 0.4475 | 0.6096 | 146 |
| V10b | 1 | hybrid | ast_based | bge-base-en-v1.5 | 0.4962 | 0.6169 | 261 |
| V10b | 1 | hybrid | ast_based | bge-base-en-v1.5 | 0.2093 | 0.2791 | 43 |
| V10b | 1 | hybrid | ast_based | bge-base-en-v1.5 | 0.2801 | 0.3392 | 401 |
| V10b | 5 | hybrid | ast_based | bge-base-en-v1.5 | 0.7861 | 0.6458 | 67 |
| V10b | 5 | hybrid | ast_based | bge-base-en-v1.5 | 0.7352 | 0.7009 | 146 |
| V10b | 5 | hybrid | ast_based | bge-base-en-v1.5 | 0.7069 | 0.6905 | 261 |
| V10b | 5 | hybrid | ast_based | bge-base-en-v1.5 | 0.4186 | 0.3508 | 43 |
| V10b | 5 | hybrid | ast_based | bge-base-en-v1.5 | 0.4746 | 0.4112 | 401 |
| V10b | 10 | hybrid | ast_based | bge-base-en-v1.5 | 0.8532 | 0.6517 | 67 |
| V10b | 10 | hybrid | ast_based | bge-base-en-v1.5 | 0.8356 | 0.7062 | 146 |
| V10b | 10 | hybrid | ast_based | bge-base-en-v1.5 | 0.7995 | 0.6983 | 261 |
| V10b | 10 | hybrid | ast_based | bge-base-en-v1.5 | 0.5814 | 0.3710 | 43 |
| V10b | 10 | hybrid | ast_based | bge-base-en-v1.5 | 0.5985 | 0.4275 | 401 |
| V12b | 1 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.4627 | 0.5075 | 67 |
| V12b | 1 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.4475 | 0.6096 | 146 |
| V12b | 1 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.4962 | 0.6169 | 261 |
| V12b | 1 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.2093 | 0.2791 | 43 |
| V12b | 1 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.2801 | 0.3292 | 401 |
| V12b | 5 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.7786 | 0.6316 | 67 |
| V12b | 5 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.7283 | 0.6999 | 146 |
| V12b | 5 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.7133 | 0.6937 | 261 |
| V12b | 5 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.3643 | 0.3477 | 43 |
| V12b | 5 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.4360 | 0.3905 | 401 |
| V12b | 10 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.8532 | 0.6392 | 67 |
| V12b | 10 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.8333 | 0.7066 | 146 |
| V12b | 10 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.7963 | 0.7007 | 261 |
| V12b | 10 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.4884 | 0.3629 | 43 |
| V12b | 10 | hierarchical_v12 | ast_based | bge-base-en-v1.5 | 0.5179 | 0.4024 | 401 |
| V16c | 1 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.4776 | 0.5224 | 67 |
| V16c | 1 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.4463 | 0.6096 | 146 |
| V16c | 1 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.4962 | 0.6169 | 261 |
| V16c | 1 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.2093 | 0.2791 | 43 |
| V16c | 1 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.3271 | 0.3940 | 401 |
| V16c | 5 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.7861 | 0.6458 | 67 |
| V16c | 5 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.7295 | 0.7018 | 146 |
| V16c | 5 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.7082 | 0.6905 | 261 |
| V16c | 5 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.4186 | 0.3547 | 43 |
| V16c | 5 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.5574 | 0.4856 | 401 |
| V16c | 10 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.8532 | 0.6517 | 67 |
| V16c | 10 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.8276 | 0.7073 | 146 |
| V16c | 10 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.7995 | 0.6983 | 261 |
| V16c | 10 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.6279 | 0.3794 | 43 |
| V16c | 10 | hierarchical_v16 | ast_based | bge-base-en-v1.5 | 0.6442 | 0.4971 | 401 |

## 2. Beste Ergebnisse pro k-Wert

### k = 1
- **Höchster Recall@1**: V12b (0.4962) – hierarchical_v12/ast_based
- **Höchster MRR@1**: V12b (0.6169) – hierarchical_v12/ast_based

### k = 5
- **Höchster Recall@5**: V16c (0.7861) – hierarchical_v16/ast_based
- **Höchster MRR@5**: V16c (0.7018) – hierarchical_v16/ast_based

### k = 10
- **Höchster Recall@10**: V12b (0.8532) – hierarchical_v12/ast_based
- **Höchster MRR@10**: V16c (0.7073) – hierarchical_v16/ast_based

## 3. Vergleich: Retriever-Typen

| Retriever | Macro R@10 | Macro MRR@10 | Micro R@10 | Micro MRR@10 |
|-----------|-----------|-------------|-----------|-------------|
| hierarchical_v12 | 0.6978 | 0.5624 | 0.6703 | 0.5510 |
| hierarchical_v16 | 0.7505 | 0.5867 | 0.7320 | 0.5935 |
| hybrid | 0.7336 | 0.5709 | 0.7111 | 0.5625 |

## 4. Vergleich: Chunking-Strategien

| Chunking | Macro R@10 | Macro MRR@10 | Micro R@10 | Micro MRR@10 |
|----------|-----------|-------------|-----------|-------------|
| ast_based | 0.7273 | 0.5733 | 0.7045 | 0.5690 |

## 5. Breakdown: Test Case Types

> TC1 (Lexikalisch) | TC2 (Semantisch) | TC3 (Strukturell)

### MRR@10 nach Variante × TC (k=10)

| Condition | Retriever | MRR TC1 (n) | MRR TC2 (n) | MRR TC3 (n) |
|-----------|-----------|----------|----------|----------|
| V10b | hybrid | 0.563 (762) | 0.578 (141) | 0.388 (15) |
| V12b | hierarchical_v12 | 0.551 (762) | 0.572 (141) | 0.343 (15) |
| V16c | hierarchical_v16 | 0.597 (762) | 0.595 (141) | 0.406 (15) |

### Recall@10 nach Variante × TC (k=10)

| Condition | Retriever | R@10 TC1 (n) | R@10 TC2 (n) | R@10 TC3 (n) |
|-----------|-----------|----------|----------|----------|
| V10b | hybrid | 0.709 (762) | 0.752 (141) | 0.456 (15) |
| V12b | hierarchical_v12 | 0.666 (762) | 0.716 (141) | 0.433 (15) |
| V16c | hierarchical_v16 | 0.734 (762) | 0.746 (141) | 0.478 (15) |

### TC-Verteilung im Dataset

| TC | Beschreibung | n | Anteil |
|----|-------------|---|--------|
| TC1 | Lexikalisch | 50 | 75% |
| TC2 | Semantisch | 16 | 24% |
| TC3 | Strukturell | 1 | 1% |

## 6. Ergebnisse pro Repository

### click (n=261)

| Condition | k | Recall@k | MRR@k |
|-----------|---|----------|-------|
| V10b | 1 | 0.4962 | 0.6169 |
| V10b | 5 | 0.7069 | 0.6905 |
| V10b | 10 | 0.7995 | 0.6983 |
| V12b | 1 | 0.4962 | 0.6169 |
| V12b | 5 | 0.7133 | 0.6937 |
| V12b | 10 | 0.7963 | 0.7007 |
| V16c | 1 | 0.4962 | 0.6169 |
| V16c | 5 | 0.7082 | 0.6905 |
| V16c | 10 | 0.7995 | 0.6983 |

### fastapi (n=43)

| Condition | k | Recall@k | MRR@k |
|-----------|---|----------|-------|
| V10b | 1 | 0.2093 | 0.2791 |
| V10b | 5 | 0.4186 | 0.3508 |
| V10b | 10 | 0.5814 | 0.3710 |
| V12b | 1 | 0.2093 | 0.2791 |
| V12b | 5 | 0.3643 | 0.3477 |
| V12b | 10 | 0.4884 | 0.3629 |
| V16c | 1 | 0.2093 | 0.2791 |
| V16c | 5 | 0.4186 | 0.3547 |
| V16c | 10 | 0.6279 | 0.3794 |

### flask (n=146)

| Condition | k | Recall@k | MRR@k |
|-----------|---|----------|-------|
| V10b | 1 | 0.4475 | 0.6096 |
| V10b | 5 | 0.7352 | 0.7009 |
| V10b | 10 | 0.8356 | 0.7062 |
| V12b | 1 | 0.4475 | 0.6096 |
| V12b | 5 | 0.7283 | 0.6999 |
| V12b | 10 | 0.8333 | 0.7066 |
| V16c | 1 | 0.4463 | 0.6096 |
| V16c | 5 | 0.7295 | 0.7018 |
| V16c | 10 | 0.8276 | 0.7073 |

### pandas (n=401)

| Condition | k | Recall@k | MRR@k |
|-----------|---|----------|-------|
| V10b | 1 | 0.2801 | 0.3392 |
| V10b | 5 | 0.4746 | 0.4112 |
| V10b | 10 | 0.5985 | 0.4275 |
| V12b | 1 | 0.2801 | 0.3292 |
| V12b | 5 | 0.4360 | 0.3905 |
| V12b | 10 | 0.5179 | 0.4024 |
| V16c | 1 | 0.3271 | 0.3940 |
| V16c | 5 | 0.5574 | 0.4856 |
| V16c | 10 | 0.6442 | 0.4971 |

### requests (n=67)

| Condition | k | Recall@k | MRR@k |
|-----------|---|----------|-------|
| V10b | 1 | 0.4776 | 0.5224 |
| V10b | 5 | 0.7861 | 0.6458 |
| V10b | 10 | 0.8532 | 0.6517 |
| V12b | 1 | 0.4627 | 0.5075 |
| V12b | 5 | 0.7786 | 0.6316 |
| V12b | 10 | 0.8532 | 0.6392 |
| V16c | 1 | 0.4776 | 0.5224 |
| V16c | 5 | 0.7861 | 0.6458 |
| V16c | 10 | 0.8532 | 0.6517 |

## 7. Macro vs. Micro Gesamtvergleich (k=10)

> Macro = Repos gleich gewichtet (Kennzahl in der BA)
> Micro = Samples gleich gewichtet (pandas mit 95 Samples dominiert)

| Condition | Retriever | Macro R@10 | Macro MRR@10 | Micro R@10 | Micro MRR@10 |
|-----------|-----------|-----------|-------------|-----------|-------------|
| V10b | hybrid | 0.7336 | 0.5709 | 0.7111 | 0.5625 |
| V12b | hierarchical_v12 | 0.6978 | 0.5624 | 0.6703 | 0.5510 |
| V16c | hierarchical_v16 | 0.7505 | 0.5867 | 0.7320 | 0.5935 |


## 9. Methodik

### Metriken
- **Recall@K**: Anteil der Ground-Truth-Targets in den Top-K Ergebnissen
- **MRR@K**: Mean Reciprocal Rank – 1/Rang des ersten relevanten Treffers

### Aggregation
- **Macro-Ø**: Berechne Metrik pro Repo → Durchschnitt der Repo-Mittelwerte
- **Micro-Ø**: Flacher Durchschnitt über alle Samples (große Repos dominieren)
- **Empfehlung**: Macro-Ø für Vergleiche (Repos gleich gewichtet)

### Test Case Types
- **TC1 (Lexikalisch)**: Query enthält Code-Identifier → BM25 sollte greifen
- **TC2 (Semantisch)**: Beschreibung ohne Code-Bezeichner → Dense/Embedding
- **TC3 (Strukturell)**: Verständnis der Repo-Architektur nötig → Struktur-Chunking
