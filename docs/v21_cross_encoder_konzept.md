# V21 – Cross-Encoder Reranking: Konzept & Umsetzungsplan

> Ziel: Recall@10 ≥ 0.80 (aktuelles Optimum: V16c = 0.5912)

---

## 1. Warum wir hier stehen

Die bisherigen Experimente haben ein klares Muster gezeigt:

| N (Stage-1 Dateien) | Stage-1 hit rate | Recall@10 | Fine-Recovery |
|---|---|---|---|
| 20 | 72.6% | 0.5772 | 79.5% |
| 40 | 81.1% | 0.5772 | 71.2% |
| 80 | 90.5% | 0.4947 | 54.6% |
| 150 | 93.7% | 0.4912 | 52.4% |

Stage-1 (BM25 + LLM-Expansion) ist gut, bei N=80 sind 90% der richtigen Dateien im
Kandidatenset. Das Problem liegt in Stage-2: Der aktuelle Bi-Encoder (Qwen3-4B) übergibt
sich, wenn N steigt, weil er ~1600 Kandidaten-Chunks gegen die Query rangieren muss,
ohne Query und Chunk dabei gleichzeitig zu "sehen".

Für 0.80 Recall@10 braucht man:

```
Stage-1-Recall (93.7%) × Fine-Recovery (85%) = 0.796 ≈ 0.80
```

Fine-Recovery muss also von 52% auf 85% steigen. Das schafft kein Bi-Encoder.

---

## 2. Bi-Encoder vs. Cross-Encoder, der Kernunterschied

### Bi-Encoder (aktuell, Stage-2)

```
Query  ──► Encoder ──► Vektor q
                               ╲
                                ──► cosine(q, c) = Score
                               ╱
Chunk  ──► Encoder ──► Vektor c
```

- Query und Chunk werden **separat** kodiert
- Chunk-Vektoren können vorberechnet und im Index gespeichert werden → schnell
- Der Encoder "weiß" beim Kodieren des Chunks nicht, was die Query fragt → Kontextverlust

### Cross-Encoder (neu, Stage-2.5)

```
[Query | Chunk] ──► Transformer ──► Score
```

- Query und Chunk gehen **gemeinsam** durch das Modell
- Das Modell kann erkennen: "Die Query fragt nach X, und in diesem Chunk steht genau X"
- Viel präziser als Bi-Encoder, weil echte Interaktion zwischen Query und Chunk
- Nachteil: Nichts vorberechenbar → jeden Kandidaten-Chunk einzeln durch das Modell
- Deshalb: **nur als zweite Stufe auf einem kleinen Kandidatenset** (nicht über den ganzen Index)

### Was der Qwen3-Reranker ist

Kein generatives LLM, ein **spezialisierter Klassifikator**, der auf Millionen von
(Query, Dokument, relevant: ja/nein)-Paaren feingetuned wurde und als Output direkt
einen Relevanz-Score liefert. Varianten:

- `Qwen3-Reranker-0.6B`, 600M Parameter, sehr schnell, läuft auf CPU/kleiner GPU
- `Qwen3-Reranker-4B`, deutlich präziser, braucht GPU

Alternative: `BAAI/bge-reranker-v2-m3`, bewährt, gut dokumentiert, schnell.

---

## 3. Der neue Flow (V21)

### Aktueller V20-Flow

```
Query
  │
  ▼
Stage-1: BM25 auf LLM-Expansion → Top-N Dateien (N=20)
  │
  ▼
Stage-2: Qwen3-4B Bi-Encoder KNN mit Terms-Filter → Top-10 Chunks
```

### Neuer V21-Flow

```
Query
  │
  ▼
Stage-1: BM25 auf LLM-Expansion → Top-N Dateien (N=80)
  │                                 [Stage-1-Recall: ~90.5%]
  ▼
Stage-2: Alle Chunks der N Dateien sammeln (~1600 Chunks)
  │
  ▼
Stage-2.5: Cross-Encoder scored jedes (Query, Chunk)-Paar
  │         → sortiert nach Relevanz-Score
  ▼
Top-10 Chunks ausgeben
```

Der entscheidende Unterschied: Stage-2 filtert nicht mehr mit einem Bi-Encoder,
sondern **rankt alle Kandidaten direkt durch den Cross-Encoder**. Dadurch können
wir N=80 nehmen (hohe Stage-1-Recall) ohne den Präzisionsverlust des Bi-Encoders.

---

## 4. Was konkret implementiert werden muss

### 4.1 Cross-Encoder-Klasse

Neue Datei: `benchmark/reranker/cross_encoder.py`

Aufgabe: Nimmt eine Query und eine Liste von Chunks, gibt sortierte (Chunk, Score)-Liste zurück.

```python
class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cuda"):
        # Lädt das Modell (z.B. Qwen3-Reranker-0.6B oder bge-reranker-v2-m3)
        # via sentence-transformers oder transformers direkt
        ...

    def rerank(self, query: str, chunks: List[Chunk], top_k: int) -> List[Tuple[Chunk, float]]:
        # Erstellt (query, chunk.content)-Paare
        # Schickt alle durch das Modell (batched)
        # Gibt top_k nach Score sortiert zurück
        ...
```

Wichtig: Batching, nicht jeden Chunk einzeln durch das Modell, sondern in Batches
von z.B. 32 oder 64 Paaren auf einmal. Das reduziert Latenz dramatisch.

### 4.2 V21-Retriever

Neue Datei: `benchmark/retrievers/hierarchical_v21_retriever.py`

Basiert auf `HierarchicalV16Retriever`, mit diesen Änderungen:

- `top_n_files` Default: 80 statt 20
- Stage-2 Bi-Encoder entfällt (oder bleibt als optionales Pre-Filter)
- Neuer Stage-2.5: CrossEncoderReranker

```python
def retrieve(self, query: str, k: int):
    # Stage 1: identisch zu V16, BM25 auf LLM-Expansion → Top-N Dateien
    candidate_files = self._stage1(query)

    # Stage 2: alle Chunks der Kandidat-Dateien sammeln
    candidate_chunks = []
    for file in candidate_files:
        candidate_chunks.extend(self._chunks_by_file[file])

    # Stage 2.5: Cross-Encoder rerankt alle Kandidat-Chunks
    results = self._reranker.rerank(query, candidate_chunks, top_k=k)
    return results
```

### 4.3 Run-Script

Neue Datei: `scripts/run_v21_cross_encoder.py`

Experiment-Matrix:

| Condition | N | Reranker | Notiz |
|---|---|---|---|
| V21a | 20 | bge-reranker-v2-m3 | Baseline: gleicher N wie V20a |
| V21b | 40 | bge-reranker-v2-m3 | |
| V21c | 80 | bge-reranker-v2-m3 | Hauptkandidat |
| V21d | 150 | bge-reranker-v2-m3 | Upper-bound |
| V21e | 80 | Qwen3-Reranker-0.6B | Modell-Vergleich |
| V21f | 80 | Qwen3-Reranker-4B | Modell-Vergleich (wenn GPU verfügbar) |

---

## 5. Erwartete Latenz

Cross-Encoder auf 1600 Chunks (N=80, ~20 Chunks/Datei):

| Modell | Hardware | Batch-Size | Latenz (geschätzt) |
|---|---|---|---|
| bge-reranker-v2-m3 | GPU (RTX 3080) | 32 | ~300-600ms |
| Qwen3-Reranker-0.6B | GPU | 32 | ~200-400ms |
| bge-reranker-v2-m3 | CPU | 16 | ~3-8s |

Zum Vergleich: V20a hat p50-Latenz von 430ms (ohne Cross-Encoder).
Mit GPU ist die Latenz also akzeptabel. Ohne GPU wird es langsam, aber für den
Benchmark spielt Latenz keine Rolle, nur für Produktion.

---

## 6. Warum TC2 (semantische Queries) ein separates Problem bleibt

TC2-Queries enthalten keine Code-Identifier, deshalb findet BM25 in Stage-1 oft schon
nicht die richtige Datei. Der Cross-Encoder hilft in Stage-2, aber wenn Stage-1 die
Datei gar nicht im Kandidatenset hat, ist sie verloren.

Für TC2 wäre der richtige Fix: **Dense Stage-1 parallel zu BM25 Stage-1**, also ein
Ensemble, das für semantische Queries auf Dense-Retrieval über Datei-Embeddings zurückgreift.
Das ist aber ein separates Experiment (V22?).

TC2 macht 11% des Datasets aus, selbst TC2-Recall von 0.0 auf 0.6 zu heben bringt
nur +0.066 im Gesamtdurchschnitt. Nicht ignorieren, aber nicht der erste Schritt.

---

## 7. Zusammenfassung: Warum 0.80 erreichbar ist

```
Stage-1 (N=80):       Stage-1-Recall = 90.5%  [bereits gemessen]
Cross-Encoder Stage-2: Fine-Recovery ≥ 85%     [typisch für gute Reranker]
─────────────────────────────────────────────
Erwartetes Recall@10:  0.905 × 0.85 = 0.769

Stage-1 (N=150):      Stage-1-Recall = 93.7%  [bereits gemessen]
Cross-Encoder Stage-2: Fine-Recovery ≥ 85%     [typisch für gute Reranker]
─────────────────────────────────────────────
Erwartetes Recall@10:  0.937 × 0.85 = 0.796 ≈ 0.80
```

Der Schlüssel: Der Cross-Encoder kann mit N=80-150 Kandidaten umgehen, ohne an
Präzision zu verlieren, genau das, woran der Bi-Encoder gescheitert ist.
