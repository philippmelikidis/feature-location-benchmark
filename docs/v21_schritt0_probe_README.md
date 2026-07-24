# Schritt 0 – Lohnt sich der Cross-Encoder? (Probe)

Bevor du die ganze V21-Matrix baust und über Nacht durchlaufen lässt, klärt dieses
eine Skript in ~einer Stunde, ob der Cross-Encoder-Weg überhaupt zu **Recall@10 ≥ 0.80**
führen kann. Wenn nicht, sparst du dir die Matrix und weißt sofort, woran du zuerst
arbeiten musst.

## Die Idee in einem Satz

Dein Endergebnis ist ein Produkt aus zwei Zahlen:

```
Recall@10_gesamt  =  Stage-1-hit-Rate  ×  Fine-Recovery
```

- **Stage-1-hit-Rate** = Anteil der Queries, bei denen die richtige Datei überhaupt
  in der Kandidatenmenge landet. Bei N=80 gemessen: **0.905**. Das ist eine Decke –
  kein Reranker kann sie überschreiten.
- **Fine-Recovery** = Recall@10, aber nur über die Samples, wo Stage-1 getroffen hat.
  Das ist der Teil, den der Reranker verbessern soll. Aktuell (Bi-Encoder, N=80): **0.547**.

Um auf 0.80 zu kommen, muss der Reranker die Fine-Recovery auf

```
0.80 / 0.905 = 0.884
```

heben. Genau diesen Zielwert misst das Skript – und sagt dir, ob dein Reranker ihn trifft.

## Was das Skript macht

Auf **denselben** Stage-1-Kandidaten vergleicht es drei Ranking-Methoden:

| Methode | Was |
|---|---|
| **(A) aktuell** | dein jetziger Stage-2 (V20c). Reproduziert die bekannten ~0.495 / 0.547 → Sanity-Check. |
| **(B) Reranker chunk-topk** | Cross-Encoder, Top-10 einzelne Chunks – so wie im V21-Plan vorgeschlagen. |
| **(C) Reranker file-maxpool** | Cross-Encoder, aber pro Datei bester Chunk, dann Top-10 **Dateien**. Passt zur datei-basierten Eval und ist bei Multi-File-Samples meist besser. |

Für jede Methode gibt es Recall@10 gesamt **und** Fine-Recovery, plus eine
TC1/TC2/TC3-Aufschlüsselung (zeigt, dass TC2 an Stage-1 hängt, nicht am Reranker)
und am Ende ein Urteil.

## Leichtgewichtig by default (wichtig)

Standardmäßig braucht die Probe das große **Qwen3-4B-Embedding NICHT**. Sie baut nur
die billige BM25-Grobstufe (Stage-1) und lässt den kleinen Cross-Encoder laufen; die
(A)-Baseline wird aus den bekannten V20c-Zahlen als Referenz angezeigt. Das ist
schnell, robust und genau das, was Schritt 0 braucht.

Nur wenn du die (A)-Baseline **live** neu rechnen willst, nimm `--with-baseline` – das
lädt Qwen3-4B (~8 GB). Auf dem Mac zwingend die Batch-Größe klein halten, sonst
sprengt es den GPU-(MPS-)Speicher (`Failed to allocate ... MTLBuffer`). Das Skript
setzt automatisch `EMBED_BATCH_SIZE=8`, falls nicht gesetzt.

## So startest du es

Voraussetzungen (auf deiner Maschine, wie bei `run_v20_largeN.py`):

- Elasticsearch läuft (`http://localhost:9200`)
- `benchmark/data/llm_expansions_pandas.json` ist vorhanden
- Reranker-Backend installieren:

```bash
pip install sentence-transformers torch
```

Dann – erst mal klein zum Ausprobieren (30 Samples, ohne Qwen):

```bash
python scripts/probe_rerank_finerecovery.py --limit 30
```

Voller Lauf über alle pandas-Samples:

```bash
python scripts/probe_rerank_finerecovery.py --out benchmark/results/probe_finerecovery.json
```

Mit echter Live-Baseline (lädt Qwen3-4B, langsamer):

```bash
EMBED_BATCH_SIZE=8 python scripts/probe_rerank_finerecovery.py --with-baseline --limit 30
```

Nützliche Schalter:

- `--condition V20d` → N=150 statt N=80 (höhere Stage-1-Decke, mehr Distraktoren)
- `--model BAAI/bge-reranker-base` → kleineres/schnelleres Modell testen
- `--backend flag` → FlagEmbedding statt sentence-transformers
- `--device cpu|mps|cuda` → erzwingen (Default: automatisch)
- `--batch-size 16` → kleiner, falls der Reranker auf MPS zu viel Speicher zieht
- `--max-chunks-per-file 8` → begrenzt die Chunks/Datei fürs Reranking

## Performance auf dem Mac (wichtig)

Der Cross-Encoder muss pro Query alle Chunks der N Kandidat-Dateien bewerten – bei
N=80 sind das ~1600 Chunks. Auf einer echten GPU (RTX 3080) sind das ~0,3–0,6 s/Query;
auf Apple-MPS ist es **deutlich** langsamer (leicht Minuten pro Query). Für einen
Schritt-0-Probe brauchst du nicht alle 95 Samples und nicht alle 1600 Chunks. Empfohlen
auf dem Mac:

```bash
python scripts/probe_rerank_finerecovery.py --limit 25 --max-chunks-per-file 12
```

Zwei Dinge, die die Messung fair halten (nicht kaputtsparen!):

- **`--max-length` NICHT unter 512 drücken.** Query + Chunk teilen sich ein Fenster;
  bei 384 wird der Code-Chunk zu stark abgeschnitten und der Reranker rät.
- **`--max-chunks-per-file` begrenzt die Chunks, aber überlappungsbasiert**, d. h. es
  behält die zur Query passendsten Chunks je Datei (nicht die ersten). So fliegt der
  Ziel-Chunk nicht raus. Trotzdem ist jede Obergrenze eine leichte Untergrenze fürs
  Ergebnis; für die *endgültige* Zahl ohne Cap laufen lassen.
- Die Reranker-Query ist standardmäßig **Titel + Anfang des Issue-Texts**
  (`--rerank-query-mode title_body`), weil der Cross-Encoder ein begrenztes Fenster hat.

Das Skript zeigt **pro Sample** Laufzeit, Chunk-Zahl und ETA. Der wirklich belastbare
Voll-Lauf (alle 95 Samples, **ohne** `--max-chunks-per-file`) gehört auf die
GPU-Maschine (die im Konzept erwähnte RTX 3080) – dort ist bge-reranker-v2-m3 in
Minuten durch statt in Stunden.

## So liest du das Ergebnis

Das Skript druckt am Ende so etwas:

```
Zielwert Fine-Recovery für Recall@10 ≥ 0.80  =  0.80 / 0.905  =  0.884

Methode                     Recall@10   Fine-Recovery   Δ vs Ziel
(A) aktuell (V20c Stage-2)     0.4947          0.5470     -0.3370
(B) Reranker chunk-topk        0.68xx          0.75xx     -0.13xx
(C) Reranker file-maxpool      0.72xx          0.80xx     -0.08xx
```

Das Urteil richtet sich nach der besten Reranker-Variante:

- **✅ GO** – Fine-Recovery ≥ Zielwert. Der Weg funktioniert → V21-Matrix bauen.
- **🟡 MARGINAL** – ≥ 0.75, aber unter Ziel. Stellschrauben: N variieren
  (`--condition V20d`), code-aware Chunks (Pfad + Signaturen an den Chunk hängen),
  oder Stage-1 via Hybrid/RRF anheben, damit die Decke höher wird.
- **❌ NO-GO** – < 0.75. Der Off-the-shelf-Reranker ist auf Code zu schwach.
  Dann zuerst code-aware Chunks oder ein code-spezifisches / feingetuntes Modell,
  danach erneut proben.

## Probe 0b: LLM-Listwise-Reranker (`probe_llm_rerank.py`)

Nachdem der Cross-Encoder-Probe gezeigt hat, dass der generische bge-reranker-v2-m3
die Fine-Recovery NICHT über die 0.55 des Qwen-Hybrids hebt (Ziel ~0.87), testet
`probe_llm_rerank.py` den vielversprechenderen Weg: ein **LLM** bekommt die N
Kandidat-Dateien (mit Symbol-Übersicht) + Issue und wählt in **einem Call** die
Top-10 Dateien. Vorteile: semantisches Issue→Code-Reasoning statt Text-Ähnlichkeit,
arbeitet auf Datei-Ebene (= Eval-Einheit), ein Call statt 1600 Forward-Passes, nicht
MPS-gebunden.

Voraussetzung: **LM Studio läuft** (wie bei `precompute_llm_expansions.py`), Default
`http://localhost:1234/v1`. Aufruf:

```bash
python scripts/probe_llm_rerank.py --limit 25
# voll + Dump:
python scripts/probe_llm_rerank.py --out benchmark/results/probe_llm.json
```

Lesart des Verdikts: **GO** (≥ Zielwert), **🟡 vielversprechend** (≥0.65, klar über
Cross-Encoder), **🟠** (>0.55, minimal über Qwen-Hybrid), **❌** (≤0.55, dann ist 0.80
via Reranking unrealistisch).

## Warum das die richtige Reihenfolge ist

Der Reranker kann **nur** aus dem herausholen, was Stage-1 liefert. Wenn (C) schon
nah an der Stage-1-Decke (0.905) liegt, ist der nächste Hebel nicht ein besserer
Reranker, sondern eine **höhere Stage-1-hit-Rate** (Hybrid-/Dense-Stage-1, hilft auch
TC2). Umgekehrt: liegt (C) weit unter der Decke, ist der Reranker das Problem und
Stage-1 anzufassen wäre verschwendete Zeit. Diese eine Messung sagt dir, welcher der
beiden Fälle vorliegt – bevor du irgendetwas Großes baust.
