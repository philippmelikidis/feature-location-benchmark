# V20 Hybrid-Coarse-N-Sweep auf Dataset v2: die N-Recall-Falle war ein v1-Artefakt

> > Schwester-Seiten: *Ground-Truth-Validierung & Dataset v2*,
> *V21 – Cross-Encoder Reranking: Konzept & Umsetzungsplan*.

## TL;DR

Der V20-N-Sweep (Stage-1 Top-N: 20/40/80/150, plus neu N=500 ≈ „alle Dateien") lief
erstmals vollständig auf dem validierten Dataset v2 (401 statt ~21 echte pandas-Samples)
— inklusive des Hybrid-Grobstufen-Arms (V20e–i), der beim ursprünglichen v1-Lauf nur
definiert, aber nie ausgeführt wurde. Ergebnis: **die zentrale Prämisse, die den Wechsel
zu Cross-Encoder-Reranking (V21) motiviert hat, hält auf sauberen Daten kaum noch.**
Auf v1 fiel der End-to-end-Recall zwischen N=20 und N=150 um **−8,6 pp** (Sparse-Grobstufe,
die einzige, die v1 je durchlief); auf v2 sind es in derselben Sparse-Grobstufe nur noch
**−0,4 pp**, praktisch flach (Hybrid-Grobstufe fällt auf v2 stärker, −2,6 pp, siehe
Abschnitt 3/5). Und die neue N=500-Condition (`V20i`, hybrid,
effektiv keine Stage-1-Filterung) erreicht **gleichzeitig** höheren Recall@10 **und**
höheren MRR@10 als die flache Suche (V8), genau das Ziel, das dieses Ticket formuliert
hat ("Recall der flachen Suche plus Präzision des Rerankings").

**Nachtrag (Abschnitt 7):** Ein direkter Vergleichslauf zeigt aber, dass selbst `V20i`
nicht die beste verfügbare Option ist, **reines flaches Dense-Retrieval mit Qwen3-4B +
`full_expanded`-Query (`V18c_QWEN34B`) schlägt den kompletten V20-Hierarchie-Sweep**,
inklusive `V20i`, um +8,2 pp Recall@10 und +7,9 pp MRR@10. Die zweistufige Architektur
bringt auf pandas/v2 gegenüber gutem Embedding + Query-Expansion allein keinen Mehrwert.

**Nachtrag 2 (Abschnitt 8):** Dass Dense hier Hybrid schlägt, widerspricht der IR-Literatur
(Hybrid gewinnt dort meist), eine gezielte Diagnose (Fusion-Code-Review, Alpha-Sweep,
Expansion-Isolation, Repo-Generalisierung auf click/flask) zeigt: **kein Fusion-Bug, kein
Kalibrierungsproblem** (der Effekt ist über α ∈ {0, 0.2, 0.5, 0.8} monoton, kein besseres
Zwischen-α), sondern ein **repo-abhängiger Effekt**, auf click und flask gewinnt Hybrid
leicht (+1,1/+2,3 pp R@10), konsistent mit der Literatur. pandas' Größe/Repetitivität
(6947 Chunks vs. 311–390) scheint BM25s Diskriminierungskraft dort zu schwächen.

## 1. Ausgangslage

Der v1-Lauf (`benchmark/results/v20_largeN/comparison_v20_largeN.txt`, ~21 echte
pandas-Samples hinter 95 nominalen) zeigte: Stage-1-Recall erreicht die flache Suche
schon bei N=20, aber der End-to-end-Recall **sinkt**, je mehr Kandidaten die Grobstufe
liefert (0,5772 bei N=20 → 0,4912 bei N=150). Interpretation damals: Der
Qwen3-4B-Bi-Encoder in Stufe 2 "übergibt sich" bei mehr Kandidaten, die Begründung für
den Wechsel zu Cross-Encoder-Reranking (`docs/v21_cross_encoder_konzept.md`). Der
Hybrid-Grobstufen-Arm (V20e–h) wurde dafür zwar in `benchmark/config.py` definiert, aber
nie tatsächlich benchmarkt (der v1-Report zeigte dort nur `—`).

Mit Dataset v2 (401 valide pandas-Samples, `docs/dataset_v2_ergebnisse.md`) ist die
Datenbasis ~19× größer und ohne die v1-Pathologien (Duplikate, tote Targets). Dieser Lauf
wiederholt den kompletten Sweep, beide Grobstufen-Familien, plus eine neue N=500-Zelle
(`V20i`), auf v2.

## 2. Stage-1-Datei-Recall vs. N (AC2)

Referenzen (bge-base, frisch auf v2 gerechnet): flache Suche V8 R@10=0,6193, V10b
R@10=0,5848; Top-20-Hierarchie V11b R@10=0,4892, Stage1-Recall=0,6733.
Zielmarke Stage-1-Recall ≈ 0,6193.

| Grobstufe | N | Stage1-Recall | ≥ Ziel? | p50-Latenz |
|---|---|---|---|---|
| Sparse | 20 | 0,8404 | JA | 332 ms |
| Sparse | 40 | 0,9202 | JA | 355 ms |
| Sparse | 80 | 0,9626 | JA | 393 ms |
| Sparse | 150 | 0,9875 | JA | 393 ms |
| Hybrid | 20 | 0,8628 | JA | 640 ms |
| Hybrid | 40 | 0,9202 | JA | 607 ms |
| Hybrid | 80 | 0,9701 | JA | 602 ms |
| Hybrid | 150 | 0,9900 | JA | 602 ms |
| **Hybrid** | **500** | **1,0000** | **JA** | **594 ms** |

**Stage-1-Recall erreicht die flache Suche bei beiden Grobstufen bereits ab N=20**, wie
schon auf v1. Bei matched N liegt Hybrid meist leicht vor Sparse (Δ siehe Abschnitt 4),
außer bei N=40 (identisch). Bei N=500 wird buchstäblich jede GT-Datei erfasst
(Stage1-Recall=1,0), konsistent mit der Interpretation "keine Stage-1-Filterung mehr".

## 3. End-to-end Recall@k / MRR@k vs. N (AC3)

| Grobstufe | N | R@1 | R@5 | R@10 | M@1 | M@5 | M@10 | ΔR@10 vs. flach | ΔR@10 vs. V11b |
|---|---|---|---|---|---|---|---|---|---|
| Sparse | 20 | 0,3441 | 0,5840 | 0,6426 | 0,4239 | 0,5189 | 0,5264 | +0,0233 | +0,1534 |
| Sparse | 40 | 0,3192 | 0,5781 | 0,6592 | 0,3890 | 0,4956 | 0,5054 | +0,0399 | +0,1700 |
| Sparse | 80 | 0,3213 | 0,5603 | 0,6455 | 0,3940 | 0,4911 | 0,5017 | +0,0262 | +0,1563 |
| Sparse | 150 | 0,3259 | 0,5495 | 0,6384 | 0,3940 | 0,4860 | 0,4969 | +0,0191 | +0,1492 |
| Hybrid | 20 | 0,3292 | 0,5998 | 0,6642 | 0,4090 | 0,5151 | 0,5237 | +0,0449 | +0,1750 |
| Hybrid | 40 | 0,3234 | 0,5806 | 0,6546 | 0,3965 | 0,5013 | 0,5104 | +0,0353 | +0,1654 |
| Hybrid | 80 | 0,3238 | 0,5665 | 0,6463 | 0,3965 | 0,4927 | 0,5025 | +0,0270 | +0,1571 |
| Hybrid | 150 | 0,3259 | 0,5495 | 0,6384 | 0,3940 | 0,4853 | 0,4963 | +0,0191 | +0,1492 |
| **Hybrid** | **500** | 0,3229 | 0,5466 | **0,6392** | 0,3865 | 0,4801 | **0,4919** | **+0,0200** | +0,1500 |

**Jede einzelne V20-Condition schlägt sowohl die flache Suche als auch V11b bei R@10** —
über den gesamten N-Bereich, nicht nur bei kleinem N. Der Rückgang von N=20 zu N=150 ist
real, aber klein: Sparse −0,0042 (0,6426→0,6384), Hybrid −0,0258 (0,6642→0,6384). Zum
Vergleich v1 (Abschnitt 5): dort war der Rückgang bei Sparse −0,0860, mehr als das
20-fache (0,0860 / 0,0042 ≈ 20,5). `V20i` (N=500) hält sich mit R@10=0,6392/M@10=0,4919
nahe an N=150 und **schlägt die flache Suche V8 (R@10=0,6193/M@10=0,4478) gleichzeitig in
Recall (+0,0200) und MRR (+0,0441)**.

## 4. Hybrid- vs. Sparse-Grobstufe bei gleichem N

| N | ΔStage1 | ΔR@10 | ΔMRR@10 | Δp50 |
|---|---|---|---|---|
| 20 | +0,0224 | +0,0216 | −0,0027 | +308 ms |
| 40 | +0,0000 | −0,0046 | +0,0051 | +251 ms |
| 80 | +0,0075 | +0,0008 | +0,0008 | +209 ms |
| 150 | +0,0025 | +0,0000 | −0,0006 | +209 ms |

Hybrid-Grobstufe bringt bei kleinem N (20) einen spürbaren Recall-Vorteil (+2,2 pp), der
mit wachsendem N verschwindet (0 bei N=150), bei gleichzeitig **~200–300 ms höherer
Latenz** durch den zusätzlichen Qwen3-4B-Dense-Leg in Stufe 1. Der Zusatzaufwand lohnt
sich also nur bei kleinem N; ab N≈80 ist Sparse praktisch gleichwertig und günstiger.

## 5. v1 vs. v2 (Sparse-Arm), war der N-Recall-Abfall ein Datenartefakt?

| N | R@10 v1 (~21 echte Samples) | R@10 v2 (401 Samples) |
|---|---|---|
| 20 | 0,5772 | 0,6426 |
| 40 | 0,5772 | 0,6592 |
| 80 | 0,4947 | 0,6455 |
| 150 | 0,4912 | 0,6384 |
| **Δ (N20→N150)** | **−0,0860** | **−0,0042** |

**Ja, überwiegend.** Auf v1 sah es so aus, als würde der Bi-Encoder in Stufe 2 bei mehr
Kandidaten regelrecht einbrechen (−8,6 pp). Auf der ~19× größeren, bereinigten v2-Basis
ist derselbe Effekt auf **−0,4 pp** geschrumpft, im Rahmen der Stichprobenschwankung bei
n=401. Der ursprüngliche Befund war real (die Richtung stimmt: leichter Abfall mit N), aber
seine **Größenordnung** war zu einem erheblichen Teil ein Artefakt der extrem kleinen
effektiven v1-Stichprobe (~21 valide Samples).

## 6. Einordnung: was bedeutet das für V21 (Cross-Encoder-Pivot)?

Die V21-Konzeption (`docs/v21_cross_encoder_konzept.md`) war explizit durch den v1-Befund
motiviert: "Stage-1 ist gut, das Problem liegt in Stage-2", mit einem beobachteten
Fine-Recovery-Fenster, das auf v1 dringend nach einem Cross-Encoder rief. Dieser Lauf zeigt:

1. **Die Dringlichkeit war überzeichnet.** Der Bi-Encoder "übergibt sich" auf v2 kaum noch
   bei wachsendem N, der Haupttreiber des V21-Konzepts war grösstenteils Datenrauschen.
2. **Das eigentliche Ziel des V21-Konzepts ist mit reinem N-Sweep + Hybrid-Coarse bereits
   erreichbar:** `V20i` liefert Recall der flachen Suche **plus** bessere Präzision
   (höherer MRR), ohne Cross-Encoder, ohne die Latenz-/Implementierungskosten aus Abschnitt
   5 des V21-Konzepts (300 ms–8 s je nach Hardware).
3. **Cross-Encoder-Reranking ist damit nicht widerlegt**, aber seine Priorität sollte
   neu bewertet werden: Der einfachere Hebel (großes N + Hybrid-Grobstufe) liefert auf
   validierten Daten bereits den geforderten Zielzustand. Ein direkter Vergleich
   `V20i` vs. Cross-Encoder-Reranking auf v2 (nicht mehr auf v1) wäre der nächste
   aussagekräftige Schritt, falls V21 weiterverfolgt wird.

## 7. Flat-Retrieval-Vergleich: braucht es die Hierarchie überhaupt?

Zusätzlich zum N-Sweep: ein direkter Vergleichslauf gegen flaches (nicht-hierarchisches)
Retrieval, mit demselben Embedding (Qwen3-4B) und demselben Query-Modus (`full_expanded`)
wie der V20-Sweep, damit der Vergleich nicht durch Embedding- oder Modus-Unterschiede
verfälscht wird, nur die Architektur (flach vs. zweistufig) variiert.

| Condition | Architektur | R@10 | MRR@10 |
|---|---|---|---|
| **V18c_QWEN34B** | **Flach, Dense**, Qwen3-4B, full_expanded | **0,7211** | **0,5711** |
| V18d_QWEN34B | Flach, Hybrid, Qwen3-4B, full_expanded | 0,6712 | 0,5284 |
| V20i (bester V20-Wert) | Hierarchisch, Hybrid-Grobstufe, N=500 | 0,6392 | 0,4919 |
| V8 (bge, ohne Expansion) | Flach, Dense, bge | 0,6193 | 0,4478 |

**Reines flaches Dense-Retrieval (`V18c_QWEN34B`) schlägt jede V20-Condition** —
+0,0819 R@10 und +0,0792 MRR@10 gegenüber `V20i`, dem bisher besten V20-Ergebnis. Zwei
Überraschungen zugleich:

1. **Die zweistufige Hierarchie (Grobstufe → Feinstufe) bringt hier keinen Mehrwert.**
   Der komplette Aufwand des N-Sweeps (9 Conditions, ~10 h Laufzeit) wird von einem
   einzigen flachen Dense-Retrieval-Lauf mit gutem Embedding + Expansion übertroffen. Der
   entscheidende Hebel war nie die Architektur, sondern Embedding-Qualität (Qwen3-4B statt
   bge) und Query-Expansion (`full_expanded`), beides bereits in früheren Läufen einzeln
   etabliert.
2. **Dense schlägt Hybrid** (V18c > V18d, +0,0499 R@10), entgegen der Intuition, dass
   BM25 + Dense immer mindestens gleichauf mit reinem Dense liegen sollte (die Literatur
   favorisiert i. d. R. Hybrid). Konsistent mit Abschnitt 4, wo der Hybrid-Vorteil in der
   Grobstufe ab N≈80 ebenfalls verschwindet: bei einem bereits starken, expansion-gestützten
   Dense-Signal scheint die BM25-Beimischung eher zu verwässern als zu helfen. **Abschnitt 8
   geht dem systematisch nach**, Ergebnis: kein Kalibrierungsproblem, aber auch kein
   generelles Muster (repo-abhängig).

**Konsequenz:** Die naheliegende nächste Frage ist nicht mehr "welches N/welche
Grobstufe für die Hierarchie", sondern ob die Hierarchie für pandas/v2 überhaupt noch
gerechtfertigt ist, ein Befund, der auch Abschnitt 6 relativiert: Der pragmatischste
nächste Schritt ist vermutlich kein V20-Fine-Tuning und kein Cross-Encoder (V21), sondern
`V18c_QWEN34B` als neue Baseline zu etablieren und andere Retriever-Familien (inkl.
Cross-Encoder) dagegen zu messen, nicht gegen die flache bge-Suche von 2026-05.

## 8. Fusion-Diagnose: warum schlägt Dense Hybrid, Kalibrierung oder etwas Grundsätzlicheres?

Die IR-Literatur favorisiert i. d. R. Hybrid (BM25 + Dense) gegenüber reinem Dense. Drei
Hypothesen gezielt geprüft: (1) fehlerhafte Score-Fusion, (2) Verwässerung durch die
Query-Expansion, (3) repo-spezifischer Effekt.

**(1) Fusion-Code-Review, keine fehlende Normalisierung.** `es_retriever.py` normalisiert
BM25- und Dense-Scores bereits per Min-Max auf [0,1], *bevor* sie mit α kombiniert werden
(`fused = α·bm25_norm + (1-α)·dense_norm`). Keine unnormalisierten Rohscores, diese
Hypothese entfällt. (RRF existiert im Repo, aber nur in einer nicht angebundenen, toten
Klasse, nicht die Ursache des beobachteten Verhaltens.)

**(2) Alpha-Sweep (pandas, α ∈ {0, 0.2, 0.5, 0.8}, `V18d_a02`/`V18d_a08` + bestehende
V8/V10a-c), Effekt ist monoton, kein Kalibrierungsproblem:**

| α (BM25-Gewicht) | R@10 mit Expansion | MRR@10 mit Expansion | R@10 ohne Expansion | MRR@10 ohne Expansion |
|---|---|---|---|---|
| 0 (reines Dense) | 0,7211 | 0,5711 | 0,7107 | 0,5573 |
| 0,2 | 0,7012 | 0,5696 | 0,6874 | 0,5531 |
| 0,5 | 0,6712 | 0,5284 | 0,6534 | 0,4911 |
| 0,8 | 0,5723 | 0,4184 | 0,4913 | 0,3168 |

Beide Metriken sinken **monoton** mit steigendem BM25-Gewicht, in beiden Query-Modi. Es
gibt kein besseres mittleres α, der Optimalwert ist α=0 (kein BM25). Das ist kein
Kalibrierungsartefakt, BM25 schadet hier durchgängig.

**(3) Query-Expansion-Effekt (isoliert bei gleichem α):** `full_expanded` hilft leicht in
beiden Architekturen, Dense +0,0104 R@10 (0,7211 vs. 0,7107), Hybrid α=0,5 +0,0178 R@10
(0,6712 vs. 0,6534). Ändert aber nichts an der Dense-vor-Hybrid-Rangfolge, Hypothese "Expansion
verwässert Hybrid" entfällt ebenfalls, Expansion hilft beiden Architekturen leicht.

**(4) Repo-Generalisierung, hier liegt der eigentliche Befund.** Derselbe
Dense-vs-Hybrid-Vergleich (α=0,5, `V8`/`V18c` vs. `V10b`/`V18d`, jeweils Qwen3-4B) auf
click (261 Samples, 390 Chunks) und flask (146 Samples, 311 Chunks) statt pandas (401
Samples, 6947 Chunks):

| Repo (Chunks) | Dense R@10 (+Exp) | Hybrid α=0,5 R@10 (+Exp) | Δ (Hybrid − Dense) | Gewinner |
|---|---|---|---|---|
| pandas (6947) | 0,7211 | 0,6712 | **−0,0499** | Dense |
| click (390) | 0,7937 | 0,8046 | **+0,0109** | Hybrid |
| flask (311) | 0,8231 | 0,8459 | **+0,0228** | Hybrid |

**Der Dense-vor-Hybrid-Befund ist pandas-spezifisch, nicht generell.** Auf den kleineren,
vokabular-diverseren Repos gewinnt Hybrid leicht, konsistent mit der Literatur. pandas ist
mit 6947 Chunks (vs. 311–390 bei click/flask) ein Ausreißer in Codebase-Größe und
-Repetitivität (viele strukturell ähnliche interne Module), BM25s IDF-Gewichtung
diskriminiert dort schlechter, weil viele Terme (`self`, interne Helper-Namen, wiederholte
Patterns) über tausende Chunks hinweg kaum selten genug sind, um stark zu gewichten.

**Fazit:** Kein Fusion-Bug, kein Kalibrierungsproblem, keine Verwässerung durch Expansion —
sondern ein echter, repo-abhängiger Effekt, plausibel erklärt durch Codebase-Größe/
-Repetitivität. Für die Praxis: Dense-only als Default für große, repetitive Codebases
(wie pandas); Hybrid bleibt für kleinere/diversere Repos konkurrenzfähig bis leicht
überlegen. Eine repo-adaptive α-Wahl (oder ein Modell, das α aus Repo-Metriken wie
Chunk-Zahl/Vokabular-Diversität vorhersagt) wäre der nächste sinnvolle Schritt, falls
Hybrid als universeller Default gewünscht ist.

## 9. Laufzeit-Overhead (AC4)

Alle Latenzen sind p50-Query-Latenz (Stufe 1 + Stufe 2 zusammen), siehe Tabelle in
Abschnitt 2. Zusammengefasst:

- **Sparse-Grobstufe:** 332–393 ms über den gesamten N-Bereich (20→150), nahezu flach,
  N-Erhöhung kostet kaum Latenz.
- **Hybrid-Grobstufe:** 594–640 ms, durchgängig ~200–300 ms teurer als Sparse (der
  zusätzliche Qwen3-4B-Dense-Leg in Stufe 1), aber ebenfalls nahezu N-unabhängig; N=500
  ist sogar leicht günstiger als N=20 (594 ms vs. 640 ms) innerhalb der Messstreuung.
- **Trade-off:** Der Recall-Gewinn durch höheres N kostet praktisch keine zusätzliche
  Latenz (beide Familien), die Latenz wird von der Grobstufen-**Art** (sparse vs. hybrid)
  bestimmt, nicht von N. Wer die ~250 ms Hybrid-Aufschlag scheut, verliert dafür bei N=20
  ~2,2 pp Stage-1-Recall (Abschnitt 4) gegenüber Sparse.

## 10. Einschränkungen

- Nur pandas (401 Samples), nicht die übrigen 4 v2-Repos (Scope-Entscheidung für diesen
  Lauf, siehe Reproduktion für den Befehl, falls das nachgezogen werden soll).
- LLM-Expansion-Coverage: 380/401 (94,8 %) der v2-pandas-Samples haben eine exakte
  `llm_expansions_pandas.json`-Übereinstimmung; die restlichen 21 fallen auf
  `title_only` zurück (bestehendes, getestetes Fallback-Verhalten, nicht neu für diesen
  Lauf).
- `top_n_files=500` für `V20i` ist eine Näherung für "alle minus Rauschen" (siehe
  `benchmark/config.py`-Kommentar bei `V20i`), keine exakte Rauschfilterung, pandas hat
  ~281 VDoc-Dateien insgesamt, 500 liegt komfortabel darüber (Stage1-Recall=1,0 bestätigt
  das empirisch).
- v1↔v2-Vergleich in Abschnitt 5 ist ein Verteilungsvergleich (andere, teilweise
  überlappende Query-Mengen), kein gepaarter Test.
- Abschnitt 7 (Flat-Vergleich) ist ein einzelner Lauf ohne Wiederholung, wie der Rest
  dieses Dokuments, keine Signifikanzangabe, aber die Effektgröße (+8,2 pp R@10) liegt
  weit über der in Abschnitt 3 beobachteten N-Sweep-Streuung (≤2,6 pp).
- Abschnitt 8s Repo-Generalisierung nutzt click (n=261) und flask (n=146), kleiner als
  pandas (n=401), also weniger statistische Power pro Repo; die click/flask-Effekte
  (+1,1/+2,3 pp) sind zudem deutlich kleiner als der pandas-Effekt (−5,0 pp) und ebenfalls
  Einzelläufe ohne Wiederholung. Die Kernaussage (Richtung kehrt sich um, nicht nur
  Größe) ist robust, die genauen click/flask-Zahlen weniger.
- Die Chunk-Zahl (6947 pandas vs. 390/311 click/flask) als Erklärung für den Repo-Effekt
  ist eine plausible Korrelation, keine kausal geprüfte Erklärung, dafür bräuchte es
  mehr Repos über die Größenspanne verteilt.

## 11. Reproduktion

```bash
# V20-N-Sweep (Abschnitt 2-6):
EMBED_BATCH_SIZE=8 python scripts/run_v20_largeN.py \
    --dataset benchmark/data/benchmark_dataset_v2.json --with-baselines
# Nur Report aus vorhandenen Ergebnissen neu erzeugen:
python scripts/run_v20_largeN.py \
    --dataset benchmark/data/benchmark_dataset_v2.json --report-only --with-baselines

# Flat-Retrieval-Vergleich (Abschnitt 7):
EMBED_BATCH_SIZE=8 python -m benchmark.runner \
    --conditions V18c_QWEN34B V18d_QWEN34B \
    --dataset benchmark/data/benchmark_dataset_v2.json --repos pandas --k 1 5 10 \
    --output benchmark/results/flat_qwen34b_fullexp_v2

# Fusion-Diagnose: Alpha-Sweep + Expansion-Isolation, pandas (Abschnitt 8.2-8.3):
EMBED_BATCH_SIZE=8 python -m benchmark.runner \
    --conditions V18d_a02_QWEN34B V18d_a08_QWEN34B V8_QWEN34B V10a_QWEN34B V10b_QWEN34B V10c_QWEN34B \
    --dataset benchmark/data/benchmark_dataset_v2.json --repos pandas --k 1 5 10 \
    --output benchmark/results/fusion_diagnosis_v2

# Fusion-Diagnose: Repo-Generalisierung, click+flask (Abschnitt 8.4):
EMBED_BATCH_SIZE=8 python -m benchmark.runner \
    --conditions V8_QWEN34B V18c_QWEN34B V10b_QWEN34B V18d_QWEN34B \
    --dataset benchmark/data/benchmark_dataset_v2.json --repos click flask --k 1 5 10 \
    --output benchmark/results/fusion_diagnosis_v2
```

Code: `scripts/run_v20_largeN.py` (`--dataset`), `benchmark/config.py` (`V20i`,
`V18d_a02`/`V18d_a08` Alpha-Sweep) ·
Daten: `benchmark/results/v20_largeN_benchmark_dataset_v2/comparison_v20_largeN.txt`,
`benchmark/results/flat_qwen34b_fullexp_v2/benchmark_report_latest.md`,
`benchmark/results/fusion_diagnosis_v2/benchmark_report_latest.md`
