# Era-Vergleich: Pre-AI vs. Post-AI Retrieval-Performance (click)

Repository: `pallets/click` | Pre-AI: 16 Samples (vor 2022) | Post-AI: 36 Samples (ab 2022) | 28 Varianten (V1–V16c)

## Kernergebnis

Retrieval auf Pre-AI-Code performt konsistent besser als auf Post-AI-Code. In 27 von 28 Varianten ist Recall@10 für Pre-AI-Samples höher. Der mittlere Unterschied beträgt **+8,6 Prozentpunkte** (Recall@10) und **+11,0 Prozentpunkte** (MRR@10) zugunsten von Pre-AI.

Bei Recall@1 kehrt sich das Bild um: Post-AI ist in 23 von 28 Varianten besser (+5,3 PP im Mittel). Das bedeutet, dass bei Post-AI-Queries der erste Treffer häufiger korrekt ist, aber die Gesamtabdeckung bei k=10 schlechter ausfällt.

*Hinweis: Delta wird als Pre-AI minus Post-AI berechnet. Positives Delta = Pre-AI besser.*

## Recall@10 nach Retriever-Typ

| Retriever-Typ  | Pre-AI | Post-AI | Delta |
|:---------------|-------:|--------:|------:|
| BM25           | 0.771  | 0.648   | +0.123 |
| Dense (A)      | 0.833  | 0.685   | +0.148 |
| Dense (B)      | 0.750  | 0.685   | +0.065 |
| Dense (C)      | 0.833  | 0.657   | +0.176 |
| Hybrid         | 0.778  | 0.653   | +0.125 |
| Hierarchical   | 0.688  | 0.667   | +0.021 |
| Hier-V12       | 0.695  | 0.630   | +0.066 |
| Hier-V16 (LLM) | 0.764  | 0.668   | +0.096 |

Dense Retriever mit höherer Dimensionalität (C: gte-large, 1024d) zeigen den größten Era-Unterschied (+17,6 PP zugunsten Pre-AI). Hierarchische Retriever sind am stabilsten über beide Eras (+2,1 PP).

## Recall über alle k-Werte

| k  | Mean Delta (Pre−Post) | Pre-AI besser | Post-AI besser |
|---:|----------------------:|--------------:|---------------:|
|  1 |                -0.053 |          5/28 |          23/28 |
|  5 |                +0.027 |         15/28 |          13/28 |
| 10 |                +0.086 |         27/28 |           1/28 |

Bei k=1 schneidet Post-AI besser ab, bei k=5 ist es ausgeglichen, bei k=10 dominiert Pre-AI klar.

## Testfall-Typen (Recall@10, gemittelt über alle Varianten)

| TC-Typ | Beschreibung | Pre-AI | Post-AI | Delta (Pre−Post) | n (pre/post) |
|:-------|:-------------|-------:|--------:|-----------------:|-------------:|
| TC1    | Lexikalisch  | 0.705  | 0.675   | +0.030 | 14 / 29 |
| TC2    | Semantisch   | 1.000  | 0.557   | +0.444 | 2 / 6 |
| TC3    | Strukturell  |,      | 0.702   |,     | 0 / 1 |

Der größte Unterschied liegt bei TC2 (semantische Queries ohne Code-Identifier). Pre-AI-TC2-Samples werden fast perfekt gefunden (1.0), Post-AI-TC2 nur mit 0.56 Recall (+44 PP zugunsten Pre-AI). Einschränkung: n=2 für Pre-AI-TC2 ist sehr klein.

## Varianten mit groesstem und geringstem Era-Unterschied

**Größter Unterschied (Pre-AI >> Post-AI):**

| Variante | Pre-AI | Post-AI | Delta |
|:---------|-------:|--------:|------:|
| V15a (Title-Only BM25) | 0.750 | 0.556 | +0.194 |
| V4 (Dense gte-large) | 0.833 | 0.657 | +0.176 |
| V10c (Hybrid AST alpha=0.8) | 0.802 | 0.644 | +0.159 |

**Geringster Unterschied (stabil über Eras):**

| Variante | Pre-AI | Post-AI | Delta |
|:---------|-------:|--------:|------:|
| V9 (Heuristic Code-Aware) | 0.688 | 0.685 | +0.002 |
| V11a (Hierarchical N=10) | 0.646 | 0.648 | -0.002 |
| V6 (Fixed-Size 512t) | 0.698 | 0.685 | +0.013 |

## Schlüsse

1. **Pre-AI-Code ist leichter zu retrieven.** Der Effekt ist über fast alle Varianten konsistent und beträgt im Mittel ~9 PP bei Recall@10. Mögliche Erklärung: Pre-AI-Issues und Code nutzen spezifischere, weniger standardisierte Formulierungen, die ein stärkeres Signal für Retriever erzeugen.

2. **Post-AI-Queries treffen häufiger den ersten Treffer.** Der umgekehrte Effekt bei k=1 (+5,3 PP) deutet darauf hin, dass Post-AI-Code und Queries homogener formuliert sind, der Top-1-Match ist öfter korrekt, aber die breitere Suche (k=5, k=10) verliert an Präzision, weil mehr Chunks ähnlich klingen.

3. **Semantische Queries (TC2) sind am stärksten betroffen.** Der Era-Gap bei TC2 (-44 PP) ist deutlich größer als bei lexikalischen Queries TC1 (-3 PP). Das stützt die Hypothese, dass KI-generierter Code generischere Beschreibungen und Docstrings produziert, die semantisches Retrieval erschweren.

4. **Hierarchische Retriever sind am robustesten.** Die zweistufige Architektur (Coarse -> Fine) scheint den Era-Effekt abzudämpfen, vermutlich weil die File-Level-Filterung in Stage 1 weniger anfällig für Formulierungsunterschiede ist.

5. **Einschränkung:** n=16 (Pre-AI) erlaubt Tendenzaussagen, aber keine harte statistische Signifikanz. Die TC2-Ergebnisse basieren auf n=2 und sind nur als Hinweis zu werten.
