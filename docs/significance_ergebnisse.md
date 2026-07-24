# Signifikanz der Kern-Befunde

> 
Gepaarte Tests auf denselben Queries (Pairing via `sample_id`):
**McNemar exakt** für Recall-Hits (binär), **gepaarter Bootstrap** (2000
Resamples, Seed 42) für MRR. α=0.05. Skript: `scripts/analyze_significance.py`.

## Kern-Vergleiche

### V16c vs. V12b (k=10)

Gepaarte Queries: **918**

| Ebene | n | Metrik | V16c | V12b | Δ | Test | p-Wert | signifikant (α=0.05) |
|---|---|---|---|---|---|---|---|---|
| **gesamt** | 918 | Recall@10 | 0.7320 | 0.6703 | +0.0617 | McNemar (b=78, c=21) | 0.0000 | ✅ ja |
| **gesamt** | 918 | MRR@10 | 0.5935 | 0.5510 | +0.0425 [+0.028, +0.057] | gepaarter Bootstrap | 0.0010 | ✅ ja |
| click | 261 | Recall@10 | 0.7995 | 0.7963 | +0.0032 | McNemar (b=1, c=0) | 1.0000 |, nein |
| click | 261 | MRR@10 | 0.6983 | 0.7007 | -0.0025 [-0.006, -0.000] | gepaarter Bootstrap | 0.0480 | ✅ ja |
| fastapi | 43 | Recall@10 | 0.6279 | 0.4884 | +0.1395 | McNemar (b=6, c=1) | 0.1250 |, nein |
| fastapi | 43 | MRR@10 | 0.3794 | 0.3629 | +0.0164 [-0.038, +0.081] | gepaarter Bootstrap | 0.5847 |, nein |
| flask | 146 | Recall@10 | 0.8276 | 0.8333 | -0.0057 | McNemar (b=0, c=0) | 1.0000 |, nein |
| flask | 146 | MRR@10 | 0.7073 | 0.7066 | +0.0007 [-0.010, +0.011] | gepaarter Bootstrap | 0.7546 |, nein |
| pandas | 401 | Recall@10 | 0.6442 | 0.5179 | +0.1264 | McNemar (b=70, c=19) | 0.0000 | ✅ ja |
| pandas | 401 | MRR@10 | 0.4971 | 0.4024 | +0.0947 [+0.064, +0.126] | gepaarter Bootstrap | 0.0010 | ✅ ja |
| requests | 67 | Recall@10 | 0.8532 | 0.8532 | +0.0000 | McNemar (b=1, c=1) | 1.0000 |, nein |
| requests | 67 | MRR@10 | 0.6517 | 0.6392 | +0.0125 [-0.004, +0.035] | gepaarter Bootstrap | 0.2289 |, nein |

### V18d (full_expanded) vs. V10b (full) (k=10)

Gepaarte Queries: **175**

| Ebene | n | Metrik | V18d (full_expanded) | V10b (full) | Δ | Test | p-Wert | signifikant (α=0.05) |
|---|---|---|---|---|---|---|---|---|
| **gesamt** | 175 | Recall@10 | 0.6067 | 0.5686 | +0.0381 | McNemar (b=10, c=3) | 0.0923 |, nein |
| **gesamt** | 175 | MRR@10 | 0.4645 | 0.4248 | +0.0397 [+0.007, +0.076] | gepaarter Bootstrap | 0.0220 | ✅ ja |
| click | 52 | Recall@10 | 0.7308 | 0.7276 | +0.0032 | McNemar (b=0, c=1) | 1.0000 |, nein |
| click | 52 | MRR@10 | 0.6285 | 0.5949 | +0.0337 [-0.008, +0.078] | gepaarter Bootstrap | 0.1069 |, nein |
| flask | 28 | Recall@10 | 0.7976 | 0.8452 | -0.0476 | McNemar (b=0, c=1) | 1.0000 |, nein |
| flask | 28 | MRR@10 | 0.7545 | 0.7364 | +0.0181 [-0.046, +0.095] | gepaarter Bootstrap | 0.6307 |, nein |
| pandas | 95 | Recall@10 | 0.4825 | 0.4000 | +0.0825 | McNemar (b=10, c=1) | 0.0117 | ✅ ja |
| pandas | 95 | MRR@10 | 0.2892 | 0.2399 | +0.0493 [-0.003, +0.104] | gepaarter Bootstrap | 0.0680 |, nein |

### V18d_QWEN3 (full_expanded) vs. V10b_QWEN3 (full) (k=10)

Gepaarte Queries: **175**

| Ebene | n | Metrik | V18d_QWEN3 (full_expanded) | V10b_QWEN3 (full) | Δ | Test | p-Wert | signifikant (α=0.05) |
|---|---|---|---|---|---|---|---|---|
| **gesamt** | 175 | Recall@10 | 0.6029 | 0.5771 | +0.0257 | McNemar (b=7, c=3) | 0.3438 |, nein |
| **gesamt** | 175 | MRR@10 | 0.5028 | 0.4622 | +0.0406 [+0.009, +0.072] | gepaarter Bootstrap | 0.0110 | ✅ ja |
| click | 52 | Recall@10 | 0.7019 | 0.7147 | -0.0128 | McNemar (b=0, c=1) | 1.0000 |, nein |
| click | 52 | MRR@10 | 0.6593 | 0.6330 | +0.0263 [-0.015, +0.072] | gepaarter Bootstrap | 0.2229 |, nein |
| flask | 28 | Recall@10 | 0.7440 | 0.7976 | -0.0536 | McNemar (b=0, c=1) | 1.0000 |, nein |
| flask | 28 | MRR@10 | 0.7262 | 0.7491 | -0.0230 [-0.064, +0.000] | gepaarter Bootstrap | 0.2289 |, nein |
| pandas | 95 | Recall@10 | 0.5070 | 0.4368 | +0.0702 | McNemar (b=7, c=1) | 0.0703 |, nein |
| pandas | 95 | MRR@10 | 0.3513 | 0.2842 | +0.0671 [+0.019, +0.121] | gepaarter Bootstrap | 0.0080 | ✅ ja |

### V16c (lean) vs. V16c (baseline) (k=10)

Gepaarte Queries: **123**

| Ebene | n | Metrik | V16c (lean) | V16c (baseline) | Δ | Test | p-Wert | signifikant (α=0.05) |
|---|---|---|---|---|---|---|---|---|
| **gesamt** | 123 | Recall@10 | 0.5176 | 0.6043 | -0.0867 | McNemar (b=6, c=15) | 0.0784 |, nein |
| **gesamt** | 123 | MRR@10 | 0.3779 | 0.3929 | -0.0150 [-0.059, +0.029] | gepaarter Bootstrap | 0.4678 |, nein |
| flask | 28 | Recall@10 | 0.8274 | 0.8452 | -0.0179 | McNemar (b=0, c=0) | 1.0000 |, nein |
| flask | 28 | MRR@10 | 0.7066 | 0.7483 | -0.0417 [-0.107, +0.000] | gepaarter Bootstrap | 0.2519 |, nein |
| pandas | 95 | Recall@10 | 0.4263 | 0.5333 | -0.1070 | McNemar (b=6, c=15) | 0.0784 |, nein |
| pandas | 95 | MRR@10 | 0.2810 | 0.2881 | -0.0071 [-0.060, +0.047] | gepaarter Bootstrap | 0.7886 |, nein |

## Einordnung

| Befund | Status nach Signifikanztest |
|---|---|
| **V16c > V12b** (v2, n=918) | ✅ **hochsignifikant**, Recall p<0.0001 (b=78/c=21), MRR p=0.001. Der zentrale Befund der Arbeit ist statistisch abgesichert. |
| **full_expanded > full** (v1, n=175) | ⚖️ **MRR signifikant** (bge p=0.022, 0.6B p=0.011), **Recall nicht** (p=0.09/0.34, zu wenig diskordante Paare bei n=175). Richtung konsistent; Recall-Absicherung braucht den v2-Re-Run. |
| **lean vs. baseline** (v1, n=123) | ❌ **nicht signifikant** (Recall p=0.078, MRR p=0.47). Der dramatische pandas-Einzelwert (−10,7 pp) ist gepoolt eine Tendenz, kein belegter Effekt. Die Default-Entscheidung „Baseline behalten" bleibt richtig (kein Wechselgrund), aber die pp-Zahl gehört als Tendenz gelabelt. |

**Lehren für die BA:**
1. Punktwerte ohne CI überschätzen die Präzision massiv (fastapi n=43: R@10-CI ±14 pp).
2. Gepaarte Tests sind der Schlüssel: Der V16c-Effekt (+6,2 pp micro) ist bei n=918 wasserdicht, derselbe Effektbetrag wäre bei v1-n (≈150) grenzwertig gewesen.
3. Mehrere frühere „Befunde" (lean-Schaden, full_expanded-Recall-Gewinn) sind bei v1-Stichprobengrößen formal Tendenzen, die v2-Re-Runs entscheiden.

## Reproduktion

```bash
python scripts/analyze_significance.py \
    --results-a benchmark/results/v2_baseline --condition-a V16c \
    --results-b benchmark/results/v2_baseline --condition-b V12b
python tests/test_statistics.py
```

CIs erscheinen automatisch in jedem neuen Benchmark-Report (reporting.py, Gesamtübersicht).
