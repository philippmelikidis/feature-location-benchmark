# Era-Split auf Dataset v2: Pre-AI vs. Post-AI, multi-repo

> > Schwester-Seiten: *Pre-/Post-AI Sample-Balance je Repository*,
> *Ground-Truth-Validierung & Dataset v2*.
> **Interpretation/Hypothesen: Research-Team, diese
> Seite liefert die Zahlen.**

## TL;DR

Erster Era-Split über **alle 5 Repos** (v2, 918 Queries) mit **beiden
Cutoffs** des Research-Teams (primär 2022-11-30 = ChatGPT-Launch;
Sensitivität 2022-01-01), je 2 Conditions (V10b flat hybrid, V16c
hierarchisch + LLM-Expansion):

1. **Der Pre-AI-Vorteil bei Recall@10 bestätigt sich multi-repo:**
   Macro Δ(pre−post) = **+6,4 bis +8,5 pp**, robust über beide Cutoffs und
   beide Conditions (click-v1-Finding: +8,6 pp, repliziert).
2. **Das v1-Gegenfinding „Post-AI besser bei Recall@1" bestätigt sich NICHT:**
   Auch R@1 ist multi-repo pre-lastig (+1,5 bis +6,1 pp). Der v1-Wert war
   vermutlich ein click-Einzelartefakt (n=36).
3. **LLM-Expansion verkleinert den Era-Gap:** V16c hebt Post-AI-pandas von
   0.530 auf 0.598 R@10 (Gap V10b +8,5 → V16c +6,4 pp Macro), die
   Expansion kompensiert einen Teil des Post-AI-Nachteils.
4. **Cutoff-Wahl ändert die Kernaussage nicht**, gute Nachricht für die
   Robustheit der Era-Methodik.

## 1. Setup

- Dataset v2 (918 Queries), Era dynamisch reklassifiziert aus
  `issue_created_at` (`--era-cutoff`, einem früheren Stand): primär **2022-11-30**,
  Sensitivität **2022-01-01**.
- Splits: 2022-11-30 → 515 pre / 403 post; 2022-01-01 → 426 pre / 492 post.
- Conditions: V10b (flat hybrid AST, bge) und V16c (hierarchisch,
  LLM-Expansion Stage 1, Hybrid Stage 2, bge). k=1/5/10.
- **Achtung:** Era-Vergleiche sind UNGEPAART (verschiedene Queries) und mit
  Issue-Mix/Repo-Alter konfundiert, Zahlen sind deskriptiv, kein Kausalbeleg.

## 2. Recall@10 je Repo × Era (primärer Cutoff 2022-11-30)

| Cond | Repo | n pre | n post | R@10 pre | R@10 post | Δ(pre−post) | MRR pre | MRR post |
|---|---|---|---|---|---|---|---|---|
| V10b | pandas | 151 | 250 | 0.711 | 0.530 | **+0.181** | 0.547 | 0.333 |
| V10b | click | 172 | 89 | 0.824 | 0.730 | +0.093 | 0.722 | 0.640 |
| V10b | flask | 112 | 34 | 0.838 | 0.828 | +0.009 | 0.682 | 0.771 |
| V10b | requests | 50 | 17 | 0.873 | 0.794 | +0.079 | 0.676 | 0.545 |
| V10b | fastapi | 30 | 13 | 0.600 | 0.538 | +0.062 | 0.421 | 0.335 |
| **V10b** | **Macro** | | | **0.769** | **0.684** | **+0.085** | | |
| V16c | pandas | 151 | 250 | 0.696 | 0.598 | +0.098 | 0.581 | 0.430 |
| V16c | click | 172 | 89 | 0.824 | 0.730 | +0.093 | 0.722 | 0.640 |
| V16c | flask | 112 | 34 | 0.827 | 0.828 | −0.001 | 0.679 | 0.786 |
| V16c | requests | 50 | 17 | 0.873 | 0.794 | +0.079 | 0.676 | 0.545 |
| V16c | fastapi | 30 | 13 | 0.667 | 0.615 | +0.051 | 0.432 | 0.344 |
| **V16c** | **Macro** | | | **0.777** | **0.713** | **+0.064** | | |

## 3. Sensitivitätscutoff 2022-01-01 (Macro)

| Cond | R@10 pre | R@10 post | Δ | R@1 pre | R@1 post | Δ |
|---|---|---|---|---|---|---|
| V10b | 0.765 | 0.701 | +0.064 | 0.416 | 0.355 | +0.061 |
| V16c | 0.780 | 0.711 | +0.069 | 0.421 | 0.371 | +0.050 |

(Primärer Cutoff, R@1 Macro: V10b +0.027, V16c +0.015, gleiche Richtung.)

## 4. Abgleich mit dem click-v1-Finding

| v1-Finding (nur click, n=16/36) | v2-multi-repo-Ergebnis |
|---|---|
| Pre-AI +8,6 pp R@10 | ✅ **repliziert**: +6,4 bis +8,5 pp Macro, 4/5 Repos positiv |
| Post-AI +5,3 pp R@1 | ❌ **nicht repliziert**: R@1 ebenfalls pre-lastig (+1,5 bis +6,1 pp) |

## 5. Beobachtungen für das Research-Team

- Der Pre-AI-Vorteil ist am größten bei pandas (bis +18 pp R@10, +21 pp MRR)
  und verschwindet bei flask (~0), repo-abhängige Stärke.
- **V16c (LLM-Expansion) schließt einen Teil des Post-AI-Gaps** (pandas post
  0.530→0.598). Mögliche Deutung (zu prüfen): Expansion kompensiert
  generischere Identifier in neuerem Code, passt zur Ausgangs-Hypothese.
- Kleine Post-Zellen beim ChatGPT-Cutoff (flask 34, requests 17, fastapi 13):
  Repo-Einzelwerte dort nur Tendenz (CI ±15–25 pp); Macro und pandas/click
  sind belastbar.
- Konfundierung beachten: Pre-Issues sind älter → mehr Zeit für Fixes/Doku,
  anderes Issue-Mix. Difficulty-Matching bleibt der richtige
  nächste Schritt.

## 6. Reproduktion

```bash
python -m benchmark.runner --conditions V10b V16c \
    --dataset benchmark/data/benchmark_dataset_v2.json \
    --era pre_ai --era-cutoff 2022-11-30 \
    --output benchmark/results/era_split_v2/chatgpt_pre_ai
# analog post_ai und Cutoff 2022-01-01
python tests/test_era_cutoff.py
```


---

## 7. Addendum: Klassifikation nach `pr_merged_at` (Einwand aus dem Team)

Für die KI-Code-Hypothese ist der **Fix-Zeitpunkt** (PR-Merge) der passendere
Proxy als die Issue-Erstellung: Ein Pre-AI-Issue kann Post-AI mit
KI-Unterstützung gefixt worden sein. Jetzt wählbar via `--era-date-field
pr_merged_at`. **Real betroffen: 193/918 Samples (21%) kippen pre→post**
(Cutoff 2022-11-30).

### Ergebnis (Cutoff 2022-11-30, Macro über 4 Repos OHNE pandas*)

| Klassifikation | Cond | R@10 Δ(pre−post) | R@1 Δ(pre−post) |
|---|---|---|---|
| issue_created_at | V10b | +0.061 | −0.016 |
| issue_created_at | V16c | +0.056 | −0.019 |
| **pr_merged_at** | V10b | **+0.015** | **−0.040** |
| **pr_merged_at** | V16c | **+0.041** | **−0.044** |

*pandas ist merged-basiert nicht auswertbar: **n=1 pre / 399 post**, unsere
v2-Extraktion zog die neuesten Issues, deren Fixes fast alle nach dem
ChatGPT-Launch gemergt wurden. Die 5-Repo-Macro-Werte mit pandas
(+9,2/+10,6 pp) sind durch diese n=1-Zelle verzerrt und NICHT belastbar.

### Interpretation (deskriptiv)

1. **Die Wahl des Datumsfelds ändert das Bild materiell:** Merged-basiert
   schrumpft der Pre-AI-R@10-Vorteil (V10b +6,1→+1,5 pp; V16c +5,6→+4,1 pp)
   und **R@1 dreht klar auf Post-Vorteil** (−4,0/−4,4 pp). das Team Einwand ist
   also nicht kosmetisch, die Era-Zuordnungs-Definition gehört als
   Methodik-Entscheidung ins Team.
2. **Für merged-basierte Analysen fehlt Pre-AI-pandas:** gezielte
   Nachextraktion alter pandas-Issues nötig (Extraktor kann das via
   Datumsfenster), deckt sich mit dem Vorschlag, feste 2-Jahres-Zeiträume zu vergleichen.

### Reproduktion

```bash
python -m benchmark.runner --conditions V10b V16c \
    --dataset benchmark/data/benchmark_dataset_v2.json \
    --era pre_ai --era-cutoff 2022-11-30 --era-date-field pr_merged_at \
    --output benchmark/results/era_split_v2/chatgpt_merged_pre_ai
```
