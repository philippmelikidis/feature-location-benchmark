# LLM-Expansion: Precision-Analyse (V16 §5.3)

> > **Insert → Markup → Markdown** (oder das „Markdown"-Makro).

## TL;DR

Die V16-Pipeline lässt ein LLM aus einem Issue **hypothetische Code-Identifier**
generieren (Funktionen, Klassen, Dateien, Imports, Keywords), die als Zusatz-Terme
die Stage-1-Suche verbessern. Diese Analyse prüft, **wie viele dieser generierten
Terme tatsächlich in der Ziel-Datei der Ground Truth vorkommen**, also wie
treffsicher das LLM rät. Hohe Precision = echtes Signal; niedrige = eher Noise.

**Ergebnis über alle 5 Repos (183 auswertbare Queries, Dataset v1): Ø Precision 0.734.**
Nur 1/183 Queries ist reiner Noise. Schwächste Kategorie: **functions (0.50)**;
treffsicherste: **imports (0.99)**.

**Update auf Dataset v2 (Abschnitt 8, 691 Queries): Ø Precision 0.839**, Muster
bestätigt sich (functions weiterhin schwächste, imports treffsicherste Kategorie),
0/691 Queries Noise. Realer Stage-1-Effekt auf v2/pandas geprüft: **0 Queries** durch
Expansion verschlechtert.

---

## 1. Worum geht es?

Kernproblem des Retrievals: Ein Issue ist natürliche Sprache („export to CSV drops
NaN"), der Code besteht aus Identifiern (`to_csv`, `_save`). BM25 matcht nur, was
wörtlich überlappt → **Vocabulary Mismatch**. V16 schließt die Lücke, indem ein LLM
vorab plausible Code-Identifier generiert.

Diese Analyse beantwortet §5.3 des FLBench-Berichts: *Welche generierten Terme
kommen wirklich in den Zieldateien vor, und bei welchen Queries schadet die
Expansion eher?*

**Precision pro Query** = (Anzahl generierter Terme, die in der/den Ziel-Datei(en)
vorkommen) / (Anzahl generierter Terme). Aggregiert als Mittelwert über alle Queries.

---

## 2. Wie ausführen

```bash
# 0. Expansionen je Repo vorberechnen (lokales LLM via Ollama)
ollama serve && ollama pull qwen2.5-coder:7b
for r in pandas flask requests click fastapi; do
  python scripts/precompute_llm_expansions.py --repo $r \
    --llm-url http://localhost:11434/v1 --model qwen2.5-coder:7b
done

# 1. Precision-Analyse über ALLE Repos (kein ES, kein LLM nötig)
python scripts/analyze_expansion_precision.py
# → benchmark/results/expansion_precision_report.md (+ .json)
```

---

## 3. Ergebnis (alle Repos, 183 auswertbare Queries)

> 183 von 201 Expansionen ausgewertet, 16 Queries ohne lesbare Ziel-Datei
> (Datei im ausgecheckten HEAD nicht (mehr) vorhanden / Pfad nicht auflösbar).

**Mittlere Precision: 0.734**

### Pro Repo

| Repo    | Queries | Ø Precision |
|---------|---------|-------------|
| click   | 42 | 0.845 |
| flask   | 25 | 0.843 |
| requests| 9  | 0.769 |
| pandas  | 94 | 0.667 |
| fastapi | 13 | 0.621 |

### Pro Kategorie (files jetzt fair gemessen, siehe §5)

| Kategorie  | Ø Precision |
|------------|-------------|
| imports    | 0.994 |
| classes    | 0.859 |
| files      | 0.843 |
| keywords   | 0.648 |
| functions  | 0.502 |

### Pro Test-Case-Typ

| TC  | Queries | Ø Precision |
|-----|---------|-------------|
| TC1 (lexikalisch) | 154 | 0.735 |
| TC2 (semantisch)  | 25  | 0.745 |
| TC3 (strukturell) | 4   | 0.629 |

### Schlechteste Queries (Expansion vermutlich wenig hilfreich)

| Repo | sample_id | #Terme | Precision | Titel |
|------|-----------|--------|-----------|-------|
| fastapi | 92789a99 | 11 | 0.000 | Use TestClient in an async fashion |
| requests | 0824816f | 14 | 0.143 | Certificate loading regression with HTTPAdapters in 2.32.3 |
| pandas | 7c29d7f7 | 14 | 0.214 | BUG: Pandas 2.2 breaks SQLAlchemy 1.4 compatibility |
| pandas | cb822a67 | 13 | 0.231 | DOC: Fix docstring validation errors (groupby) |
| fastapi | 76fee4d1 | 15 | 0.267 | Traceback stack does not show exact place of error |
| pandas | 02ea9eed | 15 | 0.267 | TRACKER: add support for Python 3.14 |
| flask | efc3fbcd | 7 | 0.286 | deprecate `__version__` |
| pandas | (mehrere) | ~13–16 | 0.36–0.40 | DOC:/TRACKER:-Issues (Docstring-Validation, Py 3.14) |

---

## 4. Interpretation

**0.73 über alle Repos ist solide**, fast drei Viertel der geratenen Terme stehen
wirklich in der Ziel-Datei. Nur **1/183** Queries ist komplett am Code vorbei.

**Repo-Größe schlägt durch:** kleine Repos (click 0.85, flask 0.84) liegen klar über
großen (pandas 0.67). Je mehr Dateien, desto mehr Raum für danebenliegende
Identifier. fastapi (0.62) und requests (0.77) haben kleine Stichproben (n=13/9) →
nur Tendenz.

**`functions` ist die echte Schwachstelle (0.50).** Das LLM erfindet plausibel
klingende Funktionsnamen, die so nicht in der Zieldatei stehen. Das ist der größte
Hebel fürs Prompt-Tuning. `imports`/`classes`/`files` sind dagegen treffsicher.

**Klares Muster bei den schlechten Queries:** pandas-**`DOC:`** und **`TRACKER:`**-
Issues (Docstring-Validation, „Python 3.14 support") dominieren die Worst-Liste.
Das sind **Maintenance-/Meta-Issues ohne einzelne Ziel-Datei**, verteilt über viele
Dateien oder rein organisatorisch. Hier kann keine Expansion treffen; solche Issues
verzerren eher die Auswertung. Kandidat zum Filtern/Sondertyp.

**TC2 (semantisch) ist unkritisch:** mit n=25 jetzt belastbar und auf TC1-Niveau
(0.745 vs 0.735), die frühere TC2-Sorge bestätigt sich auf Term-Ebene nicht.

---

## 5. Hinweis zur Mess-Methode (`files`-Fix)

Ursprünglich wurde `files` nur gegen den **Ziel-Pfad** gematcht (strikt), die
anderen Kategorien gegen den **Datei-Inhalt** (lenient). Dadurch war `files`
systematisch benachteiligt (flask-Einzellauf: 0.513). Seit dem Fix gilt ein
`files`-Term auch als Treffer, wenn sein Dateiname-Token im Inhalt der Ziel-Datei
vorkommt, konsistent zu den übrigen Kategorien. Effekt: `files` 0.513 → **0.843**.

Damit ist die Bericht-Hypothese (§5.2: „Dateipfade treffsicherer als Imports")
**relativiert**: `imports` (0.99) bleibt vorn, `files` (0.84) ist solide, und der
wahre Ausreißer nach unten ist `functions`.

---

## 6. Empfehlungen fürs Prompt-Design (datengetrieben)

- **`functions` priorisiert verbessern** (0.50): das LLM präziser auf real
  existierende Funktionsnamen lenken (z. B. Few-Shot mit echten Issue→Lösungsdatei-
  Paaren), oder generische/erfundene Funktionsnamen niedriger gewichten.
- **Maintenance-Issues abfangen:** `DOC:`/`TRACKER:`-Issues (v. a. pandas) liefern
  kaum verwertbare Expansion, als eigenen Typ behandeln oder aus der Expansion
  ausnehmen, statt das LLM raten zu lassen.
- `imports`/`classes` sind treffsicher → ruhig stärker anfordern.
- Da nur 1 % reiner Noise ist, geht es um **Feinschliff**, nicht um Grundsanierung.

---

## 7. Einschränkungen

- **Kleine Stichproben** bei fastapi (13) und requests (9) → nur Tendenz.
- **16 Queries ohne lesbare Ziel-Datei** ausgeschlossen, vermutlich HEAD-Checkout
  ≠ Issue-Commit (umbenannte/gelöschte Dateien). Wert für eine saubere Auswertung
  ggf. pro Issue-Commit auschecken.
- Precision misst **Term-Vorkommen**, nicht den Retrieval-Erfolg. Ob die Expansion
  den Stage-1-Treffer real verbessert/verschlechtert, zeigt erst der Abgleich mit
  Benchmark-Ergebnissen (siehe Next Steps).

---

## 8. Update auf Dataset v2

Die ursprüngliche Analyse (Abschnitt 3) lief auf v1-großen Stichproben (183 Queries,
~pandas n=94). Mit v2 (validierte, deutlich größere Ground Truth, siehe
`docs/dataset_v2_ergebnisse.md`) und den seither für alle 5 Repos vorberechneten
Expansionen (`llm_expansions_{pandas,click,flask,requests,fastapi}.json`, zusammen
1102 Einträge) liefert derselbe Analyse-Lauf ein deutlich breiteres Bild:

**691 auswertbare Queries (von 806 Expansionen; 115 ohne lesbare Zieldatei).
Mittlere Precision: 0.839** (v1: 0.734, +10,5 pp).

### Pro Repo (v2)

| Repo | Queries | Ø Precision | (v1) |
|---|---|---|---|
| click | 194 | 0.877 | 0.845 |
| fastapi | 30 | 0.725 | 0.621 |
| flask | 79 | 0.790 | 0.843 |
| pandas | 380 | 0.841 | 0.667 |
| requests | 8 | 0.795 | 0.769 |

pandas springt am stärksten (0.667→0.841), die größere, bereinigte v2-Stichprobe
glättet offenbar die pandas-Ausreißer (DOC:/TRACKER:-Issues), die die kleine v1-Probe
überproportional belastet hatten. requests bleibt mit n=8 eine Randnotiz (kein
gültiges Target für die meisten der 68 vorhandenen requests-Expansionen in der
aktuellen v2-Ground-Truth, nicht weiter untersucht, siehe Einschränkungen).

### Pro Kategorie (v2)

| Kategorie | Ø Precision | (v1) |
|---|---|---|
| functions | 0.602 | 0.502 |
| classes | 0.864 | 0.859 |
| files | 0.965 | 0.843 |
| imports | 0.998 | 0.994 |
| keywords | 0.842 | 0.648 |

Reihenfolge bleibt identisch (`functions` schwächste, `imports` treffsicherste
Kategorie), die Kern-Empfehlung aus Abschnitt 6 (Prompt-Tuning für `functions`)
hält auf v2 unverändert. **0/691 Queries mit Precision 0** (v1: 1/183), Noise bleibt
verschwindend selten.

### Echte Stage-1-Verschlechterung (AC3, `--results`-Abgleich)

Cross-Check gegen echte Benchmark-Ergebnisse auf v2: `V8_QWEN34B` (Dense, ohne
Expansion) vs. `V18c_QWEN34B` (Dense, `full_expanded`), beide frisch auf v2/pandas
gelaufen (siehe `docs/v20_hybrid_coarse_v2_ergebnisse.md`, Abschnitt 7).

**0 Queries** verlieren durch die Expansion ihren Stage-1-Treffer (Baseline fand die
Datei, mit Expansion nicht mehr). Konsistent mit dem aggregierten Befund dort:
`full_expanded` verbessert Recall@10 und MRR@10 gegenüber der Baseline, es gibt auf
Query-Ebene keinen erkennbaren Schaden. Dieser Cross-Check ist auf pandas beschränkt
(einzige Repo, für das `V8_QWEN34B`/`V18c_QWEN34B` bisher liefen), nicht auf
click/flask/requests/fastapi übertragbar, ohne dort dieselben zwei Conditions zu
fahren.

### Einschränkungen (v2-Update)

- requests hat auf v2 nur 8 auswertbare Queries (68 Expansionen vorhanden, aber nur 8
  mit lesbarer Zieldatei in der aktuellen `benchmark_dataset_v2.json`), Ursache nicht
  untersucht (mögliches Symptom: viele requests-Targets referenzieren Pfade, die im
  frischen HEAD-Checkout nicht mehr existieren, ähnlich dem allgemeinen
  "16/115 ohne lesbare Zieldatei"-Muster).
- fastapi (30) und requests bleiben kleine Stichproben, nur Tendenz, wie schon in v1.
- Der Stage-1-Cross-Check deckt nur `V8_QWEN34B`/`V18c_QWEN34B` (Dense, pandas) ab,
  nicht die Hybrid-Conditions oder andere Repos.

```bash
# Reproduktion (v2, alle 5 Repos, inkl. echtem Stage-1-Abgleich):
python scripts/analyze_expansion_precision.py \
    --dataset benchmark/data/benchmark_dataset_v2.json \
    --results <gemergte V8_QWEN34B+V18c_QWEN34B-Ergebnisse> \
    --expanded-condition V18c_QWEN34B --baseline-condition V8_QWEN34B \
    --output benchmark/results/expansion_precision_report_v2.md \
    --json-output benchmark/results/expansion_precision_v2.json
```

Daten: `benchmark/results/expansion_precision_report_v2.md`,
`benchmark/results/expansion_precision_v2.json` (beide gitignored, lokal reproduzierbar).

---

## 9. Next Steps

1. **Echten Effekt messen, ✅ erledigt (v1 und v2):** V12b vs V16c (v1) sowie
   V8_QWEN34B vs V18c_QWEN34B (v2, siehe Abschnitt 8) ergaben den `--results`-Abgleich.
   v1: 8 Queries mit realer Stage-1-Verschlechterung (7× pandas-Maintenance,
   1× click-Release-Plan; Details auf der Schwester-Seite „V12b vs. V16c —
   Wirksamkeit der LLM-Query-Expansion"). v2/pandas: 0 Queries verschlechtert.
2. **`functions`-Prompt-Tuning** als konkretes Folge-Experiment (Few-Shot), hält
   als Priorität auch nach dem v2-Update (Abschnitt 8).
3. **DOC:/TRACKER:-Issues** als eigenen Typ klassifizieren oder filtern.
4. Optional: Auswertung pro Issue-Commit auschecken, um die fehlenden Zieldateien
   zu retten (v1: 16, v2: 115, auf v2 relevanter geworden).
5. requests-Diskrepanz (68 Expansionen vs. nur 8 auswertbare v2-Queries) klären.
