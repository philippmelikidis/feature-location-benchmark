# Era-Split Machbarkeitsanalyse: Pre-AI vs. Post-AI

## Datengrundlage

| Repository | pre_ai (vor 2022) | post_ai (ab 2022) | Gesamt |
|:-----------|-------------------:|-------------------:|-------:|
| click      | 16                 | 36                 | 52     |
| fastapi    | 8                  | 8                  | 16     |
| flask      | 1                  | 27                 | 28     |
| pandas     | 5                  | 90                 | 95     |
| requests   | 3                  | 12                 | 15     |
| **Gesamt** | **33**             | **173**            | **206**|

## Minimum n >= 10

Recall@k und MRR@k sind Mittelwerte ueber alle Samples. Bei kleinem n dominiert die Varianz einzelner Samples das Ergebnis: Bei n = 5 verschiebt ein einziges Sample den Mittelwert um 20 Prozentpunkte, bei n = 10 nur noch um 10. Zusaetzlich sind Bootstrap-Konfidenzintervalle bei n < 10 so breit, dass sich Retriever-Konfigurationen nicht mehr unterscheiden lassen. Ab n >= 10 werden Tendenzaussagen belastbar.

## Bewertung

| Repository | Era-Vergleich | Begruendung |
|:-----------|:--------------|:------------|
| click      | Ja            | n=16 pre_ai, genuegend fuer Tendenzaussagen |
| fastapi    | Explorativ    | n=8, knapp unter Schwelle, nur ergaenzend mit Caveat |
| flask      | Nein          | n=1 |
| pandas     | Nein          | n=5 |
| requests   | Nein          | n=3 |

Fuer flask, pandas und requests kann der Pre-AI-Pool spaeter durch gezielte Nachextraktion aelterer Issues vergroessert werden (`--before 2022-01-01 --limit 200`).
