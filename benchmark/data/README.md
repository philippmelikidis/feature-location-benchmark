# Daten

Die kanonische Aufgabenliste ist `benchmark_dataset_v2.json` (918 Issues, bereinigt und validiert).

| Datei | Inhalt |
|---|---|
| `benchmark_dataset_v2.json` | Aktuelles Dataset (Standard des Runners) |
| `benchmark_dataset_v2_pre_ai.json` / `_post_ai.json` | Zeit-Splits davon |
| `benchmark_dataset.json` (+ pre/post) | Erste Version (206 Einträge, enthielt Duplikate; nur noch für die historischen Experiment-Runner) |
| `llm_expansions_<repo>.json` | Vorberechnete LLM-Begriffe pro Issue (Standard-Prompt) |
| `llm_expansions_<repo>__<variante>.json` | Dasselbe für getestete Prompt-Varianten |
| `raw_*.json` | Rohdaten aus der GitHub-Extraktion (Basis für den Dataset-Bau) |
