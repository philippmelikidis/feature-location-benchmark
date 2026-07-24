# Skripte

Aktive Werkzeuge:

| Skript | Zweck |
|---|---|
| `run_query_mode_ablation.py` | Kontrollierter Vergleich der Query-Konstruktion (full / llm_expanded / full_expanded) je Embedding-Modell |
| `analyze_significance.py` | Gepaarte Signifikanztests zwischen zwei Verfahren (McNemar, Bootstrap) |
| `precompute_llm_expansions.py` | LLM-Begriffe pro Issue vorberechnen (Ollama oder LM Studio) |
| `analyze_expansion_precision.py` | Wie oft stimmen die vom LLM geratenen Begriffe? |
| `code_parser.py`, `repo_downloader.py` | Hilfsmodule des Runners (AST-Parsing, Repo-Klonen) |

Die übrigen `run_v*.py` sind die historischen Experiment-Runner (V11 bis V20). Sie bleiben als nachvollziehbare Chronik im Repo; die Ergebnisse dazu stehen in `docs/`.
