#!/usr/bin/env python3
"""
run_query_mode_ablation.py – Kontrollierte Query-Modus-Ablation.

Löst den V8↔V18-Confound auf: Bisher wurden "rohe Query" und "LLM-erweiterte
Query" über verschiedene Läufe/Modelle hinweg verglichen. Dieses Skript fährt
die kontrollierte Matrix aus config.QUERY_MODE_ABLATION_MATRIX — bei
identischem Retriever variiert ausschließlich `query_mode`
("full" vs. "llm_expanded"), je Embedding-Modell (bge / Qwen3-0.6B / Qwen3-4B).

Der Report beantwortet pro Modell: Was bringt die LLM-Expansion wirklich?
(ΔRecall / ΔMRR), und für welche Modelle sollte Expansion Default sein.

Voraussetzungen:
  - Elasticsearch läuft (Default http://localhost:9200)
  - LLM-Expansions vorberechnet für die gewählten Repos
    (benchmark/data/llm_expansions_<repo>.json)
  - Qwen3-4B: ~8 GB — ggf. EMBED_BATCH_SIZE=8 setzen (siehe config.py, Label H)

Aufruf:
  # Volle 12er-Matrix auf pandas
  python scripts/run_query_mode_ablation.py

  # Nur einzelne Modelle / mehrere Repos
  python scripts/run_query_mode_ablation.py --models bge qwen3
  python scripts/run_query_mode_ablation.py --repos pandas click --models bge

  # Nur Report aus vorhandenen Ergebnissen
  python scripts/run_query_mode_ablation.py --report-only
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.runner import run_benchmark
from benchmark.config import (
    CONDITIONS_MAP,
    QUERY_MODE_ABLATION_MATRIX,
    EMBEDDING_MODELS,
)

OUTPUT_DIR = ROOT / "benchmark" / "results" / "query_mode_ablation"

MODEL_NAMES = {
    "bge": "bge-base-en-v1.5 (B, 768d)",
    "qwen3": "Qwen3-Embedding-0.6B (F, 1024d)",
    "qwen3-4b": "Qwen3-Embedding-4B (H, 2560d)",
}

# Schwelle, ab der ein Δ als "echt" gilt (unterhalb: neutral/Rauschen).
DELTA_EPS = 0.005


def _conditions_for(models):
    """Alle Condition-IDs der Matrix für die gewählten Modell-Keys."""
    cids = []
    for m in models:
        for modes in QUERY_MODE_ABLATION_MATRIX[m].values():
            cids.extend(modes.values())
    return cids


def _load_latest_runs(output_dir: Path):
    """Lade Runs aus ALLEN Ergebnis-Files des Ablation-Ordners (chronologisch).

    Teil-Läufe (z. B. erst --models bge qwen3, später qwen3-4b mit eigener
    Batch-Size) landen in separaten Files; der Report führt sie zusammen.
    Bei Wiederholungen gewinnt der spätere Run (siehe _build_results_map).
    """
    files = sorted(output_dir.glob("benchmark_results_*.json"))
    files = [f for f in files if "PARTIAL" not in f.name and "latest" not in f.name]
    runs = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            runs.extend(json.load(fh).get("runs", []))
    return runs


def _build_results_map(runs):
    """(condition_id, repo_id, k) → {recall, mrr, n}. Später Run gewinnt."""
    out = {}
    for run in runs:
        m = run.get("metrics", {})
        out[(run["condition_id"], run.get("repo_id") or "?", run["k"])] = {
            "recall": m.get("recall_at_k", 0.0),
            "mrr": m.get("mrr_at_k", 0.0),
            "n": m.get("num_samples", 0),
        }
    return out


def _macro(results_map, cid, repos, k):
    """Macro-Ø über Repos (Repos gleich gewichtet); None wenn ein Repo fehlt."""
    vals = [results_map.get((cid, r, k)) for r in repos]
    if any(v is None for v in vals):
        return None
    n = len(vals)
    return {
        "recall": sum(v["recall"] for v in vals) / n,
        "mrr": sum(v["mrr"] for v in vals) / n,
        "n": sum(v["n"] for v in vals),
    }


def generate_report(models, repos, k_values, results_map):
    """Δ-Report pro Modell × Retriever × Modus + Default-Empfehlung (Markdown)."""
    lines = []
    lines.append("# Query-Modus-Ablation")
    lines.append("")
    lines.append(f"Repos: {', '.join(repos)} | Macro-Ø (Repos gleich gewichtet)")
    lines.append("")
    lines.append("Modi: `full` (Baseline), `llm_expanded` (Expansion ersetzt Query), "
                 "`full_expanded` (Expansion an vollen Issue-Text angehängt). "
                 "Alle Zellen einer Zeile unterscheiden sich ausschließlich im "
                 "`query_mode` (Guard: `tests/test_query_mode_ablation.py`).")
    lines.append("")

    print()
    print("=" * 100)
    print("  QUERY-MODUS-ABLATION – reiner Expansionseffekt pro Embedding-Modell")
    print(f"  Repos: {', '.join(repos)}")
    print("=" * 100)

    # {(model, mode): {"dr": [...], "dm": [...]}} — nur am primären k=10
    deltas_at_10 = {}

    for k in k_values:
        lines.append(f"## k={k}")
        lines.append("")
        lines.append("| Modell | Retriever | Modus | R@k | ΔR vs. full | MRR | ΔMRR vs. full |")
        lines.append("|---|---|---|---|---|---|---|")
        print(f"\n  ── k={k} ──")
        print(f"  {'Modell':<10} {'Retriever':<8} {'Modus':<14} {'R@k':>8} "
              f"{'ΔR':>8} {'MRR':>8} {'ΔMRR':>8}")

        for model_key in models:
            for retriever_key, modes in QUERY_MODE_ABLATION_MATRIX[model_key].items():
                full = _macro(results_map, modes["full"], repos, k)
                if full is None:
                    print(f"  {model_key:<10} {retriever_key:<8} — Baseline "
                          f"{modes['full']} fehlt — Zeile übersprungen")
                    continue
                for mode, cid in modes.items():
                    cell = full if mode == "full" else _macro(results_map, cid, repos, k)
                    if cell is None:
                        lines.append(f"| {model_key} | {retriever_key} | {mode} "
                                     f"| – | – | – | – |")
                        continue
                    dr = cell["recall"] - full["recall"]
                    dm = cell["mrr"] - full["mrr"]
                    dr_s = "–" if mode == "full" else f"{dr:+.4f}"
                    dm_s = "–" if mode == "full" else f"{dm:+.4f}"
                    lines.append(
                        f"| {model_key} | {retriever_key} | {mode} "
                        f"| {cell['recall']:.4f} | {dr_s} "
                        f"| {cell['mrr']:.4f} | {dm_s} |"
                    )
                    print(f"  {model_key:<10} {retriever_key:<8} {mode:<14} "
                          f"{cell['recall']:>8.4f} {dr_s:>8} "
                          f"{cell['mrr']:>8.4f} {dm_s:>8}")
                    if k == 10 and mode != "full":
                        d = deltas_at_10.setdefault((model_key, mode),
                                                    {"dr": [], "dm": []})
                        d["dr"].append(dr)
                        d["dm"].append(dm)
        lines.append("")

    # ── Empfehlung: bester Modus pro Modell (k=10, Ø über dense+hybrid) ──
    lines.append("## Empfehlung (k=10, Ø über dense+hybrid)")
    lines.append("")
    lines.append("| Modell | Modus | Ø ΔRecall@10 | Ø ΔMRR@10 | Bewertung |")
    lines.append("|---|---|---|---|---|")
    print()
    print("─" * 100)
    print("  EMPFEHLUNG (k=10, Ø über dense+hybrid)")
    print("─" * 100)
    any_reco = False
    for model_key in models:
        name = MODEL_NAMES.get(model_key, model_key)
        best_mode, best_dr = "full", 0.0
        for mode in ("llm_expanded", "full_expanded"):
            d = deltas_at_10.get((model_key, mode))
            if not d or not d["dr"]:
                continue
            any_reco = True
            avg_dr = sum(d["dr"]) / len(d["dr"])
            avg_dm = sum(d["dm"]) / len(d["dm"])
            if avg_dr > DELTA_EPS and avg_dm > -DELTA_EPS:
                verdict = "besser als rohe Query"
            elif avg_dr < -DELTA_EPS:
                verdict = "schlechter als rohe Query"
            else:
                verdict = "neutral"
            if avg_dr > best_dr + DELTA_EPS:
                best_mode, best_dr = mode, avg_dr
            lines.append(f"| {name} | {mode} | {avg_dr:+.4f} | {avg_dm:+.4f} | {verdict} |")
            print(f"  {name:<36} {mode:<14} ΔR={avg_dr:+.4f}  "
                  f"ΔMRR={avg_dm:+.4f}  → {verdict}")
        if any_reco:
            lines.append(f"| {name} | **→ Default: `{best_mode}`** | | | |")
            print(f"  {name:<36} → Default-Empfehlung: {best_mode}")
    if not any_reco:
        lines.append("| – | – | – | – | keine vollständigen Zellen am k=10 |")
        print("  Keine vollständigen Zellen am k=10 — erst Benchmark laufen lassen.")
    print("=" * 100)
    lines.append("")
    return "\n".join(lines)


def generate_variant_report(models, repos, k_values, variant_name,
                             baseline_map, variant_map):
    """Δ-Report: Varianten-Expansion (z.B. 'lean') vs. Baseline-Expansion,
    bei GLEICHEM query_mode (llm_expanded/full_expanded) — isoliert den
    Expansions-Varianten-Effekt vom Modus-Effekt."""
    lines = []
    lines.append(f"# Expansion-Varianten-Kombi: `{variant_name}` vs. "
                 f"Baseline-Expansion")
    lines.append("")
    lines.append(f"Repos: {', '.join(repos)} | Macro-Ø (Repos gleich gewichtet) | "
                 f"beide Seiten nutzen denselben query_mode — nur die "
                 f"Expansions-Quelle (`EXPANSION_VARIANT`) unterscheidet sich.")
    lines.append("")

    print()
    print("=" * 100)
    print(f"  EXPANSION-VARIANTEN-KOMBI: {variant_name} vs. Baseline-Expansion")
    print("=" * 100)

    for k in k_values:
        lines.append(f"## k={k}")
        lines.append("")
        lines.append("| Modell | Retriever | Modus | R@k Baseline-Exp | "
                     f"R@k {variant_name}-Exp | ΔR | MRR Baseline-Exp | "
                     f"MRR {variant_name}-Exp | ΔMRR |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        print(f"\n  ── k={k} ──")
        for model_key in models:
            for retriever_key, modes in QUERY_MODE_ABLATION_MATRIX[model_key].items():
                for mode in ("llm_expanded", "full_expanded"):
                    cid = modes[mode]
                    base_cell = _macro(baseline_map, cid, repos, k)
                    var_cell = _macro(variant_map, cid, repos, k)
                    if base_cell is None or var_cell is None:
                        lines.append(f"| {model_key} | {retriever_key} | {mode} "
                                     f"| – | – | – | – | – | – |")
                        continue
                    dr = var_cell["recall"] - base_cell["recall"]
                    dm = var_cell["mrr"] - base_cell["mrr"]
                    lines.append(
                        f"| {model_key} | {retriever_key} | {mode} "
                        f"| {base_cell['recall']:.4f} | {var_cell['recall']:.4f} "
                        f"| {dr:+.4f} | {base_cell['mrr']:.4f} "
                        f"| {var_cell['mrr']:.4f} | {dm:+.4f} |"
                    )
                    print(f"  {model_key:<10} {retriever_key:<8} {mode:<14} "
                          f"R@base={base_cell['recall']:.4f} "
                          f"R@{variant_name}={var_cell['recall']:.4f} "
                          f"ΔR={dr:+.4f} ΔMRR={dm:+.4f}")
        lines.append("")
    print("=" * 100)
    return "\n".join(lines)


def _output_dir_for_variant(base: Path, variant: str) -> Path:
    """Non-baseline Varianten bekommen einen eigenen Unterordner, damit ihre
    Ergebnisse die Baseline-Ergebnisse desselben condition_id nicht
    überschreiben (_build_results_map() hat keine Varianten-Dimension im
    Key)."""
    variant = (variant or "baseline").strip()
    if variant in ("", "baseline"):
        return base
    return base / variant


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kontrollierte Query-Modus-Ablation"
    )
    parser.add_argument("--models", nargs="*",
                        choices=sorted(QUERY_MODE_ABLATION_MATRIX),
                        default=sorted(QUERY_MODE_ABLATION_MATRIX),
                        help="Modell-Keys der Matrix (Default: alle)")
    parser.add_argument("--repos", nargs="*", default=["pandas"],
                        help="Repos (Default: pandas)")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--es-url", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="Nur Report aus vorhandenen Ergebnissen erzeugen")
    parser.add_argument("--expansion-variant", type=str, default="baseline",
                        help="Prompt-Variante der LLM-Expansion, "
                             "z.B. 'lean'. Setzt EXPANSION_VARIANT für den Lauf "
                             "und schreibt Ergebnisse in einen eigenen "
                             "Unterordner, damit sie die Baseline-Ergebnisse "
                             "desselben condition_id nicht überschreiben. "
                             "Default 'baseline' = bisheriges Verhalten.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    variant_dir = _output_dir_for_variant(OUTPUT_DIR, args.expansion_variant)
    variant_dir.mkdir(parents=True, exist_ok=True)
    conditions = _conditions_for(args.models)

    missing = [c for c in conditions if c not in CONDITIONS_MAP]
    if missing:
        print(f"FEHLER: Conditions fehlen in config.py: {missing}")
        sys.exit(1)

    is_variant_run = args.expansion_variant and args.expansion_variant != "baseline"

    if not args.report_only:
        print(f"""
╔════════════════════════════════════════════════════════════════════╗
║  Query-Modus-Ablation                                 ║
║  Matrix: {len(args.models)} Modell(e) × dense/hybrid × full/llm_expanded          ║
║  Conditions: {len(conditions):<3} | Repos: {', '.join(args.repos):<32}║
║  Expansion-Variante: {args.expansion_variant:<46}║
╚════════════════════════════════════════════════════════════════════╝
""")
        for m in args.models:
            label = {"bge": "B", "qwen3": "F", "qwen3-4b": "H"}[m]
            print(f"  {m:<10} → {EMBEDDING_MODELS[label].name}")
        print()

        old_variant_env = os.environ.get("EXPANSION_VARIANT")
        try:
            if is_variant_run:
                os.environ["EXPANSION_VARIANT"] = args.expansion_variant
            else:
                os.environ.pop("EXPANSION_VARIANT", None)

            start = time.time()
            run_benchmark(
                conditions=conditions,
                k_values=args.k,
                repos=args.repos,
                dataset_path=str(ROOT / "benchmark" / "data" / "benchmark_dataset.json"),
                output_dir=str(variant_dir),
                es_url=args.es_url,
                verbose=args.verbose,
            )
            print(f"\n  Laufzeit: {(time.time() - start) / 60:.1f} Minuten")
        finally:
            if old_variant_env is None:
                os.environ.pop("EXPANSION_VARIANT", None)
            else:
                os.environ["EXPANSION_VARIANT"] = old_variant_env

    runs = _load_latest_runs(variant_dir)
    if not runs:
        print("\n  Keine Ergebnisse gefunden — erst ohne --report-only laufen lassen.")
        sys.exit(0 if args.report_only else 1)

    results_map = _build_results_map(runs)
    report_md = generate_report(args.models, args.repos, args.k, results_map)

    report_file = variant_dir / "query_mode_ablation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n  Report gespeichert: {report_file}")

    if is_variant_run:
        baseline_runs = _load_latest_runs(OUTPUT_DIR)
        if not baseline_runs:
            print(f"\n  Hinweis: keine Baseline-Ergebnisse in {OUTPUT_DIR} gefunden "
                  f"— Varianten-vs-Baseline-Report übersprungen. Erst die Baseline "
                  f"(ohne --expansion-variant) für dieselben Modelle/Repos laufen "
                  f"lassen.")
            return
        baseline_map = _build_results_map(baseline_runs)
        variant_report_md = generate_variant_report(
            args.models, args.repos, args.k, args.expansion_variant,
            baseline_map, results_map,
        )
        variant_report_file = (
            variant_dir / f"variant_vs_baseline_{args.expansion_variant}.md"
        )
        with open(variant_report_file, "w", encoding="utf-8") as f:
            f.write(variant_report_md)
        print(f"\n  Varianten-Vergleich gespeichert: {variant_report_file}")


if __name__ == "__main__":
    main()
