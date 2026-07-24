#!/usr/bin/env python3
"""
run_v16_llm_expansion.py – V16 LLM Query-Expansion Benchmark (pandas).

Vergleicht V16a/b/c (LLM-gestützte Query-Expansion + Dense Stage 2) gegen
V12b (bester bisheriger: Terms Filter, Recall 0.41) und V14b (Virtual-Doc
Hybrid, Recall 0.41).

Voraussetzungen:
  - Elasticsearch läuft lokal (http://localhost:9200)
  - LM Studio läuft lokal (http://localhost:1234) mit geladenem Modell
  - pip install -r requirements_benchmark.txt

Aufruf:
  python scripts/run_v16_llm_expansion.py [--verbose] [--es-url http://...]
  python scripts/run_v16_llm_expansion.py --report-only
  python scripts/run_v16_llm_expansion.py --test-llm   # nur LLM testen
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.runner import run_benchmark
from benchmark.config import CONDITIONS_MAP, K_VALUES


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Baselines zum Vergleich
BASELINE_CONDITIONS = ["V12b", "V14b"]
# V16 Varianten
V16_CONDITIONS = ["V16a", "V16b", "V16c"]
ALL_CONDITIONS = BASELINE_CONDITIONS + V16_CONDITIONS
REPO = "pandas"
OUTPUT_DIR = str(ROOT / "benchmark" / "results" / "v16_llm_expansion")


# ──────────────────────────────────────────────────────────────
# LLM Connectivity Test
# ──────────────────────────────────────────────────────────────

def check_expansions_file():
    """Check that pre-computed LLM expansions exist."""
    expansions_file = ROOT / "benchmark" / "data" / "llm_expansions_pandas.json"

    print("\nPrüfe pre-computed LLM Expansions...")
    print(f"   Datei: {expansions_file}")

    if not expansions_file.exists():
        print(f"   Expansions-Datei nicht gefunden!")
        print(f"   → Erst ausführen: python scripts/precompute_llm_expansions.py")
        return False

    with open(expansions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    expansions = data.get("expansions", {})
    total = len(expansions)
    with_terms = sum(1 for e in expansions.values()
                     if e.get("flat_terms", "").strip())
    empty = total - with_terms

    print(f"   {total} Expansions gefunden ({with_terms} mit Terms, {empty} leer/fallback)")

    if with_terms == 0:
        print(f"   Alle Expansions sind leer! LLM hat keine Terms generiert.")
        print(f"   → Nochmal ausführen: python scripts/precompute_llm_expansions.py --retry-empty")
        return False

    if empty > total * 0.5:
        print(f"    Warnung: {empty}/{total} Expansions sind leer.")
        print(f"   → Optional: python scripts/precompute_llm_expansions.py --retry-empty")

    return True


# ──────────────────────────────────────────────────────────────
# Comparison Report
# ──────────────────────────────────────────────────────────────

def generate_comparison_report(results_path: Path):
    """Parse the benchmark results JSON and produce a comparison table."""
    json_files = sorted(results_path.glob("benchmark_results_*.json"))
    # Skip PARTIAL files
    json_files = [f for f in json_files if "PARTIAL" not in f.name and "latest" not in f.name]
    if not json_files:
        print("\nKeine Ergebnisdateien gefunden.")
        return

    latest = json_files[-1]
    print(f"\nLade Ergebnisse: {latest.name}")

    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not runs:
        print("Keine Runs in den Ergebnissen.")
        return

    # Group by condition_id and k
    results_map = {}
    for run in runs:
        cid = run["condition_id"]
        k = run["k"]
        metrics = run.get("metrics", {})
        results_map[(cid, k)] = {
            "recall": metrics.get("recall_at_k", 0.0),
            "mrr": metrics.get("mrr_at_k", 0.0),
            "num_samples": metrics.get("num_samples", 0),
        }

    # ── Print Report ──
    report_lines = []
    report_lines.append("")
    report_lines.append("=" * 85)
    report_lines.append("  V16 LLM QUERY-EXPANSION – VERGLEICHSREPORT – pandas")
    report_lines.append("=" * 85)
    report_lines.append("")

    # Condition descriptions
    report_lines.append("Varianten:")
    for cid in ALL_CONDITIONS:
        cfg = CONDITIONS_MAP.get(cid)
        desc = cfg.description if cfg else "?"
        report_lines.append(f"  {cid}: {desc}")
    report_lines.append("")

    # Table
    k_vals = sorted(set(k for _, k in results_map.keys()))
    if not k_vals:
        k_vals = [10]

    for k in k_vals:
        report_lines.append(f"┌{'─'*83}┐")
        report_lines.append(f"│  Recall@{k} & MRR@{k}{' ' * (67 - len(str(k)))}│")
        report_lines.append(f"├{'─'*20}┬{'─'*14}┬{'─'*14}┬{'─'*32}┤")
        report_lines.append(f"│ {'Condition':<18} │ {'Recall@'+str(k):^12} │ {'MRR@'+str(k):^12} │ {'Δ vs V12b (Recall)':^30} │")
        report_lines.append(f"├{'─'*20}┼{'─'*14}┼{'─'*14}┼{'─'*32}┤")

        baseline = results_map.get(("V12b", k), {}).get("recall", 0.0)

        for cid in ALL_CONDITIONS:
            r = results_map.get((cid, k), {})
            recall = r.get("recall", 0.0)
            mrr = r.get("mrr", 0.0)

            if baseline > 0:
                delta = recall - baseline
                delta_pct = (delta / baseline) * 100
                delta_str = f"{delta:+.4f} ({delta_pct:+.1f}%)"
            else:
                delta_str = "—"

            report_lines.append(
                f"│ {cid:<18} │ {recall:^12.4f} │ {mrr:^12.4f} │ {delta_str:^30} │"
            )

            # Separator between baselines and V16
            if cid == BASELINE_CONDITIONS[-1]:
                report_lines.append(f"├{'─'*20}┼{'─'*14}┼{'─'*14}┼{'─'*32}┤")

        report_lines.append(f"└{'─'*20}┴{'─'*14}┴{'─'*14}┴{'─'*32}┘")
        report_lines.append("")

    # ── Stage-1 Diagnostics ──
    report_lines.append("─" * 85)
    report_lines.append("  Stage-1 Diagnostics (LLM-Expansion vs. Baseline)")
    report_lines.append("─" * 85)
    report_lines.append("")

    k_max = max(k_vals) if k_vals else 10
    for cid in ALL_CONDITIONS:
        run_data = next(
            (r for r in runs if r["condition_id"] == cid and r["k"] == k_max),
            None
        )
        if not run_data:
            continue

        per_sample = run_data.get("metrics", {}).get("per_sample", [])
        if not per_sample:
            continue

        stage1_hits = sum(1 for s in per_sample if s.get("stage1_hit"))
        total = len(per_sample)
        hit_rate = stage1_hits / total if total > 0 else 0

        report_lines.append(
            f"  {cid}: Stage-1 Hit Rate = {stage1_hits}/{total} "
            f"({hit_rate:.1%})"
        )

    report_lines.append("")

    # ── Stage-2 Diagnostics ──
    report_lines.append("─" * 85)
    report_lines.append("  Stage-2 Diagnostics (Dense vs. Hybrid/BM25)")
    report_lines.append("─" * 85)
    report_lines.append("")
    report_lines.append(f"  {'Condition':<10} │ {'Strategie':<35} │ {'Ø aus Kandidaten':>16} │ {'Ø Fallback':>10}")
    report_lines.append(f"  {'─'*10}─┼─{'─'*35}─┼─{'─'*16}─┼─{'─'*10}")

    strategy_labels = {
        "V12b": "BM25 Coarse → Hybrid Terms-Filter",
        "V14b": "VDoc Hybrid Coarse → Hybrid TF",
        "V16a": "LLM+VDoc BM25 → Dense TF",
        "V16b": "LLM+Class/File BM25 → Dense TF",
        "V16c": "LLM+VDoc BM25 → Hybrid TF",
    }

    for cid in ALL_CONDITIONS:
        run_data = next(
            (r for r in runs if r["condition_id"] == cid and r["k"] == k_max),
            None
        )
        if not run_data:
            continue

        per_sample = run_data.get("metrics", {}).get("per_sample", [])
        if not per_sample:
            continue

        s2_candidates = [
            s.get("stage2_from_candidates", k_max)
            for s in per_sample
            if s.get("stage2_from_candidates") is not None
        ]
        s2_fallback = [
            s.get("stage2_from_fallback", 0)
            for s in per_sample
            if s.get("stage2_from_fallback") is not None
        ]

        avg_cand = sum(s2_candidates) / len(s2_candidates) if s2_candidates else k_max
        avg_fb = sum(s2_fallback) / len(s2_fallback) if s2_fallback else 0

        strategy = strategy_labels.get(cid, "?")
        report_lines.append(
            f"  {cid:<10} │ {strategy:<35} │ {avg_cand:>13.1f}/{k_max} │ {avg_fb:>7.1f}/{k_max}"
        )

    report_lines.append("")

    # ── Summary ──
    report_lines.append("─" * 85)
    report_lines.append("  ZUSAMMENFASSUNG")
    report_lines.append("─" * 85)

    best_cid = None
    best_recall = -1
    for cid in ALL_CONDITIONS:
        r = results_map.get((cid, k_max), {})
        if r.get("recall", 0) > best_recall:
            best_recall = r["recall"]
            best_cid = cid

    if best_cid:
        best_mrr = results_map.get((best_cid, k_max), {}).get("mrr", 0)
        report_lines.append(
            f"\n  Bester Retriever bei k={k_max}: {best_cid} "
            f"(Recall={best_recall:.4f}, MRR={best_mrr:.4f})"
        )

        v12b_recall = results_map.get(("V12b", k_max), {}).get("recall", 0)
        if v12b_recall > 0:
            improvement = ((best_recall - v12b_recall) / v12b_recall) * 100
            report_lines.append(
                f"  Verbesserung gegenüber V12b-Baseline: {improvement:+.1f}%"
            )

        # PDF target comparison
        report_lines.append(f"\n  PDF-Prognose (Kapitel 6.4):")
        report_lines.append(f"    Recall@10 Ziel: 0.55–0.65 | Erreicht: {best_recall:.4f}")
        if best_recall >= 0.55:
            report_lines.append(f"    Ziel erreicht!")
        elif best_recall >= 0.45:
            report_lines.append(f"     Unter Ziel, aber besser als Baseline (0.42)")
        else:
            report_lines.append(f"    Unter Baseline")

    report_lines.append("")
    report_lines.append("=" * 85)

    full_report = "\n".join(report_lines)
    print(full_report)

    report_file = results_path / "comparison_v16_llm_expansion.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\nReport gespeichert: {report_file}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V16 LLM Query-Expansion Benchmark (pandas)"
    )
    parser.add_argument("--es-url", type=str, default=None,
                        help="Elasticsearch URL (default: http://localhost:9200)")
    parser.add_argument("--verbose", action="store_true",
                        help="Per-sample Ergebnisse ausgeben")
    parser.add_argument("--report-only", action="store_true",
                        help="Nur Report generieren (kein Benchmark laufen lassen)")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10],
                        help="k-Werte (default: 1 5 10)")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Nur V16 laufen (Baselines überspringen)")

    args = parser.parse_args()

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        generate_comparison_report(output_path)
        return

    # Pre-flight: check expansions file
    print("\nV16 LLM Query-Expansion Benchmark")
    if not check_expansions_file():
        sys.exit(1)

    # Select conditions
    conditions = V16_CONDITIONS if args.no_baselines else ALL_CONDITIONS

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  V16 – LLM Query-Expansion + Dense Stage 2                     ║
║  Repo: pandas | Conditions: {len(conditions)} | k={args.k}                    ║
║  Expansions: pre-computed (benchmark/data/llm_expansions_pandas.json)  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    start = time.time()

    run_benchmark(
        conditions=conditions,
        k_values=args.k,
        repos=[REPO],
        dataset_path=str(ROOT / "benchmark" / "data" / "benchmark_dataset.json"),
        output_dir=OUTPUT_DIR,
        es_url=args.es_url,
        verbose=args.verbose,
    )

    elapsed = time.time() - start
    print(f"\n⏱  Gesamtdauer: {elapsed/60:.1f} Minuten")

    # Generate comparison report
    generate_comparison_report(output_path)


if __name__ == "__main__":
    main()
