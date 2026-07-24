#!/usr/bin/env python3
"""
run_v11_vs_v12.py – Lokaler V11 vs V12 Vergleichs-Benchmark (nur pandas).

Führt V11a/b/c und V12a/b/c auf dem pandas-Repo aus und gibt einen
übersichtlichen Vergleichsreport mit Metriken-Tabelle aus.

Voraussetzungen:
  - Elasticsearch läuft lokal (http://localhost:9200)
  - pip install -r requirements_benchmark.txt

Aufruf:
  python scripts/run_v11_vs_v12.py [--verbose] [--es-url http://...]
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

V11_CONDITIONS = ["V11a", "V11b", "V11c"]
V12_CONDITIONS = ["V12a", "V12b", "V12c"]
ALL_CONDITIONS = V11_CONDITIONS + V12_CONDITIONS
REPO = "pandas"
OUTPUT_DIR = str(ROOT / "benchmark" / "results" / "v11_vs_v12")


# ──────────────────────────────────────────────────────────────
# Comparison Report
# ──────────────────────────────────────────────────────────────

def generate_comparison_report(results_path: Path):
    """Parse the benchmark results JSON and produce a comparison table."""
    # Find the latest results file
    json_files = sorted(results_path.glob("benchmark_results_*.json"))
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
    report_lines.append("=" * 80)
    report_lines.append("  V11 vs V12 VERGLEICHSREPORT – pandas")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Condition descriptions
    report_lines.append("Varianten:")
    for cid in ALL_CONDITIONS:
        cfg = CONDITIONS_MAP.get(cid)
        desc = cfg.description if cfg else "?"
        report_lines.append(f"  {cid}: {desc}")
    report_lines.append("")

    # Table header
    k_vals = sorted(set(k for _, k in results_map.keys()))
    if not k_vals:
        k_vals = K_VALUES

    for k in k_vals:
        report_lines.append(f"┌{'─'*78}┐")
        report_lines.append(f"│  Recall@{k} & MRR@{k}{' ' * (62 - len(str(k)))}│")
        report_lines.append(f"├{'─'*20}┬{'─'*14}┬{'─'*14}┬{'─'*27}┤")
        report_lines.append(f"│ {'Condition':<18} │ {'Recall@'+str(k):^12} │ {'MRR@'+str(k):^12} │ {'Δ vs V11b (Recall)':^25} │")
        report_lines.append(f"├{'─'*20}┼{'─'*14}┼{'─'*14}┼{'─'*27}┤")

        # Baseline: V11b
        baseline = results_map.get(("V11b", k), {}).get("recall", 0.0)

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

            separator = "│" if cid != "V11c" else "│"
            report_lines.append(
                f"│ {cid:<18} │ {recall:^12.4f} │ {mrr:^12.4f} │ {delta_str:^25} │"
            )

            # Visual separator between V11 and V12 block
            if cid == "V11c":
                report_lines.append(f"├{'─'*20}┼{'─'*14}┼{'─'*14}┼{'─'*27}┤")

        report_lines.append(f"└{'─'*20}┴{'─'*14}┴{'─'*14}┴{'─'*27}┘")
        report_lines.append("")

    # ── Stage-1 Diagnostics ──
    report_lines.append("─" * 80)
    report_lines.append("  Stage-1 Diagnostics (Coarse → identisch für alle)")
    report_lines.append("─" * 80)
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

    # ── Stage-2 Diagnostics (the actual comparison) ──
    report_lines.append("─" * 80)
    report_lines.append("  Stage-2 Diagnostics (WO sich V11 vs V12 unterscheidet)")
    report_lines.append("─" * 80)
    report_lines.append("")
    report_lines.append("  Frage: Wie gut nutzt Stage 2 die Kandidaten aus Stage 1?")
    report_lines.append("")
    report_lines.append(f"  {'Condition':<10} │ {'Strategie':<22} │ {'Ø aus Kandidaten':>16} │ {'Ø Fallback':>10} │ {'Conversion':>10}")
    report_lines.append(f"  {'─'*10}─┼─{'─'*22}─┼─{'─'*16}─┼─{'─'*10}─┼─{'─'*10}")

    strategy_labels = {
        "V11a": "filter (k*5, N=10)",
        "V11b": "filter (k*5, N=20)",
        "V11c": "filter (k*5, N=40)",
        "V12a": "score_propagation",
        "V12b": "terms_filter (ES)",
        "V12c": "overfetch (k*20)",
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

        # Stage-2 metrics
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

        # Stage-1 hit → final recall conversion
        stage1_hits = sum(1 for s in per_sample if s.get("stage1_hit"))
        final_hits = sum(
            1 for s in per_sample
            if s.get(f"recall@{k_max}", s.get("recall_at_k", 0)) > 0
        )
        conversion = final_hits / stage1_hits if stage1_hits > 0 else 0

        avg_cand = sum(s2_candidates) / len(s2_candidates) if s2_candidates else k_max
        avg_fb = sum(s2_fallback) / len(s2_fallback) if s2_fallback else 0

        strategy = strategy_labels.get(cid, "?")
        report_lines.append(
            f"  {cid:<10} │ {strategy:<22} │ {avg_cand:>13.1f}/{k_max} │ {avg_fb:>7.1f}/{k_max} │ {conversion:>9.1%}"
        )

        if cid == "V11c":
            report_lines.append(f"  {'─'*10}─┼─{'─'*22}─┼─{'─'*16}─┼─{'─'*10}─┼─{'─'*10}")

    report_lines.append("")
    report_lines.append("  Legende:")
    report_lines.append("    Ø aus Kandidaten = Wie viele der top-k Ergebnisse aus Stage-1 Files kommen")
    report_lines.append("    Ø Fallback       = Wie viele aus NICHT-Stage-1 Files kommen (Auffüllung)")
    report_lines.append("    Conversion       = Stage-1 Hit → finaler Recall-Hit (Stage-2 Effizienz)")
    report_lines.append("")

    # ── Winner ──
    report_lines.append("─" * 80)
    report_lines.append("  ZUSAMMENFASSUNG")
    report_lines.append("─" * 80)

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

        v11b_recall = results_map.get(("V11b", k_max), {}).get("recall", 0)
        if v11b_recall > 0 and best_cid.startswith("V12"):
            improvement = ((best_recall - v11b_recall) / v11b_recall) * 100
            report_lines.append(
                f"  Verbesserung gegenüber V11b: {improvement:+.1f}%"
            )
        elif best_cid.startswith("V11"):
            report_lines.append(
                f"  → V12 konnte V11 nicht schlagen. Stage-2 Strategien "
                f"bringen bei diesem Setup keinen Vorteil."
            )

    report_lines.append("")
    report_lines.append("=" * 80)

    # Print
    full_report = "\n".join(report_lines)
    print(full_report)

    # Save as file
    report_file = results_path / "comparison_v11_vs_v12.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\nReport gespeichert: {report_file}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V11 vs V12 Vergleichs-Benchmark (pandas)"
    )
    parser.add_argument("--es-url", type=str, default=None,
                        help="Elasticsearch URL (default: http://localhost:9200)")
    parser.add_argument("--verbose", action="store_true",
                        help="Per-sample Ergebnisse ausgeben")
    parser.add_argument("--report-only", action="store_true",
                        help="Nur Report generieren (kein Benchmark laufen lassen)")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10],
                        help="k-Werte (default: 1 5 10)")

    args = parser.parse_args()

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  V11 vs V12 – Hierarchical Retrieval Comparison            ║
║  Repo: pandas | Conditions: {len(ALL_CONDITIONS)} | k={args.k}         ║
╚══════════════════════════════════════════════════════════════╝
""")

        start = time.time()

        run_benchmark(
            conditions=ALL_CONDITIONS,
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
