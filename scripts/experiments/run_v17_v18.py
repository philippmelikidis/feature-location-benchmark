#!/usr/bin/env python3
"""
run_v17_v18.py – V17 (Ensemble Coarse) + V18 (Flat LLM-Expanded) Benchmark.

Runs all new variants on pandas and generates a unified comparison report
against V16b/V16c and the flat baselines (V3, V10b).

The report answers two key questions:
  1. Does ensemble coarse (V17) beat the best single-coarse (V16c)?
  2. Does LLM expansion alone (V18) improve flat retrieval, or is
     hierarchical filtering the real driver?

Aufruf:
  python scripts/run_v17_v18.py [--verbose] [--es-url http://...]
  python scripts/run_v17_v18.py --report-only
  python scripts/run_v17_v18.py --conditions V17a V18a  # subset
"""

import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.runner import run_benchmark
from benchmark.config import CONDITIONS_MAP


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

NEW_CONDITIONS = ["V17a", "V17b", "V17c", "V18a", "V18b"]
REPO = "pandas"
OUTPUT_DIR = str(ROOT / "benchmark" / "results" / "v17_v18")

# Previous results for comparison
PREV_RESULTS_FILES = [
    ROOT / "benchmark" / "results" / "v16_llm_expansion" / "benchmark_results_latest.json",
    ROOT / "benchmark" / "results" / "v11_vs_v12" / "benchmark_results_20260525_0131.json",
]


# ──────────────────────────────────────────────────────────────
# Comparison Report
# ──────────────────────────────────────────────────────────────

def generate_comparison_report(results_path: Path):
    """Generate unified comparison: V17/V18 vs V16 vs flat baselines."""

    # Load new results
    result_files = sorted(results_path.glob("benchmark_results_*.json"))
    result_files = [f for f in result_files
                    if "PARTIAL" not in f.name and "latest" not in f.name]
    if not result_files:
        print("\n  Keine neuen Ergebnisse gefunden.")
        return
    with open(result_files[-1], "r", encoding="utf-8") as f:
        new_data = json.load(f)

    # Load previous results
    prev_runs = []
    for prev_file in PREV_RESULTS_FILES:
        if prev_file.exists():
            with open(prev_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            prev_runs.extend(data.get("runs", []))

    all_runs = prev_runs + new_data.get("runs", [])

    # Build results map (latest per condition wins)
    results_map = {}
    for run in all_runs:
        cid = run["condition_id"]
        k = run["k"]
        m = run.get("metrics", {})
        per_sample = m.get("per_sample", [])

        s1_hits = sum(1 for s in per_sample if s.get("stage1_hit")) if per_sample else 0
        total = len(per_sample) if per_sample else 0

        results_map[(cid, k)] = {
            "recall": m.get("recall_at_k", 0.0),
            "mrr": m.get("mrr_at_k", 0.0),
            "s1_hit_rate": s1_hits / total if total > 0 else 0,
            "num_samples": m.get("num_samples", 0),
        }

    # ── Report ──
    k = 10

    strategies = {
        # Flat baselines
        "V3":   "Flat Dense Function-Level (BGE, full query)",
        "V8":   "Flat Dense AST (BGE, full query)",
        "V10b": "Flat Hybrid AST (α=0.5, full query)",
        # V16 (hierarchical + LLM expansion)
        "V12b": "Hier BM25 (N=20) → Terms Filter (no LLM)",
        "V16b": "Hier LLM-Exp Class/File (N=20) → Dense",
        "V16c": "Hier LLM-Exp VDoc (N=20) → Hybrid",
        # V17 (ensemble coarse)
        "V17a": "Ensemble (C/F ∪ VDoc) N=20 → Hybrid  ← NEU",
        "V17b": "Ensemble (C/F ∪ VDoc) N=30 → Hybrid  ← NEU",
        "V17c": "Ensemble (C/F ∪ VDoc) N=20 → Dense   ← NEU",
        # V18 (flat + LLM ablation)
        "V18a": "Flat Dense AST + LLM-Expanded  ← ABLATION",
        "V18b": "Flat Hybrid AST + LLM-Expanded  ← ABLATION",
    }

    # Group conditions for display
    sections = [
        ("Flat Baselines (kein Hierarchie, kein LLM)", ["V3", "V8", "V10b"]),
        ("V18: Flat + LLM-Expanded (Ablation)", ["V18a", "V18b"]),
        ("V16: Hierarchisch + LLM-Expanded", ["V12b", "V16b", "V16c"]),
        ("V17: Ensemble Coarse + LLM-Expanded", ["V17a", "V17b", "V17c"]),
    ]

    print()
    print("=" * 100)
    print("  V17/V18 VERGLEICHSREPORT – pandas (95 Queries, k=10)")
    print("  Frage 1: Bringt Ensemble-Coarse (V17) mehr als Single-Coarse (V16)?")
    print("  Frage 2: Wie viel bringt LLM-Expansion ohne Hierarchie (V18)?")
    print("=" * 100)
    print()

    best_recall = 0
    best_mrr = 0
    best_recall_cid = ""
    best_mrr_cid = ""

    for section_title, cids in sections:
        has_data = any((cid, k) in results_map for cid in cids)
        if not has_data:
            continue

        print(f"  ── {section_title} ──")
        print(f"  ┌{'─'*6}┬{'─'*12}┬{'─'*10}┬{'─'*12}┬{'─'*48}┐")
        print(f"  │{'':^6}│{'Recall@10':^12}│{'MRR@10':^10}│{'S1 Hit%':^12}│{'Strategie':^48}│")
        print(f"  ├{'─'*6}┼{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*48}┤")

        for cid in cids:
            r = results_map.get((cid, k))
            if not r:
                continue

            recall = r["recall"]
            mrr = r["mrr"]
            s1_hr = r["s1_hit_rate"]

            if recall > best_recall:
                best_recall = recall
                best_recall_cid = cid
            if mrr > best_mrr:
                best_mrr = mrr
                best_mrr_cid = cid

            s1_str = f"{s1_hr*100:.1f}%" if s1_hr > 0 else "—"
            strat = strategies.get(cid, "?")[:46]
            print(f"  │{cid:^6}│{recall:^12.4f}│{mrr:^10.4f}│{s1_str:^12}│{strat:^48}│")

        print(f"  └{'─'*6}┴{'─'*12}┴{'─'*10}┴{'─'*12}┴{'─'*48}┘")
        print()

    # ── Analysis ──
    print("─" * 100)
    print("  ANALYSE")
    print("─" * 100)
    print()

    # Question 1: V17 vs V16
    v16c = results_map.get(("V16c", k), {})
    v16b = results_map.get(("V16b", k), {})
    v17a = results_map.get(("V17a", k), {})
    v17b = results_map.get(("V17b", k), {})

    if v16c and v17a:
        print("  Frage 1: Ensemble-Coarse (V17) vs Single-Coarse (V16)")
        print(f"    V16c (best single):  Recall={v16c['recall']:.4f}  MRR={v16c['mrr']:.4f}")
        if v17a:
            dr = v17a['recall'] - v16c['recall']
            dm = v17a['mrr'] - v16c['mrr']
            print(f"    V17a (ensemble N=20): Recall={v17a['recall']:.4f}  MRR={v17a['mrr']:.4f}  "
                  f"Δ Recall={dr:+.4f}  Δ MRR={dm:+.4f}")
        if v17b:
            dr = v17b['recall'] - v16c['recall']
            dm = v17b['mrr'] - v16c['mrr']
            print(f"    V17b (ensemble N=30): Recall={v17b['recall']:.4f}  MRR={v17b['mrr']:.4f}  "
                  f"Δ Recall={dr:+.4f}  Δ MRR={dm:+.4f}")
        print()

    # Question 2: V18 ablation
    v18a = results_map.get(("V18a", k), {})
    v18b = results_map.get(("V18b", k), {})
    v8 = results_map.get(("V8", k), {})
    v10b = results_map.get(("V10b", k), {})

    if v18a or v18b:
        print("  Frage 2: LLM-Expansion ohne Hierarchie (V18 Ablation)")
        if v8:
            print(f"    V8  (flat dense, full query):        Recall={v8['recall']:.4f}")
        if v18a:
            dr = v18a['recall'] - v8.get('recall', 0) if v8 else 0
            print(f"    V18a (flat dense, LLM-expanded):     Recall={v18a['recall']:.4f}  "
                  f"Δ={dr:+.4f}")
        if v10b:
            print(f"    V10b (flat hybrid, full query):       Recall={v10b['recall']:.4f}")
        if v18b:
            dr = v18b['recall'] - v10b.get('recall', 0) if v10b else 0
            print(f"    V18b (flat hybrid, LLM-expanded):    Recall={v18b['recall']:.4f}  "
                  f"Δ={dr:+.4f}")
        print()

        if v18b and v16c:
            gap = v16c['recall'] - v18b['recall']
            print(f"    Differenz V18b → V16c: {gap:.4f}")
            print(f"    → Davon erklärt LLM-Expansion: {v18b['recall'] - v10b.get('recall', 0):.4f}" if v10b else "")
            print(f"    → Davon erklärt Hierarchie:    {gap:.4f}")
            print()

    # ── Summary ──
    print("─" * 100)
    print("  ZUSAMMENFASSUNG")
    print("─" * 100)
    print()
    print(f"  Bester Recall@10:  {best_recall_cid} = {best_recall:.4f}")
    print(f"  Bester MRR@10:     {best_mrr_cid} = {best_mrr:.4f}")
    print()

    # Attribution table
    if v10b and v18b and v16c:
        base = v10b.get('recall', 0)
        llm_gain = v18b['recall'] - base
        hier_gain = v16c['recall'] - v18b['recall']
        ensemble_gain = v17a['recall'] - v16c['recall'] if v17a else 0

        print("  Recall@10 Zuordnung (kumulative Verbesserungen):")
        print(f"    Flat Hybrid Baseline (V10b):         {base:.4f}")
        print(f"    + LLM Query-Expansion (V18b-V10b):   {llm_gain:+.4f}")
        print(f"    + Hierarchisches Filtering (V16c-V18b): {hier_gain:+.4f}")
        if v17a:
            print(f"    + Ensemble Coarse (V17a-V16c):       {ensemble_gain:+.4f}")
        print(f"    = Gesamt:                             {v17a.get('recall', v16c['recall']):.4f}")
    print()
    print("=" * 100)

    # ── Save report ──
    report_file = results_path / "comparison_v17_v18.txt"
    lines = []
    lines.append("=" * 100)
    lines.append("  V17/V18 VERGLEICHSREPORT – pandas (95 Queries, k=10)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("  Recall@10 + MRR@10 Vergleich:")
    for _, cids in sections:
        for cid in cids:
            r = results_map.get((cid, k))
            if not r:
                continue
            lines.append(
                f"    {cid:<6} Recall={r['recall']:.4f}  MRR={r['mrr']:.4f}  "
                f"S1={r['s1_hit_rate']*100:.1f}%  {strategies.get(cid, '')}"
            )
        lines.append("")
    lines.append(f"  Bester Recall: {best_recall_cid} = {best_recall:.4f}")
    lines.append(f"  Bester MRR:    {best_mrr_cid} = {best_mrr:.4f}")
    lines.append("=" * 100)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {report_file}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V17 Ensemble + V18 Flat-LLM Benchmark"
    )
    parser.add_argument("--es-url", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--conditions", nargs="*", default=NEW_CONDITIONS,
                        help="Conditions to run (default: V17a V17b V17c V18a V18b)")

    args = parser.parse_args()

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        cond_str = " ".join(args.conditions)
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  V17 + V18 Benchmark – Ensemble Coarse & Flat-LLM Ablation     ║
║  Repo: pandas (95 Queries) | Conditions: {cond_str:<23}║
║                                                                  ║
║  V17: Ensemble (Class/File ∪ VDoc) → Hybrid/Dense               ║
║  V18: Flat Dense/Hybrid + LLM-Expanded Query (Ablation)         ║
╚══════════════════════════════════════════════════════════════════╝
""")
        start = time.time()

        run_benchmark(
            conditions=args.conditions,
            k_values=args.k,
            repos=[REPO],
            dataset_path=str(ROOT / "benchmark" / "data" / "benchmark_dataset.json"),
            output_dir=OUTPUT_DIR,
            es_url=args.es_url,
            verbose=args.verbose,
        )

        elapsed = time.time() - start
        print(f"\n  Laufzeit: {elapsed/60:.1f} Minuten")

    generate_comparison_report(output_path)


if __name__ == "__main__":
    main()
