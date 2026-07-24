#!/usr/bin/env python3
"""
run_v19.py – V19 Benchmark: Cross-Kombination V16b×V16c + LLM Stage-2.

Runs two new variants and compares against V16b and V16c:
  V19a: V16b S1 (Class/File BM25 LLM-Exp) → V16c S2 (Hybrid)
  V19b: V16c S1 (VDoc BM25 LLM-Exp) → Hybrid S2 mit LLM-Expanded Query

Aufruf:
  python scripts/run_v19.py [--verbose] [--es-url http://...]
  python scripts/run_v19.py --report-only
  python scripts/run_v19.py --conditions V19a  # subset
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

NEW_CONDITIONS = ["V19a", "V19b"]
REPO = "pandas"
OUTPUT_DIR = str(ROOT / "benchmark" / "results" / "v19")

# Previous results for comparison (V16)
PREV_RESULTS_FILES = [
    ROOT / "benchmark" / "results" / "v16_llm_expansion" / "benchmark_results_latest.json",
]


# ──────────────────────────────────────────────────────────────
# Comparison Report
# ──────────────────────────────────────────────────────────────

def generate_comparison_report(results_path: Path):
    """Generate comparison: V19a/V19b vs V16b/V16c."""

    # Load new results
    result_files = sorted(results_path.glob("benchmark_results_*.json"))
    result_files = [f for f in result_files
                    if "PARTIAL" not in f.name and "latest" not in f.name]
    if not result_files:
        print("\n  Keine neuen Ergebnisse gefunden.")
        return
    with open(result_files[-1], "r", encoding="utf-8") as f:
        new_data = json.load(f)

    # Load V16 results
    prev_runs = []
    for prev_file in PREV_RESULTS_FILES:
        if prev_file.exists():
            with open(prev_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            prev_runs.extend(data.get("runs", []))

    all_runs = prev_runs + new_data.get("runs", [])

    # Build results map
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
    strategies = {
        "V16b": "Hier Class/File BM25 LLM-Exp (N=20) → Dense",
        "V16c": "Hier VDoc BM25 LLM-Exp (N=20) → Hybrid",
        "V19a": "V16b S1 (Class/File) → V16c S2 (Hybrid)  ← NEU",
        "V19b": "V16c + LLM-Expanded auch in Stage 2  ← NEU",
    }

    display_order = ["V16b", "V16c", "V19a", "V19b"]

    print()
    print("=" * 100)
    print("  V19 VERGLEICHSREPORT – pandas (95 Queries)")
    print("  Frage 1: Ist Class/File S1 + Hybrid S2 besser als V16b oder V16c einzeln?")
    print("  Frage 2: Hilft LLM-Expansion auch in Stage 2?")
    print("=" * 100)
    print()

    for k_val in [1, 5, 10]:
        print(f"  ── Recall@{k_val} & MRR@{k_val} ──")
        print(f"  ┌{'─'*8}┬{'─'*14}┬{'─'*12}┬{'─'*12}┬{'─'*50}┐")
        print(f"  │{'':^8}│{'Recall@'+str(k_val):^14}│{'MRR@'+str(k_val):^12}│{'S1 Hit%':^12}│{'Strategie':^50}│")
        print(f"  ├{'─'*8}┼{'─'*14}┼{'─'*12}┼{'─'*12}┼{'─'*50}┤")

        for cid in display_order:
            r = results_map.get((cid, k_val))
            if not r:
                continue

            recall = r["recall"]
            mrr = r["mrr"]
            s1_hr = r["s1_hit_rate"]
            s1_str = f"{s1_hr*100:.1f}%" if s1_hr > 0 else "—"
            strat = strategies.get(cid, "?")[:48]
            print(f"  │{cid:^8}│{recall:^14.4f}│{mrr:^12.4f}│{s1_str:^12}│{strat:^50}│")

        print(f"  └{'─'*8}┴{'─'*14}┴{'─'*12}┴{'─'*12}┴{'─'*50}┘")
        print()

    # ── Deltas ──
    k = 10
    v16b = results_map.get(("V16b", k), {})
    v16c = results_map.get(("V16c", k), {})
    v19a = results_map.get(("V19a", k), {})
    v19b = results_map.get(("V19b", k), {})

    print("─" * 100)
    print("  DELTA-ANALYSE (k=10)")
    print("─" * 100)
    print()

    if v19a and v16b:
        dr = v19a["recall"] - v16b["recall"]
        dm = v19a["mrr"] - v16b["mrr"]
        print(f"  V19a vs V16b: Δ Recall={dr:+.4f}  Δ MRR={dm:+.4f}")
    if v19a and v16c:
        dr = v19a["recall"] - v16c["recall"]
        dm = v19a["mrr"] - v16c["mrr"]
        print(f"  V19a vs V16c: Δ Recall={dr:+.4f}  Δ MRR={dm:+.4f}")
    print()
    if v19b and v16c:
        dr = v19b["recall"] - v16c["recall"]
        dm = v19b["mrr"] - v16c["mrr"]
        print(f"  V19b vs V16c: Δ Recall={dr:+.4f}  Δ MRR={dm:+.4f}")
        print(f"    → Effekt von LLM-Expansion in Stage 2")
    print()

    # ── Best overall ──
    best_recall = 0
    best_mrr = 0
    best_r_cid = ""
    best_m_cid = ""
    for cid in display_order:
        r = results_map.get((cid, k), {})
        if r.get("recall", 0) > best_recall:
            best_recall = r["recall"]
            best_r_cid = cid
        if r.get("mrr", 0) > best_mrr:
            best_mrr = r["mrr"]
            best_m_cid = cid

    print("─" * 100)
    print("  ZUSAMMENFASSUNG")
    print("─" * 100)
    print()
    print(f"  Bester Recall@10:  {best_r_cid} = {best_recall:.4f}")
    print(f"  Bester MRR@10:     {best_m_cid} = {best_mrr:.4f}")
    print()

    # ── TC Breakdown ──
    print("  TC-Breakdown (k=10):")
    print(f"  {'Condition':<10} {'R@10 TC1':>10} {'R@10 TC2':>10} {'R@10 TC3':>10} │ {'MRR TC1':>10} {'MRR TC2':>10} {'MRR TC3':>10}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} │ {'─'*10} {'─'*10} {'─'*10}")

    for cid in display_order:
        # Gather per-sample from the latest run at k=10
        tc_recalls = {"TC1": [], "TC2": [], "TC3": []}
        tc_mrrs = {"TC1": [], "TC2": [], "TC3": []}

        for run in all_runs:
            if run["condition_id"] == cid and run["k"] == 10:
                for s in run.get("metrics", {}).get("per_sample", []):
                    tc = s.get("tc_type", "TC1")
                    tc_recalls.setdefault(tc, []).append(s.get("recall_at_k", 0))
                    tc_mrrs.setdefault(tc, []).append(s.get("mrr_at_k", 0))

        if not any(tc_recalls.values()):
            continue

        parts = []
        for tc in ["TC1", "TC2", "TC3"]:
            vals = tc_recalls.get(tc, [])
            parts.append(f"{sum(vals)/len(vals):.3f}" if vals else "  —   ")
        mrr_parts = []
        for tc in ["TC1", "TC2", "TC3"]:
            vals = tc_mrrs.get(tc, [])
            mrr_parts.append(f"{sum(vals)/len(vals):.3f}" if vals else "  —   ")

        print(f"  {cid:<10} {parts[0]:>10} {parts[1]:>10} {parts[2]:>10} │ {mrr_parts[0]:>10} {mrr_parts[1]:>10} {mrr_parts[2]:>10}")

    print()
    print("=" * 100)

    # ── Save report ──
    report_file = results_path / "comparison_v19.txt"
    lines = []
    lines.append("=" * 100)
    lines.append("  V19 VERGLEICHSREPORT – pandas (95 Queries, k=10)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("  Recall@10 + MRR@10 Vergleich:")
    lines.append("")
    for cid in display_order:
        r = results_map.get((cid, k))
        if not r:
            continue
        s1_str = f"{r['s1_hit_rate']*100:.1f}%" if r["s1_hit_rate"] > 0 else "—"
        lines.append(
            f"    {cid:<6} Recall={r['recall']:.4f}  MRR={r['mrr']:.4f}  "
            f"S1={s1_str}  {strategies.get(cid, '')}"
        )
    lines.append("")

    if v19a and v16b:
        dr = v19a["recall"] - v16b["recall"]
        dm = v19a["mrr"] - v16b["mrr"]
        lines.append(f"  V19a vs V16b: Δ Recall={dr:+.4f}  Δ MRR={dm:+.4f}")
    if v19a and v16c:
        dr = v19a["recall"] - v16c["recall"]
        dm = v19a["mrr"] - v16c["mrr"]
        lines.append(f"  V19a vs V16c: Δ Recall={dr:+.4f}  Δ MRR={dm:+.4f}")
    if v19b and v16c:
        dr = v19b["recall"] - v16c["recall"]
        dm = v19b["mrr"] - v16c["mrr"]
        lines.append(f"  V19b vs V16c: Δ Recall={dr:+.4f}  Δ MRR={dm:+.4f}")
    lines.append("")
    lines.append(f"  Bester Recall: {best_r_cid} = {best_recall:.4f}")
    lines.append(f"  Bester MRR:    {best_m_cid} = {best_mrr:.4f}")
    lines.append("=" * 100)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {report_file}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V19 Benchmark: Cross-Kombination V16b×V16c + LLM Stage-2"
    )
    parser.add_argument("--es-url", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--conditions", nargs="*", default=NEW_CONDITIONS,
                        help="Conditions to run (default: V19a V19b)")

    args = parser.parse_args()

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        cond_str = " ".join(args.conditions)
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  V19 Benchmark – Cross-Kombination V16b×V16c + LLM Stage-2     ║
║  Repo: pandas (95 Queries) | Conditions: {cond_str:<23}║
║                                                                  ║
║  V19a: Class/File S1 (V16b) → Hybrid S2 (V16c)                 ║
║  V19b: VDoc S1 (V16c) → Hybrid S2 + LLM-Expanded Query         ║
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
