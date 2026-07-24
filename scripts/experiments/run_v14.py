#!/usr/bin/env python3
"""
run_v14.py – V14 Benchmark (Virtual Document Coarse) mit Vergleich.

Führt V14a + V14b auf pandas aus und vergleicht mit allen bisherigen.

Aufruf:
  python scripts/run_v14.py [--verbose] [--es-url http://...]
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

V14_CONDITIONS = ["V14a", "V14b"]
REPO = "pandas"
OUTPUT_DIR = str(ROOT / "benchmark" / "results" / "v14")

# Previous results to compare against
PREV_RESULTS_FILES = [
    ROOT / "benchmark" / "results" / "v11_vs_v12" / "benchmark_results_20260525_0131.json",
    ROOT / "benchmark" / "results" / "v13" / "benchmark_results_latest.json",
]


# ──────────────────────────────────────────────────────────────
# Comparison Report
# ──────────────────────────────────────────────────────────────

def generate_comparison_report(v14_results_path: Path):
    """Compare V14 against all previous results."""

    # Load V14 results
    v14_files = sorted(v14_results_path.glob("benchmark_results_*.json"))
    v14_files = [f for f in v14_files if "PARTIAL" not in f.name and "latest" not in f.name]
    if not v14_files:
        print("\n  Keine V14-Ergebnisse gefunden.")
        return
    with open(v14_files[-1], "r", encoding="utf-8") as f:
        v14_data = json.load(f)

    # Load all previous results
    prev_runs = []
    for prev_file in PREV_RESULTS_FILES:
        if prev_file.exists():
            with open(prev_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            prev_runs.extend(data.get("runs", []))

    # Merge
    all_runs = prev_runs + v14_data.get("runs", [])

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

    # ── Print Report ──
    # Show best of each generation + V14
    highlight = ["V11c", "V12b", "V13", "V14a", "V14b"]
    all_conditions = ["V11a", "V11b", "V11c", "V12a", "V12b", "V12c", "V13", "V14a", "V14b"]

    strategies = {
        "V11a": "BM25 coarse (N=10), post-filter k*5",
        "V11b": "BM25 coarse (N=20), post-filter k*5",
        "V11c": "BM25 coarse (N=40), post-filter k*5",
        "V12a": "BM25 coarse (N=20), score propagation",
        "V12b": "BM25 coarse (N=20), terms filter",
        "V12c": "BM25 coarse (N=20), overfetch k*20",
        "V13":  "Hybrid coarse (N=20), terms filter",
        "V14a": "Virtual-Doc BM25 coarse (N=20), terms filter  ← NEU",
        "V14b": "Virtual-Doc Hybrid coarse (N=20), terms filter  ← NEU",
    }

    print()
    print("=" * 90)
    print("  V14 VERGLEICHSREPORT – Virtual Document Coarse (pandas, 95 Queries)")
    print("=" * 90)
    print()
    print("  Varianten:")
    for cid in all_conditions:
        available = "()" if (cid, 10) in results_map else "(–)"
        print(f"    {cid:>4} {available}: {strategies.get(cid, '?')}")
    print()

    # Main comparison table (k=10 only, compact)
    k = 10
    print(f"  ┌{'─'*6}┬{'─'*12}┬{'─'*10}┬{'─'*12}┬{'─'*30}┐")
    print(f"  │{'':^6}│{'Recall@10':^12}│{'MRR@10':^10}│{'S1 Hit%':^12}│{'Strategie':^30}│")
    print(f"  ├{'─'*6}┼{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*30}┤")

    best_recall = 0
    best_cid = ""

    for cid in all_conditions:
        r = results_map.get((cid, k))
        if not r:
            continue

        recall = r["recall"]
        mrr = r["mrr"]
        s1_hr = r["s1_hit_rate"]

        if recall > best_recall:
            best_recall = recall
            best_cid = cid

        marker = " ◀" if cid in ("V14a", "V14b") else ""
        is_new = "│" if cid not in ("V14a", "V14b") else "│"

        # Separator between generations
        print(f"  │{cid:^6}│{recall:^12.4f}│{mrr:^10.4f}│{s1_hr*100:^10.1f}% │{strategies.get(cid,'')[:28]:^30}│{marker}")

        if cid in ("V11c", "V12c", "V13"):
            print(f"  ├{'─'*6}┼{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*30}┤")

    print(f"  └{'─'*6}┴{'─'*12}┴{'─'*10}┴{'─'*12}┴{'─'*30}┘")
    print()

    # K-sweep for V14
    print("  V14 Detail (alle k-Werte):")
    print(f"  {'':>6} │ {'Recall@1':>9} │ {'Recall@5':>9} │ {'Recall@10':>10} │ {'MRR@10':>8} │ {'S1 Hit%':>7}")
    print(f"  {'─'*6}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*7}")
    for cid in ["V14a", "V14b"]:
        r1 = results_map.get((cid, 1), {})
        r5 = results_map.get((cid, 5), {})
        r10 = results_map.get((cid, 10), {})
        if r10:
            print(f"  {cid:>6} │ {r1.get('recall',0):>9.4f} │ {r5.get('recall',0):>9.4f} │ {r10.get('recall',0):>10.4f} │ {r10.get('mrr',0):>8.4f} │ {r10.get('s1_hit_rate',0)*100:>5.1f}%")
    print()

    # ── Summary ──
    print("─" * 90)
    print("  ZUSAMMENFASSUNG")
    print("─" * 90)
    print()

    v14a = results_map.get(("V14a", 10), {})
    v14b = results_map.get(("V14b", 10), {})
    v12b = results_map.get(("V12b", 10), {})
    v11c = results_map.get(("V11c", 10), {})

    print(f"  Stage-1 Hit Rate Vergleich (N=20):")
    print(f"    Class/File BM25 (V12b):      {v12b.get('s1_hit_rate',0)*100:.1f}%")
    print(f"    Class/File Hybrid (V13):     {results_map.get(('V13', 10), {}).get('s1_hit_rate',0)*100:.1f}%")
    print(f"    Virtual-Doc BM25 (V14a):     {v14a.get('s1_hit_rate',0)*100:.1f}%")
    print(f"    Virtual-Doc Hybrid (V14b):   {v14b.get('s1_hit_rate',0)*100:.1f}%")
    print(f"    Class/File BM25 N=40 (V11c): {v11c.get('s1_hit_rate',0)*100:.1f}%")
    print()

    if v14b:
        gain_vs_v12b = v14b["recall"] - v12b.get("recall", 0)
        print(f"  Bester V14 Recall@10: {max(v14a.get('recall',0), v14b.get('recall',0)):.4f}")
        print(f"  vs V12b (bisheriger Benchmark): {gain_vs_v12b:+.4f}")
        print(f"  vs V11c (N=40 brute force):     {v14b['recall'] - v11c.get('recall',0):+.4f}")
    print()
    print(f"  Bester Gesamtsieger: {best_cid} (Recall@10 = {best_recall:.4f})")
    print()
    print("=" * 90)

    # ── Save report to file ──
    report_file = v14_results_path / "comparison_v14.txt"
    lines_out = []
    lines_out.append("=" * 90)
    lines_out.append("  V14 VERGLEICHSREPORT – Virtual Document Coarse (pandas, 95 Queries)")
    lines_out.append("=" * 90)
    lines_out.append("")
    lines_out.append("  Varianten:")
    for cid in all_conditions:
        available = "(ok)" if (cid, 10) in results_map else "(--)"
        lines_out.append(f"    {cid:>4} {available}: {strategies.get(cid, '?')}")
    lines_out.append("")
    lines_out.append(f"  Recall@10 Vergleich:")
    for cid in all_conditions:
        r = results_map.get((cid, 10))
        if not r:
            continue
        lines_out.append(f"    {cid:<6} Recall={r['recall']:.4f}  MRR={r['mrr']:.4f}  S1={r['s1_hit_rate']*100:.1f}%")
    lines_out.append("")
    lines_out.append(f"  Stage-1 Hit Rate Vergleich (N=20):")
    lines_out.append(f"    Class/File BM25 (V12b):      {v12b.get('s1_hit_rate',0)*100:.1f}%")
    lines_out.append(f"    Virtual-Doc BM25 (V14a):     {v14a.get('s1_hit_rate',0)*100:.1f}%")
    lines_out.append(f"    Virtual-Doc Hybrid (V14b):   {v14b.get('s1_hit_rate',0)*100:.1f}%")
    lines_out.append(f"    Class/File BM25 N=40 (V11c): {v11c.get('s1_hit_rate',0)*100:.1f}%")
    lines_out.append("")
    lines_out.append(f"  Bester: {best_cid} (Recall@10 = {best_recall:.4f})")
    lines_out.append("=" * 90)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
    print(f"\n  Report gespeichert: {report_file}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V14 Benchmark + Vergleich")
    parser.add_argument("--es-url", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])

    args = parser.parse_args()

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  V14 – Virtual Document Coarse Retrieval                   ║
║  Repo: pandas | Coarse: Virtual Docs | Stage-2: ES Filter  ║
║  V14a: BM25 auf Virtual Docs                               ║
║  V14b: Hybrid auf Virtual Docs                             ║
╚══════════════════════════════════════════════════════════════╝
""")
        start = time.time()

        run_benchmark(
            conditions=V14_CONDITIONS,
            k_values=args.k,
            repos=[REPO],
            dataset_path=str(ROOT / "benchmark" / "data" / "benchmark_dataset.json"),
            output_dir=OUTPUT_DIR,
            es_url=args.es_url,
            verbose=args.verbose,
        )

        elapsed = time.time() - start
        print(f"\n⏱  V14 Laufzeit: {elapsed/60:.1f} Minuten")

    generate_comparison_report(output_path)


if __name__ == "__main__":
    main()
