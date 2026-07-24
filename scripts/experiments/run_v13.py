#!/usr/bin/env python3
"""
run_v13.py – V13 Benchmark (Hybrid-Coarse + Terms-Filter) mit Vergleich.

Führt NUR V13 auf pandas aus und vergleicht das Ergebnis mit den
bestehenden V11/V12-Ergebnissen aus dem vorherigen Lauf.

Aufruf:
  python scripts/run_v13.py [--verbose] [--es-url http://...]
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.runner import run_benchmark
from benchmark.config import CONDITIONS_MAP, K_VALUES


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

V13_CONDITIONS = ["V13"]
REPO = "pandas"
OUTPUT_DIR = str(ROOT / "benchmark" / "results" / "v13")
PREVIOUS_RESULTS = ROOT / "benchmark" / "results" / "v11_vs_v12" / "benchmark_results_20260525_0131.json"


# ──────────────────────────────────────────────────────────────
# Comparison Report
# ──────────────────────────────────────────────────────────────

def generate_comparison_report(v13_results_path: Path):
    """Compare V13 against previously saved V11/V12 results."""

    # Load V13 results
    v13_files = sorted(v13_results_path.glob("benchmark_results_*.json"))
    v13_files = [f for f in v13_files if "PARTIAL" not in f.name and "latest" not in f.name]
    if not v13_files:
        print("\n  Keine V13-Ergebnisse gefunden.")
        return
    with open(v13_files[-1], "r", encoding="utf-8") as f:
        v13_data = json.load(f)

    # Load previous V11/V12 results
    if not PREVIOUS_RESULTS.exists():
        print(f"\n  Vorherige Ergebnisse nicht gefunden: {PREVIOUS_RESULTS}")
        print("  Zeige nur V13-Ergebnisse.")
        prev_runs = []
    else:
        with open(PREVIOUS_RESULTS, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        prev_runs = prev_data.get("runs", [])

    # Merge all runs
    all_runs = prev_runs + v13_data.get("runs", [])

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
    all_conditions = ["V11a", "V11b", "V11c", "V12a", "V12b", "V12c", "V13"]
    k_vals = [1, 5, 10]

    print()
    print("=" * 85)
    print("  V13 vs V11/V12 VERGLEICHSREPORT – pandas (95 Queries)")
    print("=" * 85)
    print()

    # Strategy descriptions
    strategies = {
        "V11a": "BM25-Coarse (N=10) → Post-Filter (k*5)",
        "V11b": "BM25-Coarse (N=20) → Post-Filter (k*5)",
        "V11c": "BM25-Coarse (N=40) → Post-Filter (k*5)",
        "V12a": "BM25-Coarse (N=20) → Score Propagation",
        "V12b": "BM25-Coarse (N=20) → Terms Filter",
        "V12c": "BM25-Coarse (N=20) → Overfetch (k*20)",
        "V13":  "HYBRID-Coarse (N=20) → Terms Filter  ← NEU",
    }

    print("  Varianten:")
    for cid in all_conditions:
        print(f"    {cid:>4}: {strategies.get(cid, '?')}")
    print()

    # Main table
    for k in k_vals:
        baseline_recall = results_map.get(("V12b", k), {}).get("recall", 0.0)

        print(f"  ┌{'─'*6}┬{'─'*12}┬{'─'*10}┬{'─'*12}┬{'─'*25}┐")
        print(f"  │{'k='+str(k):^6}│{'Recall@'+str(k):^12}│{'MRR@'+str(k):^10}│{'S1 Hit%':^12}│{'Δ vs V12b':^25}│")
        print(f"  ├{'─'*6}┼{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*25}┤")

        for cid in all_conditions:
            r = results_map.get((cid, k), {})
            if not r:
                continue

            recall = r["recall"]
            mrr = r["mrr"]
            s1_hr = r["s1_hit_rate"]

            if baseline_recall > 0:
                delta = recall - baseline_recall
                delta_pct = (delta / baseline_recall) * 100
                delta_str = f"{delta:+.4f} ({delta_pct:+.1f}%)"
            else:
                delta_str = "—"

            marker = " ◀" if cid == "V13" else ""
            print(f"  │{cid:^6}│{recall:^12.4f}│{mrr:^10.4f}│{s1_hr*100:^10.1f}% │{delta_str:^25}│{marker}")

            if cid == "V11c" or cid == "V12c":
                print(f"  ├{'─'*6}┼{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*25}┤")

        print(f"  └{'─'*6}┴{'─'*12}┴{'─'*10}┴{'─'*12}┴{'─'*25}┘")
        print()

    # ── Summary ──
    print("─" * 85)
    print("  ANALYSE")
    print("─" * 85)

    v13_r10 = results_map.get(("V13", 10), {})
    v12b_r10 = results_map.get(("V12b", 10), {})
    v11c_r10 = results_map.get(("V11c", 10), {})

    print()
    print(f"  Stage-1 Hit Rate:")
    print(f"    V11b/V12b (BM25, N=20):   {results_map.get(('V12b', 10), {}).get('s1_hit_rate', 0)*100:.1f}%")
    print(f"    V11c (BM25, N=40):         {results_map.get(('V11c', 10), {}).get('s1_hit_rate', 0)*100:.1f}%")
    print(f"    V13  (Hybrid, N=20):       {v13_r10.get('s1_hit_rate', 0)*100:.1f}%")
    print()

    if v13_r10 and v12b_r10:
        s1_gain = v13_r10["s1_hit_rate"] - v12b_r10["s1_hit_rate"]
        recall_gain = v13_r10["recall"] - v12b_r10["recall"]
        print(f"  Hybrid-Coarse vs BM25-Coarse (gleiche N=20, gleiche Stage-2):")
        print(f"    Stage-1 Hit Rate:  {s1_gain*100:+.1f} Prozentpunkte")
        print(f"    Recall@10:         {recall_gain:+.4f} ({recall_gain/v12b_r10['recall']*100:+.1f}% relativ)" if v12b_r10["recall"] > 0 else "")
        print()

    if v13_r10 and v11c_r10:
        vs_v11c = v13_r10["recall"] - v11c_r10["recall"]
        print(f"  V13 vs V11c (bester V11):")
        print(f"    V13 Recall@10:  {v13_r10['recall']:.4f}")
        print(f"    V11c Recall@10: {v11c_r10['recall']:.4f}")
        print(f"    Differenz:      {vs_v11c:+.4f}")
        if vs_v11c > 0:
            print(f"    → V13 schlägt V11c mit N=20 statt N=40 (effizienter)")
        elif vs_v11c < -0.02:
            print(f"    → V11c noch besser, aber V11c braucht N=40 (= kaum Filterung)")
        else:
            print(f"    → Etwa gleichauf, aber V13 mit halb so vielen Kandidaten")
    print()
    print("=" * 85)

    # Save report — rebuild the full text and write to file
    report_file = v13_results_path / "comparison_v13.txt"

    # Re-run the report generation into a string for file output
    import io, contextlib
    buf = io.StringIO()
    # We already printed everything above, so just capture the key data as text
    report_text = []
    report_text.append("=" * 85)
    report_text.append("  V13 vs V11/V12 VERGLEICHSREPORT – pandas (95 Queries)")
    report_text.append("=" * 85)
    report_text.append("")
    report_text.append("  Varianten:")
    for cid in all_conditions:
        report_text.append(f"    {cid:>4}: {strategies.get(cid, '?')}")
    report_text.append("")

    for k in k_vals:
        baseline_recall = results_map.get(("V12b", k), {}).get("recall", 0.0)
        report_text.append(f"  k={k}:")
        for cid in all_conditions:
            r = results_map.get((cid, k), {})
            if not r:
                continue
            recall = r["recall"]
            mrr = r["mrr"]
            s1_hr = r["s1_hit_rate"]
            if baseline_recall > 0:
                delta = recall - baseline_recall
                delta_pct = (delta / baseline_recall) * 100
                delta_str = f"{delta:+.4f} ({delta_pct:+.1f}%)"
            else:
                delta_str = "—"
            report_text.append(f"    {cid:<6} Recall={recall:.4f}  MRR={mrr:.4f}  S1={s1_hr*100:.1f}%  Δ={delta_str}")
        report_text.append("")

    report_text.append("  Stage-1 Hit Rate:")
    report_text.append(f"    V11b/V12b (BM25, N=20):   {results_map.get(('V12b', 10), {}).get('s1_hit_rate', 0)*100:.1f}%")
    report_text.append(f"    V11c (BM25, N=40):         {results_map.get(('V11c', 10), {}).get('s1_hit_rate', 0)*100:.1f}%")
    report_text.append(f"    V13  (Hybrid, N=20):       {v13_r10.get('s1_hit_rate', 0)*100:.1f}%")
    report_text.append("")
    report_text.append("=" * 85)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_text))
    print(f"\n  Report gespeichert: {report_file}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V13 Benchmark + Vergleich")
    parser.add_argument("--es-url", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="Nur Vergleichsreport (kein neuer Lauf)")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])

    args = parser.parse_args()

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  V13 – Hybrid-Coarse + Terms-Filter                        ║
║  Repo: pandas | Stage-1: Hybrid (N=20) | Stage-2: ES Filter║
╚══════════════════════════════════════════════════════════════╝
""")
        start = time.time()

        run_benchmark(
            conditions=V13_CONDITIONS,
            k_values=args.k,
            repos=[REPO],
            dataset_path=str(ROOT / "benchmark" / "data" / "benchmark_dataset.json"),
            output_dir=OUTPUT_DIR,
            es_url=args.es_url,
            verbose=args.verbose,
        )

        elapsed = time.time() - start
        print(f"\n⏱  V13 Laufzeit: {elapsed/60:.1f} Minuten")

    generate_comparison_report(output_path)


if __name__ == "__main__":
    main()
