#!/usr/bin/env python3
"""
run_v17_ensemble.py – V17 Ensemble Coarse Benchmark mit Vergleich.

Führt V17a/b/c auf pandas aus und vergleicht mit V16b/V16c.

Aufruf:
  python scripts/run_v17_ensemble.py [--verbose] [--es-url http://...]
  python scripts/run_v17_ensemble.py --report-only
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

V17_CONDITIONS = ["V17a", "V17b", "V17c"]
REPO = "pandas"
OUTPUT_DIR = str(ROOT / "benchmark" / "results" / "v17_ensemble")

PREV_RESULTS_FILES = [
    ROOT / "benchmark" / "results" / "v16_llm_expansion" / "benchmark_results_latest.json",
    ROOT / "benchmark" / "results" / "v11_vs_v12" / "benchmark_results_20260525_0131.json",
]


# ──────────────────────────────────────────────────────────────
# Comparison Report
# ──────────────────────────────────────────────────────────────

def generate_comparison_report(v17_results_path: Path):
    """Compare V17 against V16b, V16c, and best-of-breed."""

    # Load V17 results
    v17_files = sorted(v17_results_path.glob("benchmark_results_*.json"))
    v17_files = [f for f in v17_files if "PARTIAL" not in f.name and "latest" not in f.name]
    if not v17_files:
        print("\n  Keine V17-Ergebnisse gefunden.")
        return
    with open(v17_files[-1], "r", encoding="utf-8") as f:
        v17_data = json.load(f)

    # Load previous results
    prev_runs = []
    for prev_file in PREV_RESULTS_FILES:
        if prev_file.exists():
            with open(prev_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            prev_runs.extend(data.get("runs", []))

    all_runs = prev_runs + v17_data.get("runs", [])

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
    all_conditions = ["V12b", "V16b", "V16c", "V17a", "V17b", "V17c"]

    strategies = {
        "V12b": "BM25 class/file (N=20) → Terms Filter (baseline)",
        "V16b": "LLM-Exp Class/File BM25 (N=20) → Dense  [best MRR]",
        "V16c": "LLM-Exp VDoc BM25 (N=20) → Hybrid  [best Recall]",
        "V17a": "Ensemble (C/F ∪ VDoc) BM25 (N=20) → Hybrid  ← NEU",
        "V17b": "Ensemble (C/F ∪ VDoc) BM25 (N=30) → Hybrid  ← NEU",
        "V17c": "Ensemble (C/F ∪ VDoc) BM25 (N=20) → Dense   ← NEU",
    }

    print()
    print("=" * 95)
    print("  V17 VERGLEICHSREPORT – Ensemble Coarse (pandas, 95 Queries)")
    print("=" * 95)
    print()
    print("  Varianten:")
    for cid in all_conditions:
        available = "(ok)" if (cid, 10) in results_map else "(--)"
        print(f"    {cid:>4} {available}: {strategies.get(cid, '?')}")
    print()

    # Main table (k=10)
    k = 10
    v16c = results_map.get(("V16c", k), {})
    v16c_recall = v16c.get("recall", 0)
    v16c_mrr = v16c.get("mrr", 0)

    print(f"  ┌{'─'*6}┬{'─'*12}┬{'─'*10}┬{'─'*12}┬{'─'*14}┬{'─'*14}┐")
    print(f"  │{'':^6}│{'Recall@10':^12}│{'MRR@10':^10}│{'S1 Hit%':^12}│{'Δ Recall':^14}│{'Δ MRR':^14}│")
    print(f"  ├{'─'*6}┼{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*14}┼{'─'*14}┤")

    best_recall = 0
    best_mrr = 0
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
        if mrr > best_mrr:
            best_mrr = mrr

        dr = recall - v16c_recall if v16c_recall else 0
        dm = mrr - v16c_mrr if v16c_mrr else 0
        dr_s = f"{dr:+.4f}" if v16c_recall else "—"
        dm_s = f"{dm:+.4f}" if v16c_mrr else "—"

        marker = " ◀" if cid.startswith("V17") else ""
        print(f"  │{cid:^6}│{recall:^12.4f}│{mrr:^10.4f}│{s1_hr*100:^10.1f}% │{dr_s:^14}│{dm_s:^14}│{marker}")

        if cid in ("V16c",):
            print(f"  ├{'─'*6}┼{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*14}┼{'─'*14}┤")

    print(f"  └{'─'*6}┴{'─'*12}┴{'─'*10}┴{'─'*12}┴{'─'*14}┴{'─'*14}┘")
    print()

    # K-sweep for V17
    print("  V17 Detail (alle k-Werte):")
    print(f"  {'':>6} │ {'Recall@1':>9} │ {'Recall@5':>9} │ {'Recall@10':>10} │ {'MRR@10':>8} │ {'S1 Hit%':>7}")
    print(f"  {'─'*6}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*7}")
    for cid in ["V17a", "V17b", "V17c"]:
        r1 = results_map.get((cid, 1), {})
        r5 = results_map.get((cid, 5), {})
        r10 = results_map.get((cid, 10), {})
        if r10:
            print(f"  {cid:>6} │ {r1.get('recall',0):>9.4f} │ {r5.get('recall',0):>9.4f} │ {r10.get('recall',0):>10.4f} │ {r10.get('mrr',0):>8.4f} │ {r10.get('s1_hit_rate',0)*100:>5.1f}%")
    print()

    # ── Trade-off analysis ──
    print("─" * 95)
    print("  TRADE-OFF ANALYSE")
    print("─" * 95)
    print()
    print("  Latenz: Ensemble baut 3 ES-Indizes statt 2 (Coarse A + Coarse B + Fine).")
    print("          Pro Query: 2 BM25-Coarse-Queries (~1ms je) + 1 Dense/Hybrid Stage-2.")
    print("          Coarse-Overhead ist vernachlässigbar; Indexing +~30% durch doppelte Coarse.")
    print()

    v17a = results_map.get(("V17a", 10), {})
    v17b = results_map.get(("V17b", 10), {})
    if v17a and v17b:
        print(f"  Kandidaten-Pool Vergleich:")
        print(f"    V17a (N=20 cap): S1 Hit Rate = {v17a.get('s1_hit_rate',0)*100:.1f}%")
        print(f"    V17b (N=30 cap): S1 Hit Rate = {v17b.get('s1_hit_rate',0)*100:.1f}%")
        s1_diff = (v17b.get('s1_hit_rate', 0) - v17a.get('s1_hit_rate', 0)) * 100
        r_diff = v17b['recall'] - v17a['recall']
        print(f"    Δ S1 Hit Rate:   {s1_diff:+.1f}pp")
        print(f"    Δ Recall@10:     {r_diff:+.4f}")
        if abs(r_diff) < 0.02:
            print(f"    → N=30 bringt kaum Mehrwert → N=20 bevorzugen (weniger Noise).")
        elif r_diff > 0.02:
            print(f"    → N=30 lohnt sich: höhere Recall bei akzeptabler Streuung.")
    print()

    print(f"  Bester V17: {best_cid} (Recall@10 = {best_recall:.4f})")
    print()
    print("=" * 95)

    # ── Save report ──
    report_file = v17_results_path / "comparison_v17_ensemble.txt"
    lines_out = []
    lines_out.append("=" * 95)
    lines_out.append("  V17 VERGLEICHSREPORT – Ensemble Coarse (pandas, 95 Queries)")
    lines_out.append("=" * 95)
    lines_out.append("")
    lines_out.append("  Recall@10 + MRR@10 Vergleich (vs V16c):")
    for cid in all_conditions:
        r = results_map.get((cid, 10))
        if not r:
            continue
        lines_out.append(
            f"    {cid:<6} Recall={r['recall']:.4f}  MRR={r['mrr']:.4f}  "
            f"S1={r['s1_hit_rate']*100:.1f}%"
        )
    lines_out.append("")
    lines_out.append(f"  Bester: {best_cid} (Recall@10 = {best_recall:.4f})")
    lines_out.append("=" * 95)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
    print(f"\n  Saved: {report_file}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V17 Ensemble Benchmark + Vergleich")
    parser.add_argument("--es-url", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--conditions", nargs="*", default=V17_CONDITIONS,
                        help="Conditions to run (default: V17a V17b V17c)")

    args = parser.parse_args()

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  V17 – Ensemble Coarse (Class/File ∪ Virtual Document)     ║
║  Repo: pandas | Stage-1: dual BM25 | Stage-2: Terms Filter ║
║  V17a: N=20 → Hybrid | V17b: N=30 → Hybrid | V17c: Dense  ║
╚══════════════════════════════════════════════════════════════╝
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
        print(f"\n  V17 Laufzeit: {elapsed/60:.1f} Minuten")

    generate_comparison_report(output_path)


if __name__ == "__main__":
    main()
