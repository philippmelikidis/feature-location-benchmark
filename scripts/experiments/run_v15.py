#!/usr/bin/env python3
"""
run_v15.py – V15 Benchmark (Title-Only Query) mit Vergleich.

Führt V15a + V15b auf pandas aus und vergleicht mit allen bisherigen.

Hypothese: Die bisherigen Queries (~560 Zeichen GitHub-Issue-Text) enthalten
zu viel Noise. Nur der Issue-Titel als Query sollte bessere BM25-Treffer liefern.

Aufruf:
  python scripts/run_v15.py [--verbose] [--es-url http://...]
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

V15_CONDITIONS = ["V15a", "V15b"]
REPO = "pandas"
OUTPUT_DIR = str(ROOT / "benchmark" / "results" / "v15")

# Previous results to compare against
PREV_RESULTS_FILES = [
    ROOT / "benchmark" / "results" / "v11_vs_v12" / "benchmark_results_20260525_0131.json",
    ROOT / "benchmark" / "results" / "v13" / "benchmark_results_latest.json",
    ROOT / "benchmark" / "results" / "v14" / "benchmark_results_latest.json",
]


# ──────────────────────────────────────────────────────────────
# Comparison Report
# ──────────────────────────────────────────────────────────────

def generate_comparison_report(v15_results_path: Path):
    """Compare V15 against all previous results."""

    # Load V15 results
    v15_files = sorted(v15_results_path.glob("benchmark_results_*.json"))
    v15_files = [f for f in v15_files if "PARTIAL" not in f.name and "latest" not in f.name]
    if not v15_files:
        print("\n  Keine V15-Ergebnisse gefunden.")
        return
    with open(v15_files[-1], "r", encoding="utf-8") as f:
        v15_data = json.load(f)

    # Load all previous results
    prev_runs = []
    for prev_file in PREV_RESULTS_FILES:
        if prev_file.exists():
            with open(prev_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            prev_runs.extend(data.get("runs", []))

    # Merge
    all_runs = prev_runs + v15_data.get("runs", [])

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
    all_conditions = ["V11c", "V12b", "V13", "V14a", "V14b", "V15a", "V15b"]

    strategies = {
        "V11c": "BM25 coarse (N=40), post-filter k*5, full query",
        "V12b": "BM25 coarse (N=20), terms filter, full query",
        "V13":  "Hybrid coarse (N=20), terms filter, full query",
        "V14a": "Virtual-Doc BM25 coarse (N=20), terms filter, full query",
        "V14b": "Virtual-Doc Hybrid coarse (N=20), terms filter, full query",
        "V15a": "Class/File BM25 coarse (N=20), terms filter, TITLE ONLY  ← NEU",
        "V15b": "Virtual-Doc BM25 coarse (N=20), terms filter, TITLE ONLY  ← NEU",
    }

    print()
    print("=" * 95)
    print("  V15 VERGLEICHSREPORT – Title-Only Query (pandas, 95 Queries)")
    print("=" * 95)
    print()
    print("  Hypothese: Issue-Titel enthält präzisere Keywords als der Full-Body-Text.")
    print("  V15a = V12b mit title_only | V15b = V14a mit title_only")
    print()
    print("  Varianten:")
    for cid in all_conditions:
        available = "()" if (cid, 10) in results_map else "(–)"
        print(f"    {cid:>4} {available}: {strategies.get(cid, '?')}")
    print()

    # Main comparison table (k=10)
    k = 10
    print(f"  ┌{'─'*6}┬{'─'*12}┬{'─'*10}┬{'─'*12}┬{'─'*40}┐")
    print(f"  │{'':^6}│{'Recall@10':^12}│{'MRR@10':^10}│{'S1 Hit%':^12}│{'Strategie':^40}│")
    print(f"  ├{'─'*6}┼{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*40}┤")

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

        marker = " ◀" if cid in ("V15a", "V15b") else ""
        print(f"  │{cid:^6}│{recall:^12.4f}│{mrr:^10.4f}│{s1_hr*100:^10.1f}% │{strategies.get(cid,'')[:38]:^40}│{marker}")

        if cid in ("V13", "V14b"):
            print(f"  ├{'─'*6}┼{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*40}┤")

    print(f"  └{'─'*6}┴{'─'*12}┴{'─'*10}┴{'─'*12}┴{'─'*40}┘")
    print()

    # K-sweep for V15
    print("  V15 Detail (alle k-Werte):")
    print(f"  {'':>6} │ {'Recall@1':>9} │ {'Recall@5':>9} │ {'Recall@10':>10} │ {'MRR@10':>8} │ {'S1 Hit%':>7}")
    print(f"  {'─'*6}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*7}")
    for cid in ["V15a", "V15b"]:
        r1 = results_map.get((cid, 1), {})
        r5 = results_map.get((cid, 5), {})
        r10 = results_map.get((cid, 10), {})
        if r10:
            print(f"  {cid:>6} │ {r1.get('recall',0):>9.4f} │ {r5.get('recall',0):>9.4f} │ {r10.get('recall',0):>10.4f} │ {r10.get('mrr',0):>8.4f} │ {r10.get('s1_hit_rate',0)*100:>5.1f}%")
    print()

    # ── Direct Comparison: full query vs title-only ──
    print("─" * 95)
    print("  DIREKTVERGLEICH: Full Query vs Title-Only (gleiche Architektur)")
    print("─" * 95)
    print()

    v12b = results_map.get(("V12b", 10), {})
    v15a = results_map.get(("V15a", 10), {})
    v14a = results_map.get(("V14a", 10), {})
    v15b = results_map.get(("V15b", 10), {})

    if v12b and v15a:
        print(f"  Class/File BM25 (N=20) + Terms Filter:")
        print(f"    Full Query (V12b):   Recall={v12b['recall']:.4f}  S1={v12b['s1_hit_rate']*100:.1f}%")
        print(f"    Title Only (V15a):   Recall={v15a['recall']:.4f}  S1={v15a['s1_hit_rate']*100:.1f}%")
        delta = v15a['recall'] - v12b['recall']
        s1_delta = (v15a['s1_hit_rate'] - v12b['s1_hit_rate']) * 100
        print(f"    Δ Recall@10: {delta:+.4f}  |  Δ S1 Hit Rate: {s1_delta:+.1f}pp")
        print()

    if v14a and v15b:
        print(f"  Virtual-Doc BM25 (N=20) + Terms Filter:")
        print(f"    Full Query (V14a):   Recall={v14a['recall']:.4f}  S1={v14a['s1_hit_rate']*100:.1f}%")
        print(f"    Title Only (V15b):   Recall={v15b['recall']:.4f}  S1={v15b['s1_hit_rate']*100:.1f}%")
        delta = v15b['recall'] - v14a['recall']
        s1_delta = (v15b['s1_hit_rate'] - v14a['s1_hit_rate']) * 100
        print(f"    Δ Recall@10: {delta:+.4f}  |  Δ S1 Hit Rate: {s1_delta:+.1f}pp")
        print()

    # ── Summary ──
    print("─" * 95)
    print("  ZUSAMMENFASSUNG")
    print("─" * 95)
    print()
    print(f"  Bester Gesamtsieger: {best_cid} (Recall@10 = {best_recall:.4f})")
    if v15a and v12b:
        if v15a["recall"] > v12b["recall"]:
            print(f"  → Title-Only verbessert BM25-Retrieval (weniger Noise = präzisere Terme)")
        else:
            print(f"  → Full Query bleibt besser (mehr Kontext = mehr Recall-Chancen)")
    print()
    print("=" * 95)

    # ── Save report to file ──
    report_file = v15_results_path / "comparison_v15.txt"
    lines_out = []
    lines_out.append("=" * 95)
    lines_out.append("  V15 VERGLEICHSREPORT – Title-Only Query (pandas, 95 Queries)")
    lines_out.append("=" * 95)
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
    lines_out.append(f"  Direktvergleich Full vs Title-Only:")
    if v12b and v15a:
        lines_out.append(f"    V12b (full):  Recall={v12b['recall']:.4f}  S1={v12b['s1_hit_rate']*100:.1f}%")
        lines_out.append(f"    V15a (title): Recall={v15a['recall']:.4f}  S1={v15a['s1_hit_rate']*100:.1f}%")
    if v14a and v15b:
        lines_out.append(f"    V14a (full):  Recall={v14a['recall']:.4f}  S1={v14a['s1_hit_rate']*100:.1f}%")
        lines_out.append(f"    V15b (title): Recall={v15b['recall']:.4f}  S1={v15b['s1_hit_rate']*100:.1f}%")
    lines_out.append("")
    lines_out.append(f"  Bester: {best_cid} (Recall@10 = {best_recall:.4f})")
    lines_out.append("=" * 95)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
    print(f"\n  Report gespeichert: {report_file}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V15 Benchmark + Vergleich")
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
║  V15 – Title-Only Query Preprocessing                      ║
║  Repo: pandas | Query: nur Issue-Titel (1. Zeile)          ║
║  V15a: Class/File BM25 (N=20) + Terms-Filter + Title-Only  ║
║  V15b: Virtual-Doc BM25 (N=20) + Terms-Filter + Title-Only ║
╚══════════════════════════════════════════════════════════════╝
""")
        start = time.time()

        run_benchmark(
            conditions=V15_CONDITIONS,
            k_values=args.k,
            repos=[REPO],
            dataset_path=str(ROOT / "benchmark" / "data" / "benchmark_dataset.json"),
            output_dir=OUTPUT_DIR,
            es_url=args.es_url,
            verbose=args.verbose,
        )

        elapsed = time.time() - start
        print(f"\n⏱  V15 Laufzeit: {elapsed/60:.1f} Minuten")

    generate_comparison_report(output_path)


if __name__ == "__main__":
    main()
