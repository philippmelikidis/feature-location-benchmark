#!/usr/bin/env python3
"""
run_v20_largeN.py – V20: Stage-1 Top-N-Sweep + Qwen3-4B-Reranking.

Hypothese (aus dem V11-Befund): Die grobe Stufe wirft relevante Dateien zu früh
raus. Bei V11 steigt der Recall mit N (N=10→20→40: 0,472→0,498→0,579), bleibt
aber unter der flachen Suche (V8/V10b ≈ 0,610). V20 hält das beste V16c-Design
(LLM-expandierte VDoc-BM25-Grobstufe → Hybrid-Feinstufe mit Terms-Filter) bei,
tauscht das Stage-2-Embedding auf H (Qwen3-Embedding-4B) und fährt einen N-Sweep
{20, 40, 80, 150} in ZWEI Grobstufen-Familien:
  - V20a–d: sparse Grobstufe (BM25 auf LLM-Exp)        — V16-Design
  - V20e–h: hybride Grobstufe (α·BM25 + Dense/Qwen3-4B) — testet "Hybrid statt
            nur sparse hebt den Stage-1-Recall" (Dense-Leg = Qwen3-4B → teurer).

Gemessen / berichtet:
  1. Stage-1-Datei-Recall pro N (loggt der Runner als stage1_hit) → gegen N
     aufgetragen; das N, ab dem Stage-1-Recall ≈ flache Suche, wird markiert.
  2. End-to-end Recall@1/5/10 + MRR@1/5/10 pro N.
  3. Vergleich gegen flache Suche (V8/V10b) und Top-20 (V11b).
  4. Laufzeit-Overhead (p50-Query-Latenz) pro N — Trade-off Recall vs. Zeit.

Voraussetzungen (auf dem Mac, NICHT in der Sandbox):
  - Elasticsearch läuft lokal (http://localhost:9200)
  - Qwen3-Embedding-4B lädt beim ersten Lauf (~8 GB) → EMBED_BATCH_SIZE=8 setzen.
  - Pre-computed LLM-Expansions vorhanden (Stage-1 nutzt sie):
      benchmark/data/llm_expansions_pandas.json
    Falls nicht: erst `python scripts/precompute_llm_expansions.py`.

Aufruf:
  # V20a–d auf pandas (Baselines werden aus früheren Ergebnissen gezogen)
  EMBED_BATCH_SIZE=8 python scripts/run_v20_largeN.py

  # Baselines V8/V10b/V11b im selben Lauf frisch mitfahren (identische Bedingungen)
  EMBED_BATCH_SIZE=8 python scripts/run_v20_largeN.py --with-baselines

  # Nur Report aus vorhandenen Ergebnissen neu erzeugen
  python scripts/run_v20_largeN.py --report-only
"""

import sys
import json
import time
import argparse
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# NB: run_benchmark wird erst in main() importiert (zieht den vollen Retrieval-
# Stack inkl. ES/Embeddings). So läuft --report-only ohne diese Abhängigkeiten.
from benchmark.config import (
    CONDITIONS_MAP, V20_CONDITION_IDS, V20_SPARSE_IDS, V20_HYBRID_IDS,
)


REPO = "pandas"
DEFAULT_DATASET = ROOT / "benchmark" / "data" / "benchmark_dataset.json"
OUTPUT_DIR = ROOT / "benchmark" / "results" / "v20_largeN"


def _output_dir_for_dataset(dataset_path: Path) -> Path:
    """v1 (Default-Dataset) behaelt den bestehenden Ordner unveraendert;
    jedes andere Dataset (z.B. v2) bekommt einen eigenen Unterordner nach
    seinem Dateinamen, damit sich v1- und v2-Ergebnisse nie vermischen
    (dieselbe condition_id existiert in beiden)."""
    if dataset_path.resolve() == DEFAULT_DATASET.resolve():
        return OUTPUT_DIR
    return OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_{dataset_path.stem}"

# Vergleichsbaselines (bge-base, Embedding B): flache Suche + Top-20-Hierarchie.
FLAT_BASELINES = ["V8", "V10b"]   # flat dense ast / flat hybrid ast
HIER_BASELINE = "V11b"            # hierarchical sparse class/file → hybrid, N=20
BASELINE_IDS = FLAT_BASELINES + [HIER_BASELINE]

# Frühere bge-Ergebnisse als Fallback-Quelle für die Baselines (first wins).
PREV_RESULTS_FILES = [
    ROOT / "benchmark" / "results" / "v11_vs_v12" / "benchmark_results_latest.json",
    ROOT / "benchmark" / "results" / "v13" / "benchmark_results_latest.json",
    ROOT / "benchmark" / "results" / "v14" / "benchmark_results_latest.json",
    ROOT / "benchmark" / "results" / "v15" / "benchmark_results_latest.json",
    ROOT / "benchmark" / "results" / "v16_llm_expansion" / "benchmark_results_latest.json",
    ROOT / "benchmark" / "results" / "v17_v18" / "benchmark_results_latest.json",
    ROOT / "benchmark" / "results" / "v19" / "benchmark_results_latest.json",
    ROOT / "benchmark" / "results" / "benchmark_results_latest.json",
]


# ── Result loading / aggregation ──────────────────────────────

def _load_runs(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("runs", [])


def _agg_condition(runs, cid, k, repo=REPO):
    """Micro-Schnitt über die Samples einer (condition,k,repo).

    Liefert recall, mrr, Stage-1-Recall (mean stage1_hit), p50-Latenz und n.
    Nimmt den RunResult mit den meisten Samples (robust gegen Teilläufe).
    Latenz aus retrieval_results[*].retrieval_time_ms.
    """
    best = None
    for run in runs:
        if run["condition_id"] != cid or run["k"] != k:
            continue
        if repo is not None and run.get("repo_id", repo) != repo:
            continue
        m = run.get("metrics", {})
        ps = m.get("per_sample", []) or []
        n = len(ps) if ps else (m.get("num_samples", 0) or 0)
        if best is None or n > best["n"]:
            if ps:
                recall = sum(s.get("recall_at_k", 0.0) for s in ps) / n
                mrr = sum(s.get("mrr_at_k", 0.0) for s in ps) / n
                s1_vals = [s.get("stage1_hit") for s in ps if s.get("stage1_hit") is not None]
                s1 = (sum(1 for v in s1_vals if v) / len(s1_vals)) if s1_vals else None
            else:
                recall, mrr, s1 = m.get("recall_at_k", 0.0), m.get("mrr_at_k", 0.0), None
            lat = [rr.get("retrieval_time_ms", 0.0)
                   for rr in run.get("retrieval_results", [])
                   if rr.get("retrieval_time_ms")]
            p50 = statistics.median(lat) if lat else None
            best = {"recall": recall, "mrr": mrr, "s1": s1, "p50_ms": p50, "n": n}
    return best


def _baseline_lookup(new_runs, with_baselines):
    """(cid,k) -> agg. Frische Baselines aus dem Lauf haben Vorrang, sonst Prev."""
    out = {}
    if with_baselines:
        for cid in BASELINE_IDS:
            for k in (1, 5, 10):
                a = _agg_condition(new_runs, cid, k)
                if a:
                    out[(cid, k)] = a
    for prev in PREV_RESULTS_FILES:
        runs = _load_runs(prev)
        for cid in BASELINE_IDS:
            for k in (1, 5, 10):
                if (cid, k) in out:
                    continue
                a = _agg_condition(runs, cid, k)
                if a:
                    out[(cid, k)] = a
    return out


def _fmt(x, w, plus=False, pct=False):
    if x is None:
        return f"{'—':>{w}}"
    if pct:
        return f"{x*100:>{w}.1f}"
    spec = f">{'+' if plus else ''}{w}.4f"
    return format(x, spec)


# ── Report ────────────────────────────────────────────────────

def generate_report(with_baselines: bool, output_dir: Path = OUTPUT_DIR):
    result_files = sorted(output_dir.glob("benchmark_results_*.json"))
    result_files = [f for f in result_files
                    if "PARTIAL" not in f.name and "latest" not in f.name]
    if not result_files:
        print("\n  Keine V20-Ergebnisse gefunden.")
        return
    new_runs = _load_runs(result_files[-1])

    base = _baseline_lookup(new_runs, with_baselines)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # Zwei Grobstufen-Familien über denselben N-Sweep.
    families = [
        ("Sparse-Grobstufe (BM25 auf LLM-Exp)", V20_SPARSE_IDS),
        ("Hybrid-Grobstufe (α·BM25 + Dense/Qwen3-4B)", V20_HYBRID_IDS),
    ]

    def by_n(ids):
        return sorted([c for c in ids if c in CONDITIONS_MAP],
                      key=lambda c: CONDITIONS_MAP[c].top_n_files)

    out("=" * 100)
    out("  V20 – Stage-1 Top-N-Sweep + Qwen3-4B-Reranking  (Repo: pandas)")
    out("  Coarse: LLM-exp VDoc-BM25 (sparse) vs. Hybrid  ·  Fine: Hybrid + Qwen3-4B (H)")
    out("=" * 100)

    # Referenz: flache Suche (Recall@10) als Zielmarke für Stage-1-Recall.
    flat10 = [base[(c, 10)]["recall"] for c in FLAT_BASELINES if (c, 10) in base]
    flat_target = max(flat10) if flat10 else None
    v11b10 = base.get((HIER_BASELINE, 10))

    out("")
    out("  Referenzen (bge-base, Embedding B):")
    for c in FLAT_BASELINES:
        a = base.get((c, 10))
        if a:
            out(f"    flache Suche {c:<5} Recall@10={a['recall']:.4f}  MRR@10={a['mrr']:.4f}")
    if v11b10:
        out(f"    Top-20      {HIER_BASELINE:<5} Recall@10={v11b10['recall']:.4f}  "
            f"MRR@10={v11b10['mrr']:.4f}  Stage1-Recall={_fmt(v11b10['s1'],5)}")
    if flat_target is not None:
        out(f"    → Zielmarke Stage-1-Recall ≈ flache Suche Recall@10 = {flat_target:.4f}")

    # 1) Stage-1-Recall vs N + Latenz — pro Familie.
    out("")
    out("  ── (1) Stage-1-Datei-Recall & Latenz vs. N ──")
    for label, ids in families:
        sweep = by_n(ids)
        if not sweep:
            continue
        out("")
        out(f"    [{label}]")
        out(f"    {'Cond':<6} {'N':>4} {'Stage1-Recall':>14} {'≥Ziel?':>7} {'p50-Latenz[ms]':>15}")
        out(f"    {'-'*6} {'-'*4} {'-'*14} {'-'*7} {'-'*15}")
        n_reaches = None
        for c in sweep:
            N = CONDITIONS_MAP[c].top_n_files
            a = _agg_condition(new_runs, c, 10)
            s1 = a["s1"] if a else None
            reach = ""
            if s1 is not None and flat_target is not None:
                ok = s1 >= flat_target
                reach = "JA" if ok else "nein"
                if ok and n_reaches is None:
                    n_reaches = N
            p50 = a["p50_ms"] if a else None
            p50s = _fmt(p50, 15) if p50 is not None else f"{'—':>15}"
            out(f"    {c:<6} {N:>4} {_fmt(s1,14):>14} {reach:>7} {p50s}")
        if n_reaches is not None:
            out(f"    → Stage-1-Recall erreicht die flache Suche ab N = {n_reaches}.")
        elif flat_target is not None:
            out(f"    → erreicht die Zielmarke ({flat_target:.4f}) im Sweep nicht.")

    # 2) End-to-end Recall/MRR @1/5/10 vs N (+ Δ gegen flache Suche & V11b @10).
    out("")
    out("  ── (2) End-to-end Recall@k / MRR@k vs. N ──")
    HEAD = (f"    {'Cond':<6} {'N':>4} "
            f"{'R@1':>7} {'R@5':>7} {'R@10':>7}   "
            f"{'M@1':>7} {'M@5':>7} {'M@10':>7}   "
            f"{'ΔR@10 flat':>11} {'ΔR@10 V11b':>11}")
    for label, ids in families:
        sweep = by_n(ids)
        if not sweep:
            continue
        out("")
        out(f"    [{label}]")
        out(HEAD)
        out("    " + "-" * (len(HEAD) - 4))
        for c in sweep:
            N = CONDITIONS_MAP[c].top_n_files
            a = {k: _agg_condition(new_runs, c, k) for k in (1, 5, 10)}
            r1 = a[1]["recall"] if a[1] else None
            r5 = a[5]["recall"] if a[5] else None
            r10 = a[10]["recall"] if a[10] else None
            m1 = a[1]["mrr"] if a[1] else None
            m5 = a[5]["mrr"] if a[5] else None
            m10 = a[10]["mrr"] if a[10] else None
            d_flat = (r10 - flat_target) if (r10 is not None and flat_target is not None) else None
            d_v11b = (r10 - v11b10["recall"]) if (r10 is not None and v11b10) else None
            out(f"    {c:<6} {N:>4} "
                f"{_fmt(r1,7)} {_fmt(r5,7)} {_fmt(r10,7)}   "
                f"{_fmt(m1,7)} {_fmt(m5,7)} {_fmt(m10,7)}   "
                f"{_fmt(d_flat,11,True)} {_fmt(d_v11b,11,True)}")

    # 3) Hybrid- vs. Sparse-Grobstufe direkt pro N (testet "Hybrid hebt Stufe 1").
    out("")
    out("  ── (3) Hybrid- vs. Sparse-Grobstufe (gleiches N) ──")
    out(f"    {'N':>4} {'ΔStage1':>9} {'ΔR@10':>9} {'ΔMRR@10':>9}   {'Δp50[ms]':>10}")
    out(f"    {'-'*4} {'-'*9} {'-'*9} {'-'*9}   {'-'*10}")
    for cs, ch in zip(by_n(V20_SPARSE_IDS), by_n(V20_HYBRID_IDS)):
        N = CONDITIONS_MAP[cs].top_n_files
        s = _agg_condition(new_runs, cs, 10)
        h = _agg_condition(new_runs, ch, 10)
        ds = (h["s1"] - s["s1"]) if (h and s and h["s1"] is not None and s["s1"] is not None) else None
        dr = (h["recall"] - s["recall"]) if (h and s) else None
        dm = (h["mrr"] - s["mrr"]) if (h and s) else None
        dp = (h["p50_ms"] - s["p50_ms"]) if (h and s and h["p50_ms"] and s["p50_ms"]) else None
        out(f"    {N:>4} {_fmt(ds,9,True)} {_fmt(dr,9,True)} {_fmt(dm,9,True)}   "
            f"{(_fmt(dp,10,True) if dp is not None else f'{chr(8212):>10}')}")

    out("")
    out("  Lesart: ΔR@10 flat = Recall@10(V20) − beste flache Suche (V8/V10b).")
    out("          ΔR@10 V11b = Recall@10(V20) − Top-20-Hierarchie (V11b).")
    out("          (3) Δ = Hybrid-Grobstufe − Sparse-Grobstufe bei gleichem N (>0 = Hybrid besser).")
    out("          Stage-1-Recall = Anteil Queries, deren GT-Datei in der N-Kandidatenmenge liegt.")
    out("=" * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "comparison_v20_largeN.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {report_file}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V20: Stage-1 Top-N-Sweep + Qwen3-4B-Reranking"
    )
    parser.add_argument("--es-url", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--with-baselines", action="store_true",
                        help="V8/V10b/V11b (bge) im selben Lauf frisch mitfahren "
                             "(identische Bedingungen, aber langsamer).")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Override (Default: V20a–d).")
    parser.add_argument("--repos", nargs="*", default=[REPO])
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET),
                        help="Pfad zum Dataset-JSON (Default: v1). "
                             "Fuer v2: benchmark/data/benchmark_dataset_v2.json")
    args = parser.parse_args()

    conditions = args.conditions or list(V20_CONDITION_IDS)
    if args.with_baselines:
        conditions = conditions + BASELINE_IDS

    unknown = [c for c in conditions if c not in CONDITIONS_MAP]
    if unknown:
        print(f"   Unbekannte Conditions (ignoriert): {unknown}")
        conditions = [c for c in conditions if c in CONDITIONS_MAP]

    dataset_path = Path(args.dataset)
    output_dir = _output_dir_for_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        from benchmark.runner import run_benchmark  # lazy: nur für echte Läufe
        ns = [CONDITIONS_MAP[c].top_n_files for c in conditions if c in V20_CONDITION_IDS]
        print(f"""
╔════════════════════════════════════════════════════════════════════╗
║  V20 – Stage-1 Top-N-Sweep + Qwen3-4B (H, 2560d)                     ║
║  Grobstufe: LLM-exp VDoc-BM25  ·  Feinstufe: Hybrid + Qwen3-4B       ║
║  N-Sweep: {str(sorted(set(ns))):<54}║
║  Dataset: {dataset_path.name:<54}║
╚════════════════════════════════════════════════════════════════════╝
""")
        print(f"  Conditions: {conditions}")
        print(f"  Repos: {args.repos}\n")
        start = time.time()
        run_benchmark(
            conditions=conditions,
            k_values=args.k,
            repos=args.repos,
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
            es_url=args.es_url,
            verbose=args.verbose,
        )
        print(f"\n  Laufzeit: {(time.time() - start) / 60:.1f} Minuten")

    generate_report(with_baselines=args.with_baselines, output_dir=output_dir)


if __name__ == "__main__":
    main()
