#!/usr/bin/env python3
"""
run_code_embed_v11plus.py – Code-spezialisiertes Embedding vs. bge-base (ab V11).

Läuft alle V11+-Conditions noch einmal, aber mit einem code-spezialisierten
Stage-2-Embedding statt B (bge-base-en-v1.5, 768d). Wählbar via --model:

  jina  (Default): jinaai/jina-embeddings-v2-base-code  (Embedding E, ALiBi,
                   läuft stabil auf MPS/CPU)  → Conditions mit Suffix "_JINA"
  sfr            : Salesforce/SFR-Embedding-Code-400M_R (Embedding D)
                   → Suffix "_SFR".  ACHTUNG: GTE-new-impl crasht auf MPS/CPU,
                   nur auf CUDA-Hardware sinnvoll.

Die Varianten werden in config.py automatisch generiert (eigene ES-Indizes,
keine Kollision mit den bge-Läufen). Der Report stellt jede Variante ihrem
bge-Pendant gegenüber (z. B. V16c_JINA vs. V16c) und der Bestmarke V16b/V16c.

Voraussetzungen (auf dem Mac, nicht in der Sandbox):
  - Elasticsearch läuft lokal (http://localhost:9200)
  - sentence-transformers>=2.7.0
  - Erster Lauf lädt das Modell von HuggingFace (jina-code ~0.3 GB).

Aufruf:
  # Empfohlen: jina, nur die bisher beste Familie (V16)
  python scripts/run_code_embed_v11plus.py

  # Voller Sweep ab V11
  python scripts/run_code_embed_v11plus.py --all

  # Konkrete Conditions / anderes Modell
  python scripts/run_code_embed_v11plus.py --conditions V16c_JINA
  python scripts/run_code_embed_v11plus.py --model sfr --all

  # Nur Report neu erzeugen
  python scripts/run_code_embed_v11plus.py --report-only
"""

import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.runner import run_benchmark
from benchmark.config import (
    CONDITIONS_MAP, SFR_CONDITION_IDS, JINA_CONDITION_IDS,
    QWEN3_CONDITION_IDS, CODESEARCH_CONDITION_IDS, QWEN3_4B_CONDITION_IDS,
)


REPO = "pandas"


def _v16(suffix):
    return [f"V16a{suffix}", f"V16b{suffix}", f"V16c{suffix}"]


MODELS = {
    # NATIVE Modelle (kein trust_remote_code) → laufen auf neuem transformers/py3.13.
    "qwen3": {
        "suffix": "_QWEN3",
        "ids": QWEN3_CONDITION_IDS,
        "output": str(ROOT / "benchmark" / "results" / "qwen3_v11plus"),
        "name": "Qwen3-Embedding-0.6B (F, 1024d, nativ)",
        "default": _v16("_QWEN3"),
    },
    "codesearch": {
        "suffix": "_CODESEARCH",
        "ids": CODESEARCH_CONDITION_IDS,
        "output": str(ROOT / "benchmark" / "results" / "codesearch_v11plus"),
        "name": "st-codesearch-distilroberta-base (G, 768d, nativ)",
        "default": _v16("_CODESEARCH"),
    },
    "qwen3-4b": {
        "suffix": "_QWEN34B",
        "ids": QWEN3_4B_CONDITION_IDS,
        "output": str(ROOT / "benchmark" / "results" / "qwen3_4b_v11plus"),
        "name": "Qwen3-Embedding-4B (H, 2560d, nativ, stärker)",
        "default": _v16("_QWEN34B"),
    },
    # trust_remote_code-Modelle (nur mit gepinntem transformers nutzbar).
    "jina": {
        "suffix": "_JINA",
        "ids": JINA_CONDITION_IDS,
        "output": str(ROOT / "benchmark" / "results" / "jina_v11plus"),
        "name": "jina-embeddings-v2-base-code (E, 768d)",
        "default": _v16("_JINA"),
    },
    "sfr": {
        "suffix": "_SFR",
        "ids": SFR_CONDITION_IDS,
        "output": str(ROOT / "benchmark" / "results" / "sfr_v11plus"),
        "name": "SFR-Embedding-Code-400M_R (D, 1024d)",
        "default": _v16("_SFR"),
    },
}

# Alle bisherigen bge-base-Ergebnisse als Vergleichsbasis (dedizierte
# Per-Version-Ordner zuerst → "first wins"; top-level Aggregat nur Fallback).
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

_SUFFIXES = ("_QWEN34B", "_CODESEARCH", "_QWEN3", "_JINA", "_SFR")


def _load_runs(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("runs", [])


def _build_repo_map(runs):
    """(condition_id, k, repo_id) -> Metriken pro Repo.

    Der Runner speichert ein RunResult pro (condition, k, repo). Wir behalten
    die Werte je Repo getrennt (Micro-Schnitt pro Repo über dessen Samples).
    """
    out = {}
    for run in runs:
        cid, k = run["condition_id"], run["k"]
        repo = run.get("repo_id", "?")
        m = run.get("metrics", {})
        ps = m.get("per_sample", []) or []
        if ps:
            n = len(ps)
            rec = sum(s.get("recall_at_k", 0.0) for s in ps) / n
            mrr = sum(s.get("mrr_at_k", 0.0) for s in ps) / n
            s1 = sum(1 for s in ps if s.get("stage1_hit")) / n
        else:
            n = m.get("num_samples", 0) or 0
            rec, mrr, s1 = m.get("recall_at_k", 0.0), m.get("mrr_at_k", 0.0), 0.0
        key = (cid, k, repo)
        if key not in out or n > out[key]["num_samples"]:
            out[key] = {"recall": rec, "mrr": mrr, "s1_hit_rate": s1, "num_samples": n}
    return out


def _overall(repo_map, cid, k, repos=None):
    """Micro-Schnitt über die (gegebenen) Repos einer Condition."""
    rs = ms = 0.0
    tot = 0
    for (c, kk, repo), v in repo_map.items():
        if c == cid and kk == k and (repos is None or repo in repos):
            rs += v["recall"] * v["num_samples"]
            ms += v["mrr"] * v["num_samples"]
            tot += v["num_samples"]
    if tot == 0:
        return None
    return {"recall": rs / tot, "mrr": ms / tot, "num_samples": tot}


def _base_id(cid):
    for suf in _SUFFIXES:
        if cid.endswith(suf):
            return cid[: -len(suf)]
    return cid


def generate_comparison_report(results_path: Path, model_name: str):
    result_files = sorted(results_path.glob("benchmark_results_*.json"))
    result_files = [f for f in result_files
                    if "PARTIAL" not in f.name and "latest" not in f.name]
    if not result_files:
        print("\n  Keine neuen Ergebnisse gefunden.")
        return
    new_runs = _load_runs(result_files[-1])

    # bge-Baseline pro (cid,k,repo): alte Ergebnisse (pandas) + frische bge aus
    # DEMSELBEN Lauf (haben Vorrang → identische Bedingungen, alle Repos).
    base_repo = {}
    for prev_file in PREV_RESULTS_FILES:
        for key, val in _build_repo_map(_load_runs(prev_file)).items():
            base_repo.setdefault(key, val)
    base_repo.update(_build_repo_map(
        [r for r in new_runs if not r["condition_id"].endswith(_SUFFIXES)]))

    code_repo = _build_repo_map(
        [r for r in new_runs if r["condition_id"].endswith(_SUFFIXES)])
    code_ids = sorted({c for (c, _k, _r) in code_repo})
    ks = sorted({k for (_c, k, _r) in code_repo})

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    def fmt(x, w, plus=False):
        if x is None:
            return f"{'—':>{w}}"
        spec = f">{'+' if plus else ''}{w}.4f"
        return format(x, spec)

    out("=" * 104)
    out(f"  {model_name} vs. bge-base – Werte PRO REPO")
    out("  Nur das Embedding unterscheidet sich (gleiche Condition). bge = frische")
    out("  Baseline aus demselben Lauf (sofern mitgelaufen), sonst '—'.")
    out("=" * 104)

    HEAD = (f"  {'Repo':<10} {'n':>4} {'Recall(code)':>13} {'Recall(bge)':>12} "
            f"{'ΔR':>9}   {'MRR(code)':>10} {'MRR(bge)':>10} {'ΔMRR':>9}")

    for cid in code_ids:
        base = _base_id(cid)
        for k in ks:
            repos = [r for (c, kk, r) in code_repo if c == cid and kk == k]
            if not repos:
                continue
            repos = sorted(set(repos),
                           key=lambda r: -code_repo[(cid, k, r)]["num_samples"])
            out("")
            out(f"  ── {cid}  vs.  {base}  (k={k}) ──")
            out(HEAD)
            out(f"  {'-'*10} {'-'*4} {'-'*13} {'-'*12} {'-'*9}   {'-'*10} {'-'*10} {'-'*9}")
            for repo in repos:
                s = code_repo[(cid, k, repo)]
                b = base_repo.get((base, k, repo))
                br = b["recall"] if b else None
                bm = b["mrr"] if b else None
                dr = (s["recall"] - br) if br is not None else None
                dm = (s["mrr"] - bm) if bm is not None else None
                out(f"  {repo:<10} {s['num_samples']:>4} {fmt(s['recall'],13)} "
                    f"{fmt(br,12)} {fmt(dr,9,True)}   "
                    f"{fmt(s['mrr'],10)} {fmt(bm,10)} {fmt(dm,9,True)}")
            # Σ über alle gelaufenen Repos (sample-gewichtet)
            co = _overall(code_repo, cid, k, repos)
            bo = _overall(base_repo, base, k, repos)
            br = bo["recall"] if bo else None
            bm = bo["mrr"] if bo else None
            dr = (co["recall"] - br) if (co and br is not None) else None
            dm = (co["mrr"] - bm) if (co and bm is not None) else None
            out(f"  {'-'*10} {'-'*4} {'-'*13} {'-'*12} {'-'*9}   {'-'*10} {'-'*10} {'-'*9}")
            out(f"  {'Σ alle':<10} {co['num_samples']:>4} {fmt(co['recall'],13)} "
                f"{fmt(br,12)} {fmt(dr,9,True)}   "
                f"{fmt(co['mrr'],10)} {fmt(bm,10)} {fmt(dm,9,True)}")
    out("")
    out("=" * 104)

    report_file = results_path / "comparison_code_embed.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Code-spezialisiertes Embedding vs. bge-base (ab V11)"
    )
    parser.add_argument("--model",
                        choices=["qwen3", "qwen3-4b", "codesearch", "jina", "sfr"],
                        default="qwen3",
                        help="qwen3/qwen3-4b/codesearch = nativ (kein Setup nötig); "
                             "jina/sfr = trust_remote_code (gepinnte transformers-Env nötig)")
    parser.add_argument("--es-url", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--all", action="store_true",
                        help="Voller Sweep über ALLE Embedding-Conditions (V3–V19)")
    parser.add_argument("--preset", choices=["core", "v16"], default=None,
                        help="core = Dense (V3,V8) + Hybrid (V5,V10) + alle V11–V19; "
                             "v16 = nur V16-Familie")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--repos", nargs="*", default=["pandas"],
                        help="Repos (default: nur pandas, konsistent mit Baselines)")
    parser.add_argument("--all-repos", action="store_true",
                        help="Alle 5 Repos (206 Queries; klont fehlende automatisch, deutlich länger)")
    args = parser.parse_args()

    # repos=None lässt run_benchmark ALLE Repos laufen.
    repos = None if args.all_repos else args.repos

    cfg = MODELS[args.model]
    suffix = cfg["suffix"]

    # Preset "core": Dense (V3,V8) + Hybrid (V5,V10) + alle V11–V19. Nur function-
    # und ast-Embeddings → cache-freundlich (wenige Encode-Pässe pro Modell/Repo).
    CORE_BASE = [
        "V3", "V8",                       # Flat Dense (function, ast)
        "V5a", "V5b", "V5c",              # Flat Hybrid (function, α-Sweep)
        "V10a", "V10b", "V10c",           # Flat Hybrid (ast, α-Sweep)
        "V11a", "V11b", "V11c", "V12a", "V12b", "V12c", "V13",
        "V14a", "V14b", "V16a", "V16b", "V16c",
        "V17a", "V17b", "V17c", "V18a", "V18b", "V19a", "V19b",
    ]

    if args.conditions:
        conditions = args.conditions
    elif args.preset == "core":
        conditions = [f"{b}{suffix}" for b in CORE_BASE]
    elif args.preset == "v16":
        conditions = _v16(suffix)
    elif args.all:
        conditions = cfg["ids"]
    else:
        conditions = cfg["default"]

    # bge-Baseline pro Code-Variante automatisch mitnehmen (frisch, identische
    # Bedingungen) → der Report hat immer eine Vergleichsbasis pro (cid,k,repo).
    expanded = list(conditions)
    for c in conditions:
        b = _base_id(c)
        if b != c and b in CONDITIONS_MAP and b not in expanded:
            expanded.append(b)
    conditions = expanded

    unknown = [c for c in conditions if c not in CONDITIONS_MAP]
    if unknown:
        print(f"   Unbekannte Conditions (ignoriert): {unknown}")
        conditions = [c for c in conditions if c in CONDITIONS_MAP]

    output_path = Path(cfg["output"])
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        print(f"""
╔════════════════════════════════════════════════════════════════════╗
║  Code-Embedding: {cfg['name']:<48}║
║  vs. bge-base (B, 768d) — Repo: pandas (95 Queries)                  ║
║  Conditions: {len(conditions):>2} Stück                                          ║
╚════════════════════════════════════════════════════════════════════╝
""")
        print(f"  Conditions: {conditions}")
        print(f"  Repos: {'ALLE (5, 206 Queries)' if repos is None else repos}\n")
        start = time.time()
        run_benchmark(
            conditions=conditions,
            k_values=args.k,
            repos=repos,
            dataset_path=str(ROOT / "benchmark" / "data" / "benchmark_dataset.json"),
            output_dir=cfg["output"],
            es_url=args.es_url,
            verbose=args.verbose,
        )
        print(f"\n  Laufzeit: {(time.time() - start) / 60:.1f} Minuten")

    generate_comparison_report(output_path, cfg["name"])


if __name__ == "__main__":
    main()
