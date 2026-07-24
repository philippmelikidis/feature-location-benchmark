#!/usr/bin/env python3
"""
probe_llm_rerank.py – SCHRITT 0b: Schafft ein LLM-Listwise-Reranker die Konversion?

Nachdem der generische Cross-Encoder (bge-reranker-v2-m3) die Fine-Recovery NICHT
über die 0.55 des Qwen-Hybrids gehoben hat (Ziel für Recall@10≥0.80: ~0.87),
testet dieses Skript den vielversprechenderen Weg:

    Ein LLM bekommt die N Stage-1-Kandidat-Dateien (je mit Symbol-Übersicht) plus
    den Issue-Text und wählt in EINEM Call die Top-10 Dateien aus.

Gleiche Logik/Referenz wie probe_rerank_finerecovery.py, nur die Stufe 2.5 ist
das LLM statt eines Cross-Encoders. Eval ist DATEI-basiert (pandas-Ziele haben
keine function_names) → direkt vergleichbar mit der V20-Referenz.

VORAUSSETZUNGEN (auf deiner Maschine):
  - Elasticsearch läuft (Stage-1 BM25).
  - LM Studio läuft mit einem geladenen Modell (Default http://localhost:1234/v1).
  - benchmark/data/llm_expansions_pandas.json vorhanden.

AUFRUF:
  python scripts/probe_llm_rerank.py --limit 25
  python scripts/probe_llm_rerank.py --llm-url http://localhost:1234/v1 --out benchmark/results/probe_llm.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ für Reuse

# Harness aus dem Cross-Encoder-Probe wiederverwenden (identische Stage-1-Logik).
from probe_rerank_finerecovery import (
    _build_light, _stage1_candidates, _gather_candidate_chunks, V20_REFERENCE,
)

TARGET_RECALL = 0.80


def _file_outline(chunks, max_symbols=12):
    """Kompakte Symbol-Übersicht einer Datei (Klassen + Funktionen)."""
    classes, funcs = [], []
    for c in chunks:
        cn = getattr(c, "class_name", None)
        fn = getattr(c, "function_name", None)
        if cn and cn not in classes:
            classes.append(cn)
        if fn and fn not in funcs:
            funcs.append(fn)
    parts = []
    if classes:
        parts.append("class " + ", ".join(classes[:max_symbols]))
    if funcs:
        parts.append("def " + ", ".join(funcs[:max_symbols]))
    return "; ".join(parts) if parts else "(keine Symbole)"


def main():
    ap = argparse.ArgumentParser(description="Schritt-0b LLM-Listwise-Reranker Probe")
    ap.add_argument("--condition", default="V20c")
    ap.add_argument("--repo", default="pandas")
    ap.add_argument("--llm-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default="local-model")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-files-prompt", type=int, default=80,
                    help="Max. Kandidat-Dateien im Prompt (Kontext-Limit kleiner Modelle).")
    ap.add_argument("--max-symbols", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--es-url", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("EMBED_BATCH_SIZE", "8")

    from benchmark.config import CONDITIONS_MAP, REPOS_MAP
    from benchmark.runner import create_retriever, clone_repository, get_python_files, load_dataset
    from benchmark.metrics import recall_at_k, _file_matches
    from benchmark.reranker import LLMListwiseReranker

    if args.condition not in CONDITIONS_MAP:
        print(f"Unbekannte Condition: {args.condition}")
        sys.exit(1)
    condition = CONDITIONS_MAP[args.condition]
    N = getattr(condition, "top_n_files", "?")
    repo_config = REPOS_MAP[args.repo]
    reference = V20_REFERENCE.get((args.condition, args.repo))

    print("=" * 78)
    print("  SCHRITT 0b – LLM-Listwise-Reranker Probe")
    print(f"  Condition={args.condition} (N={N})  Repo={args.repo}  LLM={args.llm_url}")
    print("=" * 78)

    # LLM erreichbar?
    reranker = LLMListwiseReranker(
        api_url=args.llm_url, model=args.model, timeout=args.timeout,
    )
    if not reranker.ping():
        print("\nLM Studio nicht erreichbar. Starte das Modell und prüfe --llm-url.")
        sys.exit(1)

    # ── Indizes (leichtgewichtig: nur BM25 + Chunk-Maps) ──
    print("\n[1/3] Repo + BM25-Grobstufe bauen …")
    repo_path = clone_repository(repo_config.name)
    python_files = get_python_files(repo_path, repo_config.source_dirs)
    retriever = create_retriever(condition, es_url=args.es_url)
    _build_light(retriever, repo_path, python_files)

    dataset = load_dataset(str(ROOT / "benchmark" / "data" / "benchmark_dataset.json"))
    samples = [s for s in dataset.samples if repo_config.name in s.repo_id]
    if args.limit:
        samples = samples[: args.limit]
    print(f"[2/3] {len(samples)} Samples.")
    print(f"[3/3] LLM-Reranking läuft …\n")

    per_sample = []
    t_start = time.time()

    for i, sample in enumerate(samples, 1):
        gt_files = [t.file_path for t in sample.ground_truth.targets]
        query = sample.query

        candidate_files = _stage1_candidates(retriever, query)
        stage1_hit = any(_file_matches(cf, gt) for cf in candidate_files for gt in gt_files)

        # Kandidaten (in Stage-1-Reihenfolge) mit Symbol-Outline aufbereiten.
        cands = []
        for fp in candidate_files[: args.max_files_prompt]:
            chunks = _gather_candidate_chunks(retriever, [fp])
            cands.append((fp, _file_outline(chunks, args.max_symbols)))

        t_s = time.time()
        ranked_files = reranker.rerank(query, cands, top_k=args.k)
        dt = time.time() - t_s

        proxies = [SimpleNamespace(file_path=fp, function_name=None, content="")
                   for fp in ranked_files]
        rec = recall_at_k(proxies, gt_files, args.k, None)

        per_sample.append({
            "sample_id": sample.sample_id,
            "tc": sample.metadata.test_case_type.value,
            "stage1_hit": stage1_hit,
            "n_candidate_files": len(candidate_files),
            "recall_llm": rec,
        })

        el = time.time() - t_start
        eta = (el / i) * (len(samples) - i) / 60.0
        print(f"    {i:>3}/{len(samples)}  {dt:5.1f}s  ETA~{eta:4.0f}min  |  "
              f"s1={int(stage1_hit)} llm_recall={rec:.2f}", flush=True)

    _report(per_sample, N, args, reference, reranker)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"condition": args.condition, "N": N, "llm": args.llm_url,
                       "per_sample": per_sample}, f, indent=2, ensure_ascii=False)
        print(f"\n  Per-Sample-Dump: {args.out}")

    try:
        retriever.close()
    except Exception:
        pass


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _report(per_sample, N, args, reference, reranker):
    n = len(per_sample)
    if n == 0:
        print("Keine Samples.")
        return
    s1 = [p for p in per_sample if p["stage1_hit"]]
    s1_rate = len(s1) / n
    llm_o = _mean(p["recall_llm"] for p in per_sample)
    llm_f = _mean(p["recall_llm"] for p in s1)
    target_fine = TARGET_RECALL / s1_rate if s1_rate > 0 else float("inf")

    print("\n" + "=" * 78)
    print(f"  ERGEBNIS  (n={n}, Stage-1-hit-Rate={s1_rate:.3f}, N={N})")
    print(f"  LLM-Calls={reranker.calls}  Parse-Fehler={reranker.parse_failures}")
    print("=" * 78)
    print(f"\n  Zielwert Fine-Recovery für Recall@10 ≥ {TARGET_RECALL:.2f}"
          f"  =  {TARGET_RECALL:.2f} / {s1_rate:.3f}  =  {target_fine:.3f}\n")

    hdr = f"  {'Methode':<28} {'Recall@10':>10} {'Fine-Recovery':>15} {'Δ vs Ziel':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    if reference:
        print(f"  {'(A) Qwen-Hybrid (Referenz)':<28} {reference['recall']:>10.4f} "
              f"{reference['fine']:>15.4f} {reference['fine'] - target_fine:>+11.4f}")
    print(f"  {'(B) bge-reranker (gemessen)':<28} {'~0.43':>10} {'~0.46':>15} {'-0.41':>11}")
    print(f"  {'(C) LLM listwise':<28} {llm_o:>10.4f} {llm_f:>15.4f} {llm_f - target_fine:>+11.4f}")

    print("\n  ── Verdikt ──")
    print(f"  LLM-Fine-Recovery: {llm_f:.3f}  →  geschätztes Recall@10_gesamt ≈ {llm_f * s1_rate:.3f}")
    if llm_f >= target_fine:
        print(f"  GO – LLM erreicht den Zielwert bei N={N}. Als V21-Stufe ausbauen.")
    elif llm_f >= 0.65:
        print(f"  VIELVERSPRECHEND – klar über Cross-Encoder & Bi-Encoder, aber unter Ziel. "
              f"Prompt/Outline verbessern, N/Modell variieren.")
    elif llm_f > 0.55:
        print(f"  leicht besser als der Qwen-Hybrid (0.55), aber kein Durchbruch.")
    else:
        print(f"  Auch das LLM knackt die Konversion nicht — 0.80 via Reranking unrealistisch.")

    print("\n  ── Recall@10 nach Test-Case-Typ ──")
    by_tc = defaultdict(list)
    for p in per_sample:
        by_tc[p["tc"]].append(p)
    print(f"  {'TC':<5} {'n':>4} {'LLM':>7} {'s1-hit':>7}")
    for tc in sorted(by_tc):
        ps = by_tc[tc]
        print(f"  {tc:<5} {len(ps):>4} {_mean(p['recall_llm'] for p in ps):>7.3f} "
              f"{_mean(1.0 if p['stage1_hit'] else 0.0 for p in ps):>7.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
