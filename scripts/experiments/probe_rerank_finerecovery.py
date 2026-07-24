#!/usr/bin/env python3
"""
probe_rerank_finerecovery.py – SCHRITT 0: Lohnt sich der Cross-Encoder überhaupt?

Bevor die ganze V21-Matrix gefahren wird, beantwortet dieses Skript in EINEM
Lauf die einzige Frage, die zählt:

    "Kann ein Cross-Encoder die Fine-Recovery (Recall@10 auf den Samples, wo
     Stage-1 die richtige Datei GEFUNDEN hat) so weit anheben, dass am Ende
     Recall@10 ≥ 0.80 rauskommt?"

Hintergrund (aus den V20-Ergebnissen, Repo pandas):
    Recall@10_gesamt = Stage-1-hit-Rate × Fine-Recovery
    Bei N=80: Stage-1-hit = 0.905, Bi-Encoder-Fine-Recovery = 0.547 → 0.495.
    Für Recall@10 ≥ 0.80 braucht der Reranker bei N=80:
        Fine-Recovery ≥ 0.80 / 0.905 = 0.884
    Das ist der Zielwert, den dieses Skript misst.

──────────────────────────────────────────────────────────────────────────────
WICHTIG – LEICHTGEWICHTIGER DEFAULT:
  Standardmäßig braucht dieses Skript das große Qwen3-4B-Embedding NICHT.
  Es baut nur die billige BM25-Grobstufe (Stage-1) und lässt den kleinen
  Cross-Encoder laufen. Die (A)-Baseline wird dann aus den bekannten
  V20c-Zahlen als Referenz angezeigt.

  Mit --with-baseline wird zusätzlich der echte V20c-Stage-2 (Qwen3-4B) live
  neu gerechnet. Das lädt ein ~8-GB-Modell; auf dem Mac unbedingt
  EMBED_BATCH_SIZE klein halten (das Skript setzt automatisch 8, falls nicht
  gesetzt), sonst sprengt es den MPS-Speicher.
──────────────────────────────────────────────────────────────────────────────

Was gemessen wird – auf DENSELBEN Stage-1-Kandidaten:
    (A) aktuell (V20c)        – Referenz bzw. live mit --with-baseline
    (B) Reranker chunk-topk   – Top-10 einzelne Chunks (so schlägt es der Plan vor)
    (C) Reranker file-maxpool – pro Datei bester Chunk, Top-10 DATEIEN
                                (passt zur datei-basierten Eval → i.d.R. besser)

VORAUSSETZUNGEN (auf deiner Maschine, NICHT in der Sandbox):
  - Elasticsearch läuft (http://localhost:9200)
  - Pre-computed LLM-Expansions: benchmark/data/llm_expansions_pandas.json
  - Reranker-Backend: `pip install sentence-transformers torch`
    (bge-reranker-v2-m3 wird beim ersten Lauf geladen, ~2 GB).

AUFRUF:
  # Schneller, leichtgewichtiger Test auf 30 Samples (empfohlen):
  python scripts/probe_rerank_finerecovery.py --limit 30

  # Voller Lauf (ohne Qwen), Per-Sample-Dump:
  python scripts/probe_rerank_finerecovery.py --out benchmark/results/probe.json

  # Mit echter V20c-Baseline live (lädt Qwen3-4B, langsamer):
  EMBED_BATCH_SIZE=8 python scripts/probe_rerank_finerecovery.py --with-baseline --limit 30
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

_TOK_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _tokens(text):
    return set(_TOK_RE.findall((text or "").lower()))


def _code_aware_text(c):
    """Text mit vorangestelltem Datei-/Symbol-Kontext, den der Reranker sieht.

    Gibt dem generischen (NL-trainierten) Cross-Encoder die Code-Hinweise, die
    im nackten Chunk fehlen: Dateipfad, Klasse, Funktion. Kein Neu-Chunking nötig,
    die Felder liegen bereits am Chunk-Objekt.
    """
    parts = []
    fp = getattr(c, "file_path", None)
    if fp:
        parts.append(f"File: {fp}")
    cls = getattr(c, "class_name", None)
    if cls:
        parts.append(f"class {cls}")
    fn = getattr(c, "function_name", None)
    if fn:
        parts.append(f"def {fn}")
    header = " | ".join(parts)
    content = getattr(c, "content", "") or ""
    return (header + "\n" + content) if header else content


def _rerank_query(sample, mode="title_body", body_chars=400):
    """Kompakte Query für den Cross-Encoder.

    Cross-Encoder haben ein begrenztes gemeinsames Fenster (Query+Chunk). Ein
    kompletter Issue-Text (oft >1000 Tokens) frisst das Budget und lässt kaum
    Platz für den Chunk. Wir nehmen deshalb Titel + Anfang des Bodys.
    (Der Bi-Encoder darf lange Queries nutzen, weil er getrennt kodiert.)
    """
    q = sample.query or ""
    lines = [l.strip() for l in q.split("\n") if l.strip()]
    title = lines[0] if lines else ""
    if mode == "full":
        return q
    if mode == "title":
        return title[:body_chars]
    body = q[len(title):].strip()
    return (title + "  " + body[:body_chars]).strip()

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

TARGET_RECALL = 0.80

# Bekannte V20c-Referenz (pandas, N=80) aus results/v20_largeN — dient als
# (A)-Baseline, wenn NICHT --with-baseline genutzt wird.
V20_REFERENCE = {
    ("V20a", "pandas"): {"recall": 0.5772, "fine": 0.795, "s1": 0.726, "N": 20},
    ("V20b", "pandas"): {"recall": 0.5772, "fine": 0.712, "s1": 0.811, "N": 40},
    ("V20c", "pandas"): {"recall": 0.4947, "fine": 0.547, "s1": 0.905, "N": 80},
    ("V20d", "pandas"): {"recall": 0.4912, "fine": 0.524, "s1": 0.937, "N": 150},
}


def _stage1_candidates(retriever, query):
    """Replikiert die STAGE-1-Logik des V16-Retrievers, OHNE Stage-2/Embeddings.

    Liefert die Top-N Kandidat-Dateien (wie retriever.retrieve() sie intern
    berechnet), nutzt aber nur die billige BM25-Grobstufe.
    """
    mode = getattr(retriever, "stage1_query_mode", "llm_expanded")
    if mode == "llm_expanded":
        expanded = retriever._lookup.lookup(query)
        stage1_query = expanded if expanded else query.split("\n")[0].strip()
    elif mode == "title_only":
        stage1_query = query.split("\n")[0].strip()
    else:
        stage1_query = query

    coarse_fetch = max(retriever.top_n_files * 3, 30)
    coarse_results = retriever._coarse.retrieve(stage1_query, coarse_fetch)

    agg = getattr(retriever, "coarse_score_aggregation", "max")
    file_scores = {}
    for chunk, score in coarse_results:
        cur = file_scores.get(chunk.file_path)
        if agg == "sum":
            file_scores[chunk.file_path] = (cur or 0.0) + score
        else:
            if cur is None or score > cur:
                file_scores[chunk.file_path] = score

    ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[: retriever.top_n_files]]


def _build_light(retriever, repo_path, python_files):
    """Nur das bauen, was der Probe braucht: BM25-Grobstufe + Fein-Chunk-Maps.

    Spart das teure/fragile Qwen3-4B-Embedding komplett aus. Erzeugt exakt die
    Strukturen (_chunks_by_file, _fine_chunk_map), die index_repository() sonst
    auch anlegt.
    """
    print("  [light] coarse chunking + BM25 index …")
    coarse_chunks = retriever.coarse_chunker.chunk_repository(repo_path, python_files)
    retriever._coarse.index(coarse_chunks)
    print(f"  [light]   coarse_chunks={len(coarse_chunks)}")

    print("  [light] fine chunking (nur Chunk-Objekte, keine Embeddings) …")
    fine_chunks = retriever.fine_chunker.chunk_repository(repo_path, python_files)
    retriever._fine_chunks = fine_chunks
    retriever._fine_chunk_map = {c.chunk_id: c for c in fine_chunks}
    retriever._chunks_by_file = {}
    for c in fine_chunks:
        retriever._chunks_by_file.setdefault(c.file_path, []).append(c.chunk_id)
    retriever._indexed = True
    print(f"  [light]   fine_chunks={len(fine_chunks)}  "
          f"dateien={len(retriever._chunks_by_file)}")


def _gather_candidate_chunks(retriever, candidate_files, query=None, max_per_file=None):
    """Fein-Chunks der Stage-1-Kandidatdateien einsammeln.

    max_per_file: optionale Obergrenze an Chunks je Datei (Laufzeit-Hebel auf
    schwacher Hardware). None = alle.

    WICHTIG: bei einer Obergrenze werden NICHT die ersten N Chunks genommen
    (das würde den Ziel-Chunk in großen Dateien systematisch rauswerfen),
    sondern die N Chunks mit der höchsten lexikalischen Überlappung zur Query
    (billiger BM25-artiger Vorfilter). So bleibt der wahrscheinlich relevante
    Chunk erhalten.
    """
    chunks_by_file = getattr(retriever, "_chunks_by_file", {}) or {}
    chunk_map = getattr(retriever, "_fine_chunk_map", {}) or {}
    qtok = _tokens(query) if (query and max_per_file) else None
    out = []
    for fp in candidate_files:
        cids = chunks_by_file.get(fp, [])
        chunks = [chunk_map[cid] for cid in cids if cid in chunk_map]
        if max_per_file is not None and len(chunks) > max_per_file:
            if qtok:
                chunks.sort(
                    key=lambda c: len(qtok & _tokens(getattr(c, "content", ""))),
                    reverse=True,
                )
            chunks = chunks[:max_per_file]
        out.extend(chunks)
    return out


def main():
    ap = argparse.ArgumentParser(description="Schritt-0 Reranker Fine-Recovery Probe")
    ap.add_argument("--condition", default="V20c",
                    help="Basis-Condition (bestimmt N via top_n_files). Default V20c = N=80.")
    ap.add_argument("--repo", default="pandas")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--backend", default="auto", choices=["auto", "st", "flag"])
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (Default: auto)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-chunks-per-file", type=int, default=None,
                    help="Obergrenze Chunks/Datei fürs Reranking (die lexikalisch "
                         "relevantesten werden behalten, NICHT die ersten). Auf "
                         "Mac/MPS z. B. 12 setzen — schneller, für einen Probe ok.")
    ap.add_argument("--rerank-query-mode", default="title_body",
                    choices=["title", "title_body", "full"],
                    help="Query fürs Reranking: title_body (Default) gibt Titel + "
                         "Anfang des Issue-Texts; 'full' den ganzen Text (schlecht "
                         "für Cross-Encoder wg. begrenztem Fenster).")
    ap.add_argument("--rerank-body-chars", type=int, default=400,
                    help="Zeichen des Issue-Bodys, die in die Reranker-Query gehen.")
    ap.add_argument("--code-aware-header", action="store_true",
                    help="Dem Reranker Dateipfad + Klasse/Funktion vor den Chunk "
                         "stellen (testet den 'code-aware Chunks'-Hebel aus dem Plan).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Nur die ersten N Samples (schnell zum Ausprobieren).")
    ap.add_argument("--with-baseline", action="store_true",
                    help="Echte V20c-Baseline (A) live rechnen (lädt Qwen3-4B). "
                         "Ohne diesen Schalter wird die bekannte Referenz angezeigt.")
    ap.add_argument("--es-url", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None, help="Optionaler JSON-Dump der Per-Sample-Ergebnisse.")
    args = ap.parse_args()

    # Sicherheitsnetz: auf dem Mac sprengt EMBED_BATCH_SIZE=64 den MPS-Speicher.
    os.environ.setdefault("EMBED_BATCH_SIZE", "8")

    # Lazy imports (ziehen den vollen Stack: ES, ggf. Embeddings, torch).
    from benchmark.config import CONDITIONS_MAP, REPOS_MAP
    from benchmark.runner import create_retriever, clone_repository, get_python_files, load_dataset
    from benchmark.metrics import recall_at_k, _file_matches
    from benchmark.reranker import CrossEncoderReranker

    if args.condition not in CONDITIONS_MAP:
        print(f"Unbekannte Condition: {args.condition}")
        sys.exit(1)
    condition = CONDITIONS_MAP[args.condition]
    N = getattr(condition, "top_n_files", "?")
    repo_config = REPOS_MAP[args.repo]
    reference = V20_REFERENCE.get((args.condition, args.repo))

    print("=" * 78)
    print("  SCHRITT 0 – Reranker Fine-Recovery Probe")
    print(f"  Condition={args.condition} (N={N})  Repo={args.repo}  Modell={args.model}")
    print(f"  Modus={'MIT Qwen-Baseline (live)' if args.with_baseline else 'leicht (BM25 + Reranker)'}")
    print("=" * 78)

    # ── 1) Retriever bauen & indexieren ──
    print("\n[1/4] Repo klonen + Indizes bauen …")
    repo_path = clone_repository(repo_config.name)
    python_files = get_python_files(repo_path, repo_config.source_dirs)
    retriever = create_retriever(condition, es_url=args.es_url)
    if args.with_baseline:
        retriever.index_repository(repo_path, python_files)   # inkl. Qwen3-4B
    else:
        _build_light(retriever, repo_path, python_files)      # nur BM25 + Chunk-Maps

    # ── 2) Samples laden ──
    dataset = load_dataset(str(ROOT / "benchmark" / "data" / "benchmark_dataset.json"))
    samples = [s for s in dataset.samples if repo_config.name in s.repo_id]
    if args.limit:
        samples = samples[: args.limit]
    print(f"[2/4] {len(samples)} Samples geladen.")

    # ── 3) Reranker laden ──
    print("[3/4] Reranker laden …")
    reranker = CrossEncoderReranker(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        backend=args.backend,
    )

    # ── 4) Über die Samples laufen ──
    print(f"[4/4] Reranking läuft über {len(samples)} Samples …\n")
    per_sample = []
    t_start = time.time()
    warned_empty = False

    for i, sample in enumerate(samples, 1):
        gt_files = [t.file_path for t in sample.ground_truth.targets]
        gt_funcs = [t.function_name for t in sample.ground_truth.targets if t.function_name]
        gt_funcs = gt_funcs or None
        query = sample.query

        # Stage-1-Kandidaten (+ ggf. Live-Baseline)
        rec_bi = None
        if args.with_baseline:
            bi_results = retriever.retrieve(query, args.k)
            candidate_files = list(getattr(retriever, "last_coarse_files", []) or [])
            rec_bi = recall_at_k([c for c, _ in bi_results], gt_files, args.k, gt_funcs)
        else:
            candidate_files = _stage1_candidates(retriever, query)

        stage1_hit = any(_file_matches(cf, gt) for cf in candidate_files for gt in gt_files)

        # Chunk-Auswahl nutzt den vollen Query-Text für die Überlappung (mehr Signal);
        # der Cross-Encoder bekommt eine kompakte Query (begrenztes Fenster).
        cand_chunks = _gather_candidate_chunks(
            retriever, candidate_files, query=query, max_per_file=args.max_chunks_per_file
        )
        if not cand_chunks and candidate_files and not warned_empty:
            print("   Warnung: Kandidat-Dateien liefern keine Chunks – "
                  "Pfad-Format-Mismatch? (nur einmal gemeldet)")
            warned_empty = True

        rr_query = _rerank_query(sample, args.rerank_query_mode, args.rerank_body_chars)
        text_fn = _code_aware_text if args.code_aware_header else None

        # EINMAL scoren, beide Rankings ableiten (spart die Hälfte der Rechenzeit).
        t_s = time.time()
        variants = reranker.rerank_variants(rr_query, cand_chunks, top_k=args.k, text_fn=text_fn)
        dt = time.time() - t_s
        rec_rr_chunk = recall_at_k([c for c, _ in variants["none"]], gt_files, args.k, gt_funcs)
        rec_rr_file = recall_at_k([c for c, _ in variants["file_max"]], gt_files, args.k, gt_funcs)

        per_sample.append({
            "sample_id": sample.sample_id,
            "tc": sample.metadata.test_case_type.value,
            "stage1_hit": stage1_hit,
            "n_candidate_files": len(candidate_files),
            "n_candidate_chunks": len(cand_chunks),
            "recall_bi": rec_bi,
            "recall_rr_chunk": rec_rr_chunk,
            "recall_rr_file": rec_rr_file,
        })

        # Fortschritt JE Sample (auf langsamer Hardware wichtig), inkl. ETA.
        el = time.time() - t_start
        eta_min = (el / i) * (len(samples) - i) / 60.0
        bi_str = f"{rec_bi:.2f}" if rec_bi is not None else "–"
        print(f"    {i:>3}/{len(samples)}  {dt:5.1f}s ({len(cand_chunks)} chunks)  "
              f"ETA~{eta_min:4.0f}min  |  s1={int(stage1_hit)} bi={bi_str} "
              f"rr_chunk={rec_rr_chunk:.2f} rr_file={rec_rr_file:.2f}", flush=True)

    _report(per_sample, N, args, reference)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"condition": args.condition, "N": N, "model": args.model,
                       "with_baseline": args.with_baseline, "per_sample": per_sample},
                      f, indent=2, ensure_ascii=False)
        print(f"\n  Per-Sample-Dump: {args.out}")

    try:
        retriever.close()
    except Exception:
        pass


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _report(per_sample, N, args, reference):
    n = len(per_sample)
    if n == 0:
        print("Keine Samples ausgewertet.")
        return

    s1 = [p for p in per_sample if p["stage1_hit"]]
    s1_rate = len(s1) / n

    def block(key):
        return _mean(p[key] for p in per_sample), _mean(p[key] for p in s1)

    rc_o, rc_f = block("recall_rr_chunk")
    rf_o, rf_f = block("recall_rr_file")

    # (A)-Zeile: live gerechnet oder aus Referenz
    if any(p["recall_bi"] is not None for p in per_sample):
        bi_o, bi_f = block("recall_bi")
        bi_note = "live"
    elif reference:
        bi_o, bi_f = reference["recall"], reference["fine"]
        bi_note = "Referenz V20"
    else:
        bi_o = bi_f = None
        bi_note = "n/a"

    target_fine = TARGET_RECALL / s1_rate if s1_rate > 0 else float("inf")

    print("\n" + "=" * 78)
    print(f"  ERGEBNIS  (n={n}, Stage-1-hit-Rate={s1_rate:.3f}, N={N})")
    if reference:
        print(f"  Referenz-Stage-1-hit (V20, voller Lauf): {reference['s1']:.3f}"
              f"  – deine hier: {s1_rate:.3f}"
              + ("  (Teilmenge via --limit)" if args.limit else ""))
    print("=" * 78)
    print(f"\n  Zielwert Fine-Recovery für Recall@10 ≥ {TARGET_RECALL:.2f}"
          f"  =  {TARGET_RECALL:.2f} / {s1_rate:.3f}  =  {target_fine:.3f}\n")

    hdr = f"  {'Methode':<28} {'Recall@10':>10} {'Fine-Recovery':>15} {'Δ vs Ziel':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def row(label, o, fdr):
        if o is None:
            print(f"  {label:<28} {'—':>10} {'—':>15} {'—':>11}")
        else:
            print(f"  {label:<28} {o:>10.4f} {fdr:>15.4f} {fdr - target_fine:>+11.4f}")

    row(f"(A) aktuell ({bi_note})", bi_o, bi_f)
    row("(B) Reranker chunk-topk", rc_o, rc_f)
    row("(C) Reranker file-maxpool", rf_o, rf_f)

    best_fine = max(rc_f, rf_f)
    best_overall_est = best_fine * s1_rate
    print("\n  ── Verdikt ──")
    print(f"  Beste Reranker-Fine-Recovery: {best_fine:.3f}"
          f"  →  geschätztes Recall@10_gesamt ≈ {best_overall_est:.3f}")
    if best_fine >= target_fine:
        print(f"  GO – Reranker erreicht den Zielwert bei N={N}. V21-Matrix fahren.")
    elif best_fine >= 0.75:
        print(f"  MARGINAL – nah dran, aber unter Ziel. Optionen: N variieren "
              f"(--condition V20d=N150), code-aware Chunks, Stage-1 via Hybrid heben.")
    else:
        print(f"  NO-GO bei diesem Modell – Off-the-shelf-Reranker zu schwach auf Code.")
        print(f"     Zuerst: code-aware Chunks (Pfad+Signaturen) oder Fine-Tuning, "
              f"dann erneut proben.")

    print("\n  ── Recall@10 nach Test-Case-Typ (gesamt) ──")
    by_tc = defaultdict(list)
    for p in per_sample:
        by_tc[p["tc"]].append(p)
    print(f"  {'TC':<5} {'n':>4} {'RR-chunk':>9} {'RR-file':>9} {'s1-hit':>7}")
    for tc in sorted(by_tc):
        ps = by_tc[tc]
        print(f"  {tc:<5} {len(ps):>4} "
              f"{_mean(p['recall_rr_chunk'] for p in ps):>9.3f} "
              f"{_mean(p['recall_rr_file'] for p in ps):>9.3f} "
              f"{_mean(1.0 if p['stage1_hit'] else 0.0 for p in ps):>7.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
