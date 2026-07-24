#!/usr/bin/env python3
"""
analyze_expansion_precision.py – Precision der LLM-Query-Expansion (V16 §5.3).

Misst pro Query, welcher Anteil der LLM-generierten Terme (functions, classes,
files, imports, keywords) tatsächlich in der/den Ziel-Datei(en) der Ground Truth
vorkommt. Daraus:

  * Precision pro Query (overall + pro Kategorie)
  * Aggregation pro Repo / Kategorie / Test-Case-Typ
  * Liste der Queries mit niedrigster Precision (Expansion bringt nur Noise →
    schadet Stage 1 tendenziell mehr als sie hilft)
  * Optionaler Abgleich mit echten Benchmark-Ergebnissen (--results): listet
    Queries, bei denen die LLM-Expansion den Stage-1-Treffer verschlechtert
    (stage1_hit True ohne, False mit Expansion)
  * Datengetriebene Empfehlungen fürs Prompt-Design

KEIN Elasticsearch, KEIN LLM nötig — nur stdlib. Läuft, sobald die Expansionen
vorberechnet sind (scripts/precompute_llm_expansions.py --repo <repo>).

Aufruf (aus dem Repo-Root):
    python scripts/analyze_expansion_precision.py
    python scripts/analyze_expansion_precision.py --repo pandas
    python scripts/analyze_expansion_precision.py --results benchmark/results/benchmark_results_latest.json \\
        --expanded-condition V16c --baseline-condition V12b
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "benchmark" / "data"
REPOS_DIR = ROOT / "benchmark" / "repos"
RESULTS_DIR = ROOT / "benchmark" / "results"
CATEGORIES = ["functions", "classes", "files", "imports", "keywords"]
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ─── Loading ────────────────────────────────────────────────────────

def load_dataset(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {}
    for s in data["samples"]:
        by_id[s["sample_id"]] = {
            "repo_id": s["repo_id"],
            "query": s.get("query", ""),
            "targets": [t["file_path"] for t in s["ground_truth"]["targets"]],
            "tc": (s.get("metadata") or {}).get("test_case_type"),
        }
    return by_id


def find_expansion_files(data_dir: Path, only_repo: str = None,
                         variant: str = None):
    """Expansion-Files je Repo finden.

    variant=None/"baseline" → nur Baseline-Dateien (llm_expansions_<repo>.json);
    Varianten-Dateien (…__<variante>.json) werden dann explizit
    ausgeschlossen, damit sie nicht als eigene "Repos" mitgezählt werden.
    variant="lean" etc. → nur llm_expansions_<repo>__lean.json (Repo-Name
    ohne Suffix).
    """
    files = sorted(data_dir.glob("llm_expansions_*.json"))
    out = {}
    for fp in files:
        stem = fp.stem.replace("llm_expansions_", "")
        if variant and variant != "baseline":
            if not stem.endswith(f"__{variant}"):
                continue
            repo = stem[: -len(f"__{variant}")]
        else:
            if "__" in stem:  # Varianten-Datei → gehört nicht zur Baseline
                continue
            repo = stem
        if only_repo and repo != only_repo:
            continue
        out[repo] = fp
    return out


def repo_root_for(repo: str, repos_dir: Path) -> Path:
    """Best-effort path to the cloned repo source (file_path is relative to it)."""
    short = repo.split("/")[-1]
    for cand in (repos_dir / short, repos_dir / repo):
        if cand.exists():
            return cand
    return repos_dir / short


# ─── Term extraction & matching ─────────────────────────────────────

def terms_by_category(entry: dict) -> dict:
    """Return {category: [terms]} from a precomputed expansion entry."""
    parsed = entry.get("parsed")
    out = {c: [] for c in CATEGORIES}
    if isinstance(parsed, dict) and "raw" not in parsed:
        for c in CATEGORIES:
            items = parsed.get(c, [])
            if isinstance(items, list):
                out[c] = [str(x) for x in items if str(x).strip()]
        return out
    # Fallback: no structured parse → treat flat terms as keywords
    flat = entry.get("flat_terms", "") or ""
    out["keywords"] = [t for t in flat.split() if t.strip()]
    return out


def term_tokens(term: str):
    """Significant identifier tokens (len>=3) of a term, lowercased."""
    return [t.lower() for t in _IDENT.findall(term) if len(t) >= 3]


def term_hit(term: str, category: str, target_text_lc: str, target_paths_lc) -> bool:
    """True if the generated term plausibly occurs in the target file(s)."""
    if category == "files":
        # Hit if the filename matches a GT target path (strict) OR — to be
        # consistent with the other categories — if a filename token appears in
        # the target file content (lenient). Without the content fallback, files
        # were systematically penalised vs. imports/functions/keywords.
        base = term.replace("\\", "/").lower().rsplit("/", 1)[-1].replace(".py", "")
        if any(base and base in p for p in target_paths_lc):
            return True
        toks = term_tokens(term)
        return any(tok in target_text_lc for tok in toks)
    toks = term_tokens(term)
    if not toks:
        return False
    # Hit if any significant identifier token appears in the target file text.
    return any(tok in target_text_lc for tok in toks)


def read_targets_text(repo_root: Path, target_paths):
    texts = []
    ok = False
    for rel in target_paths:
        fp = repo_root / rel
        if fp.exists() and fp.is_file():
            try:
                texts.append(fp.read_text(encoding="utf-8", errors="replace"))
                ok = True
            except Exception:
                pass
    return ("\n".join(texts), ok)


# ─── Per-query precision ────────────────────────────────────────────

def analyze_query(entry: dict, targets, repo_root: Path) -> dict:
    cats = terms_by_category(entry)
    total_terms = sum(len(v) for v in cats.values())
    target_text, have_text = read_targets_text(repo_root, targets)
    target_text_lc = target_text.lower()
    target_paths_lc = [p.replace("\\", "/").lower() for p in targets]

    per_cat = {}
    hits_total = 0
    for c in CATEGORIES:
        terms = cats[c]
        if not terms:
            per_cat[c] = {"n": 0, "hits": 0, "precision": None}
            continue
        h = sum(1 for t in terms if term_hit(t, c, target_text_lc, target_paths_lc))
        hits_total += h
        per_cat[c] = {"n": len(terms), "hits": h, "precision": h / len(terms)}

    precision = (hits_total / total_terms) if total_terms else None
    return {
        "n_terms": total_terms,
        "hits": hits_total,
        "precision": precision,
        "per_category": per_cat,
        "have_target_text": have_text,
    }


# ─── Optional: cross-reference real benchmark results ───────────────

def stage1_hits_by_condition(results_path: Path):
    """{condition_id: {sample_id: stage1_hit}} from a benchmark results JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs", data if isinstance(data, list) else [])
    out = defaultdict(dict)
    for run in runs:
        cid = run.get("condition_id")
        per = (run.get("metrics") or {}).get("per_sample", [])
        for m in per:
            sid = m.get("sample_id")
            if sid is not None and m.get("stage1_hit") is not None:
                out[cid][sid] = m["stage1_hit"]
    return out


# ─── Reporting ──────────────────────────────────────────────────────

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x):
    return "–" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description="LLM expansion precision analysis (V16 §5.3)")
    ap.add_argument("--repo", default=None, help="Nur dieses Repo analysieren")
    ap.add_argument("--dataset", default=str(DATA_DIR / "benchmark_dataset.json"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--repos-dir", default=str(REPOS_DIR))
    ap.add_argument("--bottom", type=int, default=15, help="Anzahl Queries in Worst-Liste")
    ap.add_argument("--results", default=None, help="Benchmark-Results-JSON für echten Stage-1-Abgleich")
    ap.add_argument("--expanded-condition", default=None, help="Kondition MIT LLM-Expansion (z.B. V16c)")
    ap.add_argument("--baseline-condition", default=None, help="Kondition OHNE Expansion (z.B. V12b)")
    ap.add_argument("--output", default=None,
                    help="Report-MD (default: expansion_precision_report[__<variante>].md)")
    ap.add_argument("--json-output", default=None,
                    help="Report-JSON (default: expansion_precision[__<variante>].json)")
    ap.add_argument("--variant", default=None,
                    help="Prompt-Variante analysieren, z.B. 'lean' — "
                         "liest llm_expansions_<repo>__<variante>.json")
    args = ap.parse_args()

    _suffix = f"__{args.variant}" if args.variant and args.variant != "baseline" else ""
    if args.output is None:
        args.output = str(RESULTS_DIR / f"expansion_precision_report{_suffix}.md")
    if args.json_output is None:
        args.json_output = str(RESULTS_DIR / f"expansion_precision{_suffix}.json")

    repos_dir = Path(args.repos_dir)
    dataset = load_dataset(Path(args.dataset))
    exp_files = find_expansion_files(Path(args.data_dir), args.repo, args.variant)

    if not exp_files:
        print(f"Keine llm_expansions_*.json in {args.data_dir} gefunden "
              f"(Repo-Filter: {args.repo}). Erst precompute_llm_expansions.py laufen lassen.")
        return

    per_query = []      # flat list of dicts
    missing_text = 0
    for repo, fp in exp_files.items():
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        expansions = payload.get("expansions", {})
        repo_root = repo_root_for(repo, repos_dir)
        for sid, entry in expansions.items():
            meta = dataset.get(sid)
            if not meta:
                continue
            res = analyze_query(entry, meta["targets"], repo_root)
            if not res["have_target_text"]:
                missing_text += 1
            per_query.append({
                "sample_id": sid,
                "repo": repo,
                "tc": meta["tc"],
                "title": entry.get("title", meta["query"].split("\n")[0][:80]),
                **res,
            })

    scored = [q for q in per_query if q["precision"] is not None and q["have_target_text"]]

    # ── Aggregates ──
    overall = _mean([q["precision"] for q in scored])
    by_repo = defaultdict(list)
    by_tc = defaultdict(list)
    by_cat = defaultdict(list)
    for q in scored:
        by_repo[q["repo"]].append(q["precision"])
        by_tc[q["tc"] or "?"].append(q["precision"])
        for c in CATEGORIES:
            pc = q["per_category"][c]["precision"]
            if pc is not None:
                by_cat[c].append(pc)

    cat_means = {c: _mean(by_cat[c]) for c in CATEGORIES}
    zero_prec = [q for q in scored if q["precision"] == 0.0 and q["n_terms"] > 0]
    worst = sorted(scored, key=lambda q: (q["precision"], -q["n_terms"]))[:args.bottom]

    # ── Optional real Stage-1 cross-reference ──
    harmful = None
    if args.results and args.expanded_condition and args.baseline_condition:
        hits = stage1_hits_by_condition(Path(args.results))
        exp_h = hits.get(args.expanded_condition, {})
        base_h = hits.get(args.baseline_condition, {})
        harmful = []
        for sid in set(exp_h) & set(base_h):
            if base_h[sid] and not exp_h[sid]:  # baseline found file, expansion lost it
                meta = dataset.get(sid, {})
                harmful.append({"sample_id": sid, "repo": meta.get("repo_id"),
                                "title": meta.get("query", "").split("\n")[0][:80]})

    # ── Recommendations (data-driven) ──
    ranked_cats = sorted(
        [(c, m) for c, m in cat_means.items() if m is not None], key=lambda x: x[1]
    )
    recs = []
    if ranked_cats:
        worst_cat, worst_m = ranked_cats[0]
        best_cat, best_m = ranked_cats[-1]
        recs.append(
            f"Kategorie **{worst_cat}** hat die niedrigste Precision ({worst_m:.3f}) — "
            f"diese Terme landen am seltensten in der Zieldatei. Im Prompt deemphasizen "
            f"oder präziser anweisen.")
        recs.append(
            f"Kategorie **{best_cat}** ist am treffsichersten ({best_m:.3f}) — im Prompt "
            f"stärker gewichten / mehr davon anfordern.")
    if cat_means.get("files") is not None and cat_means.get("imports") is not None:
        if cat_means["files"] > cat_means["imports"]:
            recs.append(
                "Bestätigt die Bericht-Hypothese (§5.2): Dateipfade treffsicherer als "
                "Imports → Prompt-Fokus von imports auf files verschieben.")
    if scored:
        recs.append(
            f"{len(zero_prec)}/{len(scored)} Queries ({100*len(zero_prec)/len(scored):.0f}%) "
            f"haben Precision 0 — hier ist die Expansion reiner Noise; Few-Shot-Beispiele "
            f"mit echten Issue→Lösungsdatei-Paaren könnten gezielt gegensteuern.")

    # ── Write report ──
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# LLM-Expansion: Precision-Analyse (V16 §5.3)\n")
    L.append(f"Analysierte Queries (mit Termen + lesbarer Zieldatei): **{len(scored)}** "
             f"(von {len(per_query)} Expansionen; {missing_text} ohne lesbare Zieldatei).\n")
    L.append(f"**Mittlere Precision (Anteil generierter Terme in der Zieldatei): {_fmt(overall)}**\n")

    L.append("\n## Precision pro Repo\n")
    L.append("| Repo | Queries | Ø Precision |")
    L.append("|---|---|---|")
    for r in sorted(by_repo):
        L.append(f"| {r} | {len(by_repo[r])} | {_fmt(_mean(by_repo[r]))} |")

    L.append("\n## Precision pro Kategorie\n")
    L.append("| Kategorie | Ø Precision |")
    L.append("|---|---|")
    for c in CATEGORIES:
        L.append(f"| {c} | {_fmt(cat_means[c])} |")

    L.append("\n## Precision pro Test-Case-Typ\n")
    L.append("| TC | Queries | Ø Precision |")
    L.append("|---|---|---|")
    for t in sorted(by_tc):
        L.append(f"| {t} | {len(by_tc[t])} | {_fmt(_mean(by_tc[t]))} |")

    L.append(f"\n## Schlechteste {len(worst)} Queries (Expansion vermutlich schädlich)\n")
    L.append("| Repo | sample_id | #Terme | Precision | Titel |")
    L.append("|---|---|---|---|---|")
    for q in worst:
        L.append(f"| {q['repo']} | {q['sample_id']} | {q['n_terms']} | "
                 f"{_fmt(q['precision'])} | {q['title'][:60]} |")

    if harmful is not None:
        L.append(f"\n## Echte Stage-1-Verschlechterung durch Expansion "
                 f"({args.baseline_condition} → {args.expanded_condition})\n")
        L.append(f"{len(harmful)} Queries: Baseline fand die Datei in Stage 1, mit Expansion nicht mehr.\n")
        L.append("| Repo | sample_id | Titel |")
        L.append("|---|---|---|")
        for h in harmful:
            L.append(f"| {h['repo']} | {h['sample_id']} | {h['title'][:60]} |")

    L.append("\n## Empfehlungen fürs Prompt-Design\n")
    for r in recs:
        L.append(f"- {r}")

    out.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── Write JSON detail ──
    jp = Path(args.json_output)
    jp.write_text(json.dumps({
        "overall_precision": overall,
        "by_repo": {r: _mean(v) for r, v in by_repo.items()},
        "by_category": cat_means,
        "by_tc": {t: _mean(v) for t, v in by_tc.items()},
        "n_scored": len(scored),
        "n_zero_precision": len(zero_prec),
        "missing_target_text": missing_text,
        "per_query": per_query,
        "harmful_queries": harmful,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Console summary ──
    print(f"Queries analysiert: {len(scored)} | Ø Precision: {_fmt(overall)}")
    print("Precision pro Kategorie:",
          ", ".join(f"{c}={_fmt(cat_means[c])}" for c in CATEGORIES))
    print(f"Precision 0 (Noise): {len(zero_prec)}/{len(scored)}")
    if harmful is not None:
        print(f"Echte Stage-1-Verschlechterung: {len(harmful)} Queries")
    print(f"Report:  {out}")
    print(f"Detail:  {jp}")


if __name__ == "__main__":
    main()
