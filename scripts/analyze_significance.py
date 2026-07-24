#!/usr/bin/env python3
"""
analyze_significance.py – Gepaarte Signifikanztests für Condition-Vergleiche.

Vergleicht zwei Conditions auf DENSELBEN Queries (Pairing über sample_id):
  * Recall-Hits (binär, recall_at_k > 0): exakter McNemar-Test
  * MRR (kontinuierlich): gepaarter Bootstrap-Test (Differenzen)

KEIN Elasticsearch/LLM nötig — arbeitet auf gespeicherten Ergebnis-JSONs
(benchmark_results_*.json mit per_sample-Einträgen).

Aufruf:
  # Beide Conditions im selben Ergebnis-Ordner (z. B. v2-Baseline):
  python scripts/analyze_significance.py \
      --results-a benchmark/results/v2_baseline --condition-a V16c \
      --results-b benchmark/results/v2_baseline --condition-b V12b

  # Conditions aus verschiedenen Läufen (z. B. lean vs. baseline):
  python scripts/analyze_significance.py \
      --results-a benchmark/results/variant_lean --condition-a V16c \
      --results-b benchmark/results/variant_lean_baseline --condition-b V16c \
      --label-a "V16c (lean)" --label-b "V16c (baseline)"
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.metrics import mcnemar_exact, paired_bootstrap_diff, bootstrap_ci  # noqa: E402

ALPHA = 0.05


def load_per_sample(results_dir: Path, condition_id: str, k: int):
    """{sample_id: {"recall": …, "mrr": …, "repo": …}} für eine Condition.

    Liest alle vollständigen Ergebnis-Files des Ordners chronologisch;
    bei Wiederholungen gewinnt der spätere Run.
    """
    files = sorted(results_dir.glob("benchmark_results_*.json"))
    files = [f for f in files if "PARTIAL" not in f.name and "latest" not in f.name]
    out = {}
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for run in data.get("runs", []):
            if run.get("condition_id") != condition_id or run.get("k") != k:
                continue
            for sm in (run.get("metrics") or {}).get("per_sample", []):
                sid = sm.get("sample_id")
                if sid is None:
                    continue
                out[sid] = {
                    "recall": sm.get("recall_at_k", 0.0),
                    "mrr": sm.get("mrr_at_k", 0.0),
                    "repo": (sm.get("repo_id") or "?").split("/")[-1],
                }
    return out


def compare(a, b, label_a, label_b, k):
    """Gepaarter Vergleich zweier per-Sample-Maps. Gibt Markdown-Zeilen zurück."""
    paired_ids = sorted(set(a) & set(b))
    only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
    if not paired_ids:
        print("  Keine gemeinsamen sample_ids — Vergleich nicht möglich.")
        return []

    lines = []
    lines.append(f"### {label_a} vs. {label_b} (k={k})")
    lines.append("")
    lines.append(f"Gepaarte Queries: **{len(paired_ids)}**"
                 + (f" (ungepaart ignoriert: {only_a}/{only_b})" if only_a or only_b else ""))
    lines.append("")
    lines.append("| Ebene | n | Metrik | " + label_a + " | " + label_b +
                 " | Δ | Test | p-Wert | signifikant (α=0.05) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    by_repo = defaultdict(list)
    for sid in paired_ids:
        by_repo[a[sid]["repo"]].append(sid)

    def _rows(scope, ids):
        ra = [a[s]["recall"] for s in ids]
        rb = [b[s]["recall"] for s in ids]
        ha = [x > 0 for x in ra]
        hb = [x > 0 for x in rb]
        mc = mcnemar_exact(ha, hb)
        mean_ra, mean_rb = sum(ra) / len(ra), sum(rb) / len(rb)
        sig_r = "ja" if mc["p_value"] < ALPHA else "— nein"
        lines.append(
            f"| {scope} | {len(ids)} | Recall@{k} | {mean_ra:.4f} | {mean_rb:.4f} "
            f"| {mean_ra - mean_rb:+.4f} | McNemar (b={mc['b']}, c={mc['c']}) "
            f"| {mc['p_value']:.4f} | {sig_r} |")

        ma = [a[s]["mrr"] for s in ids]
        mb = [b[s]["mrr"] for s in ids]
        bt = paired_bootstrap_diff(ma, mb)
        sig_m = "ja" if bt["p_value"] < ALPHA else "— nein"
        lines.append(
            f"| {scope} | {len(ids)} | MRR@{k} | {sum(ma)/len(ma):.4f} | {sum(mb)/len(mb):.4f} "
            f"| {bt['mean_diff']:+.4f} [{bt['ci_lo']:+.3f}, {bt['ci_hi']:+.3f}] "
            f"| gepaarter Bootstrap | {bt['p_value']:.4f} | {sig_m} |")

    _rows("**gesamt**", paired_ids)
    for repo in sorted(by_repo):
        _rows(repo, by_repo[repo])
    lines.append("")

    for ln in lines:
        print("  " + ln)
    return lines


def main():
    ap = argparse.ArgumentParser(description="Gepaarte Signifikanztests")
    ap.add_argument("--results-a", required=True, help="Ergebnis-Ordner Condition A")
    ap.add_argument("--condition-a", required=True)
    ap.add_argument("--results-b", required=True, help="Ergebnis-Ordner Condition B")
    ap.add_argument("--condition-b", required=True)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--output", default=None, help="Markdown-Report anhängen/schreiben")
    args = ap.parse_args()

    label_a = args.label_a or args.condition_a
    label_b = args.label_b or args.condition_b

    a = load_per_sample(Path(args.results_a), args.condition_a, args.k)
    b = load_per_sample(Path(args.results_b), args.condition_b, args.k)
    print(f"\n{label_a}: {len(a)} Samples | {label_b}: {len(b)} Samples")

    lines = compare(a, b, label_a, label_b, args.k)

    if args.output and lines:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n  Angehängt an: {out}")


if __name__ == "__main__":
    main()
