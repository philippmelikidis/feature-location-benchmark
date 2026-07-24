#!/usr/bin/env python3
"""
reporting.py – Benchmark Report Generation (NS1)

Generates Markdown reports with:
- Macro-Average (repos equally weighted) vs Micro-Average (all samples)
- Breakdown by Repository (with query counts)
- Breakdown by Test Case Type (TC1/TC2/TC3)
- Retriever and Chunking comparison
- Aggregation logic clearly documented
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from benchmark.schemas import BenchmarkReport, RunResult, SampleMetrics
from benchmark.metrics import bootstrap_ci


# ──────────────────────────────────────────────────────────────
# Helper: compute mean safely
# ──────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _collect_per_sample(runs: List[RunResult]) -> List[SampleMetrics]:
    """Flatten per-sample metrics from multiple runs."""
    all_sm = []
    for run in runs:
        all_sm.extend(run.metrics.per_sample)
    return all_sm


def _macro_avg(runs: List[RunResult], metric_fn) -> float:
    """
    Macro-Average: compute metric per repo, then average across repos.
    Each repo contributes equally regardless of sample count.
    """
    repo_values: Dict[str, List[float]] = defaultdict(list)
    for run in runs:
        for sm in run.metrics.per_sample:
            if sm.repo_id:
                repo_values[sm.repo_id].append(metric_fn(sm))
    if not repo_values:
        return _mean([metric_fn(sm) for run in runs for sm in run.metrics.per_sample])
    return _mean([_mean(vals) for vals in repo_values.values()])


def _micro_avg(runs: List[RunResult], metric_fn) -> float:
    """Micro-Average: flat average over all samples (pandas dominates)."""
    all_vals = [metric_fn(sm) for run in runs for sm in run.metrics.per_sample]
    return _mean(all_vals)


# ──────────────────────────────────────────────────────────────
# Main Report Generator
# ──────────────────────────────────────────────────────────────

def generate_report(report: BenchmarkReport, output_path: str):
    """Generate a comprehensive Markdown report with NS1 improvements."""
    lines = []

    # ── Header ────────────────────────────────────────────
    lines.append("# FLBench – Feature Location Benchmark v1 – Ergebnisse")
    lines.append("")
    lines.append(f"> Generiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> Dataset Version: {report.dataset_version}")
    lines.append(f"> Runs: {len(report.runs)}")
    lines.append("")

    # ── Aggregation Note ──────────────────────────────────
    # Count samples per repo
    repo_counts: Dict[str, int] = defaultdict(int)
    for run in report.runs:
        for sm in run.metrics.per_sample:
            if sm.repo_id:
                repo_counts[sm.repo_id] = max(repo_counts[sm.repo_id],
                    sum(1 for s in run.metrics.per_sample if s.repo_id == sm.repo_id))

    total_samples = sum(repo_counts.values()) if repo_counts else sum(r.metrics.num_samples for r in report.runs)

    lines.append("> [!IMPORTANT]")
    lines.append("> **Aggregationshinweis:**")
    if repo_counts:
        counts_str = ", ".join(f"{repo}={n}" for repo, n in sorted(repo_counts.items()))
        lines.append(f"> Query-Verteilung: {counts_str}")
    lines.append(f"> Macro-Ø = Repos gleich gewichtet | Micro-Ø = alle Samples gleich gewichtet")
    lines.append("")

    # ── 1. Gesamtübersicht ────────────────────────────────
    # 95%-Bootstrap-CIs aus den per-Sample-Werten — macht die
    # Unsicherheit der Punktwerte sichtbar (bei n=50 ist das CI ±~14 pp!).
    lines.append("## 1. Gesamtübersicht")
    lines.append("")
    lines.append("> Recall/MRR mit 95%-Bootstrap-CI (Percentile, 1000 Resamples).")
    lines.append("")
    lines.append("| Condition | k | Retriever | Chunking | Embedding | Recall@k [95%-CI] | MRR@k [95%-CI] | n |")
    lines.append("|-----------|---|-----------|----------|-----------|----------|-------|---|")

    for run in sorted(report.runs, key=lambda r: (r.condition_id, r.k)):
        embed = run.embedding_model.split("/")[-1] if run.embedding_model else "–"
        recalls = [sm.recall_at_k for sm in run.metrics.per_sample]
        mrrs = [sm.mrr_at_k for sm in run.metrics.per_sample]
        if recalls:
            _, r_lo, r_hi = bootstrap_ci(recalls, n_resamples=1000)
            _, m_lo, m_hi = bootstrap_ci(mrrs, n_resamples=1000)
            recall_str = f"{run.metrics.recall_at_k:.4f} [{r_lo:.3f}, {r_hi:.3f}]"
            mrr_str = f"{run.metrics.mrr_at_k:.4f} [{m_lo:.3f}, {m_hi:.3f}]"
        else:
            recall_str = f"{run.metrics.recall_at_k:.4f}"
            mrr_str = f"{run.metrics.mrr_at_k:.4f}"
        lines.append(
            f"| {run.condition_id} | {run.k} | {run.retriever_type} | "
            f"{run.chunking_strategy} | {embed} | "
            f"{recall_str} | {mrr_str} | "
            f"{run.metrics.num_samples} |"
        )
    lines.append("")

    # ── 2. Beste Ergebnisse pro k ─────────────────────────
    lines.append("## 2. Beste Ergebnisse pro k-Wert")
    lines.append("")

    for k in sorted(set(r.k for r in report.runs)):
        k_runs = [r for r in report.runs if r.k == k]
        if k_runs:
            best_recall = max(k_runs, key=lambda r: r.metrics.recall_at_k)
            best_mrr = max(k_runs, key=lambda r: r.metrics.mrr_at_k)
            lines.append(f"### k = {k}")
            lines.append(f"- **Höchster Recall@{k}**: {best_recall.condition_id} "
                         f"({best_recall.metrics.recall_at_k:.4f}) – "
                         f"{best_recall.retriever_type}/{best_recall.chunking_strategy}")
            lines.append(f"- **Höchster MRR@{k}**: {best_mrr.condition_id} "
                         f"({best_mrr.metrics.mrr_at_k:.4f}) – "
                         f"{best_mrr.retriever_type}/{best_mrr.chunking_strategy}")
            lines.append("")

    # ── 3. Retriever-Vergleich (Macro + Micro) ────────────
    lines.append("## 3. Vergleich: Retriever-Typen")
    lines.append("")
    lines.append("| Retriever | Macro R@10 | Macro MRR@10 | Micro R@10 | Micro MRR@10 |")
    lines.append("|-----------|-----------|-------------|-----------|-------------|")

    retriever_groups: Dict[str, List[RunResult]] = defaultdict(list)
    for run in report.runs:
        retriever_groups[run.retriever_type].append(run)

    for ret_type in sorted(retriever_groups):
        runs_k10 = [r for r in retriever_groups[ret_type] if r.k == 10]
        if not runs_k10:
            runs_k10 = retriever_groups[ret_type]

        macro_r = _macro_avg(runs_k10, lambda sm: sm.recall_at_k)
        macro_m = _macro_avg(runs_k10, lambda sm: sm.mrr_at_k)
        micro_r = _micro_avg(runs_k10, lambda sm: sm.recall_at_k)
        micro_m = _micro_avg(runs_k10, lambda sm: sm.mrr_at_k)

        lines.append(f"| {ret_type} | {macro_r:.4f} | {macro_m:.4f} | {micro_r:.4f} | {micro_m:.4f} |")
    lines.append("")

    # ── 4. Chunking-Vergleich (Macro + Micro) ─────────────
    lines.append("## 4. Vergleich: Chunking-Strategien")
    lines.append("")
    lines.append("| Chunking | Macro R@10 | Macro MRR@10 | Micro R@10 | Micro MRR@10 |")
    lines.append("|----------|-----------|-------------|-----------|-------------|")

    chunking_groups: Dict[str, List[RunResult]] = defaultdict(list)
    for run in report.runs:
        chunking_groups[run.chunking_strategy].append(run)

    for chunk_type in sorted(chunking_groups):
        runs_k10 = [r for r in chunking_groups[chunk_type] if r.k == 10]
        if not runs_k10:
            runs_k10 = chunking_groups[chunk_type]

        macro_r = _macro_avg(runs_k10, lambda sm: sm.recall_at_k)
        macro_m = _macro_avg(runs_k10, lambda sm: sm.mrr_at_k)
        micro_r = _micro_avg(runs_k10, lambda sm: sm.recall_at_k)
        micro_m = _micro_avg(runs_k10, lambda sm: sm.mrr_at_k)

        lines.append(f"| {chunk_type} | {macro_r:.4f} | {macro_m:.4f} | {micro_r:.4f} | {micro_m:.4f} |")
    lines.append("")

    # ── 5. TC1/TC2/TC3 Breakdown ──────────────────────────
    lines.append("## 5. Breakdown: Test Case Types")
    lines.append("")
    lines.append("> TC1 (Lexikalisch) | TC2 (Semantisch) | TC3 (Strukturell)")
    lines.append("")

    # Collect per-condition TC breakdown at k=10
    k10_runs = [r for r in report.runs if r.k == 10]
    if not k10_runs:
        k10_runs = report.runs

    # Group by condition
    cond_runs: Dict[str, List[RunResult]] = defaultdict(list)
    for run in k10_runs:
        cond_runs[run.condition_id].append(run)

    # Check if we have TC info
    has_tc = any(sm.tc_type for run in k10_runs for sm in run.metrics.per_sample)

    if has_tc:
        # Per-condition, per-TC table
        tc_types = sorted(set(sm.tc_type for run in k10_runs
                              for sm in run.metrics.per_sample if sm.tc_type))

        lines.append(f"### MRR@10 nach Variante × TC (k=10)")
        lines.append("")
        tc_header = " | ".join(f"MRR {tc} (n)" for tc in tc_types)
        lines.append(f"| Condition | Retriever | {tc_header} |")
        lines.append(f"|-----------|-----------|{'|'.join(['----------'] * len(tc_types))}|")

        for cid in sorted(cond_runs):
            runs = cond_runs[cid]
            ret_type = runs[0].retriever_type if runs else "–"
            tc_cells = []
            for tc in tc_types:
                tc_samples = [sm for run in runs for sm in run.metrics.per_sample
                              if sm.tc_type == tc]
                if tc_samples:
                    avg_mrr = _mean([sm.mrr_at_k for sm in tc_samples])
                    tc_cells.append(f"{avg_mrr:.3f} ({len(tc_samples)})")
                else:
                    tc_cells.append("– (0)")
            lines.append(f"| {cid} | {ret_type} | {' | '.join(tc_cells)} |")

        lines.append("")

        # Also add Recall@10 TC table
        lines.append(f"### Recall@10 nach Variante × TC (k=10)")
        lines.append("")
        tc_header = " | ".join(f"R@10 {tc} (n)" for tc in tc_types)
        lines.append(f"| Condition | Retriever | {tc_header} |")
        lines.append(f"|-----------|-----------|{'|'.join(['----------'] * len(tc_types))}|")

        for cid in sorted(cond_runs):
            runs = cond_runs[cid]
            ret_type = runs[0].retriever_type if runs else "–"
            tc_cells = []
            for tc in tc_types:
                tc_samples = [sm for run in runs for sm in run.metrics.per_sample
                              if sm.tc_type == tc]
                if tc_samples:
                    avg_r = _mean([sm.recall_at_k for sm in tc_samples])
                    tc_cells.append(f"{avg_r:.3f} ({len(tc_samples)})")
                else:
                    tc_cells.append("– (0)")
            lines.append(f"| {cid} | {ret_type} | {' | '.join(tc_cells)} |")

        lines.append("")

        # TC Distribution summary
        lines.append("### TC-Verteilung im Dataset")
        lines.append("")
        lines.append("| TC | Beschreibung | n | Anteil |")
        lines.append("|----|-------------|---|--------|")
        all_tc_samples = [sm for run in k10_runs for sm in run.metrics.per_sample if sm.tc_type]
        # Deduplicate by sample_id per condition (count unique per condition group)
        # Actually just use the first run per condition to count
        first_runs = {cid: runs[0] for cid, runs in cond_runs.items() if runs}
        if first_runs:
            sample_tc = defaultdict(int)
            first_run = list(first_runs.values())[0]
            for sm in first_run.metrics.per_sample:
                if sm.tc_type:
                    sample_tc[sm.tc_type] += 1
            total_tc = sum(sample_tc.values())
            tc_desc = {"TC1": "Lexikalisch", "TC2": "Semantisch", "TC3": "Strukturell"}
            for tc in sorted(sample_tc):
                pct = sample_tc[tc] / total_tc * 100 if total_tc else 0
                lines.append(f"| {tc} | {tc_desc.get(tc, '–')} | {sample_tc[tc]} | {pct:.0f}% |")
            lines.append("")
    else:
        lines.append("*Keine TC-Daten in per-sample Metriken verfügbar.*")
        lines.append("")

    # ── 6. Repo Breakdown ─────────────────────────────────
    lines.append("## 6. Ergebnisse pro Repository")
    lines.append("")

    repo_groups: Dict[str, List[RunResult]] = defaultdict(list)
    for run in report.runs:
        repo_groups[run.repo_id].append(run)

    for repo in sorted(repo_groups):
        runs = repo_groups[repo]
        n_samples = runs[0].metrics.num_samples if runs else 0
        lines.append(f"### {repo} (n={n_samples})")
        lines.append("")
        lines.append("| Condition | k | Recall@k | MRR@k |")
        lines.append("|-----------|---|----------|-------|")
        for run in sorted(runs, key=lambda r: (r.condition_id, r.k)):
            lines.append(f"| {run.condition_id} | {run.k} | "
                         f"{run.metrics.recall_at_k:.4f} | {run.metrics.mrr_at_k:.4f} |")
        lines.append("")

    # ── 7. Macro vs Micro Gesamtvergleich ────────────────
    lines.append("## 7. Macro vs. Micro Gesamtvergleich (k=10)")
    lines.append("")
    lines.append("> Macro = Repos gleich gewichtet (Kennzahl in der BA)")
    lines.append("> Micro = Samples gleich gewichtet (pandas mit 95 Samples dominiert)")
    lines.append("")
    lines.append("| Condition | Retriever | Macro R@10 | Macro MRR@10 | Micro R@10 | Micro MRR@10 |")
    lines.append("|-----------|-----------|-----------|-------------|-----------|-------------|")

    for cid in sorted(cond_runs):
        runs = cond_runs[cid]
        ret_type = runs[0].retriever_type if runs else "–"

        macro_r = _macro_avg(runs, lambda sm: sm.recall_at_k)
        macro_m = _macro_avg(runs, lambda sm: sm.mrr_at_k)
        micro_r = _micro_avg(runs, lambda sm: sm.recall_at_k)
        micro_m = _micro_avg(runs, lambda sm: sm.mrr_at_k)

        lines.append(f"| {cid} | {ret_type} | {macro_r:.4f} | {macro_m:.4f} | {micro_r:.4f} | {micro_m:.4f} |")
    lines.append("")

    # ── 8. Hierarchical Retrieval (V11) Diagnostics ──────────────
    v11_runs_k10 = [r for r in k10_runs if r.condition_id.startswith("V11")]
    if v11_runs_k10:
        lines.append("## 8. Hierarchical Retrieval (V11) Diagnostics")
        lines.append("")
        lines.append("> Diese Sektion existiert nur, wenn V11 in diesem Run war.")
        lines.append("> Sie trennt **Coarse-Miss** (Stage 1 hatte die GT-Datei nicht im")
        lines.append("> Kandidatenset) von **Fine-Miss** (Stage 1 fand sie, Stage 2 nicht")
        lines.append("> hochgeranked) — das ist die wichtigste Frage für jede V11-Iteration.")
        lines.append("")

        # ── 8a. Stage-1 hit rate per V11 variant per repo ──
        lines.append("### 8a. Stage-1 File-Hit-Rate (gefiltert auf V11-Runs, k=10)")
        lines.append("")
        lines.append("> stage1_hit = mind. eine GT-Datei lag im Stage-1-Kandidatenset.")
        lines.append("> Theoretischer Recall@10-Cap = stage1_hit_rate.")
        lines.append("")
        v11_repo_set = sorted({r.repo_id for r in v11_runs_k10})
        header = "| Condition | N | " + " | ".join(f"{repo}" for repo in v11_repo_set) + " | Macro Ø |"
        sep = "|-----------|---|" + "|".join(["----------"] * (len(v11_repo_set) + 1)) + "|"
        lines.append(header)
        lines.append(sep)

        v11_by_cond: Dict[str, List[RunResult]] = defaultdict(list)
        for r in v11_runs_k10:
            v11_by_cond[r.condition_id].append(r)

        for cid in sorted(v11_by_cond):
            runs = v11_by_cond[cid]
            # top_n_files: read off the first sample that has it set
            n_files = next(
                (sm.stage1_n_files for run in runs for sm in run.metrics.per_sample
                 if sm.stage1_n_files is not None),
                "–",
            )
            cells = []
            macro_vals = []
            for repo in v11_repo_set:
                samples = [sm for run in runs for sm in run.metrics.per_sample
                           if sm.repo_id == repo and sm.stage1_hit is not None]
                if samples:
                    rate = sum(1 for sm in samples if sm.stage1_hit) / len(samples)
                    cells.append(f"{rate:.3f} ({len(samples)})")
                    macro_vals.append(rate)
                else:
                    cells.append("– (0)")
            macro = _mean(macro_vals) if macro_vals else 0.0
            lines.append(f"| {cid} | {n_files} | {' | '.join(cells)} | {macro:.3f} |")
        lines.append("")

        # ── 8b. Coarse-vs-Fine miss decomposition (per condition, micro) ──
        lines.append("### 8b. Miss-Decomposition (Recall@10)")
        lines.append("")
        lines.append("Spalte | Bedeutung")
        lines.append("--- | ---")
        lines.append("Stage-1 hit % | % Samples mit ≥1 GT-Datei in Coarse-Kandidaten")
        lines.append("R@10 | tatsächliches Recall@10 nach Stage-2")
        lines.append("Cap (= Stage-1 hit %) | obere Schranke — Stage-2 kann nicht über Stage-1 hinaus")
        lines.append("Fine-loss | Stage-1 hit % − R@10. Differenz zwischen Cap und Realität.")
        lines.append("")
        lines.append("| Condition | Stage-1 hit % | R@10 | Fine-loss |")
        lines.append("|-----------|---------------|------|-----------|")
        for cid in sorted(v11_by_cond):
            runs = v11_by_cond[cid]
            samples = [sm for run in runs for sm in run.metrics.per_sample
                       if sm.stage1_hit is not None]
            if not samples:
                continue
            stage1_rate = sum(1 for sm in samples if sm.stage1_hit) / len(samples)
            r10 = _mean([sm.recall_at_k for sm in samples])
            fine_loss = stage1_rate - r10
            lines.append(f"| {cid} | {stage1_rate:.3f} | {r10:.3f} | {fine_loss:+.3f} |")
        lines.append("")
        lines.append("> Hohe Fine-loss → Stage-2-Ranking ist das Problem (mehr Embedding-Power, größeres k).")
        lines.append("> Niedrige Stage-1 hit % → Stage-1 ist das Problem (top_n_files erhöhen, hybrid statt sparse).")
        lines.append("")

        # ── 8c. V11 by TC type ──
        if has_tc:
            lines.append("### 8c. V11 nach TC-Typ (Recall@10 / MRR@10)")
            lines.append("")
            lines.append("> Erwartung: V11 zieht v.a. bei TC2/TC3 (semantisch/strukturell),")
            lines.append("> wo der File-Filter aus Stage 1 das semantische Rauschen wegnimmt.")
            lines.append("> Bei TC1 (lexikalisch) sollte V11 nicht schlechter als V1/V5 sein.")
            lines.append("")
            tc_types = sorted(set(sm.tc_type for run in v11_runs_k10
                                  for sm in run.metrics.per_sample if sm.tc_type))
            tc_header = " | ".join(f"R@10 {tc} | MRR@10 {tc}" for tc in tc_types)
            lines.append(f"| Condition | {tc_header} |")
            lines.append("|-----------|" + "|".join(["----------"] * (2 * len(tc_types))) + "|")
            for cid in sorted(v11_by_cond):
                runs = v11_by_cond[cid]
                cells = []
                for tc in tc_types:
                    tc_samples = [sm for run in runs for sm in run.metrics.per_sample
                                  if sm.tc_type == tc]
                    if tc_samples:
                        cells.append(f"{_mean([sm.recall_at_k for sm in tc_samples]):.3f}")
                        cells.append(f"{_mean([sm.mrr_at_k for sm in tc_samples]):.3f}")
                    else:
                        cells.append("–")
                        cells.append("–")
                lines.append(f"| {cid} | {' | '.join(cells)} |")
            lines.append("")

    # ── 9. Methodik ──────────────────────────────────────
    lines.append("")
    lines.append("## 9. Methodik")
    lines.append("")
    lines.append("### Metriken")
    lines.append("- **Recall@K**: Anteil der Ground-Truth-Targets in den Top-K Ergebnissen")
    lines.append("- **MRR@K**: Mean Reciprocal Rank – 1/Rang des ersten relevanten Treffers")
    lines.append("")
    lines.append("### Aggregation")
    lines.append("- **Macro-Ø**: Berechne Metrik pro Repo → Durchschnitt der Repo-Mittelwerte")
    lines.append("- **Micro-Ø**: Flacher Durchschnitt über alle Samples (große Repos dominieren)")
    lines.append("- **Empfehlung**: Macro-Ø für Vergleiche (Repos gleich gewichtet)")
    lines.append("")
    lines.append("### Test Case Types")
    lines.append("- **TC1 (Lexikalisch)**: Query enthält Code-Identifier → BM25 sollte greifen")
    lines.append("- **TC2 (Semantisch)**: Beschreibung ohne Code-Bezeichner → Dense/Embedding")
    lines.append("- **TC3 (Strukturell)**: Verständnis der Repo-Architektur nötig → Struktur-Chunking")
    lines.append("")

    # Write
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report generiert: {output_path}")


def generate_json_export(report: BenchmarkReport, output_path: str):
    """Export full results as JSON (for further analysis)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--input", type=str, default="benchmark/results/benchmark_results.json",
                        help="Path to results JSON")
    parser.add_argument("--output", type=str, default="benchmark/results/benchmark_report.md",
                        help="Output report path")

    args = parser.parse_args()

    import json
    with open(args.input) as f:
        data = json.load(f)
    report = BenchmarkReport(**data)
    generate_report(report, args.output)
