#!/usr/bin/env python3
"""
metrics.py – Retrieval Evaluation Metrics

Primary metrics (spec Section 7):
- Recall@K (K = 1, 5, 10)
- MRR@K (Mean Reciprocal Rank)

Statistik:
- bootstrap_ci: 95%-Konfidenzintervall per Bootstrap (Percentile-Methode)
- mcnemar_exact: gepaarter Test für binäre Hits (exakter Binomialtest)
- paired_bootstrap_diff: gepaarter Bootstrap-Test für kontinuierliche
  Metriken (MRR), zweiseitiger p-Wert
Alle deterministisch (fester Seed) und stdlib-only.
"""

import math
import random
from typing import List, Dict, Optional, Any, Tuple

from benchmark.chunking.base import Chunk


def recall_at_k(
    retrieved: List[Chunk],
    ground_truth_files: List[str],
    k: int,
    ground_truth_functions: Optional[List[str]] = None,
) -> float:
    """
    Compute Recall@K: fraction of ground-truth targets found in top-K results.

    Args:
        retrieved: List of retrieved Chunk objects (already sorted by relevance).
        ground_truth_files: List of ground-truth file paths.
        k: Number of top results to consider.
        ground_truth_functions: Optional list of function names for finer matching.

    Returns:
        Recall@K score between 0.0 and 1.0.
    """
    if not ground_truth_files:
        return 0.0

    top_k = retrieved[:k]

    if ground_truth_functions:
        # Function-level matching
        targets = set(zip(ground_truth_files, ground_truth_functions))
        hits = 0
        for chunk in top_k:
            for gt_file, gt_func in targets:
                if _file_matches(chunk.file_path, gt_file):
                    if gt_func and chunk.function_name:
                        if gt_func in chunk.function_name or chunk.function_name in gt_func:
                            hits += 1
                            break
                    else:
                        hits += 1
                        break
        return hits / len(targets)
    else:
        # File-level matching
        gt_set = set(ground_truth_files)
        retrieved_files = set()
        for chunk in top_k:
            for gt_file in gt_set:
                if _file_matches(chunk.file_path, gt_file):
                    retrieved_files.add(gt_file)

        return len(retrieved_files) / len(gt_set)


def mrr_at_k(
    retrieved: List[Chunk],
    ground_truth_files: List[str],
    k: int,
    ground_truth_functions: Optional[List[str]] = None,
) -> float:
    """
    Compute MRR@K: 1/rank of first relevant result.

    Args:
        retrieved: List of retrieved Chunk objects (sorted by relevance).
        ground_truth_files: List of ground-truth file paths.
        k: Maximum rank to consider.
        ground_truth_functions: Optional list of function names.

    Returns:
        MRR@K score between 0.0 and 1.0.
    """
    if not ground_truth_files:
        return 0.0

    top_k = retrieved[:k]
    gt_set = set(ground_truth_files)

    for rank, chunk in enumerate(top_k, start=1):
        for gt_file in gt_set:
            if _file_matches(chunk.file_path, gt_file):
                if ground_truth_functions:
                    # Need function match too
                    for gt_func in ground_truth_functions:
                        if gt_func and chunk.function_name:
                            if gt_func in chunk.function_name or chunk.function_name in gt_func:
                                return 1.0 / rank
                else:
                    return 1.0 / rank

    return 0.0


def first_hit_rank(
    retrieved: List[Chunk],
    ground_truth_files: List[str],
    ground_truth_functions: Optional[List[str]] = None,
) -> Optional[int]:
    """
    Find the rank of the first ground-truth hit.

    Returns:
        Rank (1-indexed) or None if no hit found.
    """
    gt_set = set(ground_truth_files)

    for rank, chunk in enumerate(retrieved, start=1):
        for gt_file in gt_set:
            if _file_matches(chunk.file_path, gt_file):
                if ground_truth_functions:
                    for gt_func in ground_truth_functions:
                        if gt_func and chunk.function_name:
                            if gt_func in chunk.function_name or chunk.function_name in gt_func:
                                return rank
                else:
                    return rank

    return None


def compute_all_metrics(
    retrieved: List[Chunk],
    ground_truth_files: List[str],
    ground_truth_functions: Optional[List[str]] = None,
    k_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Compute all metrics for a single query result.

    Returns:
        Dict with recall@1, recall@5, recall@10, mrr@10, first_hit_rank.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    results = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(
            retrieved, ground_truth_files, k, ground_truth_functions
        )
        results[f"mrr@{k}"] = mrr_at_k(
            retrieved, ground_truth_files, k, ground_truth_functions
        )

    results["first_hit_rank"] = first_hit_rank(
        retrieved, ground_truth_files, ground_truth_functions
    )

    return results


def aggregate_metrics(
    all_sample_metrics: List[Dict[str, Any]],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Aggregate metrics across all samples (mean).

    Args:
        all_sample_metrics: List of per-sample metric dicts.

    Returns:
        Dict with mean values for each metric.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    if not all_sample_metrics:
        return {f"recall@{k}": 0.0 for k in k_values} | {f"mrr@{k}": 0.0 for k in k_values}

    aggregated = {}
    for k in k_values:
        recall_key = f"recall@{k}"
        mrr_key = f"mrr@{k}"

        recalls = [m[recall_key] for m in all_sample_metrics if recall_key in m]
        mrrs = [m[mrr_key] for m in all_sample_metrics if mrr_key in m]

        aggregated[recall_key] = sum(recalls) / len(recalls) if recalls else 0.0
        aggregated[mrr_key] = sum(mrrs) / len(mrrs) if mrrs else 0.0

    # Average first hit rank (excluding None)
    hit_ranks = [
        m["first_hit_rank"]
        for m in all_sample_metrics
        if m.get("first_hit_rank") is not None
    ]
    aggregated["avg_first_hit_rank"] = (
        sum(hit_ranks) / len(hit_ranks) if hit_ranks else float("inf")
    )
    aggregated["hit_rate"] = len(hit_ranks) / len(all_sample_metrics) if all_sample_metrics else 0.0

    return aggregated


# ──────────────────────────────────────────────────────────────
# Statistik: CIs + gepaarte Signifikanztests
# ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: List[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """95%-Bootstrap-Konfidenzintervall (Percentile-Methode) für den Mittelwert.

    Returns:
        (mean, ci_lo, ci_hi). Bei leerer Liste (0.0, 0.0, 0.0); bei n=1
        degeneriert das CI auf den Punktwert.
    """
    if not values:
        return (0.0, 0.0, 0.0)
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return (mean, mean, mean)

    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(0, int(alpha * n_resamples))]
    hi = means[min(n_resamples - 1, int((1.0 - alpha) * n_resamples))]
    return (mean, lo, hi)


def mcnemar_exact(hits_a: List[bool], hits_b: List[bool]) -> Dict[str, Any]:
    """Exakter McNemar-Test für gepaarte binäre Ergebnisse (Recall-Hits).

    Vergleicht Condition A und B auf DENSELBEN Queries (gleiche Reihenfolge!).
    Nur die diskordanten Paare tragen Information:
        b = A trifft, B nicht   |   c = B trifft, A nicht
    Exakter zweiseitiger Binomialtest auf b ~ Bin(b+c, 0.5) — korrekt auch
    bei kleinen Fallzahlen (im Gegensatz zur Chi²-Approximation).

    Returns:
        {"n": Paare, "b": …, "c": …, "n_discordant": b+c, "p_value": …}
    """
    if len(hits_a) != len(hits_b):
        raise ValueError(f"Ungepaarte Listen: {len(hits_a)} vs. {len(hits_b)}")
    b = sum(1 for x, y in zip(hits_a, hits_b) if x and not y)
    c = sum(1 for x, y in zip(hits_a, hits_b) if y and not x)
    n_disc = b + c
    if n_disc == 0:
        p = 1.0  # keine diskordanten Paare → kein Unterschied messbar
    else:
        k = min(b, c)
        # zweiseitig: 2 * P(X <= k), X ~ Bin(n_disc, 0.5); bei b == c ist p = 1
        cdf = sum(math.comb(n_disc, i) for i in range(k + 1)) * (0.5 ** n_disc)
        p = min(1.0, 2.0 * cdf)
    return {"n": len(hits_a), "b": b, "c": c, "n_discordant": n_disc, "p_value": p}


def paired_bootstrap_diff(
    values_a: List[float],
    values_b: List[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Gepaarter Bootstrap-Test für kontinuierliche Metriken (z. B. MRR).

    Resampelt die per-Query-DIFFERENZEN (a_i − b_i) und liefert Mittelwert,
    CI (Percentile) und zweiseitigen p-Wert (Anteil der Bootstrap-Mittel auf
    der „falschen" Seite von 0, ×2, mit +1-Glättung gegen p=0).
    """
    if len(values_a) != len(values_b):
        raise ValueError(f"Ungepaarte Listen: {len(values_a)} vs. {len(values_b)}")
    if not values_a:
        return {"n": 0, "mean_diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_value": 1.0}

    diffs = [a - b for a, b in zip(values_a, values_b)]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    if n == 1 or all(d == diffs[0] for d in diffs):
        # degeneriert: keine Varianz → CI = Punkt, p informationslos
        return {"n": n, "mean_diff": mean_diff, "ci_lo": mean_diff,
                "ci_hi": mean_diff, "p_value": 1.0 if mean_diff == 0 else 0.0}

    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_resamples):
        resample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(resample) / n)
    boot_means.sort()

    alpha = (1.0 - confidence) / 2.0
    lo = boot_means[max(0, int(alpha * n_resamples))]
    hi = boot_means[min(n_resamples - 1, int((1.0 - alpha) * n_resamples))]

    n_le = sum(1 for m in boot_means if m <= 0)
    n_ge = sum(1 for m in boot_means if m >= 0)
    p = 2.0 * (min(n_le, n_ge) + 1) / (n_resamples + 1)
    return {"n": n, "mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi,
            "p_value": min(1.0, p)}


def _file_matches(retrieved_path: str, ground_truth_path: str) -> bool:
    """
    Check if a retrieved file path matches a ground-truth path.
    Handles relative/absolute path differences.
    """
    # Normalize paths
    r = retrieved_path.replace("\\", "/").strip("/")
    g = ground_truth_path.replace("\\", "/").strip("/")

    # Exact match
    if r == g:
        return True

    # One could be a suffix of the other
    if r.endswith(g) or g.endswith(r):
        return True

    # Just filename match as fallback
    r_name = r.split("/")[-1]
    g_name = g.split("/")[-1]
    if r_name == g_name and r_name != "__init__.py":
        # Same filename (but not generic __init__.py)
        return True

    return False
