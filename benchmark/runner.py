#!/usr/bin/env python3
"""
runner.py – Benchmark Pipeline Orchestrator (Elasticsearch-based)

Runs the full Feature Location benchmark using Elasticsearch:
1. Load/clone repository
2. Load dataset
3. For each condition: chunk -> index in ES -> retrieve via ES -> compute metrics
4. Aggregate results and generate report

Usage:
    python -m benchmark.runner --conditions all --repos requests flask
    python -m benchmark.runner --condition V1 --k 5 --repo requests --verbose
    python -m benchmark.runner --dry-run
"""

import gc
import json
import os
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import (
    CONDITIONS, CONDITIONS_MAP, K_VALUES, REPOSITORIES, REPOS_MAP,
    ConditionConfig, ES_URL, ES_INDEX_PREFIX, get_all_run_configs,
)
from benchmark.schemas import (
    Sample, BenchmarkDataset, RunResult, RunMetrics, SampleMetrics,
    RetrievalResult, RetrievedItem, BenchmarkReport,
)
from benchmark.chunking import CHUNKER_REGISTRY
from benchmark.chunking.base import Chunk
from benchmark.metrics import compute_all_metrics, aggregate_metrics
from benchmark.reporting import generate_report

import re

# ──────────────────────────────────────────────────────────────
# Query Preprocessing
# ──────────────────────────────────────────────────────────────

def preprocess_query(query: str, mode: str = "full",
                     expansion_lookup=None) -> str:
    """Transform the raw query based on the preprocessing mode.

    Modes:
        "full":         Use the entire query as-is (default, backward compat).
        "title_only":   Use only the first line (issue title).
        "cleaned":      Remove boilerplate, checkboxes, code blocks.
        "llm_expanded": Use pre-computed LLM expansion (title + code terms).
                        Requires expansion_lookup to be set. Falls back to
                        title_only if no expansion found.
        "full_expanded": Full issue text + appended LLM expansion terms.
                        Gives dense/hybrid both semantic context and code terms.
    """
    if mode == "full":
        return query

    if mode == "title_only":
        # First non-empty line = issue title
        for line in query.split("\n"):
            line = line.strip()
            if line:
                return line
        return query

    if mode == "llm_expanded":
        if expansion_lookup is not None:
            expanded = expansion_lookup.lookup(query)
            if expanded:
                return expanded
        # Fallback: title only
        for line in query.split("\n"):
            line = line.strip()
            if line:
                return line
        return query

    if mode == "full_expanded":
        if expansion_lookup is not None:
            expanded = expansion_lookup.lookup(query)
            if expanded:
                # Append LLM terms to full query
                return f"{query}\n\n{expanded}"
        return query

    if mode == "cleaned":
        lines = query.split("\n")
        cleaned = []
        in_code_block = False
        for line in lines:
            # Toggle code blocks
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            # Skip markdown checkboxes
            if re.match(r'\s*-\s*\[[ xX]\]', line):
                continue
            # Skip common boilerplate headers
            if line.strip() in (
                "### Pandas version checks",
                "### Feature Type",
                "### Reproducible Example",
                "### Expected Behavior",
                "### Installed Versions",
                "### Issue Description",
                "### Steps to Reproduce",
            ):
                continue
            # Skip empty markdown headers with nothing useful
            if re.match(r'^###?\s*$', line.strip()):
                continue
            cleaned.append(line)

        result = "\n".join(cleaned).strip()
        # Limit to ~300 chars to keep embedding focused
        if len(result) > 300:
            result = result[:300]
        return result if result else query

    return query


# Shared expansion lookups, cached per repo (lazy-loaded once per process/repo)
_expansion_lookups: Dict[str, object] = {}

def _get_expansion_lookup(repo_id: str = None):
    """Get or create the ExpansionLookup for flat LLM-expanded variants.

    Repo-aware: each repo loads data/llm_expansions_<repo>.json. Falls back to
    the pandas default when repo_id is None (backward compatible).
    """
    from benchmark.retrievers.hierarchical_v16_retriever import (
        ExpansionLookup, expansions_file_for_repo,
    )
    key = str(repo_id or "pandas")
    if key not in _expansion_lookups:
        _expansion_lookups[key] = ExpansionLookup(expansions_file_for_repo(repo_id))
    return _expansion_lookups[key]


def create_retriever(condition: ConditionConfig, es_url: str = None, repo_id: str = None):
    """Create the appropriate retriever for the given condition.

    repo_id selects the repo-specific pre-computed LLM expansion file
    (data/llm_expansions_<repo>.json) for V16/Ensemble retrievers. If None,
    the pandas default is used (backward compatible).
    """
    if condition.is_hierarchical:
        from benchmark.config import RetrieverType
        from benchmark.retrievers.hierarchical_v16_retriever import expansions_file_for_repo
        _exp_file = expansions_file_for_repo(repo_id)
        coarse_chunker = create_chunker_by_strategy(condition.coarse_chunking_strategy.value)
        fine_chunker = create_chunker(condition)  # uses chunking_strategy = fine

        if condition.retriever_type == RetrieverType.HIERARCHICAL_ENSEMBLE:
            from benchmark.retrievers.hierarchical_ensemble_retriever import HierarchicalEnsembleRetriever
            coarse_chunker_b = create_chunker_by_strategy(
                condition.coarse_chunking_strategy_b.value
            )
            return HierarchicalEnsembleRetriever(
                es_url=es_url or ES_URL,
                index_prefix=ES_INDEX_PREFIX,
                condition_id=condition.condition_id,
                coarse_chunker_a=coarse_chunker,
                coarse_retriever_type_a=condition.coarse_retriever_type.value
                    if condition.coarse_retriever_type else "sparse",
                coarse_chunker_b=coarse_chunker_b,
                coarse_retriever_type_b=condition.coarse_retriever_type_b.value
                    if condition.coarse_retriever_type_b else "sparse",
                fine_chunker=fine_chunker,
                fine_retriever_type=condition.fine_retriever_type.value
                    if condition.fine_retriever_type else "hybrid",
                embedding_model_name=condition.embedding_name,
                embedding_dims=condition.embedding_dims,
                hybrid_alpha=condition.hybrid_alpha,
                top_n_files=condition.top_n_files,
                ensemble_top_n=condition.ensemble_top_n,
                merge_strategy=condition.ensemble_merge,
                expansions_file=_exp_file,
                stage1_query_mode=condition.stage1_query_mode,
            )

        if condition.retriever_type == RetrieverType.HIERARCHICAL_V16:
            from benchmark.retrievers.hierarchical_v16_retriever import HierarchicalV16Retriever
            return HierarchicalV16Retriever(
                es_url=es_url or ES_URL,
                index_prefix=ES_INDEX_PREFIX,
                condition_id=condition.condition_id,
                coarse_chunker=coarse_chunker,
                fine_chunker=fine_chunker,
                coarse_retriever_type=condition.coarse_retriever_type.value
                    if condition.coarse_retriever_type else "sparse",
                fine_retriever_type=condition.fine_retriever_type.value
                    if condition.fine_retriever_type else "dense",
                embedding_model_name=condition.embedding_name,
                embedding_dims=condition.embedding_dims,
                hybrid_alpha=condition.hybrid_alpha,
                top_n_files=condition.top_n_files,
                expansions_file=_exp_file,
                repo_name=(repo_id or "pandas"),
                llm_api_url=condition.llm_api_url,
                llm_model=condition.llm_model,
                llm_temperature=condition.llm_temperature,
                llm_timeout=condition.llm_timeout,
                stage1_query_mode=condition.stage1_query_mode,
                stage2_query_mode=condition.stage2_query_mode,
            )

        if condition.retriever_type == RetrieverType.HIERARCHICAL_V12:
            from benchmark.retrievers.hierarchical_v12_retriever import HierarchicalV12Retriever
            return HierarchicalV12Retriever(
                es_url=es_url or ES_URL,
                index_prefix=ES_INDEX_PREFIX,
                condition_id=condition.condition_id,
                coarse_chunker=coarse_chunker,
                fine_chunker=fine_chunker,
                coarse_retriever_type=condition.coarse_retriever_type.value
                    if condition.coarse_retriever_type else "sparse",
                fine_retriever_type=condition.fine_retriever_type.value
                    if condition.fine_retriever_type else "hybrid",
                embedding_model_name=condition.embedding_name,
                embedding_dims=condition.embedding_dims,
                hybrid_alpha=condition.hybrid_alpha,
                top_n_files=condition.top_n_files,
                stage2_strategy=condition.stage2_strategy or "score_propagation",
                score_lambda=condition.score_lambda,
                overfetch_multiplier=condition.overfetch_multiplier,
            )

        from benchmark.retrievers.hierarchical_retriever import HierarchicalRetriever
        return HierarchicalRetriever(
            es_url=es_url or ES_URL,
            index_prefix=ES_INDEX_PREFIX,
            condition_id=condition.condition_id,
            coarse_chunker=coarse_chunker,
            fine_chunker=fine_chunker,
            coarse_retriever_type=condition.coarse_retriever_type.value
                if condition.coarse_retriever_type else "sparse",
            fine_retriever_type=condition.fine_retriever_type.value
                if condition.fine_retriever_type else "hybrid",
            embedding_model_name=condition.embedding_name,
            embedding_dims=condition.embedding_dims,
            hybrid_alpha=condition.hybrid_alpha,
            top_n_files=condition.top_n_files,
        )

    from benchmark.retrievers.es_retriever import ElasticsearchRetriever
    return ElasticsearchRetriever(
        es_url=es_url or ES_URL,
        index_prefix=ES_INDEX_PREFIX,
        condition_id=condition.condition_id,
        retriever_type=condition.retriever_type.value,
        embedding_model_name=condition.embedding_name,
        embedding_dims=condition.embedding_dims,
        hybrid_alpha=condition.hybrid_alpha,
    )


def create_chunker_by_strategy(strategy: str):
    """Create a chunker by strategy name string (used by HierarchicalRetriever)."""
    chunker_cls = CHUNKER_REGISTRY.get(strategy)
    if not chunker_cls:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    return chunker_cls()


def create_chunker(condition: ConditionConfig):
    """Create a chunker instance based on condition config."""
    strategy = condition.chunking_strategy.value
    chunker_cls = CHUNKER_REGISTRY.get(strategy)
    if not chunker_cls:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    return chunker_cls()


def load_dataset(dataset_path: str) -> BenchmarkDataset:
    """Load the benchmark dataset."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BenchmarkDataset(**data)


def clone_repository(repo_name: str, target_dir: str = "benchmark/repos") -> Path:
    """Clone a repository using the existing RepoDownloader."""
    repo_config = REPOS_MAP.get(repo_name)
    if not repo_config:
        raise ValueError(f"Unknown repository: {repo_name}")

    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    from repo_downloader import RepoDownloader

    downloader = RepoDownloader(target_dir=target_dir)
    info = downloader.clone(repo_config.url, depth=1)
    return info.path


def get_python_files(repo_path: Path, source_dirs: List[str] = None) -> List[Path]:
    """Get Python files from a repository, optionally filtered by source dirs."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    from repo_downloader import RepoDownloader

    downloader = RepoDownloader.__new__(RepoDownloader)
    all_files = downloader.get_python_files(repo_path, exclude_tests=True)

    if source_dirs:
        filtered = []
        for f in all_files:
            rel = str(f.relative_to(repo_path))
            if any(rel.startswith(d) for d in source_dirs):
                filtered.append(f)
        return filtered if filtered else all_files

    return all_files


def run_single_condition_all_k(
    condition: ConditionConfig,
    k_values: List[int],
    samples: List[Sample],
    repo_path: Path,
    python_files: List[Path],
    repo_config=None,
    dataset_version: str = "1.0.0",
    es_url: str = None,
    verbose: bool = False,
) -> List[RunResult]:
    """
    Run the benchmark for a single condition, retrieving once with k_max
    and deriving metrics for all k-values from the same results.

    Returns:
        List of RunResult, one per k-value.
    """
    k_max = max(k_values)
    emb = condition.embedding_model
    emb_label = f"{emb.label}: {emb.name}" if emb else "-"
    commit_hash = repo_config.commit_hash if repo_config else "HEAD"
    repo_id = repo_config.name if repo_config else str(repo_path.name)

    print(f"\n{'='*60}")
    print(f"RUN  {condition.condition_id} | repo={repo_id} | k={k_values} (retrieve k={k_max})")
    print(f"     retriever={condition.retriever_type.value} | "
          f"chunking={condition.chunking_strategy.value} | "
          f"embedding={emb_label}")
    if condition.is_hybrid:
        print(f"     hybrid_alpha={condition.hybrid_alpha}")
    if condition.is_hierarchical:
        print(f"     coarse={condition.coarse_chunking_strategy.value}/"
              f"{condition.coarse_retriever_type.value if condition.coarse_retriever_type else 'sparse'} "
              f"| top_n_files={condition.top_n_files}")
    print(f"     dataset_version={dataset_version} | commit={commit_hash}")
    print(f"{'='*60}")

    start_time = time.time()

    # Step 1+2: Build the index. Hierarchical retrievers do their own chunking
    # internally (they need two chunkers — coarse and fine), so we hand them
    # the repo and skip the outer chunker call.
    print(f"\n[INDEX] target=elasticsearch")
    t_index_start = time.time()
    retriever = create_retriever(condition, es_url, repo_id=repo_id)

    if condition.is_hierarchical:
        retriever.index_repository(repo_path, python_files)
        # Use fine-chunk count as the headline number for the run record.
        chunks_indexed = len(getattr(retriever, "_fine_chunks", []))
        if chunks_indexed == 0:
            print("  WARN: no chunks produced, skipping")
            retriever.cleanup()
            retriever.close()
            return [_empty_result(condition, k, samples) for k in k_values]
    else:
        print(f"\n[CHUNK] strategy={condition.chunking_strategy.value}")
        chunker = create_chunker(condition)
        chunks = chunker.chunk_repository(repo_path, python_files)
        print(f"  chunks_created={len(chunks)}")
        if not chunks:
            print("  WARN: no chunks produced, skipping")
            return [_empty_result(condition, k, samples) for k in k_values]
        # Pass cache context so V5a/V5b/V5c (same model+chunker+repo) can
        # share embedding vectors instead of re-encoding 3 times.
        cache_context = {
            "chunker_strategy": condition.chunking_strategy.value,
            "repo_id": repo_id,
            "commit": commit_hash or "HEAD",
        }
        retriever.index(chunks, cache_context=cache_context)
        chunks_indexed = len(chunks)

    indexing_time_s = time.time() - t_index_start
    print(f"  indexing_time={indexing_time_s:.2f}s | chunks_indexed={chunks_indexed}")

    # Step 3: Retrieve ONCE with k_max & evaluate for each sample
    print(f"\n[RETRIEVE] samples={len(samples)} | k_retrieve={k_max} | k_derive={k_values}")
    all_sample_metrics = []       # per-sample metrics dict with all k-values
    all_retrieval_results = []    # raw retrieval results per sample
    query_times_ms = []           # for p50 calculation

    for i, sample in enumerate(samples):
        t0 = time.time()
        if condition.is_hierarchical:
            # Hierarchical retrievers own stage-specific query preprocessing
            # via stage1_query_mode / stage2_query_mode.  Pass the raw query
            # so each stage can independently apply its preparation function.
            query = sample.query
        else:
            # Flat retrievers use a single query_mode applied here.
            exp_lookup = None
            if condition.query_mode in ("llm_expanded", "full_expanded"):
                exp_lookup = _get_expansion_lookup(repo_id)
            query = preprocess_query(sample.query, condition.query_mode,
                                     expansion_lookup=exp_lookup)
        results = retriever.retrieve(query, k_max)
        retrieval_time = (time.time() - t0) * 1000
        query_times_ms.append(retrieval_time)

        # Extract ground truth file paths
        gt_files = [t.file_path for t in sample.ground_truth.targets]
        gt_funcs = [t.function_name for t in sample.ground_truth.targets if t.function_name]

        # Get chunks from results
        retrieved_chunks = [chunk for chunk, _ in results]

        # Compute metrics for ALL k-values at once from the same top-k_max results
        metrics = compute_all_metrics(
            retrieved_chunks, gt_files,
            gt_funcs if gt_funcs else None,
            k_values=k_values,
        )

        # Hierarchical diagnostics: did Stage-1 surface at least one GT file?
        # We use the same fuzzy file-matcher as recall@k for consistency.
        if condition.is_hierarchical:
            from benchmark.metrics import _file_matches
            coarse_files = list(getattr(retriever, "last_coarse_files", []) or [])
            stage1_hit = any(
                _file_matches(cand, gt) for cand in coarse_files for gt in gt_files
            )
            metrics["stage1_hit"] = stage1_hit
            metrics["stage1_n_files"] = len(coarse_files)

            # Stage-2 diagnostics (V12): how did Stage 2 use the candidate set?
            metrics["stage2_from_candidates"] = getattr(
                retriever, "last_stage2_from_candidate_files", None
            )
            metrics["stage2_from_fallback"] = getattr(
                retriever, "last_stage2_from_fallback", None
            )

        all_sample_metrics.append(metrics)

        # Build retrieval result (full k_max results)
        retrieved_items = []
        for rank, (chunk, score) in enumerate(results, start=1):
            retrieved_items.append(RetrievedItem(
                chunk_id=chunk.chunk_id,
                file_path=chunk.file_path,
                score=score,
                rank=rank,
                function_name=chunk.function_name,
            ))

        tc_type = sample.metadata.test_case_type.value if sample.metadata.test_case_type else "-"
        first_hit = metrics.get("first_hit_rank")

        all_retrieval_results.append({
            "sample": sample,
            "retrieved_items": retrieved_items,
            "retrieval_time_ms": retrieval_time,
            "metrics": metrics,
        })

        if verbose:
            hit_marker = "HIT" if metrics.get(f"recall@{k_max}", 0) > 0 else "MISS"
            top_k_summary = " | ".join(
                f"r{rank}={item.file_path.split('/')[-1]}({item.score:.3f})"
                for rank, item in enumerate(retrieved_items[:5], start=1)
            )
            print(f"  [{hit_marker}] query_id={sample.sample_id} | "
                  f"tc={tc_type} | "
                  f"R@{k_max}={metrics.get(f'recall@{k_max}', 0):.2f} | "
                  f"MRR@{k_max}={metrics.get(f'mrr@{k_max}', 0):.2f} | "
                  f"rank_first_hit={first_hit if first_hit else '-'} | "
                  f"query_time={retrieval_time:.1f}ms")
            print(f"         top_k: {top_k_summary}")

    # Compute timing stats
    query_time_p50 = statistics.median(query_times_ms) if query_times_ms else 0.0
    elapsed = time.time() - start_time

    print(f"\n[TIMING] indexing_time={indexing_time_s:.2f}s | "
          f"query_time_p50={query_time_p50:.1f}ms | "
          f"total={elapsed:.1f}s")

    # Aggregate and build one RunResult per k-value
    run_results = []

    for k in k_values:
        agg = aggregate_metrics(all_sample_metrics, k_values=[k])

        print(f"\n[RESULT] {condition.condition_id}_k{k}:")
        print(f"  Recall@{k}={agg.get(f'recall@{k}', 0):.4f} | "
              f"MRR@{k}={agg.get(f'mrr@{k}', 0):.4f} | "
              f"hit_rate={agg.get('hit_rate', 0):.4f}")

        run_result = RunResult(
            condition_id=condition.condition_id,
            k=k,
            dataset_version=dataset_version,
            repo_id=repo_id,
            chunking_strategy=condition.chunking_strategy.value,
            retriever_type=condition.retriever_type.value,
            embedding_model=condition.embedding_name,
            metrics=RunMetrics(
                condition_id=condition.condition_id,
                k=k,
                recall_at_k=agg.get(f"recall@{k}", 0),
                mrr_at_k=agg.get(f"mrr@{k}", 0),
                num_samples=len(samples),
                per_sample=[SampleMetrics(
                    sample_id=r["sample"].sample_id,
                    recall_at_k=r["metrics"].get(f"recall@{k}", 0.0),
                    mrr_at_k=r["metrics"].get(f"mrr@{k}", 0.0),
                    first_hit_rank=r["metrics"].get("first_hit_rank"),
                    tc_type=r["sample"].metadata.test_case_type.value if r["sample"].metadata.test_case_type else None,
                    repo_id=r["sample"].repo_id,
                    stage1_hit=r["metrics"].get("stage1_hit"),
                    stage1_n_files=r["metrics"].get("stage1_n_files"),
                    stage2_from_candidates=r["metrics"].get("stage2_from_candidates"),
                    stage2_from_fallback=r["metrics"].get("stage2_from_fallback"),
                ) for r in all_retrieval_results],
            ),
            completed_at=datetime.now().isoformat(),
            retrieval_results=[RetrievalResult(
                sample_id=r["sample"].sample_id,
                condition_id=condition.condition_id,
                k=k,
                retrieved=r["retrieved_items"][:k],
                retrieval_time_ms=r["retrieval_time_ms"],
            ) for r in all_retrieval_results],
        )
        run_results.append(run_result)

    # Cleanup ES index synchronously (NOT via __del__/GC).
    # Releasing here matters: under memory pressure, the next variant's
    # index_create() can otherwise race with this delete.
    retriever.cleanup()
    retriever.close()
    gc.collect()

    return run_results


def _persist_partial_and_exit(
    all_results: List[RunResult],
    dataset: BenchmarkDataset,
    output_dir: str,
    reason: str = "aborted",
):
    """Persist results gathered so far, then exit non-zero so CI flags the run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    if all_results:
        report = BenchmarkReport(
            dataset_version=dataset.version,
            runs=all_results,
        )
        results_file = output_path / f"benchmark_results_{ts}_PARTIAL.json"
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"\n[SAVE] partial results -> {results_file}")

    # Stamp the reason into the artifact path so the CI dump is useful.
    (output_path / f"ABORTED_{ts}.txt").write_text(
        f"Run aborted at {datetime.now().isoformat()}\nReason: {reason}\n",
        encoding="utf-8",
    )
    sys.exit(1)


def _empty_result(condition: ConditionConfig, k: int, samples: List[Sample]) -> RunResult:
    """Create an empty result for a skipped run."""
    return RunResult(
        condition_id=condition.condition_id,
        k=k,
        dataset_version="1.0.0",
        repo_id="unknown",
        chunking_strategy=condition.chunking_strategy.value,
        retriever_type=condition.retriever_type.value,
        embedding_model=condition.embedding_name,
        metrics=RunMetrics(
            condition_id=condition.condition_id,
            k=k,
            recall_at_k=0.0,
            mrr_at_k=0.0,
            num_samples=len(samples),
        ),
    )


def run_benchmark(
    conditions: Optional[List[str]] = None,
    k_values: Optional[List[int]] = None,
    repos: Optional[List[str]] = None,
    dataset_path: str = "benchmark/data/benchmark_dataset_v2.json",
    output_dir: str = "benchmark/results",
    es_url: str = None,
    verbose: bool = False,
    dry_run: bool = False,
    era: Optional[str] = None,
    era_cutoff: Optional[str] = None,
    era_date_field: str = "issue_created_at",
):
    """
    Run the full benchmark using Elasticsearch.

    Args:
        conditions: List of condition IDs (e.g., ["V1", "V3"]) or None for all.
        k_values: List of k values or None for [1, 5, 10].
        repos: List of repo names or None for all.
        dataset_path: Path to the benchmark dataset JSON.
        output_dir: Directory for results output.
        es_url: Elasticsearch URL (default from config).
        verbose: Print per-sample results.
        dry_run: Only print what would be done.
        era: Filter samples by era ("pre_ai", "post_ai", or None for all).
        era_cutoff: Optionaler ISO-Cutoff (z. B. "2022-11-30" = ChatGPT-Launch,
            primärer Research-Cutoff). Reklassifiziert die Era
            dynamisch aus issue_created_at statt der eingebauten Labels.
    """
    # Resolve parameters
    selected_conditions = (
        [CONDITIONS_MAP[c] for c in conditions if c in CONDITIONS_MAP]
        if conditions else CONDITIONS
    )
    selected_k = k_values or K_VALUES
    selected_repos = (
        [REPOS_MAP[r] for r in repos if r in REPOS_MAP]
        if repos else REPOSITORIES
    )

    total_retrievals = len(selected_conditions) * len(selected_repos)
    total_results = total_retrievals * len(selected_k)
    effective_es_url = es_url or ES_URL

    print(f"\n{'#'*60}")
    print(f"# FLBench – Feature Location Benchmark v1 (Elasticsearch)")
    print(f"{'#'*60}")
    print(f"\nConfiguration:")
    print(f"  conditions:  {[c.condition_id for c in selected_conditions]}")
    print(f"  k_values:    {selected_k}")
    print(f"  repos:       {[r.name for r in selected_repos]}")
    print(f"  es_url:      {effective_es_url}")
    print(f"  era:         {era or 'all'}")
    print(f"  retrievals:  {total_retrievals} (k={selected_k} derived -> {total_results} RunResults)")

    if dry_run:
        print(f"\n[DRY RUN] listing all {total_retrievals} retrieval runs:")
        for repo in selected_repos:
            for condition in selected_conditions:
                print(f"  {repo.name} x {condition.condition_id} "
                      f"(retrieve k={max(selected_k)}, derive k={selected_k})")
        return

    # Load dataset
    print(f"\n[DATASET] path={dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"  samples_loaded={dataset.total_samples} | version={dataset.version}")

    # Filter by era if requested
    if era_cutoff and not era:
        raise ValueError("era_cutoff braucht era ('pre_ai'|'post_ai')")
    if era:
        original_count = dataset.total_samples
        dataset = dataset.filter_by_era(era, cutoff=era_cutoff,
                                        date_field=era_date_field)
        cutoff_str = (f"@{era_cutoff}({era_date_field})" if era_cutoff else "")
        print(f"  era_filter={era}{cutoff_str} | {original_count} -> {dataset.total_samples} samples")
        if dataset.total_samples == 0:
            print(f"  Keine Samples für era='{era}' gefunden. Abbruch.")
            return

    # Run all combinations
    all_results: List[RunResult] = []

    for repo_config in selected_repos:
        print(f"\n{'='*60}")
        print(f"REPO {repo_config.owner}/{repo_config.name}")
        print(f"  domain={repo_config.domain} | phase={repo_config.phase} | "
              f"commit={repo_config.commit_hash or 'HEAD'}")
        print(f"{'='*60}")

        # Clone repo
        repo_path = clone_repository(repo_config.name)
        python_files = get_python_files(repo_path, repo_config.source_dirs)
        print(f"  python_files={len(python_files)} | source_dirs={repo_config.source_dirs}")

        # Filter samples for this repo
        repo_samples = [
            s for s in dataset.samples
            if repo_config.name in s.repo_id
        ]

        if not repo_samples:
            print(f"  WARN: no samples for {repo_config.name}, skipping")
            continue

        print(f"  samples={len(repo_samples)}")

        # Fail-fast counter: if ES dies, every subsequent variant will hit
        # ConnectionError on the healthcheck (~1s each). After N consecutive
        # ES-related failures, abort the whole job and let the CI report a
        # real failure instead of churning for 80 minutes.
        fail_fast_threshold = int(os.environ.get("BENCHMARK_FAIL_FAST_AFTER", "2"))
        consecutive_es_failures = 0

        for condition in selected_conditions:
            try:
                results = run_single_condition_all_k(
                    condition=condition,
                    k_values=selected_k,
                    samples=repo_samples,
                    repo_path=repo_path,
                    python_files=python_files,
                    repo_config=repo_config,
                    dataset_version=dataset.version,
                    es_url=effective_es_url,
                    verbose=verbose,
                )
                all_results.extend(results)
                consecutive_es_failures = 0  # reset on success
            except Exception as e:
                msg = str(e)
                is_es_error = (
                    "Connection" in msg
                    or "timed out" in msg.lower()
                    or "Cannot reach ES" in msg
                    or "ConnectionTimeout" in msg
                    or isinstance(e, ConnectionError)
                )
                print(f"\n  ERROR {condition.condition_id}: {e}")

                if is_es_error:
                    consecutive_es_failures += 1
                    print(f"    ES failure {consecutive_es_failures}/{fail_fast_threshold}")
                else:
                    consecutive_es_failures = 0  # non-ES bug, don't fail-fast

                for k in selected_k:
                    all_results.append(_empty_result(condition, k, repo_samples))

                if consecutive_es_failures >= fail_fast_threshold:
                    print(f"\n  Fail-fast: {consecutive_es_failures} consecutive ES failures — "
                          f"aborting run, ES service is unhealthy.")
                    # Persist whatever we have, then bail with non-zero exit.
                    _persist_partial_and_exit(
                        all_results, dataset, output_dir,
                        reason=f"ES service died after {consecutive_es_failures} consecutive failures"
                    )

                print(f"    skipping, continuing...")

        # End of variant loop for this repo: free the embedding-model cache so
        # the next repo's variants don't pile RAM on top of the previous load.
        try:
            from benchmark.retrievers.es_retriever import free_embedding_cache
            free_embedding_cache()
        except Exception:
            pass
        gc.collect()

    # Generate report
    report = BenchmarkReport(
        dataset_version=dataset.version,
        runs=all_results,
    )

    # Save results with timestamp (never overwrite old runs)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M")

    results_file = output_path / f"benchmark_results_{ts}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"\n[SAVE] results -> {results_file}")

    # Also write a "latest" copy for easy access
    latest_json = output_path / "benchmark_results_latest.json"
    with open(latest_json, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # Generate readable report
    report_file = output_path / f"benchmark_report_{ts}.md"
    generate_report(report, str(report_file))

    latest_md = output_path / "benchmark_report_latest.md"
    generate_report(report, str(latest_md))

    print(f"[SAVE] report  -> {report_file}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FLBench Feature Location Benchmark Runner (Elasticsearch)"
    )
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Condition IDs (e.g., V1 V3 V5). Default: all")
    parser.add_argument("--k", type=int, nargs="*", default=None,
                        help="k values (e.g., 1 5 10). Default: 1 5 10")
    parser.add_argument("--repos", nargs="*", default=None,
                        help="Repository names (e.g., requests flask). Default: all")
    parser.add_argument("--dataset", type=str, default="benchmark/data/benchmark_dataset_v2.json",
                        help="Path to benchmark dataset")
    parser.add_argument("--output", type=str, default="benchmark/results",
                        help="Output directory for results")
    parser.add_argument("--es-url", type=str, default=None,
                        help="Elasticsearch URL (default: http://localhost:9200)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only list what would be run")
    parser.add_argument("--era", type=str, default=None,
                        choices=["pre_ai", "post_ai"],
                        help="Only benchmark samples from this era "
                             "(pre_ai = vor 2022, post_ai = ab 2022)")
    parser.add_argument("--era-cutoff", type=str, default=None,
                        help="ISO-Cutoff für dynamische Era-Reklassifikation, "
                             "z. B. 2022-11-30 (ChatGPT-Launch, primärer "
                             "Research-Cutoff). Default: eingebaute Labels (2022-01-01)")
    parser.add_argument("--era-date-field", type=str, default="issue_created_at",
                        choices=["issue_created_at", "pr_merged_at"],
                        help="Datumsfeld für die Era-Klassifikation: "
                             "issue_created_at = Query-Seite (wann Issue formuliert), "
                             "pr_merged_at = Code-Seite (wann Fix gemergt — "
                             "passender Proxy für KI-generierten Code)")

    args = parser.parse_args()

    run_benchmark(
        conditions=args.conditions,
        k_values=args.k,
        repos=args.repos,
        dataset_path=args.dataset,
        output_dir=args.output,
        es_url=args.es_url,
        verbose=args.verbose,
        dry_run=args.dry_run,
        era=args.era,
        era_cutoff=args.era_cutoff,
        era_date_field=args.era_date_field,
    )
