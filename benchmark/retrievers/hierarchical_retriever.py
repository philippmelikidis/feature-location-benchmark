#!/usr/bin/env python3
"""
hierarchical_retriever.py – Two-stage hierarchical retrieval (V11).

Design (per Findings F3 + NS3):
    Stage 1 (coarse): index at class/file-level, retrieve top-N files/classes
    Stage 2 (fine):   index at function/AST-level, restrict to chunks whose
                      file_path is in the Stage-1 candidate set, return top-K

Why: pandas (the largest repo, 95 queries) consistently scores Recall@10 ≤ 0.43
because the relevant function gets buried under semantic near-duplicates from
unrelated files. Filtering Stage 2 by Stage-1 file candidates removes most of
that noise without ditching dense semantic matching.

Each Stage owns its own ES index so coarse and fine can use independent
chunkers and retriever types (e.g. sparse coarse + hybrid fine).

Stage-1 candidate set is computed by aggregating chunk-level scores per file
(max-pooling) then taking the top-N files. This avoids needing a separate
"file-level" search API on top of ES.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional

from .base import BaseRetriever
from .es_retriever import ElasticsearchRetriever
from benchmark.chunking.base import BaseChunker, Chunk


def _apply_query_mode(query: str, mode: str) -> str:
    """Return a stage-specific representation of the raw query.

    "full"       → query unchanged (default)
    "title_only" → first non-empty line (issue title)

    Used by all hierarchical retrievers to decouple Stage-1 (compact) from
    Stage-2 (rich) query representations without touching the runner.
    """
    if mode == "title_only":
        for line in query.split("\n"):
            if line.strip():
                return line.strip()
    return query


class HierarchicalRetriever(BaseRetriever):
    """
    Two-stage retriever: coarse file selection → fine within-file ranking.

    Args:
        es_url, index_prefix, condition_id, embedding_*: same semantics as
            ElasticsearchRetriever — applied to BOTH stages.
        coarse_chunker / fine_chunker: chunking strategies for each stage.
            Coarse should produce a few large chunks per file (class/file-level);
            fine should produce many small chunks (function/AST).
        coarse_retriever_type / fine_retriever_type: "sparse", "dense", or "hybrid".
        top_n_files: number of file candidates kept after Stage 1 (default 20).
        coarse_score_aggregation: "max" or "sum" — how to roll up chunk scores
            to file scores (max is more robust to chunk-count imbalance).
    """

    def __init__(
        self,
        es_url: str,
        index_prefix: str,
        condition_id: str,
        coarse_chunker: BaseChunker,
        fine_chunker: BaseChunker,
        coarse_retriever_type: str = "sparse",
        fine_retriever_type: str = "hybrid",
        embedding_model_name: Optional[str] = None,
        embedding_dims: int = 768,
        hybrid_alpha: float = 0.5,
        top_n_files: int = 20,
        coarse_score_aggregation: str = "max",
        # Asymmetric query preparation
        stage1_query_mode: str = "full",
        stage2_query_mode: str = "full",
    ):
        super().__init__(name=f"hier_{condition_id}")
        self.es_url = es_url
        self.condition_id = condition_id
        self.coarse_chunker = coarse_chunker
        self.fine_chunker = fine_chunker
        self.top_n_files = top_n_files
        self.coarse_score_aggregation = coarse_score_aggregation
        self.stage1_query_mode = stage1_query_mode
        self.stage2_query_mode = stage2_query_mode

        # Coarse stage: simpler retriever, no embedding needed if sparse.
        self._coarse = ElasticsearchRetriever(
            es_url=es_url,
            index_prefix=index_prefix,
            condition_id=f"{condition_id}_coarse",
            retriever_type=coarse_retriever_type,
            embedding_model_name=embedding_model_name if coarse_retriever_type != "sparse" else None,
            embedding_dims=embedding_dims,
            hybrid_alpha=hybrid_alpha,
        )
        # Fine stage: usually hybrid+ast for max recall on relevant files.
        self._fine = ElasticsearchRetriever(
            es_url=es_url,
            index_prefix=index_prefix,
            condition_id=f"{condition_id}_fine",
            retriever_type=fine_retriever_type,
            embedding_model_name=embedding_model_name,
            embedding_dims=embedding_dims,
            hybrid_alpha=hybrid_alpha,
        )

        self._fine_chunks: List[Chunk] = []
        self._fine_chunk_map: Dict[str, Chunk] = {}
        # Group fine chunks by file so we can filter Stage 2 results in-memory
        # without a per-query ES `terms` filter (which scales worse on large
        # candidate sets).
        self._chunks_by_file: Dict[str, List[str]] = {}

        # For instrumentation: record stage-1 hit-rate so we can tell whether
        # bad final scores come from coarse misses or fine ranking failure.
        self.last_coarse_files: List[str] = []
        self.last_coarse_hit_count: int = 0
        # Stage-2 diagnostics (compatible with V12 report)
        self.last_stage2_from_candidate_files: int = 0
        self.last_stage2_from_fallback: int = 0

    # ─── Indexing ──────────────────────────────────────────────────

    def index_repository(self, repo_path: Path, python_files: List[Path],
                         cache_context: Optional[dict] = None):
        """Build BOTH coarse and fine indices from the repo.

        cache_context is passed to each stage with the corresponding
        chunker strategy substituted so coarse and fine vectors get
        cached separately (and can be reused by single-stage variants
        with the same chunker, e.g. V7 reuses V11's coarse vectors).
        """
        coarse_ctx = None
        fine_ctx = None
        if cache_context:
            coarse_ctx = {**cache_context, "chunker_strategy": self.coarse_chunker.name}
            fine_ctx = {**cache_context, "chunker_strategy": self.fine_chunker.name}

        # Stage 1: coarse chunks (class/file-level)
        print(f"  [HIER] coarse chunking ({self.coarse_chunker.name})...")
        coarse_chunks = self.coarse_chunker.chunk_repository(repo_path, python_files)
        print(f"  [HIER]   coarse_chunks={len(coarse_chunks)}")

        print(f"  [HIER] coarse indexing (retriever={self._coarse.retriever_type})...")
        self._coarse.index(coarse_chunks, cache_context=coarse_ctx)

        # Stage 2: fine chunks
        print(f"  [HIER] fine chunking ({self.fine_chunker.name})...")
        fine_chunks = self.fine_chunker.chunk_repository(repo_path, python_files)
        print(f"  [HIER]   fine_chunks={len(fine_chunks)}")

        print(f"  [HIER] fine indexing (retriever={self._fine.retriever_type})...")
        self._fine.index(fine_chunks, cache_context=fine_ctx)

        self._fine_chunks = fine_chunks
        self._fine_chunk_map = {c.chunk_id: c for c in fine_chunks}
        self._chunks_by_file = {}
        for c in fine_chunks:
            self._chunks_by_file.setdefault(c.file_path, []).append(c.chunk_id)

        self._indexed = True

    def index(self, chunks: List[Chunk]):
        """Not used directly — call index_repository() with repo path + files."""
        raise NotImplementedError(
            "HierarchicalRetriever needs both coarse and fine chunks. "
            "Use index_repository(repo_path, python_files) instead of index(chunks)."
        )

    # ─── Retrieval ─────────────────────────────────────────────────

    def retrieve(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        if not self._indexed:
            raise RuntimeError("Call index_repository() first.")

        stage1_query = _apply_query_mode(query, self.stage1_query_mode)
        stage2_query = _apply_query_mode(query, self.stage2_query_mode)

        # Stage 1: coarse retrieval. We over-fetch to compensate for files
        # with many chunks (max-pooling concentrates the signal).
        coarse_fetch = max(self.top_n_files * 3, 30)
        coarse_results = self._coarse.retrieve(stage1_query, coarse_fetch)

        # Aggregate to file-level scores
        file_scores: Dict[str, float] = {}
        for chunk, score in coarse_results:
            cur = file_scores.get(chunk.file_path)
            if self.coarse_score_aggregation == "sum":
                file_scores[chunk.file_path] = (cur or 0.0) + score
            else:  # "max"
                if cur is None or score > cur:
                    file_scores[chunk.file_path] = score

        candidate_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_files = candidate_files[: self.top_n_files]
        candidate_set = {f for f, _ in candidate_files}
        self.last_coarse_files = [f for f, _ in candidate_files]

        if not candidate_set:
            # Coarse stage returned nothing — fall back to plain fine retrieval
            # so we at least produce some ranking instead of an empty list.
            return self._fine.retrieve(stage2_query, k)

        # Stage 2: fine retrieval. Over-fetch enough to have at least k
        # survivors after the file-filter. 5x is empirically safe.
        fine_fetch = min(max(k * 5, 50), len(self._fine_chunks))
        fine_results = self._fine.retrieve(stage2_query, fine_fetch)

        # Filter to chunks whose file_path is in the candidate set
        filtered: List[Tuple[Chunk, float]] = [
            (chunk, score) for chunk, score in fine_results
            if chunk.file_path in candidate_set
        ]

        # If filtering wiped out too much, top up from unfiltered results
        # (ranked below the filtered ones to preserve Stage-1 priority).
        if len(filtered) < k:
            remaining = [(c, s) for c, s in fine_results if c.file_path not in candidate_set]
            filtered = filtered + remaining[: k - len(filtered)]

        final = filtered[:k]
        from_cand = [c for c, _ in final if c.file_path in candidate_set]
        from_fb = [c for c, _ in final if c.file_path not in candidate_set]
        self.last_coarse_hit_count = len(from_cand)
        self.last_stage2_from_candidate_files = len(from_cand)
        self.last_stage2_from_fallback = len(from_fb)
        return final

    # ─── Lifecycle ─────────────────────────────────────────────────

    def cleanup(self):
        try:
            self._coarse.cleanup()
        finally:
            self._fine.cleanup()

    def close(self):
        try:
            self._coarse.close()
        finally:
            self._fine.close()
        self._fine_chunks = []
        self._fine_chunk_map = {}
        self._chunks_by_file = {}

    def __del__(self):
        # See ElasticsearchRetriever.__del__: explicitly do nothing here.
        pass
