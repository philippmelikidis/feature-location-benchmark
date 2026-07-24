#!/usr/bin/env python3
"""
hierarchical_v12_retriever.py – Improved two-stage hierarchical retrieval (V12).

Three variants addressing V11's core problem: Stage 1 finds good files,
but Stage 2 degrades results back to single-retriever level because it
ignores Stage-1 scores and searches globally.

V12a – Score Propagation:
    final_score = fine_score + lambda * coarse_file_score
    Stage-1 relevance directly boosts Stage-2 ranking.

V12b – Targeted Fine Search (ES terms filter):
    Stage-2 knn/BM25 is restricted to chunks within Stage-1 files
    via an ES `terms` filter. This finds the best chunks WITHIN the
    good files, instead of finding globally best chunks and hoping
    they happen to be in the right files.

V12c – Aggressive Over-Fetching:
    Same structure as V11 but with fine_fetch = k * 20 instead of k * 5.
    The hypothesis: V11's problem is that the relevant chunk sits at
    position 50+ in the global fine ranking, and k*5=50 just barely
    misses it. This is the cheapest fix to test.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional

from .base import BaseRetriever
from .es_retriever import ElasticsearchRetriever
from .hierarchical_retriever import _apply_query_mode
from benchmark.chunking.base import BaseChunker, Chunk


class HierarchicalV12Retriever(BaseRetriever):
    """
    Improved two-stage retriever with three fusion strategies.

    Args:
        stage2_strategy: "score_propagation" (V12a), "terms_filter" (V12b),
                         or "overfetch" (V12c).
        score_lambda: Weight for coarse score in V12a (default 0.3).
        overfetch_multiplier: For V12c, how many times k to fetch (default 20).
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
        # V12-specific parameters
        stage2_strategy: str = "score_propagation",
        score_lambda: float = 0.3,
        overfetch_multiplier: int = 20,
        # Asymmetric query preparation
        stage1_query_mode: str = "full",
        stage2_query_mode: str = "full",
    ):
        super().__init__(name=f"hier_v12_{condition_id}")
        self.es_url = es_url
        self.condition_id = condition_id
        self.coarse_chunker = coarse_chunker
        self.fine_chunker = fine_chunker
        self.top_n_files = top_n_files
        self.coarse_score_aggregation = coarse_score_aggregation
        self.stage2_strategy = stage2_strategy
        self.score_lambda = score_lambda
        self.overfetch_multiplier = overfetch_multiplier
        self.stage1_query_mode = stage1_query_mode
        self.stage2_query_mode = stage2_query_mode

        # Coarse stage
        self._coarse = ElasticsearchRetriever(
            es_url=es_url,
            index_prefix=index_prefix,
            condition_id=f"{condition_id}_coarse",
            retriever_type=coarse_retriever_type,
            embedding_model_name=embedding_model_name if coarse_retriever_type != "sparse" else None,
            embedding_dims=embedding_dims,
            hybrid_alpha=hybrid_alpha,
        )
        # Fine stage
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
        self._chunks_by_file: Dict[str, List[str]] = {}

        # Instrumentation — Stage 1
        self.last_coarse_files: List[str] = []
        self.last_coarse_hit_count: int = 0
        self.last_strategy_used: str = stage2_strategy

        # Instrumentation — Stage 2 (what changed between coarse and final?)
        self.last_stage2_total_candidates: int = 0  # how many fine results before filter
        self.last_stage2_from_candidate_files: int = 0  # final results from Stage-1 files
        self.last_stage2_from_fallback: int = 0  # final results NOT in Stage-1 files
        self.last_stage2_rank_boost: float = 0.0  # avg rank improvement vs unfiltered (V12a)

    # ─── Indexing ──────────────────────────────────────────────────

    def index_repository(self, repo_path: Path, python_files: List[Path],
                         cache_context: Optional[dict] = None):
        """Build BOTH coarse and fine indices from the repo."""
        coarse_ctx = None
        fine_ctx = None
        if cache_context:
            coarse_ctx = {**cache_context, "chunker_strategy": self.coarse_chunker.name}
            fine_ctx = {**cache_context, "chunker_strategy": self.fine_chunker.name}

        # Stage 1: coarse chunks (class/file-level)
        print(f"  [HIER-V12] coarse chunking ({self.coarse_chunker.name})...")
        coarse_chunks = self.coarse_chunker.chunk_repository(repo_path, python_files)
        print(f"  [HIER-V12]   coarse_chunks={len(coarse_chunks)}")

        print(f"  [HIER-V12] coarse indexing (retriever={self._coarse.retriever_type})...")
        self._coarse.index(coarse_chunks, cache_context=coarse_ctx)

        # Stage 2: fine chunks
        print(f"  [HIER-V12] fine chunking ({self.fine_chunker.name})...")
        fine_chunks = self.fine_chunker.chunk_repository(repo_path, python_files)
        print(f"  [HIER-V12]   fine_chunks={len(fine_chunks)}")

        print(f"  [HIER-V12] fine indexing (retriever={self._fine.retriever_type})...")
        self._fine.index(fine_chunks, cache_context=fine_ctx)

        self._fine_chunks = fine_chunks
        self._fine_chunk_map = {c.chunk_id: c for c in fine_chunks}
        self._chunks_by_file = {}
        for c in fine_chunks:
            self._chunks_by_file.setdefault(c.file_path, []).append(c.chunk_id)

        self._indexed = True

    def index(self, chunks: List[Chunk]):
        raise NotImplementedError(
            "HierarchicalV12Retriever needs both coarse and fine chunks. "
            "Use index_repository(repo_path, python_files) instead of index(chunks)."
        )

    # ─── Retrieval ─────────────────────────────────────────────────

    def retrieve(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        if not self._indexed:
            raise RuntimeError("Call index_repository() first.")

        stage1_query = _apply_query_mode(query, self.stage1_query_mode)
        stage2_query = _apply_query_mode(query, self.stage2_query_mode)

        # ── Stage 1: coarse retrieval + file-level aggregation ──
        coarse_fetch = max(self.top_n_files * 3, 30)
        coarse_results = self._coarse.retrieve(stage1_query, coarse_fetch)

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
        # Normalize coarse file scores to [0, 1] for score propagation
        coarse_file_scores = dict(candidate_files)
        if coarse_file_scores:
            max_cs = max(coarse_file_scores.values())
            min_cs = min(coarse_file_scores.values())
            spread = max_cs - min_cs
            if spread > 0:
                coarse_file_scores = {
                    f: (s - min_cs) / spread for f, s in coarse_file_scores.items()
                }
            else:
                coarse_file_scores = {f: 1.0 for f in coarse_file_scores}

        self.last_coarse_files = [f for f, _ in candidate_files]

        if not candidate_set:
            return self._fine.retrieve(stage2_query, k)

        # ── Stage 2: dispatch based on strategy ──
        if self.stage2_strategy == "score_propagation":
            results = self._retrieve_score_propagation(
                stage2_query, k, candidate_set, coarse_file_scores
            )
        elif self.stage2_strategy == "terms_filter":
            results = self._retrieve_terms_filter(stage2_query, k, candidate_set)
        elif self.stage2_strategy == "overfetch":
            results = self._retrieve_overfetch(stage2_query, k, candidate_set)
        else:
            raise ValueError(f"Unknown stage2_strategy: {self.stage2_strategy}")

        # Stage-2 diagnostics
        from_candidates = [c for c, _ in results if c.file_path in candidate_set]
        from_fallback = [c for c, _ in results if c.file_path not in candidate_set]
        self.last_coarse_hit_count = len(from_candidates)
        self.last_stage2_from_candidate_files = len(from_candidates)
        self.last_stage2_from_fallback = len(from_fallback)
        return results

    # ─── V12a: Score Propagation ───────────────────────────────────

    def _retrieve_score_propagation(
        self,
        query: str,
        k: int,
        candidate_set: set,
        coarse_file_scores: Dict[str, float],
    ) -> List[Tuple[Chunk, float]]:
        """
        Stage 2 with score propagation: fine_score + lambda * coarse_file_score.

        The coarse score acts as a prior — chunks in highly-ranked files get
        a relevance boost even if their individual fine score is mediocre.
        """
        # Fetch enough from fine to have good coverage
        fine_fetch = min(max(k * 10, 100), len(self._fine_chunks))
        fine_results = self._fine.retrieve(query, fine_fetch)

        # Normalize fine scores to [0, 1]
        if fine_results:
            fine_scores_raw = [s for _, s in fine_results]
            min_fs = min(fine_scores_raw)
            max_fs = max(fine_scores_raw)
            spread_fs = max_fs - min_fs
        else:
            spread_fs = 0

        boosted: List[Tuple[Chunk, float]] = []
        for chunk, score in fine_results:
            # Normalize fine score
            if spread_fs > 0:
                norm_fine = (score - min_fs) / spread_fs
            else:
                norm_fine = 1.0

            # Add coarse file boost (0 if file not in candidate set)
            coarse_boost = coarse_file_scores.get(chunk.file_path, 0.0)
            combined = norm_fine + self.score_lambda * coarse_boost
            boosted.append((chunk, combined))

        # Re-rank by combined score
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted[:k]

    # ─── V12b: Terms Filter (targeted fine search) ─────────────────

    def _retrieve_terms_filter(
        self,
        query: str,
        k: int,
        candidate_set: set,
    ) -> List[Tuple[Chunk, float]]:
        """
        Stage 2 with ES terms filter: only search within candidate files.

        Instead of searching globally and filtering post-hoc, we constrain
        ES to only consider chunks from Stage-1 files. This finds the best
        chunks WITHIN the relevant files.
        """
        # Direct ES query with terms filter on document_id (= file_path)
        self._fine._init_es()
        if self._fine._embedding_model is None and self._fine.embedding_model_name:
            self._fine._init_embedding_model()

        file_list = list(candidate_set)

        if self._fine.retriever_type == "hybrid":
            results = self._hybrid_with_filter(query, k, file_list)
        elif self._fine.retriever_type == "dense":
            results = self._dense_with_filter(query, k, file_list)
        else:
            results = self._bm25_with_filter(query, k, file_list)

        # Map back to Chunk objects
        chunk_results: List[Tuple[Chunk, float]] = []
        for chunk_id, score in results:
            chunk = self._fine._chunk_map.get(chunk_id)
            if chunk:
                chunk_results.append((chunk, score))

        # If we got fewer than k (small candidate set), top up from unfiltered
        if len(chunk_results) < k:
            unfiltered = self._fine.retrieve(query, k)
            seen = {c.chunk_id for c, _ in chunk_results}
            for chunk, score in unfiltered:
                if chunk.chunk_id not in seen:
                    chunk_results.append((chunk, score * 0.5))  # penalty
                    if len(chunk_results) >= k:
                        break

        return chunk_results[:k]

    def _bm25_with_filter(self, query: str, k: int, file_list: List[str]) -> List[Tuple[str, float]]:
        """BM25 restricted to specific files."""
        body = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^3", "keywords^2", "content"],
                            "type": "best_fields",
                        }
                    },
                    "filter": {
                        "terms": {"document_id": file_list}
                    }
                }
            },
            "size": k,
        }
        res = self._fine._es_client.search(index=self._fine.index_name, body=body)
        return [
            (hit["_source"]["chunk_id"], float(hit["_score"]))
            for hit in res["hits"]["hits"]
        ]

    def _dense_with_filter(self, query: str, k: int, file_list: List[str]) -> List[Tuple[str, float]]:
        """Dense knn restricted to specific files."""
        query_embedding = self._fine._embedding_model.encode(query).tolist()
        body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": min(k * 10, len(self._fine_chunks)),
                "filter": {
                    "terms": {"document_id": file_list}
                }
            },
            "_source": True,
        }
        res = self._fine._es_client.search(index=self._fine.index_name, body=body)
        return [
            (hit["_source"]["chunk_id"], float(hit["_score"]))
            for hit in res["hits"]["hits"]
        ]

    def _hybrid_with_filter(self, query: str, k: int, file_list: List[str]) -> List[Tuple[str, float]]:
        """Hybrid (BM25 + Dense) both restricted to candidate files."""
        fetch_k = min(k * 3, len(self._fine_chunks))

        bm25_results = self._bm25_with_filter(query, fetch_k, file_list)
        dense_results = self._dense_with_filter(query, fetch_k, file_list)

        # Min-max normalize
        bm25_norm = self._normalize_scores_dict(bm25_results)
        dense_norm = self._normalize_scores_dict(dense_results)

        # Fuse
        all_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
        alpha = self._fine.hybrid_alpha
        fused = {}
        for cid in all_ids:
            bm25_s = bm25_norm.get(cid, 0.0)
            dense_s = dense_norm.get(cid, 0.0)
            fused[cid] = alpha * bm25_s + (1 - alpha) * dense_s

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return ranked

    @staticmethod
    def _normalize_scores_dict(results: List[Tuple[str, float]]) -> Dict[str, float]:
        if not results:
            return {}
        scores = [s for _, s in results]
        min_s, max_s = min(scores), max(scores)
        spread = max_s - min_s
        if spread == 0:
            return {cid: 1.0 for cid, _ in results}
        return {cid: (s - min_s) / spread for cid, s in results}

    # ─── V12c: Aggressive Over-Fetching ───────────────────────────

    def _retrieve_overfetch(
        self,
        query: str,
        k: int,
        candidate_set: set,
    ) -> List[Tuple[Chunk, float]]:
        """
        Same as V11 but with much more aggressive over-fetching.
        fine_fetch = k * overfetch_multiplier (default 20 vs V11's 5).
        """
        fine_fetch = min(k * self.overfetch_multiplier, len(self._fine_chunks))
        fine_results = self._fine.retrieve(query, fine_fetch)

        # Filter to candidate files
        filtered: List[Tuple[Chunk, float]] = [
            (chunk, score) for chunk, score in fine_results
            if chunk.file_path in candidate_set
        ]

        # Top up if needed
        if len(filtered) < k:
            remaining = [(c, s) for c, s in fine_results if c.file_path not in candidate_set]
            filtered = filtered + remaining[: k - len(filtered)]

        return filtered[:k]

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
        pass
