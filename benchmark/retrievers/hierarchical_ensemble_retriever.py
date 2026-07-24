#!/usr/bin/env python3
"""
hierarchical_ensemble_retriever.py – Ensemble Coarse Stage (V17).

Combines TWO coarse retrieval strategies (e.g. Class/File-BM25 and VDoc-BM25)
in Stage 1, unions their candidate file sets, and runs a single fine Stage 2
on the merged candidates.

Motivation (from V16 analysis):
    V16b (Class/File → Dense):  best MRR@10 = 0.466
    V16c (VDoc → Hybrid):      best Recall@10 = 0.591
    Each coarse strategy finds files the other misses.
    Union of both should yield Recall ≥ V16c AND MRR ≥ V16c.

Score-merge strategy:
    1. Normalize each coarse retriever's file scores to [0, 1].
    2. For files in both sets: combined_score = max(score_A, score_B).
       (Max preserves the strongest signal; mean would dilute a clear hit.)
    3. Rank merged files by combined score, take top N.

The LLM is NOT called at benchmark time — pre-computed expansions are used
for the Stage-1 query (same as V16).
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

from .base import BaseRetriever
from .es_retriever import ElasticsearchRetriever
from .hierarchical_retriever import _apply_query_mode
from .hierarchical_v16_retriever import ExpansionLookup, _DEFAULT_EXPANSIONS_FILE
from benchmark.chunking.base import BaseChunker, Chunk


class HierarchicalEnsembleRetriever(BaseRetriever):
    """
    Two-stage retriever with ensemble coarse stage.

    Stage 1: Run two coarse retrievers in parallel, union candidate files.
    Stage 2: Dense or Hybrid retrieval with Terms Filter on merged candidates.

    Args:
        coarse_chunker_a / coarse_chunker_b: Two chunking strategies for coarse.
        coarse_retriever_type_a / _b: Retriever type for each coarse leg.
        merge_strategy: "max" (default) or "mean" for score combination.
        ensemble_top_n: Max files after merge (default: top_n_files, i.e. 20).
            Set higher to allow more candidates from the union.
    """

    def __init__(
        self,
        es_url: str,
        index_prefix: str,
        condition_id: str,
        # Coarse A (e.g. Class/File-Level)
        coarse_chunker_a: BaseChunker,
        # Coarse B (e.g. Virtual Document)
        coarse_chunker_b: BaseChunker,
        # Fine stage
        fine_chunker: BaseChunker = None,
        coarse_retriever_type_a: str = "sparse",
        coarse_retriever_type_b: str = "sparse",
        fine_retriever_type: str = "hybrid",
        # Shared
        embedding_model_name: Optional[str] = None,
        embedding_dims: int = 768,
        hybrid_alpha: float = 0.5,
        top_n_files: int = 20,
        ensemble_top_n: Optional[int] = None,
        merge_strategy: str = "max",
        coarse_score_aggregation: str = "max",
        # Asymmetric query preparation
        expansions_file: Optional[str] = None,
        stage1_query_mode: str = "llm_expanded",
        stage2_query_mode: str = "full",
        # Kept for config compat (ignored)
        llm_api_url: str = "http://localhost:1234/v1",
        llm_model: Optional[str] = None,
        llm_temperature: float = 0.3,
        llm_timeout: float = 120.0,
    ):
        super().__init__(name=f"hier_ensemble_{condition_id}")
        self.es_url = es_url
        self.condition_id = condition_id
        self.coarse_chunker_a = coarse_chunker_a
        self.coarse_chunker_b = coarse_chunker_b
        self.fine_chunker = fine_chunker
        self.top_n_files = top_n_files
        self.ensemble_top_n = ensemble_top_n or top_n_files
        self.merge_strategy = merge_strategy
        self.coarse_score_aggregation = coarse_score_aggregation
        self.stage1_query_mode = stage1_query_mode
        self.stage2_query_mode = stage2_query_mode

        # Pre-computed expansions
        self._lookup = ExpansionLookup(expansions_file or _DEFAULT_EXPANSIONS_FILE)

        # Expansion stats
        self._expansion_hits = 0
        self._expansion_misses = 0

        # ── Coarse A ──
        self._coarse_a = ElasticsearchRetriever(
            es_url=es_url,
            index_prefix=index_prefix,
            condition_id=f"{condition_id}_coarseA",
            retriever_type=coarse_retriever_type_a,
            embedding_model_name=(
                embedding_model_name if coarse_retriever_type_a != "sparse" else None
            ),
            embedding_dims=embedding_dims,
            hybrid_alpha=hybrid_alpha,
        )

        # ── Coarse B ──
        self._coarse_b = ElasticsearchRetriever(
            es_url=es_url,
            index_prefix=index_prefix,
            condition_id=f"{condition_id}_coarseB",
            retriever_type=coarse_retriever_type_b,
            embedding_model_name=(
                embedding_model_name if coarse_retriever_type_b != "sparse" else None
            ),
            embedding_dims=embedding_dims,
            hybrid_alpha=hybrid_alpha,
        )

        # ── Fine stage ──
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

        # Instrumentation
        self.last_coarse_files: List[str] = []
        self.last_coarse_hit_count: int = 0
        self.last_stage2_from_candidate_files: int = 0
        self.last_stage2_from_fallback: int = 0
        self.last_coarse_a_files: List[str] = []
        self.last_coarse_b_files: List[str] = []
        self.last_union_size: int = 0
        self.last_overlap_size: int = 0

    # ─── Indexing ──────────────────────────────────────────────────

    def index_repository(self, repo_path: Path, python_files: List[Path],
                         cache_context: Optional[dict] = None):
        """Build three ES indices: coarse A, coarse B, and fine."""

        def _ctx(chunker_name):
            if not cache_context:
                return None
            return {**cache_context, "chunker_strategy": chunker_name}

        # Coarse A
        print(f"  [ENS] coarse-A chunking ({self.coarse_chunker_a.name})...")
        chunks_a = self.coarse_chunker_a.chunk_repository(repo_path, python_files)
        print(f"  [ENS]   coarse-A chunks={len(chunks_a)}")
        print(f"  [ENS] coarse-A indexing ({self._coarse_a.retriever_type})...")
        self._coarse_a.index(chunks_a, cache_context=_ctx(self.coarse_chunker_a.name))

        # Coarse B
        print(f"  [ENS] coarse-B chunking ({self.coarse_chunker_b.name})...")
        chunks_b = self.coarse_chunker_b.chunk_repository(repo_path, python_files)
        print(f"  [ENS]   coarse-B chunks={len(chunks_b)}")
        print(f"  [ENS] coarse-B indexing ({self._coarse_b.retriever_type})...")
        self._coarse_b.index(chunks_b, cache_context=_ctx(self.coarse_chunker_b.name))

        # Fine
        print(f"  [ENS] fine chunking ({self.fine_chunker.name})...")
        fine_chunks = self.fine_chunker.chunk_repository(repo_path, python_files)
        print(f"  [ENS]   fine chunks={len(fine_chunks)}")
        print(f"  [ENS] fine indexing ({self._fine.retriever_type})...")
        self._fine.index(fine_chunks, cache_context=_ctx(self.fine_chunker.name))

        self._fine_chunks = fine_chunks
        self._fine_chunk_map = {c.chunk_id: c for c in fine_chunks}
        self._chunks_by_file = {}
        for c in fine_chunks:
            self._chunks_by_file.setdefault(c.file_path, []).append(c.chunk_id)

        self._indexed = True

    def index(self, chunks: List[Chunk]):
        raise NotImplementedError(
            "HierarchicalEnsembleRetriever needs three index passes. "
            "Use index_repository(repo_path, python_files)."
        )

    # ─── Retrieval ─────────────────────────────────────────────────

    def retrieve(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        if not self._indexed:
            raise RuntimeError("Call index_repository() first.")

        # ── Build Stage-1 query (LLM-expanded or fallback) ──
        if self.stage1_query_mode == "llm_expanded":
            expanded = self._lookup.lookup(query)
            if expanded:
                stage1_query = expanded
                self._expansion_hits += 1
            else:
                stage1_query = query.split("\n")[0].strip()
                self._expansion_misses += 1
        elif self.stage1_query_mode == "title_only":
            stage1_query = query.split("\n")[0].strip()
        else:
            stage1_query = query

        # ── Stage 1A: coarse retriever A ──
        files_a = self._coarse_to_file_scores(
            self._coarse_a, stage1_query, self.top_n_files
        )
        # ── Stage 1B: coarse retriever B ──
        files_b = self._coarse_to_file_scores(
            self._coarse_b, stage1_query, self.top_n_files
        )

        # ── Merge: union with score combination ──
        merged = self._merge_file_scores(files_a, files_b)

        # Take top N from merged ranking
        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[: self.ensemble_top_n]
        candidate_set = {f for f, _ in ranked}

        # Instrumentation
        set_a = set(files_a.keys())
        set_b = set(files_b.keys())
        self.last_coarse_a_files = sorted(set_a)
        self.last_coarse_b_files = sorted(set_b)
        self.last_coarse_files = [f for f, _ in ranked]
        self.last_union_size = len(set_a | set_b)
        self.last_overlap_size = len(set_a & set_b)

        if not candidate_set:
            return self._fine.retrieve(query, k)

        stage2_query = _apply_query_mode(query, self.stage2_query_mode)

        # ── Stage 2: Dense/Hybrid with Terms Filter ──
        results = self._retrieve_with_terms_filter(stage2_query, k, candidate_set)

        # Diagnostics
        from_candidates = [c for c, _ in results if c.file_path in candidate_set]
        from_fallback = [c for c, _ in results if c.file_path not in candidate_set]
        self.last_coarse_hit_count = len(from_candidates)
        self.last_stage2_from_candidate_files = len(from_candidates)
        self.last_stage2_from_fallback = len(from_fallback)
        return results

    # ─── Coarse → file scores ─────────────────────────────────────

    def _coarse_to_file_scores(
        self,
        coarse: ElasticsearchRetriever,
        query: str,
        top_n: int,
    ) -> Dict[str, float]:
        """Run a coarse retriever, aggregate to file-level, normalize to [0,1]."""
        coarse_fetch = max(top_n * 3, 30)
        results = coarse.retrieve(query, coarse_fetch)

        file_scores: Dict[str, float] = {}
        for chunk, score in results:
            cur = file_scores.get(chunk.file_path)
            if self.coarse_score_aggregation == "sum":
                file_scores[chunk.file_path] = (cur or 0.0) + score
            else:  # "max"
                if cur is None or score > cur:
                    file_scores[chunk.file_path] = score

        # Take top N
        ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        scores = dict(ranked)

        # Normalize to [0, 1]
        if scores:
            vals = list(scores.values())
            mn, mx = min(vals), max(vals)
            spread = mx - mn
            if spread > 0:
                scores = {f: (s - mn) / spread for f, s in scores.items()}
            else:
                scores = {f: 1.0 for f in scores}

        return scores

    def _merge_file_scores(
        self,
        scores_a: Dict[str, float],
        scores_b: Dict[str, float],
    ) -> Dict[str, float]:
        """Merge two normalized file-score dicts via max or mean."""
        all_files = set(scores_a.keys()) | set(scores_b.keys())
        merged = {}
        for f in all_files:
            sa = scores_a.get(f)
            sb = scores_b.get(f)
            if sa is not None and sb is not None:
                if self.merge_strategy == "mean":
                    merged[f] = (sa + sb) / 2
                else:  # "max"
                    merged[f] = max(sa, sb)
            else:
                merged[f] = sa if sa is not None else sb
        return merged

    # ─── Stage 2: Terms-filtered retrieval ─────────────────────────

    def _retrieve_with_terms_filter(
        self,
        query: str,
        k: int,
        candidate_set: Set[str],
    ) -> List[Tuple[Chunk, float]]:
        """Dense/Hybrid retrieval restricted to candidate files."""
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

        # Top up if needed
        if len(chunk_results) < k:
            unfiltered = self._fine.retrieve(query, k)
            seen = {c.chunk_id for c, _ in chunk_results}
            for chunk, score in unfiltered:
                if chunk.chunk_id not in seen:
                    chunk_results.append((chunk, score * 0.5))
                    if len(chunk_results) >= k:
                        break

        return chunk_results[:k]

    def _bm25_with_filter(self, query: str, k: int, file_list: List[str]) -> List[Tuple[str, float]]:
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
                    "filter": {"terms": {"document_id": file_list}}
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
        query_embedding = self._fine._encode_query(query).tolist()
        body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": min(k * 10, len(self._fine_chunks)),
                "filter": {"terms": {"document_id": file_list}}
            },
            "_source": True,
        }
        res = self._fine._es_client.search(index=self._fine.index_name, body=body)
        return [
            (hit["_source"]["chunk_id"], float(hit["_score"]))
            for hit in res["hits"]["hits"]
        ]

    def _hybrid_with_filter(self, query: str, k: int, file_list: List[str]) -> List[Tuple[str, float]]:
        fetch_k = min(k * 3, len(self._fine_chunks))
        bm25_results = self._bm25_with_filter(query, fetch_k, file_list)
        dense_results = self._dense_with_filter(query, fetch_k, file_list)

        bm25_norm = self._normalize_scores_dict(bm25_results)
        dense_norm = self._normalize_scores_dict(dense_results)

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

    # ─── Lifecycle ─────────────────────────────────────────────────

    def cleanup(self):
        for stage in (self._coarse_a, self._coarse_b, self._fine):
            try:
                stage.cleanup()
            except Exception:
                pass

    def close(self):
        for stage in (self._coarse_a, self._coarse_b, self._fine):
            try:
                stage.close()
            except Exception:
                pass
        self._fine_chunks = []
        self._fine_chunk_map = {}
        self._chunks_by_file = {}
        total = self._expansion_hits + self._expansion_misses
        if total > 0:
            print(f"  [ENS] Expansion stats: {self._expansion_hits}/{total} hits "
                  f"({self._expansion_misses} misses)")
            print(f"  [ENS] Avg union size: coarse-A ∪ coarse-B")

    def __del__(self):
        pass
