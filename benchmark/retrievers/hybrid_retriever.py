#!/usr/bin/env python3
"""
hybrid_retriever.py – Hybrid Retriever (V5, V10)

Combines BM25 + Dense retrieval with score fusion.
Supports weighted combination (alpha parameter) or Reciprocal Rank Fusion (RRF).
"""

from typing import List, Tuple, Dict

from .base import BaseRetriever
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from benchmark.chunking.base import Chunk


class HybridRetriever(BaseRetriever):
    """Hybrid retriever: BM25 + Dense with score fusion."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        alpha: float = 0.5,
        fusion_method: str = "weighted",  # "weighted" or "rrf"
        device: str = "cpu",
    ):
        """
        Initialize hybrid retriever.

        Args:
            embedding_model: HuggingFace model for dense component.
            alpha: Weight for dense scores. BM25 weight = 1 - alpha.
                   alpha=0.5 means equal weighting.
            fusion_method: "weighted" (score combination) or "rrf" (Reciprocal Rank Fusion).
            device: Device for embedding model.
        """
        super().__init__(name=f"hybrid_a{alpha}")
        self.alpha = alpha
        self.fusion_method = fusion_method

        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(
            model_name=embedding_model, device=device
        )
        self.chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]):
        """Build both BM25 and dense indices."""
        self.chunks = chunks
        print("  Building BM25 index...")
        self.bm25.index(chunks)
        print("  Building Dense index...")
        self.dense.index(chunks)
        self._indexed = True

    def retrieve(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """Retrieve using score fusion of BM25 and dense results."""
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")

        # Get more candidates from each retriever for better fusion
        candidate_k = min(k * 3, len(self.chunks))

        bm25_results = self.bm25.retrieve(query, candidate_k)
        dense_results = self.dense.retrieve(query, candidate_k)

        if self.fusion_method == "rrf":
            return self._rrf_fusion(bm25_results, dense_results, k)
        else:
            return self._weighted_fusion(bm25_results, dense_results, k)

    def _weighted_fusion(
        self,
        bm25_results: List[Tuple[Chunk, float]],
        dense_results: List[Tuple[Chunk, float]],
        k: int,
    ) -> List[Tuple[Chunk, float]]:
        """Combine scores with weighted sum."""
        # Normalize scores to [0, 1]
        bm25_scores = self._normalize_scores(bm25_results)
        dense_scores = self._normalize_scores(dense_results)

        # Merge by chunk_id
        combined: Dict[str, Tuple[Chunk, float]] = {}

        for chunk, score in bm25_scores:
            key = f"{chunk.file_path}:{chunk.start_line}"
            bm25_s = score * (1 - self.alpha)
            combined[key] = (chunk, bm25_s)

        for chunk, score in dense_scores:
            key = f"{chunk.file_path}:{chunk.start_line}"
            dense_s = score * self.alpha
            if key in combined:
                existing_chunk, existing_score = combined[key]
                combined[key] = (existing_chunk, existing_score + dense_s)
            else:
                combined[key] = (chunk, dense_s)

        # Sort by combined score
        sorted_results = sorted(
            combined.values(), key=lambda x: x[1], reverse=True
        )

        return sorted_results[:k]

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[Chunk, float]],
        dense_results: List[Tuple[Chunk, float]],
        k: int,
        rrf_k: int = 60,
    ) -> List[Tuple[Chunk, float]]:
        """Reciprocal Rank Fusion."""
        combined: Dict[str, Tuple[Chunk, float]] = {}

        for rank, (chunk, _) in enumerate(bm25_results):
            key = f"{chunk.file_path}:{chunk.start_line}"
            rrf_score = 1.0 / (rrf_k + rank + 1)
            combined[key] = (chunk, rrf_score)

        for rank, (chunk, _) in enumerate(dense_results):
            key = f"{chunk.file_path}:{chunk.start_line}"
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if key in combined:
                existing_chunk, existing_score = combined[key]
                combined[key] = (existing_chunk, existing_score + rrf_score)
            else:
                combined[key] = (chunk, rrf_score)

        sorted_results = sorted(
            combined.values(), key=lambda x: x[1], reverse=True
        )

        return sorted_results[:k]

    @staticmethod
    def _normalize_scores(results: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return []

        scores = [s for _, s in results]
        min_s = min(scores)
        max_s = max(scores)
        range_s = max_s - min_s

        if range_s == 0:
            return [(chunk, 1.0) for chunk, _ in results]

        return [(chunk, (score - min_s) / range_s) for chunk, score in results]

    def reset(self):
        super().reset()
        self.bm25.reset()
        self.dense.reset()
        self.chunks = []
