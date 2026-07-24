#!/usr/bin/env python3
"""
bm25_retriever.py – BM25 Sparse Retriever (V1)

Classic term-based search using rank-bm25.
Good for exact keyword matches, API names, error messages.
"""

import re
from typing import List, Tuple

from .base import BaseRetriever
from benchmark.chunking.base import Chunk

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, filter short tokens."""
    text = text.lower()
    tokens = re.findall(r"[a-z_][a-z0-9_]*", text)
    # Also split camelCase and snake_case
    expanded = []
    for token in tokens:
        # Split camelCase
        parts = re.sub(r"([A-Z])", r" \1", token).split()
        for part in parts:
            # Split snake_case
            for sub in part.split("_"):
                if len(sub) >= 2:
                    expanded.append(sub.lower())
    return expanded if expanded else tokens


class BM25Retriever(BaseRetriever):
    """BM25 sparse retriever using rank-bm25."""

    def __init__(self):
        super().__init__(name="bm25")
        if BM25Okapi is None:
            raise ImportError("rank-bm25 not installed. Run: pip install rank-bm25")
        self.bm25 = None
        self.chunks: List[Chunk] = []
        self.corpus: List[List[str]] = []

    def index(self, chunks: List[Chunk]):
        """Build BM25 index from chunks."""
        self.chunks = chunks
        self.corpus = [_tokenize(chunk.content) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus)
        self._indexed = True

    def retrieve(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k chunks using BM25 scoring."""
        if not self._indexed or self.bm25 is None:
            raise RuntimeError("Index not built. Call index() first.")

        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]

        results = []
        for idx, score in scored_indices:
            if score > 0:
                results.append((self.chunks[idx], float(score)))

        return results

    def reset(self):
        super().reset()
        self.bm25 = None
        self.chunks = []
        self.corpus = []
