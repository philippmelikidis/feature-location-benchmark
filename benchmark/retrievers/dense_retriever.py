#!/usr/bin/env python3
"""
dense_retriever.py – Dense Embedding Retriever (V2–V4, V6–V9)

Uses sentence-transformers for embedding generation and numpy cosine
similarity for vector search. Works reliably on all platforms including
Apple Silicon (avoids FAISS segfault issues).
"""

import numpy as np
from typing import List, Tuple, Optional

from .base import BaseRetriever
from benchmark.chunking.base import Chunk

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class DenseRetriever(BaseRetriever):
    """Dense embedding retriever using sentence-transformers + numpy."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu",
                 batch_size: int = 32):
        """
        Initialize dense retriever.

        Args:
            model_name: HuggingFace model name for embeddings.
            device: 'cpu' or 'cuda'.
            batch_size: Batch size for encoding.
        """
        super().__init__(name=f"dense_{model_name.split('/')[-1]}")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model: Optional[object] = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[Chunk] = []
        self.dimension: int = 0

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self.model is not None:
            return

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        print(f"  Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"  Model loaded (dim={self.model.get_sentence_embedding_dimension()})")

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to normalized embeddings."""
        self._load_model()
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def index(self, chunks: List[Chunk]):
        """Build embedding matrix from chunks."""
        self.chunks = chunks

        # Encode all chunk contents
        texts = [chunk.content for chunk in chunks]
        self.embeddings = self._encode(texts)
        self.dimension = self.embeddings.shape[1]

        self._indexed = True
        print(f"  Index built: {len(chunks)} vectors, dim={self.dimension}")

    def retrieve(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k chunks using cosine similarity (numpy)."""
        if not self._indexed or self.embeddings is None:
            raise RuntimeError("Index not built. Call index() first.")

        # Encode query
        query_embedding = self._encode([query])  # shape: (1, dim)

        # Cosine similarity via dot product (embeddings are already normalized)
        scores = np.dot(self.embeddings, query_embedding.T).flatten()

        # Get top-k indices
        actual_k = min(k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:actual_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results

    def reset(self):
        super().reset()
        self.embeddings = None
        self.chunks = []
        # Keep model loaded to avoid reloading
