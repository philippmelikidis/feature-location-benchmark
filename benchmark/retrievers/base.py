#!/usr/bin/env python3
"""
base.py – Abstract base class for all retrievers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from benchmark.chunking.base import Chunk


class BaseRetriever(ABC):
    """Abstract base class for code retrievers."""

    def __init__(self, name: str):
        self.name = name
        self._indexed = False

    @abstractmethod
    def index(self, chunks: List[Chunk]):
        """
        Build the retrieval index from a list of chunks.

        Args:
            chunks: List of Chunk objects to index.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: The search query (natural language feature request).
            k: Number of results to return.

        Returns:
            List of (Chunk, score) tuples, sorted by relevance descending.
        """
        pass

    def reset(self):
        """Reset the index."""
        self._indexed = False

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', indexed={self._indexed})"
