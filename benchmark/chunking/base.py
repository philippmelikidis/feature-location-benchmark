#!/usr/bin/env python3
"""
base.py – Abstract base class for all chunking strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Chunk:
    """A code chunk produced by a chunking strategy."""
    chunk_id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str = "code"  # "function", "class", "file", "block"
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    token_count: Optional[int] = None

    def overlaps_with(self, file_path: str, start: int, end: int) -> bool:
        """Check if this chunk overlaps with a given span."""
        if self.file_path != file_path:
            return False
        return self.start_line <= end and self.end_line >= start

    def contains_function(self, file_path: str, function_name: Optional[str]) -> bool:
        """Check if this chunk contains a given function."""
        if self.file_path != file_path:
            return False
        if function_name and self.function_name:
            return self.function_name == function_name
        return True  # File-level match


class BaseChunker(ABC):
    """Abstract base class for code chunking strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def chunk_file(self, file_path: Path, repo_root: Path) -> List[Chunk]:
        """
        Split a single file into chunks.

        Args:
            file_path: Absolute path to the Python file.
            repo_root: Root directory of the repository (for relative paths).

        Returns:
            List of Chunk objects.
        """
        pass

    def chunk_repository(self, repo_root: Path, python_files: List[Path]) -> List[Chunk]:
        """
        Chunk all Python files in a repository.

        Args:
            repo_root: Root directory of the repository.
            python_files: List of Python file paths.

        Returns:
            List of all Chunk objects.
        """
        all_chunks = []
        chunk_counter = 0

        for file_path in python_files:
            try:
                chunks = self.chunk_file(file_path, repo_root)
                for chunk in chunks:
                    if not chunk.chunk_id:
                        chunk.chunk_id = f"chunk_{chunk_counter:06d}"
                    chunk_counter += 1
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"   Fehler beim Chunking von {file_path}: {e}")

        return all_chunks

    def _count_tokens(self, text: str) -> int:
        """Approximate token count (whitespace-based, fast)."""
        return len(text.split())

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
