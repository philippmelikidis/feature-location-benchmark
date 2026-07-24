#!/usr/bin/env python3
"""
heuristic_code_aware.py – Heuristic Code-Aware Chunking (V9)

Uses regex/heuristic patterns to split code at function/class boundaries.
No AST parser needed — faster, more robust against syntax errors.
Chunks are ≤1500 tokens.
"""

import re
from pathlib import Path
from typing import List, Tuple
import tiktoken

from .base import BaseChunker, Chunk


# Patterns that mark structural boundaries in Python
BOUNDARY_PATTERNS = [
    re.compile(r"^(class\s+\w+)", re.MULTILINE),
    re.compile(r"^(def\s+\w+)", re.MULTILINE),
    re.compile(r"^(async\s+def\s+\w+)", re.MULTILINE),
    re.compile(r"^(\s{4}def\s+\w+)", re.MULTILINE),   # Methods (4-space indent)
    re.compile(r"^(\s{4}async\s+def\s+\w+)", re.MULTILINE),
]

# Pattern to detect function/class definitions
DEF_PATTERN = re.compile(
    r"^(\s*)((?:async\s+)?def|class)\s+(\w+)", re.MULTILINE
)


class HeuristicCodeAwareChunker(BaseChunker):
    """Code-aware heuristic chunker: split at function/class boundaries, ≤1500 tokens."""

    MAX_TOKENS = 1500

    def __init__(self, max_tokens: int = 1500):
        super().__init__(name="heuristic_code_aware")
        self.max_tokens = max_tokens
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def _count_tokens_exact(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        return len(text.split())

    def chunk_file(self, file_path: Path, repo_root: Path) -> List[Chunk]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        if not content.strip():
            return []

        rel_path = str(file_path.relative_to(repo_root))
        lines = content.split("\n")

        # Find structural boundaries
        boundaries = self._find_boundaries(lines)

        # Create chunks from boundary segments
        chunks = self._create_chunks_from_boundaries(lines, boundaries, rel_path)

        return chunks

    def _find_boundaries(self, lines: List[str]) -> List[Tuple[int, str, str]]:
        """
        Find line indices where structural boundaries occur.

        Returns:
            List of (line_index, type, name) tuples.
        """
        boundaries = []

        for i, line in enumerate(lines):
            match = DEF_PATTERN.match(line)
            if match:
                indent = match.group(1)
                keyword = match.group(2)
                name = match.group(3)
                deftype = "class" if keyword == "class" else "function"

                # Only top-level and first-level indented definitions
                if len(indent) <= 4:
                    boundaries.append((i, deftype, name))

        return boundaries

    def _create_chunks_from_boundaries(
        self, lines: List[str], boundaries: List[Tuple[int, str, str]], file_path: str
    ) -> List[Chunk]:
        """Create chunks by splitting at structural boundaries."""
        if not boundaries:
            # No boundaries found: whole file is one chunk
            content = "\n".join(lines)
            token_count = self._count_tokens_exact(content)
            if token_count > self.max_tokens:
                return self._split_by_token_limit(lines, 0, len(lines), file_path)
            return [Chunk(
                chunk_id="", file_path=file_path, content=content,
                start_line=1, end_line=len(lines),
                chunk_type="file", token_count=token_count,
            )]

        chunks = []

        # Handle content before first boundary (module header / imports)
        if boundaries[0][0] > 0:
            header_content = "\n".join(lines[:boundaries[0][0]])
            htokens = self._count_tokens_exact(header_content)
            if htokens > 0:
                chunks.append(Chunk(
                    chunk_id="", file_path=file_path, content=header_content,
                    start_line=1, end_line=boundaries[0][0],
                    chunk_type="module_header", token_count=htokens,
                ))

        # Process each boundary segment
        for idx, (start_line, deftype, name) in enumerate(boundaries):
            # End is either next boundary or end of file
            if idx + 1 < len(boundaries):
                end_line = boundaries[idx + 1][0]
            else:
                end_line = len(lines)

            segment_content = "\n".join(lines[start_line:end_line])
            token_count = self._count_tokens_exact(segment_content)

            if token_count <= self.max_tokens:
                func_name = name if deftype == "function" else None
                class_name = name if deftype == "class" else None

                chunks.append(Chunk(
                    chunk_id="", file_path=file_path, content=segment_content,
                    start_line=start_line + 1, end_line=end_line,
                    chunk_type=deftype, function_name=func_name,
                    class_name=class_name, token_count=token_count,
                ))
            else:
                # Too large: split further
                sub_chunks = self._split_by_token_limit(
                    lines, start_line, end_line, file_path, name=name, deftype=deftype
                )
                chunks.extend(sub_chunks)

        return chunks

    def _split_by_token_limit(
        self, lines: List[str], start: int, end: int,
        file_path: str, name: str = None, deftype: str = "block"
    ) -> List[Chunk]:
        """Split a segment that exceeds the token limit."""
        chunks = []
        current_start = start

        while current_start < end:
            current_end = current_start
            current_tokens = 0

            while current_end < end:
                line_tokens = self._count_tokens_exact(lines[current_end])
                if current_tokens + line_tokens > self.max_tokens and current_end > current_start:
                    break
                current_tokens += line_tokens
                current_end += 1

            content = "\n".join(lines[current_start:current_end])
            chunks.append(Chunk(
                chunk_id="", file_path=file_path, content=content,
                start_line=current_start + 1, end_line=current_end,
                chunk_type=f"{deftype}_part",
                function_name=name if deftype == "function" else None,
                class_name=name if deftype == "class" else None,
                token_count=current_tokens,
            ))
            current_start = current_end

        return chunks
