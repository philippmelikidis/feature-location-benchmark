#!/usr/bin/env python3
"""
fixed_size.py – Fixed-Size Token Chunking (V6)

512 tokens per chunk with 20% overlap (~102 token overlap).
Uses tiktoken for accurate token counting.
"""

from pathlib import Path
from typing import List
import tiktoken

from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """Fixed-size sliding window chunker: 512 tokens, 20% overlap."""

    def __init__(self, chunk_size: int = 512, overlap_pct: float = 0.20):
        super().__init__(name="fixed_size")
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_pct)
        self.stride = chunk_size - self.overlap
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def _tokenize(self, text: str) -> List[int]:
        if self.tokenizer:
            return self.tokenizer.encode(text, disallowed_special=())
        # Fallback: whitespace split
        return list(range(len(text.split())))

    def _detokenize(self, tokens: List[int]) -> str:
        if self.tokenizer:
            return self.tokenizer.decode(tokens)
        return ""  # Fallback won't use this path

    def chunk_file(self, file_path: Path, repo_root: Path) -> List[Chunk]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        if not content.strip():
            return []

        rel_path = str(file_path.relative_to(repo_root))
        lines = content.split("\n")

        # We work line-by-line to track line numbers accurately
        chunks = []
        current_tokens = []
        current_lines_content = []
        current_start_line = 1
        line_token_counts = []

        # Pre-compute token counts per line
        for i, line in enumerate(lines):
            if self.tokenizer:
                tokens = self.tokenizer.encode(line + "\n", disallowed_special=())
            else:
                tokens = (line + "\n").split()
            line_token_counts.append(len(tokens))

        # Sliding window over lines
        window_start = 0
        while window_start < len(lines):
            window_tokens = 0
            window_end = window_start

            # Extend window until we reach chunk_size tokens
            while window_end < len(lines) and window_tokens + line_token_counts[window_end] <= self.chunk_size:
                window_tokens += line_token_counts[window_end]
                window_end += 1

            # If we didn't advance at all (single line > chunk_size), take at least one line
            if window_end == window_start:
                window_end = window_start + 1
                window_tokens = line_token_counts[window_start]

            chunk_content = "\n".join(lines[window_start:window_end])

            chunks.append(Chunk(
                chunk_id="",
                file_path=rel_path,
                content=chunk_content,
                start_line=window_start + 1,
                end_line=window_end,
                chunk_type="fixed_size",
                token_count=window_tokens,
            ))

            # Advance by stride (in lines)
            # Calculate how many lines correspond to the overlap
            overlap_tokens = 0
            overlap_lines = 0
            for i in range(window_end - 1, window_start - 1, -1):
                if overlap_tokens + line_token_counts[i] > self.overlap:
                    break
                overlap_tokens += line_token_counts[i]
                overlap_lines += 1

            next_start = window_end - overlap_lines
            if next_start <= window_start:
                next_start = window_start + 1  # Prevent infinite loop

            window_start = next_start

        return chunks
