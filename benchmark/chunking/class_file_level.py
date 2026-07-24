#!/usr/bin/env python3
"""
class_file_level.py – Class/File-Level Chunking (V7)

Whole class or file as one chunk, max 4k tokens.
If a class exceeds 4k tokens, it gets split into its methods.
"""

import tree_sitter_python as ts_python
from tree_sitter import Language, Parser, Node
from pathlib import Path
from typing import List, Optional
import tiktoken

from .base import BaseChunker, Chunk


class ClassFileLevelChunker(BaseChunker):
    """Class or file = 1 chunk, max 4k tokens. Splits oversized classes into methods."""

    MAX_TOKENS = 4096

    def __init__(self, max_tokens: int = 4096):
        super().__init__(name="class_file_level")
        self.max_tokens = max_tokens
        self.parser = Parser()
        self.parser.language = Language(ts_python.language())
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
        tree = self.parser.parse(content.encode("utf-8"))

        chunks = []

        # Find top-level classes
        classes_found = False
        for child in tree.root_node.children:
            actual_node = child
            if child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type == "class_definition":
                        actual_node = child  # Keep decorated node for full range
                        classes_found = True
                        self._process_class(actual_node, sub, lines, rel_path, chunks)
                        break
            elif child.type == "class_definition":
                classes_found = True
                self._process_class(child, child, lines, rel_path, chunks)

        # If no classes found, treat the whole file as one chunk
        if not classes_found:
            token_count = self._count_tokens_exact(content)
            if token_count <= self.max_tokens:
                chunks.append(Chunk(
                    chunk_id="",
                    file_path=rel_path,
                    content=content,
                    start_line=1,
                    end_line=len(lines),
                    chunk_type="file",
                    token_count=token_count,
                ))
            else:
                # File too large: split by top-level functions
                self._split_file_by_functions(tree.root_node, lines, rel_path, chunks)

        # Also capture top-level functions not in classes
        for child in tree.root_node.children:
            if child.type == "function_definition":
                self._add_function_chunk(child, child, lines, rel_path, chunks, None)
            elif child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type == "function_definition":
                        self._add_function_chunk(child, sub, lines, rel_path, chunks, None)

        return chunks

    def _process_class(
        self, outer_node: Node, class_node: Node,
        lines: List[str], file_path: str, chunks: List[Chunk]
    ):
        """Process a class: if within token limit, one chunk; else split into methods."""
        class_name = self._get_name(class_node)
        start = outer_node.start_point[0]
        end = outer_node.end_point[0]
        class_content = "\n".join(lines[start:end + 1])
        token_count = self._count_tokens_exact(class_content)

        if token_count <= self.max_tokens:
            chunks.append(Chunk(
                chunk_id="",
                file_path=file_path,
                content=class_content,
                start_line=start + 1,
                end_line=end + 1,
                chunk_type="class",
                class_name=class_name,
                token_count=token_count,
            ))
        else:
            # Split into methods
            body = self._get_body(class_node)
            if body:
                for child in body.children:
                    if child.type == "function_definition":
                        self._add_function_chunk(child, child, lines, file_path, chunks, class_name)
                    elif child.type == "decorated_definition":
                        for sub in child.children:
                            if sub.type == "function_definition":
                                self._add_function_chunk(child, sub, lines, file_path, chunks, class_name)

    def _add_function_chunk(
        self, outer_node: Node, func_node: Node,
        lines: List[str], file_path: str, chunks: List[Chunk],
        parent_class: Optional[str]
    ):
        name = self._get_name(func_node)
        start = outer_node.start_point[0]
        end = outer_node.end_point[0]
        content = "\n".join(lines[start:end + 1])
        qualified = f"{parent_class}.{name}" if parent_class else name

        chunks.append(Chunk(
            chunk_id="",
            file_path=file_path,
            content=content,
            start_line=start + 1,
            end_line=end + 1,
            chunk_type="method" if parent_class else "function",
            function_name=qualified,
            class_name=parent_class,
            token_count=self._count_tokens_exact(content),
        ))

    def _split_file_by_functions(self, root: Node, lines: List[str], file_path: str, chunks: List[Chunk]):
        """Split a large file into individual function chunks."""
        for child in root.children:
            if child.type in ("function_definition", "class_definition"):
                self._add_function_chunk(child, child, lines, file_path, chunks, None)
            elif child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type in ("function_definition", "class_definition"):
                        self._add_function_chunk(child, sub, lines, file_path, chunks, None)

    def _get_name(self, node: Node) -> Optional[str]:
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _get_body(self, node: Node) -> Optional[Node]:
        for child in node.children:
            if child.type == "block":
                return child
        return None
