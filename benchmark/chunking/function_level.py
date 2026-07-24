#!/usr/bin/env python3
"""
function_level.py – Function/Method-Level Chunking (V1–V5)

Each function or method becomes one chunk, including its docstring.
Uses tree-sitter for accurate parsing.
"""

import tree_sitter_python as ts_python
from tree_sitter import Language, Parser, Node
from pathlib import Path
from typing import List, Optional

from .base import BaseChunker, Chunk


class FunctionLevelChunker(BaseChunker):
    """1 Function/Method = 1 Chunk, including docstring and decorators."""

    def __init__(self):
        super().__init__(name="function_level")
        self.parser = Parser()
        self.parser.language = Language(ts_python.language())

    def chunk_file(self, file_path: Path, repo_root: Path) -> List[Chunk]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        if not content.strip():
            return []

        tree = self.parser.parse(content.encode("utf-8"))
        lines = content.split("\n")
        rel_path = str(file_path.relative_to(repo_root))

        chunks = []
        self._extract_functions(tree.root_node, lines, rel_path, chunks, parent_class=None)

        return chunks

    def _extract_functions(
        self,
        node: Node,
        lines: List[str],
        file_path: str,
        chunks: List[Chunk],
        parent_class: Optional[str],
    ):
        """Recursively extract functions and methods."""
        for child in node.children:
            if child.type == "decorated_definition":
                # Get the actual definition inside
                for sub in child.children:
                    if sub.type == "function_definition":
                        chunk = self._make_function_chunk(
                            child, sub, lines, file_path, parent_class
                        )
                        if chunk:
                            chunks.append(chunk)
                    elif sub.type == "class_definition":
                        class_name = self._get_name(sub)
                        # Recurse into decorated class body
                        body = self._get_body(sub)
                        if body:
                            self._extract_functions(
                                body, lines, file_path, chunks, parent_class=class_name
                            )

            elif child.type == "function_definition":
                chunk = self._make_function_chunk(
                    child, child, lines, file_path, parent_class
                )
                if chunk:
                    chunks.append(chunk)

            elif child.type == "class_definition":
                class_name = self._get_name(child)
                body = self._get_body(child)
                if body:
                    self._extract_functions(
                        body, lines, file_path, chunks, parent_class=class_name
                    )

    def _make_function_chunk(
        self,
        outer_node: Node,  # May include decorators
        func_node: Node,
        lines: List[str],
        file_path: str,
        parent_class: Optional[str],
    ) -> Optional[Chunk]:
        """Create a Chunk from a function/method node."""
        name = self._get_name(func_node)
        if not name:
            return None

        start_line = outer_node.start_point[0] + 1  # 1-indexed
        end_line = outer_node.end_point[0] + 1

        content = "\n".join(lines[start_line - 1 : end_line])

        qualified_name = f"{parent_class}.{name}" if parent_class else name

        return Chunk(
            chunk_id="",  # Will be assigned by chunk_repository
            file_path=file_path,
            content=content,
            start_line=start_line,
            end_line=end_line,
            chunk_type="method" if parent_class else "function",
            function_name=qualified_name,
            class_name=parent_class,
            token_count=self._count_tokens(content),
        )

    def _get_name(self, node: Node) -> Optional[str]:
        """Extract the name from a function/class definition node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _get_body(self, node: Node) -> Optional[Node]:
        """Get the body block of a class/function."""
        for child in node.children:
            if child.type == "block":
                return child
        return None
