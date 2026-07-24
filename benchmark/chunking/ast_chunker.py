#!/usr/bin/env python3
"""
ast_chunker.py – AST-Based Chunking (V8, V10)

Uses tree-sitter to create structure-aware chunks between 200–1500 tokens.
Chunks follow the AST structure: functions, classes, blocks.
"""

import tree_sitter_python as ts_python
from tree_sitter import Language, Parser, Node
from pathlib import Path
from typing import List, Optional
import tiktoken

from .base import BaseChunker, Chunk


class ASTChunker(BaseChunker):
    """Tree-sitter based AST chunker: 200–1500 tokens per chunk."""

    MIN_TOKENS = 200
    MAX_TOKENS = 1500

    def __init__(self, min_tokens: int = 200, max_tokens: int = 1500):
        super().__init__(name="ast_based")
        self.min_tokens = min_tokens
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
        self._walk_ast(tree.root_node, lines, rel_path, chunks)

        return chunks

    def _walk_ast(self, node: Node, lines: List[str], file_path: str, chunks: List[Chunk]):
        """Walk the AST and create chunks from structural elements."""
        # Collect top-level structural nodes
        structural_nodes = []
        pending_lines = []  # Non-structural lines to potentially merge

        for child in node.children:
            if child.type in (
                "function_definition", "class_definition",
                "decorated_definition", "if_statement",
                "for_statement", "while_statement", "try_statement",
                "with_statement", "async_function_definition",
            ):
                # If we have accumulated non-structural lines, flush them
                if pending_lines:
                    structural_nodes.append(("pending", pending_lines))
                    pending_lines = []
                structural_nodes.append(("structural", child))
            else:
                # Accumulate imports, assignments, comments, etc.
                pending_lines.append(child)

        if pending_lines:
            structural_nodes.append(("pending", pending_lines))

        for node_type, item in structural_nodes:
            if node_type == "structural":
                self._process_structural_node(item, lines, file_path, chunks)
            else:
                # Merge non-structural lines into a chunk if large enough
                self._process_pending_nodes(item, lines, file_path, chunks)

    def _process_structural_node(
        self, node: Node, lines: List[str], file_path: str, chunks: List[Chunk]
    ):
        """Process a structural AST node (function, class, etc.)."""
        # Get the actual definition from decorated nodes
        actual_node = node
        func_name = None
        class_name = None

        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "async_function_definition"):
                    actual_node = child
                    func_name = self._get_name(actual_node)
                    break
                elif child.type == "class_definition":
                    actual_node = child
                    class_name = self._get_name(actual_node)
                    break
        elif node.type in ("function_definition", "async_function_definition"):
            func_name = self._get_name(node)
        elif node.type == "class_definition":
            class_name = self._get_name(node)

        start = node.start_point[0]
        end = node.end_point[0]
        content = "\n".join(lines[start:end + 1])
        token_count = self._count_tokens_exact(content)

        if token_count <= self.max_tokens:
            # Fits in one chunk
            chunk_type = "class" if class_name else ("function" if func_name else "block")
            chunks.append(Chunk(
                chunk_id="",
                file_path=file_path,
                content=content,
                start_line=start + 1,
                end_line=end + 1,
                chunk_type=chunk_type,
                function_name=func_name,
                class_name=class_name,
                token_count=token_count,
            ))
        elif class_name and actual_node.type == "class_definition":
            # Large class: recurse into methods
            body = self._get_body(actual_node)
            if body:
                # Add class header (up to body) as one chunk
                header_end = body.start_point[0] - 1
                if header_end >= start:
                    header = "\n".join(lines[start:header_end + 1])
                    htokens = self._count_tokens_exact(header)
                    if htokens >= self.min_tokens:
                        chunks.append(Chunk(
                            chunk_id="", file_path=file_path, content=header,
                            start_line=start + 1, end_line=header_end + 1,
                            chunk_type="class_header", class_name=class_name,
                            token_count=htokens,
                        ))

                self._walk_ast(body, lines, file_path, chunks)
        elif func_name:
            # Large function: split by sub-blocks
            self._split_large_function(node, lines, file_path, chunks, func_name)
        else:
            # Large block: just use it as-is (capped)
            chunks.append(Chunk(
                chunk_id="", file_path=file_path, content=content[:6000],
                start_line=start + 1, end_line=end + 1,
                chunk_type="block", token_count=min(token_count, self.max_tokens),
            ))

    def _split_large_function(
        self, node: Node, lines: List[str], file_path: str,
        chunks: List[Chunk], func_name: str
    ):
        """Split a large function into sub-chunks."""
        start = node.start_point[0]
        end = node.end_point[0]

        # Simple approach: split into max_tokens-sized line groups
        current_start = start
        while current_start <= end:
            # Find how many lines fit
            current_end = current_start
            current_tokens = 0
            while current_end <= end:
                line_tokens = self._count_tokens_exact(lines[current_end])
                if current_tokens + line_tokens > self.max_tokens and current_end > current_start:
                    break
                current_tokens += line_tokens
                current_end += 1

            content = "\n".join(lines[current_start:current_end])
            chunks.append(Chunk(
                chunk_id="", file_path=file_path, content=content,
                start_line=current_start + 1, end_line=current_end,
                chunk_type="function_part", function_name=func_name,
                token_count=current_tokens,
            ))
            current_start = current_end

    def _process_pending_nodes(
        self, nodes: List[Node], lines: List[str], file_path: str, chunks: List[Chunk]
    ):
        """Merge small non-structural nodes (imports, assignments) into chunks."""
        if not nodes:
            return

        start = nodes[0].start_point[0]
        end = nodes[-1].end_point[0]
        content = "\n".join(lines[start:end + 1])
        token_count = self._count_tokens_exact(content)

        if token_count >= self.min_tokens:
            chunks.append(Chunk(
                chunk_id="", file_path=file_path, content=content,
                start_line=start + 1, end_line=end + 1,
                chunk_type="module_header", token_count=token_count,
            ))

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
