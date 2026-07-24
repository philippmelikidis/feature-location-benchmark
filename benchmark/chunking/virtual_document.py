#!/usr/bin/env python3
"""
virtual_document.py – Virtual Document Chunker (V14 coarse stage).

Produces ONE chunk per file: a compact "virtual document" that summarizes
what the file does, using only AST-extractable information (no LLM needed).

The virtual document contains:
  - File path and module name
  - All class names
  - All function/method names (with signatures)
  - First line of each docstring (the "summary sentence")
  - Import statements (show domain/context)

This gives BM25 and dense retrievers a much richer signal about *what* a
file is for, compared to embedding 500 lines of raw code into one vector.

Typical output size: 30–80 lines per file (vs 200–2000 for raw code).
"""

import tree_sitter_python as ts_python
from tree_sitter import Language, Parser, Node
from pathlib import Path
from typing import List, Optional, Tuple

from .base import BaseChunker, Chunk


class VirtualDocumentChunker(BaseChunker):
    """Produces one virtual document per file for coarse-stage retrieval."""

    def __init__(self):
        super().__init__(name="virtual_document")
        self.parser = Parser()
        self.parser.language = Language(ts_python.language())

    def chunk_file(self, file_path: Path, repo_root: Path) -> List[Chunk]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        if not content.strip():
            return []

        rel_path = str(file_path.relative_to(repo_root))
        tree = self.parser.parse(content.encode("utf-8"))
        lines = content.split("\n")

        # Extract structured information
        module_name = rel_path.replace("/", ".").replace(".py", "")
        imports = self._extract_imports(tree.root_node, lines)
        classes = self._extract_classes(tree.root_node, lines)
        functions = self._extract_functions(tree.root_node, lines)
        module_docstring = self._extract_docstring(tree.root_node, lines)

        # Build the virtual document
        doc_parts = []

        # Header
        doc_parts.append(f"File: {rel_path}")
        doc_parts.append(f"Module: {module_name}")
        doc_parts.append("")

        # Module docstring
        if module_docstring:
            doc_parts.append(f"Description: {module_docstring}")
            doc_parts.append("")

        # Classes with their methods
        if classes:
            doc_parts.append("Classes:")
            for cls_name, cls_doc, methods in classes:
                if cls_doc:
                    doc_parts.append(f"  {cls_name}: {cls_doc}")
                else:
                    doc_parts.append(f"  {cls_name}")
                for method_name, method_sig, method_doc in methods:
                    if method_doc:
                        doc_parts.append(f"    .{method_name}({method_sig}): {method_doc}")
                    else:
                        doc_parts.append(f"    .{method_name}({method_sig})")
            doc_parts.append("")

        # Module-level functions
        if functions:
            doc_parts.append("Functions:")
            for func_name, func_sig, func_doc in functions:
                if func_doc:
                    doc_parts.append(f"  {func_name}({func_sig}): {func_doc}")
                else:
                    doc_parts.append(f"  {func_name}({func_sig})")
            doc_parts.append("")

        # Imports (show what domain this file operates in)
        if imports:
            doc_parts.append("Imports: " + ", ".join(imports[:30]))
            doc_parts.append("")

        virtual_doc = "\n".join(doc_parts)

        return [Chunk(
            chunk_id="",
            file_path=rel_path,
            content=virtual_doc,
            start_line=1,
            end_line=len(lines),
            chunk_type="virtual_document",
            function_name=None,
            class_name=None,
            token_count=len(virtual_doc.split()),
        )]

    # ─── Extraction helpers ────────────────────────────────────────

    def _extract_imports(self, root: Node, lines: List[str]) -> List[str]:
        """Extract imported module names."""
        imports = []
        for child in root.children:
            if child.type == "import_statement":
                text = self._node_text(child, lines)
                # "import pandas" → "pandas"
                parts = text.replace("import ", "").split(",")
                for p in parts:
                    name = p.strip().split(" as ")[0].strip()
                    if name:
                        imports.append(name)
            elif child.type == "import_from_statement":
                text = self._node_text(child, lines)
                # "from pandas.core import foo" → "pandas.core.foo"
                if "from " in text and " import " in text:
                    module = text.split("from ")[1].split(" import ")[0].strip()
                    imported = text.split(" import ")[1].strip()
                    for item in imported.split(","):
                        name = item.strip().split(" as ")[0].strip()
                        if name and name != "*":
                            imports.append(f"{module}.{name}")
                        elif name == "*":
                            imports.append(module)
        return imports

    def _extract_classes(self, root: Node, lines: List[str]) -> List[Tuple]:
        """Extract classes with their docstrings and methods.

        Returns: [(class_name, class_docstring, [(method_name, signature, docstring), ...])]
        """
        classes = []
        for child in root.children:
            node = child
            if child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type == "class_definition":
                        node = sub
                        break
                else:
                    continue

            if node.type != "class_definition":
                continue

            cls_name = self._get_identifier(node)
            if not cls_name:
                continue

            cls_doc = self._get_node_docstring(node, lines)
            methods = self._extract_methods(node, lines)
            classes.append((cls_name, cls_doc, methods))

        return classes

    def _extract_methods(self, class_node: Node, lines: List[str]) -> List[Tuple]:
        """Extract methods from a class body.

        Returns: [(method_name, signature, docstring)]
        """
        methods = []
        body = self._get_body(class_node)
        if not body:
            return methods

        for child in body.children:
            node = child
            if child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type in ("function_definition", "async_function_definition"):
                        node = sub
                        break
                else:
                    continue

            if node.type not in ("function_definition", "async_function_definition"):
                continue

            name = self._get_identifier(node)
            if not name:
                continue
            # Skip dunder methods except __init__
            if name.startswith("__") and name != "__init__":
                continue

            sig = self._get_signature(node, lines)
            doc = self._get_node_docstring(node, lines)
            methods.append((name, sig, doc))

        return methods

    def _extract_functions(self, root: Node, lines: List[str]) -> List[Tuple]:
        """Extract module-level functions.

        Returns: [(func_name, signature, docstring)]
        """
        functions = []
        for child in root.children:
            node = child
            if child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type in ("function_definition", "async_function_definition"):
                        node = sub
                        break
                else:
                    continue

            if node.type not in ("function_definition", "async_function_definition"):
                continue

            name = self._get_identifier(node)
            if not name:
                continue

            sig = self._get_signature(node, lines)
            doc = self._get_node_docstring(node, lines)
            functions.append((name, sig, doc))

        return functions

    def _extract_docstring(self, root: Node, lines: List[str]) -> Optional[str]:
        """Extract module-level docstring (first expression_statement with a string)."""
        return self._get_node_docstring(root, lines)

    # ─── Node utilities ────────────────────────────────────────────

    def _get_identifier(self, node: Node) -> Optional[str]:
        """Get the name identifier from a function/class definition."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _get_body(self, node: Node) -> Optional[Node]:
        """Get the block (body) of a class/function."""
        for child in node.children:
            if child.type == "block":
                return child
        return None

    def _get_signature(self, func_node: Node, lines: List[str]) -> str:
        """Extract function parameters (without 'self'/'cls')."""
        for child in func_node.children:
            if child.type == "parameters":
                text = child.text.decode("utf-8")
                # Remove outer parens
                text = text.strip("()")
                # Remove self/cls
                params = [p.strip() for p in text.split(",")]
                params = [p for p in params if p and p not in ("self", "cls")]
                # Shorten type annotations for readability
                shortened = []
                for p in params[:6]:  # Max 6 params shown
                    # Just keep param name + short type hint
                    if ":" in p:
                        name = p.split(":")[0].strip()
                        shortened.append(name)
                    elif "=" in p:
                        name = p.split("=")[0].strip()
                        shortened.append(name + "=...")
                    else:
                        shortened.append(p)
                result = ", ".join(shortened)
                if len(params) > 6:
                    result += ", ..."
                return result
        return ""

    def _get_node_docstring(self, node: Node, lines: List[str]) -> Optional[str]:
        """Extract the first line of the docstring from a function/class/module."""
        body = self._get_body(node)
        target = body if body else node

        for child in target.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        raw = sub.text.decode("utf-8")
                        # Strip triple-quote markers
                        doc = raw.strip('"""').strip("'''").strip()
                        # Take first line only
                        first_line = doc.split("\n")[0].strip()
                        if first_line and len(first_line) > 5:
                            # Truncate very long docstrings
                            if len(first_line) > 120:
                                first_line = first_line[:117] + "..."
                            return first_line
                        return None
                break  # Only check the first statement
            elif child.type not in ("comment",):
                break  # Docstring must be the first statement

        return None

    def _node_text(self, node: Node, lines: List[str]) -> str:
        """Get the text of a node from the lines array."""
        start = node.start_point[0]
        end = node.end_point[0]
        if start == end:
            return lines[start][node.start_point[1]:node.end_point[1]]
        return " ".join(lines[start:end + 1]).strip()
