#!/usr/bin/env python3
"""
test_virtual_document.py – Regressionstests für den Virtual-Document-Chunker (V14).

Der VDoc-Chunker erzeugt EIN kompaktes Dokument pro Datei (Imports, Klassen,
Methoden-/Funktionssignaturen, Docstring-Erstzeilen) statt rohen Code zu chunken.
Diese Tests parsen echten Python-Code, daher wird tree-sitter benötigt — im
Projekt-venv vorhanden. Aus dem Repo-Root:

    python -m pytest tests/test_virtual_document.py -v
    # oder standalone:
    python tests/test_virtual_document.py

Fehlt tree-sitter, überspringt sich der Test selbst (kein Fehler).
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from benchmark.chunking.virtual_document import VirtualDocumentChunker
    _IMPORT_ERROR = None
except Exception as e:  # tree-sitter not installed in this interpreter
    VirtualDocumentChunker = None
    _IMPORT_ERROR = e


SAMPLE = '''"""Module summary line for the parser."""
import os
from typing import List

def standalone(x, y):
    """Standalone helper function."""
    return x + y

class Widget(Base):
    """A configurable widget."""

    def __init__(self, name):
        """Init the widget."""
        self.name = name

    def render(self, mode):
        """Render the widget in a mode."""
        return mode

    def __repr__(self):
        return "Widget"
'''


def _chunk(code: str):
    chunker = VirtualDocumentChunker()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        f = root / "mod.py"
        f.write_text(code, encoding="utf-8")
        return chunker.chunk_file(f, root)


def test_one_virtual_document_per_file():
    chunks = _chunk(SAMPLE)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_type == "virtual_document"
    assert c.file_path == "mod.py"
    assert c.token_count and c.token_count > 0


def test_contains_header_module_and_description():
    body = _chunk(SAMPLE)[0].content
    assert "File: mod.py" in body
    assert "Module: mod" in body
    assert "Module summary line for the parser." in body


def test_contains_classes_methods_signatures_self_stripped():
    body = _chunk(SAMPLE)[0].content
    assert "Classes:" in body
    assert "Widget" in body
    assert "A configurable widget." in body
    # self is stripped from method signatures
    assert ".render(mode)" in body
    assert ".__init__(name)" in body
    # raw method bodies must NOT leak into the summary
    assert "return mode" not in body


def test_excludes_dunder_methods_except_init():
    body = _chunk(SAMPLE)[0].content
    assert "__repr__" not in body


def test_contains_functions_and_imports():
    body = _chunk(SAMPLE)[0].content
    assert "Functions:" in body
    assert "standalone(x, y)" in body
    assert "Standalone helper function." in body
    assert "Imports:" in body
    assert "os" in body
    assert "typing.List" in body


def test_empty_file_returns_no_chunk():
    assert _chunk("   \n\n  ") == []


if __name__ == "__main__":
    if VirtualDocumentChunker is None:
        print(f"SKIP  tree-sitter not available: {_IMPORT_ERROR}")
        sys.exit(0)
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:
                failed += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL  {name}: {e}")
    print(f"\n{'ALL PASS' if not failed else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
