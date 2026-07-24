#!/usr/bin/env python3
"""
test_asymmetric_query.py – Regressionstests für asymmetrische Query-Aufbereitung.

Kernfinding aus V15: title_only für beide Stages senkte den Recall, weil Stage 2
zu wenige Terme bekam.  Diese Tests stellen sicher, dass:
  1. _apply_query_mode korrekt arbeitet
  2. HierarchicalV12Retriever Stage 1 und Stage 2 mit den konfigurierten Modes bedient
  3. Stage 2 den vollen Issue-Body erhält, wenn stage2_query_mode="full"
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# Insert project root so benchmark.* is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub heavy optional dependencies before any benchmark import triggers them
for _mod in ("tree_sitter", "tree_sitter_python", "sentence_transformers", "elasticsearch"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Import only the modules under test — use importlib to bypass __init__.py
import importlib.util as _ilu

def _load(rel: str):
    path = Path(__file__).resolve().parent.parent / rel
    spec = _ilu.spec_from_file_location(rel.replace("/", ".").rstrip(".py"), path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

# Minimal Chunk stub — avoids pulling in tree_sitter via benchmark.chunking
class Chunk:
    def __init__(self, chunk_id, file_path, content, function_name, start_line, end_line):
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.content = content
        self.function_name = function_name
        self.start_line = start_line
        self.end_line = end_line

# Stub benchmark.chunking.base so hierarchical_retriever can import it
_chunking_base = types.ModuleType("benchmark.chunking.base")
_chunking_base.Chunk = Chunk
_chunking_base.BaseChunker = MagicMock
sys.modules["benchmark.chunking.base"] = _chunking_base
sys.modules["benchmark.chunking"] = types.ModuleType("benchmark.chunking")

# Stub benchmark.retrievers.base
_ret_base = types.ModuleType("benchmark.retrievers.base")
class _BaseRetriever:
    def __init__(self, name=""):
        self.name = name
        self._indexed = False
_ret_base.BaseRetriever = _BaseRetriever
sys.modules["benchmark.retrievers.base"] = _ret_base

# Stub ElasticsearchRetriever
_es_ret = types.ModuleType("benchmark.retrievers.es_retriever")
class _ESRetriever:
    def __init__(self, **kwargs):
        self.retriever_type = kwargs.get("retriever_type", "sparse")
        self.hybrid_alpha = kwargs.get("hybrid_alpha", 0.5)
        self._chunk_map = {}
        self.embedding_model_name = kwargs.get("embedding_model_name")
        self._embedding_model = None
_es_ret.ElasticsearchRetriever = _ESRetriever
sys.modules["benchmark.retrievers.es_retriever"] = _es_ret

# Now import the actual modules under test
_hier_mod = _load("benchmark/retrievers/hierarchical_retriever.py")
_apply_query_mode = _hier_mod._apply_query_mode

# Patch the import inside hierarchical_v12_retriever before loading it
sys.modules["benchmark.retrievers.hierarchical_retriever"] = _hier_mod
_v12_mod = _load("benchmark/retrievers/hierarchical_v12_retriever.py")
HierarchicalV12Retriever = _v12_mod.HierarchicalV12Retriever


# ─── Fixtures ─────────────────────────────────────────────────────────────────

FULL_QUERY = (
    "Fix DataFrame.merge on duplicate column names\n"
    "\n"
    "When calling df.merge() with duplicate column names in the right frame "
    "the resulting DataFrame silently drops one of them. Expected behaviour: "
    "both columns should appear with _x / _y suffixes as documented."
)
TITLE_ONLY = "Fix DataFrame.merge on duplicate column names"


def _fake_chunk(file_path: str = "pandas/core/reshape/merge.py") -> Chunk:
    return Chunk(
        chunk_id="c1",
        file_path=file_path,
        content="def merge(left, right, ...): ...",
        function_name="merge",
        start_line=1,
        end_line=50,
    )


def _make_retriever(**kwargs) -> HierarchicalV12Retriever:
    """Build a V12 retriever with mocked chunkers; does not touch ES."""
    defaults = dict(
        es_url="http://localhost:9200",
        index_prefix="test",
        condition_id="test_asym",
        coarse_chunker=MagicMock(),
        fine_chunker=MagicMock(),
        stage2_strategy="score_propagation",
        stage1_query_mode="full",
        stage2_query_mode="full",
    )
    defaults.update(kwargs)
    return HierarchicalV12Retriever(**defaults)


# ─── Unit tests: _apply_query_mode ────────────────────────────────────────────

def test_apply_query_mode_full_returns_unchanged():
    assert _apply_query_mode(FULL_QUERY, "full") == FULL_QUERY


def test_apply_query_mode_title_only_returns_first_line():
    result = _apply_query_mode(FULL_QUERY, "title_only")
    assert result == TITLE_ONLY
    assert "\n" not in result


def test_apply_query_mode_title_only_skips_leading_blank_lines():
    query_with_blank = "\n\n  \nActual Title\n\nBody text"
    assert _apply_query_mode(query_with_blank, "title_only") == "Actual Title"


def test_apply_query_mode_unknown_mode_returns_full():
    assert _apply_query_mode(FULL_QUERY, "nonexistent_mode") == FULL_QUERY


# ─── Regression test: Stage 2 erhält vollen Issue-Body ────────────────────────

def test_stage2_receives_full_query_when_stage2_mode_is_full():
    """
    Kernregression: stage2_query_mode='full' → Stage 2 muss den gesamten
    Issue-Body bekommen, nicht nur den Titel.
    """
    retriever = _make_retriever(
        stage1_query_mode="title_only",
        stage2_query_mode="full",
        stage2_strategy="score_propagation",
    )

    fake = _fake_chunk()
    retriever._indexed = True
    retriever._fine_chunks = [fake]
    retriever._fine_chunk_map = {fake.chunk_id: fake}
    retriever._chunks_by_file = {fake.file_path: [fake.chunk_id]}

    coarse_queries: list = []
    fine_queries: list = []

    retriever._coarse.retrieve = lambda q, k: (coarse_queries.append(q), [(fake, 1.0)])[1]
    retriever._fine.retrieve = lambda q, k: (fine_queries.append(q), [(fake, 0.9)])[1]

    retriever.retrieve(FULL_QUERY, k=5)

    assert coarse_queries, "Stage 1 (coarse) wurde nicht aufgerufen"
    assert fine_queries, "Stage 2 (fine) wurde nicht aufgerufen"

    assert coarse_queries[0] == TITLE_ONLY, (
        f"Stage 1 sollte nur den Titel erhalten, bekam: {coarse_queries[0]!r}"
    )
    assert fine_queries[0] == FULL_QUERY, (
        f"Stage 2 sollte den vollen Issue-Body erhalten, bekam: {fine_queries[0]!r}"
    )


def test_stage1_receives_full_query_when_both_modes_are_full():
    """Standardverhalten (beide Modes = 'full'): beide Stages bekommen die volle Query."""
    retriever = _make_retriever(
        stage1_query_mode="full",
        stage2_query_mode="full",
    )

    fake = _fake_chunk()
    retriever._indexed = True
    retriever._fine_chunks = [fake]
    retriever._fine_chunk_map = {fake.chunk_id: fake}
    retriever._chunks_by_file = {fake.file_path: [fake.chunk_id]}

    coarse_queries: list = []
    fine_queries: list = []

    retriever._coarse.retrieve = lambda q, k: (coarse_queries.append(q), [(fake, 1.0)])[1]
    retriever._fine.retrieve = lambda q, k: (fine_queries.append(q), [(fake, 0.9)])[1]

    retriever.retrieve(FULL_QUERY, k=5)

    assert coarse_queries[0] == FULL_QUERY
    assert fine_queries[0] == FULL_QUERY


def test_v15_bug_title_only_for_both_stages_would_lose_body():
    """
    Dokumentiert das ursprüngliche V15-Problem: wenn query_mode='title_only'
    extern angewendet wurde, bekam Stage 2 nur den Titel.
    Mit stage2_query_mode='full' wird dieses Verhalten verhindert.
    """
    retriever = _make_retriever(
        stage1_query_mode="title_only",
        stage2_query_mode="full",
    )

    fake = _fake_chunk()
    retriever._indexed = True
    retriever._fine_chunks = [fake]
    retriever._fine_chunk_map = {fake.chunk_id: fake}
    retriever._chunks_by_file = {fake.file_path: [fake.chunk_id]}

    fine_queries: list = []
    retriever._coarse.retrieve = lambda q, k: [(fake, 1.0)]
    retriever._fine.retrieve = lambda q, k: (fine_queries.append(q), [(fake, 0.9)])[1]

    retriever.retrieve(FULL_QUERY, k=5)

    # Stage 2 darf nicht nur den Titel bekommen
    assert fine_queries[0] != TITLE_ONLY, (
        "V15-Bug reproduziert: Stage 2 bekam nur den Titel statt des vollen Issue-Bodys"
    )
    assert fine_queries[0] == FULL_QUERY


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_apply_query_mode_full_returns_unchanged,
        test_apply_query_mode_title_only_returns_first_line,
        test_apply_query_mode_title_only_skips_leading_blank_lines,
        test_apply_query_mode_unknown_mode_returns_full,
        test_stage2_receives_full_query_when_stage2_mode_is_full,
        test_stage1_receives_full_query_when_both_modes_are_full,
        test_v15_bug_title_only_for_both_stages_would_lose_body,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} Tests bestanden")
    sys.exit(0 if failed == 0 else 1)
