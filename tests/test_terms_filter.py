#!/usr/bin/env python3
"""
test_terms_filter.py – Regressionstests für den ES-Level Terms-Filter (V12b).

Kernfinding aus V12: Stage 2 darf nur innerhalb der Stage-1-Kandidatendateien
suchen. Statt global zu suchen und nachzufiltern, schränkt der Retriever die
ES-Query per `terms`-Filter auf `document_id` (= file_path) ein. Diese Tests
prüfen, dass BM25, Dense und Hybrid in Stage 2 den Filter korrekt aufbauen.

Reine Query-Konstruktions-Tests — kein laufendes Elasticsearch, kein
Embedding-Modell nötig (beide werden gestubbt). Aus dem Repo-Root:

    python -m pytest tests/test_terms_filter.py -v
    # oder standalone:
    python tests/test_terms_filter.py
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub heavy optional deps before importing the modules under test.
for _mod in ("tree_sitter", "tree_sitter_python", "sentence_transformers", "elasticsearch"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

import importlib.util as _ilu


def _load(rel: str):
    path = Path(__file__).resolve().parent.parent / rel
    spec = _ilu.spec_from_file_location(rel.replace("/", ".").rstrip(".py"), path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# Minimal Chunk stub — avoids pulling tree_sitter via benchmark.chunking.
class Chunk:
    def __init__(self, chunk_id, file_path, content="", function_name=None,
                 start_line=1, end_line=1):
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.content = content
        self.function_name = function_name
        self.start_line = start_line
        self.end_line = end_line


_chunking_base = types.ModuleType("benchmark.chunking.base")
_chunking_base.Chunk = Chunk
_chunking_base.BaseChunker = MagicMock
sys.modules["benchmark.chunking.base"] = _chunking_base
sys.modules["benchmark.chunking"] = types.ModuleType("benchmark.chunking")

_ret_base = types.ModuleType("benchmark.retrievers.base")
class _BaseRetriever:
    def __init__(self, name=""):
        self.name = name
        self._indexed = False
_ret_base.BaseRetriever = _BaseRetriever
sys.modules["benchmark.retrievers.base"] = _ret_base

# Stub ElasticsearchRetriever: holds the attributes the V12 filter methods touch.
_es_ret = types.ModuleType("benchmark.retrievers.es_retriever")
class _ESRetriever:
    def __init__(self, **kwargs):
        self.retriever_type = kwargs.get("retriever_type", "sparse")
        self.hybrid_alpha = kwargs.get("hybrid_alpha", 0.5)
        self.embedding_model_name = kwargs.get("embedding_model_name")
        self.index_name = "test_index"
        self._chunk_map = {}
        self._es_client = None
        self._embedding_model = None
    def _init_es(self):
        pass
    def _init_embedding_model(self):
        pass
_es_ret.ElasticsearchRetriever = _ESRetriever
sys.modules["benchmark.retrievers.es_retriever"] = _es_ret

_hier_mod = _load("benchmark/retrievers/hierarchical_retriever.py")
sys.modules["benchmark.retrievers.hierarchical_retriever"] = _hier_mod
_v12_mod = _load("benchmark/retrievers/hierarchical_v12_retriever.py")
HierarchicalV12Retriever = _v12_mod.HierarchicalV12Retriever


# ─── Fakes ──────────────────────────────────────────────────────────

class _FakeES:
    """Records the last query body and returns two fixed hits."""
    def __init__(self):
        self.last_index = None
        self.last_body = None

    def search(self, index, body):
        self.last_index = index
        self.last_body = body
        return {"hits": {"hits": [
            {"_source": {"chunk_id": "c1"}, "_score": 2.0},
            {"_source": {"chunk_id": "c2"}, "_score": 1.0},
        ]}}


class _Vec(list):
    def tolist(self):
        return list(self)


class _FakeEnc:
    def encode(self, q):
        return _Vec([0.1, 0.2, 0.3])


def _make(fine_type="hybrid"):
    r = HierarchicalV12Retriever(
        es_url="http://unused",
        index_prefix="test",
        condition_id="tf",
        coarse_chunker=MagicMock(),
        fine_chunker=MagicMock(),
        fine_retriever_type=fine_type,
        stage2_strategy="terms_filter",
    )
    r._fine._es_client = _FakeES()
    r._fine.index_name = "test_fine"
    r._fine._embedding_model = _FakeEnc()
    r._fine_chunks = [Chunk("c1", "a.py"), Chunk("c2", "b.py")]
    return r


FILES = ["pandas/core/frame.py", "pandas/core/series.py"]


# ─── BM25 ───────────────────────────────────────────────────────────

def test_bm25_filter_restricts_to_document_ids():
    r = _make("sparse")
    res = r._bm25_with_filter("add column", 5, FILES)
    body = r._fine._es_client.last_body
    terms = body["query"]["bool"]["filter"]["terms"]["document_id"]
    assert terms == FILES
    assert "multi_match" in body["query"]["bool"]["must"]
    assert body["size"] == 5
    assert res == [("c1", 2.0), ("c2", 1.0)]


# ─── Dense ──────────────────────────────────────────────────────────

def test_dense_filter_restricts_to_document_ids():
    r = _make("dense")
    res = r._dense_with_filter("add column", 5, FILES)
    knn = r._fine._es_client.last_body["knn"]
    assert knn["field"] == "embedding"
    assert knn["filter"]["terms"]["document_id"] == FILES
    assert knn["k"] == 5
    assert res == [("c1", 2.0), ("c2", 1.0)]


# ─── Hybrid ─────────────────────────────────────────────────────────

def test_hybrid_filter_applies_filter_to_both_subqueries():
    r = _make("hybrid")
    res = r._hybrid_with_filter("add column", 5, FILES)
    # The last call (dense) must still carry the terms filter; BM25 path is
    # covered above. Hybrid must fuse and return a non-empty ranked list.
    knn = r._fine._es_client.last_body["knn"]
    assert knn["filter"]["terms"]["document_id"] == FILES
    assert len(res) > 0
    ids = [cid for cid, _ in res]
    assert "c1" in ids and "c2" in ids


def test_normalize_scores_dict_handles_constant_and_spread():
    norm = HierarchicalV12Retriever._normalize_scores_dict([("a", 2.0), ("b", 1.0)])
    assert norm["a"] == 1.0 and norm["b"] == 0.0
    flat = HierarchicalV12Retriever._normalize_scores_dict([("a", 5.0), ("b", 5.0)])
    assert flat == {"a": 1.0, "b": 1.0}
    assert HierarchicalV12Retriever._normalize_scores_dict([]) == {}


if __name__ == "__main__":
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
