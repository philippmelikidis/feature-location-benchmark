"""
Benchmark Retrievers Package.

Provides retriever implementations for the benchmark:
- ElasticsearchRetriever: BM25 text search via Elasticsearch (production setup)
"""

from benchmark.retrievers.base import BaseRetriever

RETRIEVER_REGISTRY = {}

# Lazy imports to avoid dependency issues
def _register_retrievers():
    """Register retrievers lazily."""
    global RETRIEVER_REGISTRY
    try:
        from benchmark.retrievers.es_retriever import ElasticsearchRetriever
        RETRIEVER_REGISTRY["elasticsearch"] = ElasticsearchRetriever
    except ImportError:
        pass
    try:
        from benchmark.retrievers.hierarchical_retriever import HierarchicalRetriever
        RETRIEVER_REGISTRY["hierarchical"] = HierarchicalRetriever
    except ImportError:
        pass
    try:
        from benchmark.retrievers.hierarchical_v12_retriever import HierarchicalV12Retriever
        RETRIEVER_REGISTRY["hierarchical_v12"] = HierarchicalV12Retriever
    except ImportError:
        pass
    try:
        from benchmark.retrievers.hierarchical_ensemble_retriever import HierarchicalEnsembleRetriever
        RETRIEVER_REGISTRY["hierarchical_ensemble"] = HierarchicalEnsembleRetriever
    except ImportError:
        pass

_register_retrievers()

__all__ = ["BaseRetriever", "RETRIEVER_REGISTRY"]
