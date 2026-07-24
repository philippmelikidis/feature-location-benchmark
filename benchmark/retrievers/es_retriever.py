#!/usr/bin/env python3
"""
es_retriever.py – Elasticsearch Retriever with Sparse, Dense, and Hybrid modes.

Three retrieval strategies on ES 8.11:
  - Sparse (BM25):  multi_match best_fields on text fields
  - Dense:          knn search on dense_vector (sentence-transformers embeddings)
  - Hybrid:         α·BM25_score + (1-α)·Dense_score (manual score fusion)

Hybrid does NOT use ES's built-in RRF/score combination. Instead, both queries
run independently, scores are min-max normalized, and then fused with configurable α.

Each benchmark variant creates its own ES index with both text fields
and a dense_vector embedding field.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from .base import BaseRetriever
from benchmark.chunking.base import Chunk

APP_ROOT = str(Path(__file__).parent.parent.parent)
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)


# ─── Custom ES mapping with dense_vector ───────────────────────

def _get_benchmark_mapping(dims: int = 768, with_embedding: bool = True) -> dict:
    """ES index mapping with text fields + dense_vector for embeddings.

    Lean settings to keep the heap footprint small in CI:
      - 1 primary shard (defaults to 1 in 8.x but explicit for older clusters)
      - 0 replicas (we never need redundancy in a benchmark run)
      - refresh_interval=-1 during bulk index, then re-enabled before search
      - codec=best_compression to reduce JVM heap pressure on segments
    """
    properties = {
        "chunk_id":     {"type": "keyword"},
        "document_id":  {"type": "keyword"},
        "title":        {"type": "text", "analyzer": "standard"},
        "content":      {"type": "text", "analyzer": "standard"},
        "keywords":     {"type": "keyword"},
        "topics":       {"type": "keyword"},
        "visibility":   {"type": "keyword"},
        "category":     {"type": "keyword"},
        "sub_category": {"type": "keyword"},
        "token_count":  {"type": "integer"},
    }
    if with_embedding:
        properties["embedding"] = {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": "cosine",
        }
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1",
                "codec": "best_compression",
            }
        },
        "mappings": {"properties": properties},
    }


# ─── Module-level embedding-model cache ────────────────────────
# Loading sentence-transformers per-variant burns >400 MB of RAM each time
# and never frees it (the GC doesn't always run between variants in CI).
# We share a single instance per (model_name, device) tuple across all
# ElasticsearchRetriever instances in the same process.
_EMBEDDING_CACHE: Dict[str, "object"] = {}


def _patch_transformers_compat():
    """Shim für sehr neue/dev transformers (z. B. 5.x).

    custom-Code-Modelle (jina-bert, gte new-impl) importieren
    `find_pruneable_heads_and_indices` aus `transformers.pytorch_utils`. In
    neueren transformers wurde die Funktion verschoben/entfernt → ImportError.
    Falls sie noch irgendwo existiert, injizieren wir sie zurück in
    pytorch_utils, sodass die Modelle auf der vorhandenen transformers-Version
    laden (ohne separate Env). Bringt nichts, wenn die Funktion komplett fehlt —
    dann hilft nur die gepinnte Env (siehe scripts/setup_code_env.sh).
    """
    try:
        import transformers.pytorch_utils as _pu
    except Exception:
        return
    if hasattr(_pu, "find_pruneable_heads_and_indices"):
        return
    fn = None
    for modpath in ("transformers.utils.modeling_utils",
                    "transformers.modeling_utils"):
        try:
            mod = __import__(modpath, fromlist=["find_pruneable_heads_and_indices"])
            fn = getattr(mod, "find_pruneable_heads_and_indices", None)
            if fn is not None:
                break
        except Exception:
            continue
    if fn is not None:
        _pu.find_pruneable_heads_and_indices = fn


# Query-seitige Instructions (nur auf die Query, NICHT auf die Dokumente).
# Beide Modelle haben eine empfohlene Instruction → fairer Vergleich, wenn man
# sie aktiviert (EMBED_USE_INSTRUCTION=1). Default aus (= bisheriges Verhalten).
_QUERY_INSTRUCTIONS = {
    "qwen3-embedding": ("Instruct: Given a search query, retrieve relevant code "
                        "that answers the query\nQuery: "),
    "bge": "Represent this sentence for searching relevant passages: ",
}


def _use_instruction() -> bool:
    return os.environ.get("EMBED_USE_INSTRUCTION", "").lower() in ("1", "true", "yes")


def _query_instruction(model_name: Optional[str]) -> str:
    if not model_name:
        return ""
    n = model_name.lower()
    for key, instr in _QUERY_INSTRUCTIONS.items():
        if key in n:
            return instr
    return ""


def _needs_gte_newimpl_fix(model_name: str) -> bool:
    """True for models on Alibaba's GTE "new-impl" architecture.

    SFR-Embedding-Code-* und gte-*-v1.5 nutzen custom Modeling-Code, dessen
    Unpadding- + Memory-Efficient-Attention-Pfade CUDA+xformers voraussetzen.
    Auf MPS (Apple Silicon) crashen sie ("index out of bounds for dimension 0",
    "Invalid buffer size: … GiB"). Wir schalten beide Pfade ab und kappen die
    Sequenzlänge.
    """
    n = model_name.lower()
    return ("sfr-embedding-code" in n) or ("new-impl" in n) or ("gte" in n and "v1.5" in n)


def _get_or_load_embedding_model(model_name: str, device: str):
    # Optionaler Override: EMBED_DEVICE=cpu|mps|cuda erzwingt das Gerät.
    forced = os.environ.get("EMBED_DEVICE")
    if forced:
        device = forced

    gte_fix = _needs_gte_newimpl_fix(model_name)

    # MPS ist mit der GTE-"new-impl"-Architektur (RoPE-/Attention-Index-Kernel)
    # defekt: garbage-Indizes → "index out of bounds". Ohne expliziten Override
    # automatisch auf CPU ausweichen (korrekt, nur langsamer). CUDA gibt es auf
    # dem Mac nicht; auf einer NVIDIA-Maschine liefe das Modell schnell auf MPS-
    # Ersatz "cuda".
    if gte_fix and device == "mps" and not forced:
        print("   GTE-new-impl-Modell auf MPS instabil → fällt auf CPU zurück "
              "(erzwingen mit EMBED_DEVICE=mps)")
        device = "cpu"

    key = f"{model_name}|{device}"
    if key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[key]
    _patch_transformers_compat()  # rescue custom-code models on very new transformers
    from sentence_transformers import SentenceTransformer

    # trust_remote_code=True ist nötig für Modelle mit eigenem Modeling-Code
    # (z. B. Salesforce/SFR-Embedding-Code-*). Für bge/MiniLM ein harmloser No-op.
    st_kwargs = {"device": device, "trust_remote_code": True}
    if gte_fix:
        # Standard-(gepaddete) Attention erzwingen → kein MPS-Crash.
        st_kwargs["config_kwargs"] = {
            "unpad_inputs": False,
            "use_memory_efficient_attention": False,
        }

    try:
        model = SentenceTransformer(model_name, **st_kwargs)
    except TypeError:
        # Ältere sentence-transformers ohne trust_remote_code/config_kwargs.
        model = SentenceTransformer(model_name, device=device)

    # Sequenzlänge kappen für Code-Embeddings mit großer Default-Länge (SFR/jina
    # = 8192). 512 entspricht der effektiven bge-Länge → fairer Vergleich, kleiner
    # Attention-Buffer, schneller. Überschreibbar via EMBED_MAX_SEQ_LEN.
    _ln = model_name.lower()
    cap_seq = gte_fix or ("jina-embeddings-v2-base-code" in _ln) or ("qwen3-embedding" in _ln)
    if cap_seq:
        try:
            cap = int(os.environ.get("EMBED_MAX_SEQ_LEN", "512"))
            if getattr(model, "max_seq_length", 0) and model.max_seq_length > cap:
                model.max_seq_length = cap
        except Exception:
            pass

    _EMBEDDING_CACHE[key] = model
    return model


def free_embedding_cache():
    """Free all cached embedding models — call between repos to reclaim RAM."""
    global _EMBEDDING_CACHE, _VECTOR_CACHE
    _EMBEDDING_CACHE.clear()
    _VECTOR_CACHE.clear()
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ─── Vector cache ──────────────────────────────────────────────
# Cache the encoded vectors per (model, chunker_strategy, repo_id, commit).
# V5a/b/c all use (BGE, function_level), V10a/b/c all use (BGE, ast_based) —
# without this cache they re-encode the same chunks 3 times each.
# Empirical: ~30-50% wall-time saved on the full V1-V11c sweep.
_VECTOR_CACHE: Dict[str, "object"] = {}


def _vector_cache_key(model_name: str, chunker_strategy: str,
                      repo_id: str, commit: str, num_chunks: int,
                      first_chunk_id: str = "", last_chunk_id: str = "") -> str:
    """Build a stable cache key for an embedding batch.

    Includes first/last chunk_id as a cheap content fingerprint — guards
    against accidentally reusing vectors for a different chunk-list with
    matching count (e.g. if a chunker is non-deterministic).
    """
    return (
        f"{model_name}|{chunker_strategy}|{repo_id}|{commit}"
        f"|n={num_chunks}|fst={first_chunk_id}|lst={last_chunk_id}"
    )


def vector_cache_get(key: str):
    return _VECTOR_CACHE.get(key)


def vector_cache_put(key: str, vectors):
    _VECTOR_CACHE[key] = vectors


def vector_cache_stats() -> Dict[str, int]:
    return {"entries": len(_VECTOR_CACHE)}


class ElasticsearchRetriever(BaseRetriever):
    """
    Elasticsearch retriever supporting Sparse (BM25), Dense, and Hybrid search.

    Hybrid mode:
        Runs BM25 and knn independently, min-max normalizes scores,
        then fuses: final = α·bm25_norm + (1-α)·dense_norm.
    """

    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index_prefix: str = "benchmark",
        condition_id: str = "v1",
        retriever_type: str = "sparse",
        embedding_model_name: Optional[str] = None,
        embedding_dims: int = 768,
        hybrid_alpha: float = 0.5,
    ):
        super().__init__(name=f"es_{retriever_type}_{condition_id}")
        self.es_url = es_url
        self.index_name = f"{index_prefix}_{condition_id}".lower()
        self.retriever_type = retriever_type
        self.embedding_model_name = embedding_model_name
        self.embedding_dims = embedding_dims
        self.hybrid_alpha = hybrid_alpha

        self.chunks: List[Chunk] = []
        self._chunk_map: Dict[str, Chunk] = {}
        self._es_client = None
        self._embedding_model = None

    # ─── Lazy init ──────────────────────────────────────────

    def _init_es(self):
        if self._es_client is not None:
            return
        import elasticsearch
        from elasticsearch import Elasticsearch

        # Hard fail-fast on incompatible client versions BEFORE we connect.
        # ES 8.x server only accepts elasticsearch-py 7.x or 8.x; the 9.x
        # client sends `Accept: ...; compatible-with=9` and the server
        # responds with `media_type_header_exception` — which surfaces as
        # `ping() returns False` and is hell to debug. Detect it up front.
        client_major = int(elasticsearch.__version__[0]) if elasticsearch.__version__ else 0
        if client_major >= 9:
            raise RuntimeError(
                f"elasticsearch-py {elasticsearch.__version__} is incompatible "
                f"with ES 8.x. Run:\n"
                f"  pip install --force-reinstall 'elasticsearch>=8.0,<9.0' "
                f"'elastic-transport>=8.0,<9.0'"
            )

        # request_timeout: any single ES op (search/index/delete) that hangs
        # longer than this raises instead of waiting for kernel TCP timeout (~120s).
        # max_retries=2 with the default exponential backoff catches transient
        # blips without burning minutes when ES is dead.
        self._es_client = Elasticsearch(
            self.es_url,
            request_timeout=15,
            max_retries=2,
            retry_on_timeout=True,
        )
        # Healthcheck: ping() with a short-but-tolerant timeout so a dead
        # ES container fails the variant fast, but a busy local ES with a
        # large heap still gets through.
        try:
            ok = self._es_client.options(request_timeout=10).ping()
        except Exception as e:
            raise ConnectionError(f"ES healthcheck failed at {self.es_url}: {e}")
        if not ok:
            raise ConnectionError(
                f"Cannot reach ES at {self.es_url} — ping() returned False. "
                "Common causes: ES has security enabled (try https:// + auth), "
                "wrong port, wrong host, or client/server version mismatch."
            )

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect best device: CUDA (H100/GPU) > MPS (Mac) > CPU."""
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  GPU detected: {name} (CUDA)")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(f"  Apple Silicon detected (MPS)")
            return "mps"
        print(f"  Using CPU")
        return "cpu"

    def _init_embedding_model(self):
        if self._embedding_model is not None:
            return
        if not self.embedding_model_name:
            return
        device = self._detect_device()
        print(f"  Loading embedding model: {self.embedding_model_name} → {device}")
        # Use module-level cache: load once per (model, device), reuse across variants.
        self._embedding_model = _get_or_load_embedding_model(
            self.embedding_model_name, device
        )
        # `get_sentence_embedding_dimension` was renamed in newer
        # sentence-transformers; fall back gracefully to silence the FutureWarning.
        if hasattr(self._embedding_model, "get_embedding_dimension"):
            actual_dims = self._embedding_model.get_embedding_dimension()
        else:
            actual_dims = self._embedding_model.get_sentence_embedding_dimension()
        if actual_dims != self.embedding_dims:
            print(f"   Adjusting dims: {self.embedding_dims} → {actual_dims}")
            self.embedding_dims = actual_dims

    # ─── Chunk → ES document ───────────────────────────────

    def _chunk_to_es_doc(self, chunk: Chunk, embedding: list = None) -> dict:
        title = chunk.function_name or Path(chunk.file_path).stem
        keywords = self._extract_keywords(chunk.content)
        doc = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.file_path,
            "title": title,
            "content": chunk.content,
            "keywords": keywords,
            "topics": ["benchmark", "__all__"],
            "visibility": ["private", "protected", "public"],
            "category": "content",
            "sub_category": "code",
            "token_count": chunk.token_count or 0,
        }
        if embedding is not None:
            doc["embedding"] = embedding
        return doc

    @staticmethod
    def _extract_keywords(code: str) -> List[str]:
        identifiers = set()
        for m in re.finditer(r'def\s+(\w+)', code):
            identifiers.add(m.group(1))
        for m in re.finditer(r'class\s+(\w+)', code):
            identifiers.add(m.group(1))
        for m in re.finditer(r'(?:from\s+\S+\s+)?import\s+(\w+)', code):
            identifiers.add(m.group(1))
        stop = {'self', 'cls', 'None', 'True', 'False', 'def', 'class',
                'return', 'import', 'from', 'if', 'else', 'for', 'while',
                'try', 'except', 'with', 'as', 'in', 'not', 'and', 'or'}
        return list(identifiers - stop)[:20]

    # ─── Indexing ──────────────────────────────────────────

    def index(self, chunks: List[Chunk], cache_context: Optional[dict] = None):
        """Create ES index and bulk-index chunks with text + optional embeddings.

        cache_context (optional) keys:
            chunker_strategy: str (e.g. "function_level")
            repo_id: str
            commit: str (or "HEAD")
        If provided, computed embedding vectors are cached and reused on
        subsequent index() calls with the same (model, chunker, repo, commit).
        """
        self._init_es()
        self.chunks = chunks
        self._chunk_map = {c.chunk_id: c for c in chunks}

        needs_embeddings = self.retriever_type in ("dense", "hybrid")

        if needs_embeddings:
            self._init_embedding_model()

        # Create index with custom mapping. Tear down the previous index for THIS
        # condition_id explicitly — never rely on __del__/GC, which fires lazily
        # and can collide with the next variant's setup.
        print(f"  Creating ES index: {self.index_name}")
        if self._es_client.indices.exists(index=self.index_name):
            self._es_client.indices.delete(index=self.index_name)
        self._es_client.indices.create(
            index=self.index_name,
            body=_get_benchmark_mapping(
                self.embedding_dims if needs_embeddings else 1,
                with_embedding=needs_embeddings,
            ),
        )

        # Compute embeddings if needed (with cache).
        embeddings = None
        if needs_embeddings and self._embedding_model:
            cache_key = None
            if cache_context:
                cache_key = _vector_cache_key(
                    model_name=self.embedding_model_name or "",
                    chunker_strategy=cache_context.get("chunker_strategy", ""),
                    repo_id=cache_context.get("repo_id", ""),
                    commit=cache_context.get("commit", "HEAD"),
                    num_chunks=len(chunks),
                    first_chunk_id=chunks[0].chunk_id if chunks else "",
                    last_chunk_id=chunks[-1].chunk_id if chunks else "",
                )
                cached = vector_cache_get(cache_key)
                if cached is not None:
                    print(f"  Embedding cache HIT — reusing {len(cached)} vectors "
                          f"({self.embedding_model_name})")
                    embeddings = cached

            if embeddings is None:
                print(f"  Computing embeddings for {len(chunks)} chunks...")
                texts = [c.content for c in chunks]
                # Batch-Größe per Env steuerbar (EMBED_BATCH_SIZE). Bei großem
                # max_seq_length (z. B. 2048) sprengt Batch 64 den MPS-Speicher
                # ("Failed to allocate private MTLBuffer 16 GiB") → kleiner setzen.
                try:
                    bs = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
                except Exception:
                    bs = 64
                embeddings = self._embedding_model.encode(
                    texts, show_progress_bar=True, batch_size=max(1, bs)
                )
                print(f"  Embeddings: shape={embeddings.shape}")
                if cache_key is not None:
                    vector_cache_put(cache_key, embeddings)

        # Bulk index. refresh_interval=-1 is set on the index, so we issue an
        # explicit _refresh once at the end instead of forcing a refresh per batch
        # (cheaper on ES heap when the index is large).
        from elasticsearch.helpers import bulk
        actions = []
        for i, chunk in enumerate(chunks):
            emb = embeddings[i].tolist() if embeddings is not None else None
            doc = self._chunk_to_es_doc(chunk, embedding=emb)
            actions.append({
                "_op_type": "index",
                "_index": self.index_name,
                "_id": chunk.chunk_id,
                "_source": doc,
            })

        if actions:
            # raise_on_error=False: einzelne fehlgeschlagene Dokumente (z. B.
            # transiente ES-Timeouts) sollen NICHT die ganze Condition abbrechen.
            # Die wenigen fehlenden Chunks haben vernachlässigbaren Recall-Effekt;
            # wichtiger ist, dass unbeaufsichtigte Läufe durchlaufen.
            success, errors = bulk(
                self._es_client,
                actions,
                refresh=False,
                request_timeout=120,
                chunk_size=500,
                raise_on_error=False,
                raise_on_exception=False,
            )
            if errors:
                print(f"   {len(errors)}/{len(actions)} Dokumente nicht indexiert "
                      f"(übersprungen, Lauf läuft weiter). Beispiel: {errors[0]}")
            else:
                print(f"  Indexed {success} chunks")
            # One explicit refresh + re-enable for searches.
            self._es_client.indices.put_settings(
                index=self.index_name,
                body={"index": {"refresh_interval": "1s"}},
            )
            self._es_client.indices.refresh(index=self.index_name)

        self._indexed = True

    # ─── Retrieval: Sparse (BM25) ─────────────────────────

    def _retrieve_bm25(self, query: str, k: int) -> List[Tuple[str, float]]:
        """BM25 text search. Returns list of (chunk_id, score)."""
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "keywords^2", "content"],
                    "type": "best_fields",
                }
            },
            "size": k,
        }
        res = self._es_client.search(index=self.index_name, body=body)
        return [
            (hit["_source"]["chunk_id"], float(hit["_score"]))
            for hit in res["hits"]["hits"]
        ]

    # ─── Retrieval: Dense (knn) ───────────────────────────

    def _encode_query(self, query: str):
        """Encode the QUERY, optional mit modell-spezifischer Instruction.

        Dokumente werden weiterhin ohne Instruction encodiert/indiziert — die
        Instruction gilt nur query-seitig (gilt für bge wie für Qwen3).
        Aktivierung: EMBED_USE_INSTRUCTION=1.
        """
        if _use_instruction():
            instr = _query_instruction(self.embedding_model_name)
            if instr:
                return self._embedding_model.encode(instr + query)
        return self._embedding_model.encode(query)

    def _retrieve_dense(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Dense vector search via knn. Returns list of (chunk_id, score)."""
        query_embedding = self._encode_query(query).tolist()
        body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": min(k * 10, len(self.chunks)),
            },
            "_source": True,
        }
        res = self._es_client.search(index=self.index_name, body=body)
        return [
            (hit["_source"]["chunk_id"], float(hit["_score"]))
            for hit in res["hits"]["hits"]
        ]

    # ─── Retrieval: Hybrid (α-weighted fusion) ────────────

    def _retrieve_hybrid(self, query: str, k: int) -> List[Tuple[str, float]]:
        """
        Hybrid: α·BM25_norm + (1-α)·Dense_norm

        Runs both queries with k*3 candidates, normalizes scores (min-max),
        fuses with α, re-ranks, returns top-k.
        """
        # Fetch more candidates for fusion
        fetch_k = min(k * 3, len(self.chunks))

        bm25_results = self._retrieve_bm25(query, fetch_k)
        dense_results = self._retrieve_dense(query, fetch_k)

        # Min-max normalize
        bm25_scores = self._normalize_scores(bm25_results)
        dense_scores = self._normalize_scores(dense_results)

        # Fuse: collect all chunk_ids from both results
        all_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        fused = {}
        for cid in all_ids:
            bm25_s = bm25_scores.get(cid, 0.0)
            dense_s = dense_scores.get(cid, 0.0)
            fused[cid] = self.hybrid_alpha * bm25_s + (1 - self.hybrid_alpha) * dense_s

        # Sort by fused score, return top-k
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return ranked

    @staticmethod
    def _normalize_scores(results: List[Tuple[str, float]]) -> Dict[str, float]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return {}
        scores = [s for _, s in results]
        min_s, max_s = min(scores), max(scores)
        spread = max_s - min_s
        if spread == 0:
            return {cid: 1.0 for cid, _ in results}
        return {cid: (s - min_s) / spread for cid, s in results}

    # ─── Main retrieve dispatch ───────────────────────────

    def retrieve(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        if not self._indexed:
            raise RuntimeError("Call index() first.")

        if self.retriever_type == "dense":
            raw = self._retrieve_dense(query, k)
        elif self.retriever_type == "hybrid":
            raw = self._retrieve_hybrid(query, k)
        else:  # sparse / bm25
            raw = self._retrieve_bm25(query, k)

        # Map chunk_ids back to Chunk objects
        results = []
        for chunk_id, score in raw:
            chunk = self._chunk_map.get(chunk_id)
            if chunk:
                results.append((chunk, score))
        return results

    # ─── Lifecycle ────────────────────────────────────────

    def reset(self):
        super().reset()
        self.chunks = []
        self._chunk_map = {}

    def cleanup(self):
        """Delete the ES index for this variant. Idempotent and timeout-safe.

        Important: callers MUST invoke this synchronously between variants.
        We deliberately do NOT delete in __del__ anymore — relying on GC
        caused index deletes to race with the next variant's create() under
        memory pressure (the V5→V6 hang in CI).
        """
        if not self._es_client:
            return
        try:
            client = self._es_client.options(request_timeout=10, ignore_status=[404])
            if client.indices.exists(index=self.index_name):
                print(f"  Deleting ES index: {self.index_name}")
                client.indices.delete(index=self.index_name)
        except Exception as e:
            # Don't propagate cleanup failures — they shouldn't kill the run,
            # but they also shouldn't be silent.
            print(f"   Cleanup failed for {self.index_name}: {e}")

    def close(self):
        """Release the ES HTTP connection and drop references."""
        try:
            if self._es_client is not None:
                self._es_client.close()
        except Exception:
            pass
        self._es_client = None
        self._embedding_model = None  # don't free the cached model, just our ref
        self.chunks = []
        self._chunk_map = {}

    def __del__(self):
        # Intentionally a no-op: __del__ runs in a non-deterministic order
        # under GC pressure and previously caused races with the next
        # variant's index creation. Cleanup is the runner's responsibility.
        pass
