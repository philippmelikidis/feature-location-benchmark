#!/usr/bin/env python3
"""
cross_encoder.py – Cross-Encoder Reranker (Stage 2.5).

Nimmt eine Query und eine Liste von Chunks, schickt jedes (Query, Chunk)-Paar
GEMEINSAM durch ein Cross-Encoder-Modell (echte Interaktion zwischen Query und
Chunk, anders als der Bi-Encoder) und gibt eine nach Relevanz sortierte Liste
zurück.

Diese Klasse ist bewusst eigenständig gehalten: der Schritt-0-Probe
(scripts/probe_rerank_finerecovery.py) benutzt sie, und der spätere
V21-Retriever kann sie 1:1 wiederverwenden.

Zwei Backends werden unterstützt (Auto-Erkennung):
  - "st"   → sentence_transformers.CrossEncoder  (breit installiert, einfach)
  - "flag" → FlagEmbedding.FlagReranker          (offizieller Weg für bge-*)

Aggregations-Modi in rerank():
  - "none"      → Top-k einzelne Chunks (das, was der V21-Plan vorschlägt)
  - "file_max"  → pro Datei den besten Chunk-Score (max-pool), dann Top-k DATEIEN
                  je durch ihren besten Chunk repräsentiert. Das passt zur
                  datei-basierten Eval (metrics.recall_at_k) und ist wichtig für
                  Samples mit mehreren Ziel-Dateien.

Beispiel:
    rr = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    ranked = rr.rerank(query, chunks, top_k=10, aggregate="file_max")
    # ranked: List[Tuple[Chunk, float]]  (absteigend nach Score)
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional, Sequence

# Chunk nur für Typannotationen — kein harter Import-Zwang beim reinen Scoring.
try:
    from benchmark.chunking.base import Chunk
except Exception:  # pragma: no cover - Fallback falls Paketpfad anders
    Chunk = object  # type: ignore


def _pick_device(requested: Optional[str]) -> str:
    """cuda > mps (Apple) > cpu, sofern nichts explizit verlangt wird."""
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class CrossEncoderReranker:
    """Rerankt (Query, Chunk)-Paare mit einem Cross-Encoder.

    Args:
        model_name:     HF-Modellname. Default: BAAI/bge-reranker-v2-m3.
        device:         "cuda" | "mps" | "cpu" | None (=auto).
        max_length:     Max. Tokenlänge pro (Query, Chunk)-Paar (Tokenizer kürzt).
        batch_size:     Anzahl Paare pro Forward-Pass (Latenz-Hebel).
        backend:        "auto" | "st" | "flag".
        query_max_chars: Query wird vorab hart auf so viele Zeichen gekürzt,
                        damit bei sehr langen Issue-Texten noch Budget für den
                        Chunk bleibt (der Tokenizer kürzt sonst den Chunk weg).
        content_max_chars: analoge harte Kürzung des Chunk-Textes.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        backend: str = "auto",
        query_max_chars: int = 1000,
        content_max_chars: int = 4000,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.device = _pick_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.query_max_chars = query_max_chars
        self.content_max_chars = content_max_chars
        self.verbose = verbose

        self._backend = self._resolve_backend(backend)
        self._model = None
        self._load()

    # ── Backend-Wahl & Laden ──────────────────────────────────

    def _resolve_backend(self, backend: str) -> str:
        if backend in ("st", "flag"):
            return backend
        # auto: sentence-transformers zuerst (einfachste, häufigste Installation)
        if self._have("sentence_transformers"):
            return "st"
        if self._have("FlagEmbedding"):
            return "flag"
        raise ImportError(
            "Kein Reranker-Backend gefunden. Installiere eines davon:\n"
            "  pip install sentence-transformers      (Backend 'st')\n"
            "  pip install FlagEmbedding               (Backend 'flag')\n"
            "…plus torch (pip install torch)."
        )

    @staticmethod
    def _have(module: str) -> bool:
        import importlib.util
        return importlib.util.find_spec(module) is not None

    def _load(self):
        if self.verbose:
            print(f"  [Reranker] backend={self._backend} model={self.model_name} "
                  f"device={self.device} batch={self.batch_size} max_len={self.max_length}")
        if self._backend == "st":
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
        else:  # flag
            from FlagEmbedding import FlagReranker
            use_fp16 = self.device in ("cuda",)
            self._model = FlagReranker(self.model_name, use_fp16=use_fp16)

    # ── Scoring ───────────────────────────────────────────────

    def score_pairs(self, query: str, texts: Sequence[str]) -> List[float]:
        """Roh-Relevanzscores für [(query, text) …], gebatcht. Höher = relevanter."""
        q = (query or "")[: self.query_max_chars]
        pairs = [[q, (t or "")[: self.content_max_chars]] for t in texts]
        if not pairs:
            return []

        if self._backend == "st":
            # CrossEncoder.predict batcht selbst; show_progress_bar aus für Logs.
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [float(s) for s in scores]
        else:  # flag
            # FlagReranker.compute_score nimmt Paar-Listen; normalize=True → Sigmoid.
            scores = self._model.compute_score(
                pairs, batch_size=self.batch_size, normalize=True
            )
            if isinstance(scores, float):
                scores = [scores]
            return [float(s) for s in scores]

    # ── Reranking ─────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        chunks: List["Chunk"],
        top_k: Optional[int] = None,
        aggregate: str = "none",
    ) -> List[Tuple["Chunk", float]]:
        """Sortiert Chunks nach Cross-Encoder-Score.

        aggregate="none"     → Top-k einzelne Chunks.
        aggregate="file_max" → pro Datei bester Chunk, Top-k DATEIEN
                               (je durch besten Chunk repräsentiert).
        """
        if not chunks:
            return []

        texts = [getattr(c, "content", "") or "" for c in chunks]
        scores = self.score_pairs(query, texts)
        scored = list(zip(chunks, scores))

        if aggregate == "file_max":
            best_by_file = {}  # file_path -> (chunk, score)
            for c, s in scored:
                fp = getattr(c, "file_path", None)
                if fp is None:
                    continue
                cur = best_by_file.get(fp)
                if cur is None or s > cur[1]:
                    best_by_file[fp] = (c, s)
            ranked = sorted(best_by_file.values(), key=lambda x: x[1], reverse=True)
        else:  # "none"
            ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    def rerank_variants(self, query, chunks, top_k=None, text_fn=None):
        """Scored die Chunks EINMAL und liefert beide Rankings zurück.

        Vermeidet doppeltes Scoren, wenn man chunk-topk UND file-maxpool braucht
        (z. B. im Schritt-0-Probe). Rückgabe:
            {"none": [(chunk, score) …], "file_max": [(chunk, score) …]}

        text_fn: optionale Funktion Chunk -> str, die den Text baut, der dem
        Modell gezeigt wird (z. B. mit vorangestelltem Dateipfad/Signatur für
        "code-aware" Reranking). Default: chunk.content.
        """
        if not chunks:
            return {"none": [], "file_max": []}

        if text_fn is None:
            texts = [getattr(c, "content", "") or "" for c in chunks]
        else:
            texts = [text_fn(c) for c in chunks]
        scores = self.score_pairs(query, texts)
        scored = list(zip(chunks, scores))

        none = sorted(scored, key=lambda x: x[1], reverse=True)

        best_by_file = {}
        for c, s in scored:
            fp = getattr(c, "file_path", None)
            if fp is None:
                continue
            cur = best_by_file.get(fp)
            if cur is None or s > cur[1]:
                best_by_file[fp] = (c, s)
        file_max = sorted(best_by_file.values(), key=lambda x: x[1], reverse=True)

        if top_k is not None:
            none = none[:top_k]
            file_max = file_max[:top_k]
        return {"none": none, "file_max": file_max}
