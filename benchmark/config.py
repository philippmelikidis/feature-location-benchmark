#!/usr/bin/env python3
"""
config.py – Benchmark Configuration (v1 Kernset-Matrix)

10 Varianten über 2 Achsen:
  V1–V5:  Modell-/Retriever-Vergleich bei konstantem Chunking (Function-Level)
  V6–V9:  Chunking-Vergleich bei konstantem Modell (Kandidat B)
  V10:    Hybrid + strukturtreues Chunking (AST)

Embedding-Kandidaten:
  A: all-MiniLM-L6-v2                  (384d, Speed-Baseline)
  B: BAAI/bge-base-en-v1.5             (768d, praxisnahe Baseline)
  C: thenlper/gte-large                (1024d, stärkerer Retriever)
  D: Salesforce/SFR-Embedding-Code-400M_R (1024d, code-spezialisiert, CoIR #1-Familie)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


# ──────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────

class RetrieverType(str, Enum):
    SPARSE = "sparse"          # BM25, rein lexikalisch
    DENSE = "dense"            # Embedding-basiert, semantische Ähnlichkeit
    HYBRID = "hybrid"          # α·BM25 + (1-α)·Dense
    HIERARCHICAL = "hierarchical"  # Stage1 (file) → Stage2 (function) — V11
    HIERARCHICAL_V12 = "hierarchical_v12"  # Improved Stage1→Stage2 — V12
    HIERARCHICAL_V16 = "hierarchical_v16"  # LLM Query-Expansion + Dense Stage2 — V16
    HIERARCHICAL_ENSEMBLE = "hierarchical_ensemble"  # Ensemble Coarse (dual index) — V17

class ChunkingStrategy(str, Enum):
    FUNCTION_LEVEL = "function_level"
    FIXED_SIZE = "fixed_size"
    CLASS_FILE_LEVEL = "class_file_level"
    AST_BASED = "ast_based"
    HEURISTIC_CODE_AWARE = "heuristic_code_aware"
    VIRTUAL_DOCUMENT = "virtual_document"


# ──────────────────────────────────────────────────────────────
# Embedding Model Definitions
# ──────────────────────────────────────────────────────────────

@dataclass
class EmbeddingModelConfig:
    """An embedding model candidate."""
    name: str           # HuggingFace model identifier
    label: str          # Short label for display (A, B, C)
    dims: int           # Embedding dimensions
    description: str = ""

# v1 Kandidaten (austauschbar via Konfiguration)
EMBEDDING_MODELS = {
    "A": EmbeddingModelConfig(
        name="all-MiniLM-L6-v2",
        label="A",
        dims=384,
        description="Speed-Baseline, 22M params",
    ),
    "B": EmbeddingModelConfig(
        name="BAAI/bge-base-en-v1.5",
        label="B",
        dims=768,
        description="Praxisnahe Baseline, robuster Retriever",
    ),
    "C": EmbeddingModelConfig(
        name="thenlper/gte-large",
        label="C",
        dims=1024,
        description="Stärkerer Retriever, höhere Dimensionalität",
    ),
    # Code-spezialisiertes Embedding (CodeXEmbed / SFR-Embedding-Code-Familie,
    # #1 auf dem CoIR Code-Retrieval-Benchmark). 400M-Variante als lokaler
    # Drop-in-Test gegen die bge-base-Baseline (B). Benötigt trust_remote_code=True
    # und sentence-transformers>=2.7.0. dims werden zur Laufzeit aus dem Modell
    # verifiziert/angepasst (siehe ElasticsearchRetriever._init_embedding_model).
    "D": EmbeddingModelConfig(
        name="Salesforce/SFR-Embedding-Code-400M_R",
        label="D",
        dims=1024,
        description="Code-spezialisiert (SFR-Embedding-Code-400M_R), CoIR #1-Familie. "
                    "ACHTUNG: GTE-new-impl-Architektur crasht auf MPS/CPU "
                    "(transformers-Inkompatibilität) → praktisch nicht nutzbar.",
    ),
    # Code-spezialisiertes Embedding mit ALiBi statt RoPE/Unpadding → läuft
    # stabil auf MPS/CPU (kein 'index out of bounds'-Bug wie SFR). 161M, 768d,
    # trainiert auf github-code + 150M Code-QA/Docstring-Paaren. Braucht
    # trust_remote_code=True (JinaBERT), aber Standard-Attention.
    "E": EmbeddingModelConfig(
        name="jinaai/jina-embeddings-v2-base-code",
        label="E",
        dims=768,
        description="Code-spezialisiert (jina-embeddings-v2-base-code), ALiBi. "
                    "ACHTUNG: trust_remote_code-Modell → bricht auf sehr neuen "
                    "transformers (find_pruneable_heads_and_indices entfernt). "
                    "Nur mit gepinntem transformers nutzbar.",
    ),
    # NATIVE Modelle (kein trust_remote_code) — laufen auch auf sehr neuen
    # transformers/py3.13, weil ihre Architektur in transformers enthalten ist.
    # F: Qwen3-Embedding-0.6B — generalistisch, aber stark bei Code-Retrieval;
    #    nativ ab transformers>=4.51. Last-Token-Pooling, instruct-fähig.
    "F": EmbeddingModelConfig(
        name="Qwen/Qwen3-Embedding-0.6B",
        label="F",
        dims=1024,
        description="Qwen3-Embedding-0.6B (nativ, code-stark, MPS-tauglich)",
    ),
    # G: st-codesearch-distilroberta-base — DistilRoBERTa, auf CodeSearchNet
    #    trainiert, sentence-transformers-nativ. Schwächer/älter, aber kugelsicher.
    "G": EmbeddingModelConfig(
        name="flax-sentence-embeddings/st-codesearch-distilroberta-base",
        label="G",
        dims=768,
        description="st-codesearch-distilroberta-base (nativ, Code-Search, Fallback)",
    ),
    # H: Qwen3-Embedding-4B — gleiche Familie wie F, aber deutlich stärker.
    #    Nativ (kein trust_remote_code). ~8 GB fp16 → passt auf 24 GB, aber
    #    Batch klein halten (EMBED_BATCH_SIZE=8) und Seq-Cap nutzen.
    "H": EmbeddingModelConfig(
        name="Qwen/Qwen3-Embedding-4B",
        label="H",
        dims=2560,
        description="Qwen3-Embedding-4B (nativ, stärker als 0.6B; ~8GB, kleine Batch)",
    ),
}


# ──────────────────────────────────────────────────────────────
# Condition Config
# ──────────────────────────────────────────────────────────────

@dataclass
class ConditionConfig:
    """A single benchmark variant (V1–V11)."""
    condition_id: str
    retriever_type: RetrieverType
    chunking_strategy: ChunkingStrategy
    description: str
    embedding_model_label: Optional[str] = None  # "A", "B", "C" or None
    is_hybrid: bool = False
    hybrid_alpha: float = 0.5  # α: weight for BM25 in hybrid fusion

    # ── Hierarchical-only fields (V11+). Ignored by other retriever types. ──
    # Stage 1 (coarse) chunking; Stage 2 (fine) uses `chunking_strategy`.
    coarse_chunking_strategy: Optional[ChunkingStrategy] = None
    coarse_retriever_type: Optional[RetrieverType] = None
    fine_retriever_type: Optional[RetrieverType] = None
    top_n_files: int = 20  # Stage 1 candidate file count

    # ── Asymmetric query preparation (V15+, all hierarchical variants) ──
    # Stage 1 needs a compact, high-precision signal (title or LLM-expanded code
    # terms) to maximise file-level hit rate.  Stage 2 profits from the full
    # issue body — many lexical and semantic terms improve chunk ranking within
    # the candidate files.  Both modes are independent so every hierarchical
    # variant can express its intended design without affecting the other stage.
    #
    # "full"        → raw query unchanged (default for both stages)
    # "title_only"  → first non-empty line only  (compact Stage-1 signal)
    # "llm_expanded"→ pre-computed code-term expansion (V16/V17, Stage-1 only)
    stage1_query_mode: str = "full"
    stage2_query_mode: str = "full"

    # ── V12-specific fields ──
    stage2_strategy: Optional[str] = None  # "score_propagation", "terms_filter", "overfetch"
    score_lambda: float = 0.3  # V12a: weight for coarse score boost
    overfetch_multiplier: int = 20  # V12c: fine_fetch = k * this

    # ── V17-specific: Ensemble coarse (dual-index) ──
    coarse_chunking_strategy_b: Optional[ChunkingStrategy] = None  # second coarse leg
    coarse_retriever_type_b: Optional[RetrieverType] = None
    ensemble_top_n: Optional[int] = None  # merged candidate cap (default = top_n_files)
    ensemble_merge: str = "max"  # "max" or "mean"

    # ── Query preprocessing (flat retrievers, V15+) ──
    query_mode: str = "full"  # "full" (default), "title_only", "cleaned", "llm_expanded"

    # ── V16-specific: LLM Query Expansion ──
    llm_api_url: str = "http://localhost:1234/v1"
    llm_model: Optional[str] = None
    llm_temperature: float = 0.3
    llm_timeout: float = 30.0

    @property
    def is_hierarchical(self) -> bool:
        return self.retriever_type in (
            RetrieverType.HIERARCHICAL, RetrieverType.HIERARCHICAL_V12,
            RetrieverType.HIERARCHICAL_V16, RetrieverType.HIERARCHICAL_ENSEMBLE,
        )

    @property
    def embedding_model(self) -> Optional[EmbeddingModelConfig]:
        if self.embedding_model_label and self.embedding_model_label in EMBEDDING_MODELS:
            return EMBEDDING_MODELS[self.embedding_model_label]
        return None

    @property
    def embedding_name(self) -> Optional[str]:
        model = self.embedding_model
        return model.name if model else None

    @property
    def embedding_dims(self) -> int:
        model = self.embedding_model
        return model.dims if model else 0

    def __str__(self):
        model_str = self.embedding_model_label or "–"
        return (
            f"{self.condition_id}: {self.retriever_type.value} | "
            f"Embedding={model_str} | "
            f"Chunk={self.chunking_strategy.value} | "
            f"{self.description}"
        )


# ──────────────────────────────────────────────────────────────
# v1-Kernset-Matrix (14 Varianten)
# ──────────────────────────────────────────────────────────────

# α-Werte für Hybrid-Fusion: α·BM25 + (1-α)·Dense
HYBRID_ALPHAS = [0.2, 0.5, 0.8]

CONDITIONS: List[ConditionConfig] = [
    # --- Modell-/Retrieververgleich (Function-Level) ---
    ConditionConfig(
        condition_id="V1",
        retriever_type=RetrieverType.SPARSE,
        chunking_strategy=ChunkingStrategy.FUNCTION_LEVEL,
        description="BM25 Baseline",
    ),
    ConditionConfig(
        condition_id="V2",
        retriever_type=RetrieverType.DENSE,
        chunking_strategy=ChunkingStrategy.FUNCTION_LEVEL,
        embedding_model_label="A",
        description="Modellvergleich (A: all-MiniLM-L6-v2)",
    ),
    ConditionConfig(
        condition_id="V3",
        retriever_type=RetrieverType.DENSE,
        chunking_strategy=ChunkingStrategy.FUNCTION_LEVEL,
        embedding_model_label="B",
        description="Modellvergleich (B: bge-base-en-v1.5)",
    ),
    ConditionConfig(
        condition_id="V4",
        retriever_type=RetrieverType.DENSE,
        chunking_strategy=ChunkingStrategy.FUNCTION_LEVEL,
        embedding_model_label="C",
        description="Modellvergleich (C: gte-large)",
    ),
    # --- Hybrid α-Sweep (Function-Level, Embedding B) ---
    ConditionConfig(
        condition_id="V5a",
        retriever_type=RetrieverType.HYBRID,
        chunking_strategy=ChunkingStrategy.FUNCTION_LEVEL,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.2,
        description="Hybrid Function-Level (α=0.2, Dense-dominant)",
    ),
    ConditionConfig(
        condition_id="V5b",
        retriever_type=RetrieverType.HYBRID,
        chunking_strategy=ChunkingStrategy.FUNCTION_LEVEL,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        description="Hybrid Function-Level (α=0.5, balanced)",
    ),
    ConditionConfig(
        condition_id="V5c",
        retriever_type=RetrieverType.HYBRID,
        chunking_strategy=ChunkingStrategy.FUNCTION_LEVEL,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.8,
        description="Hybrid Function-Level (α=0.8, BM25-dominant)",
    ),
    # --- Chunking-Vergleich (Embedding B, Dense) ---
    ConditionConfig(
        condition_id="V6",
        retriever_type=RetrieverType.DENSE,
        chunking_strategy=ChunkingStrategy.FIXED_SIZE,
        embedding_model_label="B",
        description="Chunking-Vergleich (Fixed-Size 512t)",
    ),
    ConditionConfig(
        condition_id="V7",
        retriever_type=RetrieverType.DENSE,
        chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        embedding_model_label="B",
        description="Chunking-Vergleich (Class/File-Level)",
    ),
    ConditionConfig(
        condition_id="V8",
        retriever_type=RetrieverType.DENSE,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        embedding_model_label="B",
        description="Chunking-Vergleich (AST-basiert)",
    ),
    ConditionConfig(
        condition_id="V9",
        retriever_type=RetrieverType.DENSE,
        chunking_strategy=ChunkingStrategy.HEURISTIC_CODE_AWARE,
        embedding_model_label="B",
        description="Chunking-Vergleich (Heuristic Code-Aware)",
    ),
    # --- Hybrid α-Sweep + AST (Chunking-Interaktion) ---
    ConditionConfig(
        condition_id="V10a",
        retriever_type=RetrieverType.HYBRID,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.2,
        description="Hybrid AST (α=0.2, Dense-dominant)",
    ),
    ConditionConfig(
        condition_id="V10b",
        retriever_type=RetrieverType.HYBRID,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        description="Hybrid AST (α=0.5, balanced)",
    ),
    ConditionConfig(
        condition_id="V10c",
        retriever_type=RetrieverType.HYBRID,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.8,
        description="Hybrid AST (α=0.8, BM25-dominant)",
    ),
    # --- Hierarchical Retrieval (NS3) ---
    # Stage 1 (coarse): sparse on class/file-level chunks, top-N files.
    # Stage 2 (fine):   hybrid on AST-based chunks, restricted to Stage-1 files.
    # Sweep over N to characterise the recall/latency trade-off.
    ConditionConfig(
        condition_id="V11a",
        retriever_type=RetrieverType.HIERARCHICAL,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=10,
        description="Hierarchical (sparse class/file → hybrid AST, N=10)",
    ),
    ConditionConfig(
        condition_id="V11b",
        retriever_type=RetrieverType.HIERARCHICAL,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        description="Hierarchical (sparse class/file → hybrid AST, N=20)",
    ),
    ConditionConfig(
        condition_id="V11c",
        retriever_type=RetrieverType.HIERARCHICAL,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=40,
        description="Hierarchical (sparse class/file → hybrid AST, N=40)",
    ),
    # --- Hierarchical V12: Improved Stage-2 strategies (NS3 follow-up) ---
    # All use same Stage-1 as V11b (sparse class/file, N=20) but fix Stage-2.
    ConditionConfig(
        condition_id="V12a",
        retriever_type=RetrieverType.HIERARCHICAL_V12,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        stage2_strategy="score_propagation",
        score_lambda=0.3,
        description="Hier-V12 Score Propagation (λ=0.3, fine + coarse boost)",
    ),
    ConditionConfig(
        condition_id="V12b",
        retriever_type=RetrieverType.HIERARCHICAL_V12,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        stage2_strategy="terms_filter",
        description="Hier-V12 Terms Filter (ES-level file restriction in Stage 2)",
    ),
    ConditionConfig(
        condition_id="V12c",
        retriever_type=RetrieverType.HIERARCHICAL_V12,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        stage2_strategy="overfetch",
        overfetch_multiplier=20,
        description="Hier-V12 Aggressive Overfetch (k*20 statt k*5)",
    ),
    # --- V13: Hybrid-Coarse + Terms-Filter (fix Stage-1 misses) ---
    # Key insight: 50% of queries fail because BM25-Coarse can't find the
    # right file semantically. Using Hybrid for Stage 1 bridges the
    # vocabulary gap (query language ≠ code identifiers).
    ConditionConfig(
        condition_id="V13",
        retriever_type=RetrieverType.HIERARCHICAL_V12,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.HYBRID,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        stage2_strategy="terms_filter",
        description="Hybrid-Coarse (N=20) → Terms-Filter Stage-2",
    ),
    # --- V14: Virtual Document Coarse + Terms-Filter ---
    # Key insight: embedding raw 500-line files loses specificity.
    # Virtual documents (function names + docstrings + imports) give both
    # BM25 and dense a much richer signal about what a file does.
    ConditionConfig(
        condition_id="V14a",
        retriever_type=RetrieverType.HIERARCHICAL_V12,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        stage2_strategy="terms_filter",
        description="Virtual-Doc BM25-Coarse (N=20) → Terms-Filter",
    ),
    ConditionConfig(
        condition_id="V14b",
        retriever_type=RetrieverType.HIERARCHICAL_V12,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.HYBRID,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        stage2_strategy="terms_filter",
        description="Virtual-Doc Hybrid-Coarse (N=20) → Terms-Filter",
    ),

    # ── V15: Title-Only Query (weniger Noise) ──────────────────────
    ConditionConfig(
        condition_id="V15a",
        retriever_type=RetrieverType.HIERARCHICAL_V12,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.SPARSE,
        top_n_files=20,
        stage2_strategy="terms_filter",
        stage1_query_mode="title_only",
        stage2_query_mode="full",
        description="Class/File BM25-Coarse (N=20) → Terms-Filter, Title-Only S1 / Full S2",
    ),
    ConditionConfig(
        condition_id="V15b",
        retriever_type=RetrieverType.HIERARCHICAL_V12,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.SPARSE,
        top_n_files=20,
        stage2_strategy="terms_filter",
        stage1_query_mode="title_only",
        stage2_query_mode="full",
        description="Virtual-Doc BM25-Coarse (N=20) → Terms-Filter, Title-Only S1 / Full S2",
    ),

    # ── V16: LLM Query-Expansion + Dense Stage 2 (asymmetrisches Design) ──
    # Stage 1: LLM generiert Code-Terme → BM25 auf Virtual-Doc-Index
    # Stage 2: Full Issue-Text → Dense Retrieval mit Terms-Filter
    ConditionConfig(
        condition_id="V16a",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.DENSE,
        embedding_model_label="B",
        top_n_files=20,
        stage1_query_mode="llm_expanded",
        description="V16 LLM-Expanded VDoc-BM25 (N=20) → Dense Terms-Filter",
    ),
    # V16b: Same but with Class/File-Level coarse (vergleich zu V12b baseline)
    ConditionConfig(
        condition_id="V16b",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.DENSE,
        embedding_model_label="B",
        top_n_files=20,
        stage1_query_mode="llm_expanded",
        description="V16 LLM-Expanded Class/File-BM25 (N=20) → Dense Terms-Filter",
    ),
    # V16c: LLM-Expanded + Hybrid Stage 2 (best of both worlds?)
    ConditionConfig(
        condition_id="V16c",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        stage1_query_mode="llm_expanded",
        description="V16 LLM-Expanded VDoc-BM25 (N=20) → Hybrid Terms-Filter",
    ),

    # ── V17: Ensemble Coarse (dual Stage-1) ───────────────────────
    # Combines V16b's Class/File-BM25 and V16c's VDoc-BM25 in Stage 1.
    # Union of candidate files captures files that either strategy alone misses.
    # V17a: N=20 cap (strict), Hybrid Stage 2
    ConditionConfig(
        condition_id="V17a",
        retriever_type=RetrieverType.HIERARCHICAL_ENSEMBLE,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        coarse_chunking_strategy_b=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type_b=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        ensemble_top_n=20,
        ensemble_merge="max",
        stage1_query_mode="llm_expanded",
        description="Ensemble (Class/File ∪ VDoc) BM25 (N=20) → Hybrid, LLM-Expanded",
    ),
    # V17b: N=30 cap (wider net), Hybrid Stage 2
    ConditionConfig(
        condition_id="V17b",
        retriever_type=RetrieverType.HIERARCHICAL_ENSEMBLE,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        coarse_chunking_strategy_b=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type_b=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        ensemble_top_n=30,
        ensemble_merge="max",
        stage1_query_mode="llm_expanded",
        description="Ensemble (Class/File ∪ VDoc) BM25 (N=30) → Hybrid, LLM-Expanded",
    ),
    # V17c: N=20 cap, Dense Stage 2 (V16b's MRR strength + broader coarse)
    ConditionConfig(
        condition_id="V17c",
        retriever_type=RetrieverType.HIERARCHICAL_ENSEMBLE,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        coarse_chunking_strategy_b=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type_b=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.DENSE,
        embedding_model_label="B",
        top_n_files=20,
        ensemble_top_n=20,
        ensemble_merge="max",
        stage1_query_mode="llm_expanded",
        description="Ensemble (Class/File ∪ VDoc) BM25 (N=20) → Dense, LLM-Expanded",
    ),

    # ── V18: Flat Retrieval + LLM-Expanded Query (Ablation) ───────
    # Tests whether LLM expansion alone (without hierarchical filtering)
    # improves flat retrieval. Isolates the contribution of query enrichment
    # from the contribution of the two-stage architecture.
    # Baseline comparison: V3 (flat dense, full query) = ~0.35 Recall@10
    ConditionConfig(
        condition_id="V18a",
        retriever_type=RetrieverType.DENSE,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        embedding_model_label="B",
        query_mode="llm_expanded",
        description="Flat Dense AST + LLM-Expanded Query (Ablation: expansion ohne Hierarchie)",
    ),
    ConditionConfig(
        condition_id="V18b",
        retriever_type=RetrieverType.HYBRID,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        query_mode="llm_expanded",
        description="Flat Hybrid AST + LLM-Expanded Query (Ablation: expansion ohne Hierarchie)",
    ),
    # V18c/V18d: dritter Query-Modus für die Ablation.
    # "full_expanded" hängt die LLM-Terme an den vollen Issue-Text AN, statt
    # ihn zu ersetzen. Beantwortet die Folge-Frage aus der Ablation: schadet
    # die Expansion code-starken Embeddings (Qwen3), weil der Issue-Kontext
    # wegfällt — oder auch als reine Ergänzung?
    ConditionConfig(
        condition_id="V18c",
        retriever_type=RetrieverType.DENSE,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        embedding_model_label="B",
        query_mode="full_expanded",
        description="Flat Dense AST + Full+Expanded Query (Ablation: Expansion angehängt)",
    ),
    ConditionConfig(
        condition_id="V18d",
        retriever_type=RetrieverType.HYBRID,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        query_mode="full_expanded",
        description="Flat Hybrid AST + Full+Expanded Query (Ablation: Expansion angehängt)",
    ),

    # ── V18d Alpha-Sweep: ist α=0.5 nur ein schlechter Mittelwert? ──
    # V18d (α=0.5) verlor auf v2/pandas + Qwen3-4B gegen reines Dense (V18c).
    # Hypothese: bei bereits starkem, expansion-gestütztem Dense-Signal zieht
    # BM25 den Fusion-Score eher runter. Testet dieselbe α-Spreizung wie
    # V10a/c (0.2/0.8), aber MIT full_expanded statt der reinen Query.
    ConditionConfig(
        condition_id="V18d_a02",
        retriever_type=RetrieverType.HYBRID,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.2,
        query_mode="full_expanded",
        description="Flat Hybrid AST + Full+Expanded Query, α=0.2 (Dense-lastig, Alpha-Sweep ggü. V18d)",
    ),
    ConditionConfig(
        condition_id="V18d_a08",
        retriever_type=RetrieverType.HYBRID,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.8,
        query_mode="full_expanded",
        description="Flat Hybrid AST + Full+Expanded Query, α=0.8 (BM25-lastig, Alpha-Sweep ggü. V18d)",
    ),

    # ── V19: Cross-Kombination V16b×V16c + LLM Stage-2 ───────────
    # V19a: V16b's Stage 1 (Class/File BM25) + V16c's Stage 2 (Hybrid)
    # Tests whether the more precise Class/File coarse + the stronger
    # Hybrid reranking yields the best of both worlds.
    ConditionConfig(
        condition_id="V19a",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.CLASS_FILE_LEVEL,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        stage1_query_mode="llm_expanded",
        description="V16b S1 (Class/File BM25 LLM-Exp) → V16c S2 (Hybrid)",
    ),
    # V19b: Like V16c but with LLM-expanded query ALSO in Stage 2.
    # Tests whether the enriched code terms also improve reranking.
    ConditionConfig(
        condition_id="V19b",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="B",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        stage1_query_mode="llm_expanded",
        stage2_query_mode="llm_expanded",
        description="V16c + LLM-Expanded Query auch in Stage 2 (Hybrid)",
    ),

    # ── V20: Stage-1 Top-N-Sweep + Qwen3-4B Reranking ────────────
    # Befund: Bei V11 steigt der Recall mit N (N=10→20→40: 0,472→0,498→0,579),
    # bleibt aber unter der flachen Suche (V8/V10b ≈ 0,610). Hypothese: Die grobe
    # Stufe wirft relevante Dateien zu früh raus. V20 hält das (beste) V16c-Design
    # — LLM-expandierte VDoc-BM25-Grobstufe → Hybrid-Feinstufe mit Terms-Filter —
    # bei, tauscht aber das Stage-2-Embedding auf H (Qwen3-4B) und fährt einen
    # reinen N-Sweep {20, 40, 80, 150}. Ziel: das N finden, ab dem der
    # Stage-1-Datei-Recall ≈ flache Suche erreicht, während die feine 4B-Stufe
    # den MRR-Vorteil hält. Stage-1-Recall pro Sample loggt der Runner bereits
    # (stage1_hit / stage1_n_files), sodass Recall(Stufe 1) gegen N aufgetragen
    # werden kann. Vergleich: flache Suche (V8/V10b) und Top-20 (V11b).
    ConditionConfig(
        condition_id="V20a",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="H",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        stage1_query_mode="llm_expanded",
        description="[V20 N-Sweep] V16c-Design (LLM-Exp VDoc-BM25 → Hybrid) + Qwen3-4B, N=20",
    ),
    ConditionConfig(
        condition_id="V20b",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="H",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=40,
        stage1_query_mode="llm_expanded",
        description="[V20 N-Sweep] V16c-Design (LLM-Exp VDoc-BM25 → Hybrid) + Qwen3-4B, N=40",
    ),
    ConditionConfig(
        condition_id="V20c",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="H",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=80,
        stage1_query_mode="llm_expanded",
        description="[V20 N-Sweep] V16c-Design (LLM-Exp VDoc-BM25 → Hybrid) + Qwen3-4B, N=80",
    ),
    ConditionConfig(
        condition_id="V20d",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.SPARSE,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="H",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=150,
        stage1_query_mode="llm_expanded",
        description="[V20 N-Sweep] V16c-Design (LLM-Exp VDoc-BM25 → Hybrid) + Qwen3-4B, N=150",
    ),

    # ── V20 Hybrid-Coarse-Arm (V20e–h) ──────────────────────────
    # Identisch zu V20a–d, aber die GROBE Stufe ist hybrid (α·BM25 + (1-α)·Dense)
    # statt nur sparse — testet die ursprüngliche Hypothese "Hybrid statt nur
    # sparse in Stufe 1 hebt den Stage-1-Recall". Der Dense-Leg der Grobstufe
    # nutzt dasselbe Embedding wie die Feinstufe (H, Qwen3-4B) → höherer
    # Index-/Laufzeit-Aufwand; der p50-Latenz-Report macht den Trade-off sichtbar.
    # Direkt vergleichbar: V20e↔V20a, V20f↔V20b, V20g↔V20c, V20h↔V20d.
    ConditionConfig(
        condition_id="V20e",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.HYBRID,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="H",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=20,
        stage1_query_mode="llm_expanded",
        description="[V20 N-Sweep] V16c-Design, HYBRID-Grobstufe + Qwen3-4B, N=20",
    ),
    ConditionConfig(
        condition_id="V20f",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.HYBRID,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="H",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=40,
        stage1_query_mode="llm_expanded",
        description="[V20 N-Sweep] V16c-Design, HYBRID-Grobstufe + Qwen3-4B, N=40",
    ),
    ConditionConfig(
        condition_id="V20g",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.HYBRID,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="H",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=80,
        stage1_query_mode="llm_expanded",
        description="[V20 N-Sweep] V16c-Design, HYBRID-Grobstufe + Qwen3-4B, N=80",
    ),
    ConditionConfig(
        condition_id="V20h",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.HYBRID,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="H",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=150,
        stage1_query_mode="llm_expanded",
        description="[V20 N-Sweep] V16c-Design, HYBRID-Grobstufe + Qwen3-4B, N=150",
    ),
    ConditionConfig(
        condition_id="V20i",
        retriever_type=RetrieverType.HIERARCHICAL_V16,
        chunking_strategy=ChunkingStrategy.AST_BASED,
        coarse_chunking_strategy=ChunkingStrategy.VIRTUAL_DOCUMENT,
        coarse_retriever_type=RetrieverType.HYBRID,
        fine_retriever_type=RetrieverType.HYBRID,
        embedding_model_label="H",
        is_hybrid=True,
        hybrid_alpha=0.5,
        top_n_files=500,
        stage1_query_mode="llm_expanded",
        description="[V20 N-Sweep] V16c-Design, HYBRID-Grobstufe + Qwen3-4B, "
                     "N=500 (≈ alle Dateien, effektiv keine Stage-1-Filterung "
                     "— Näherung für \"alle minus Rauschen\")",
    ),
]

# ──────────────────────────────────────────────────────────────
# SFR-Embedding-Code Experiment (Branch: experiment/sfr-embedding-code-400m)
# ──────────────────────────────────────────────────────────────
# Ziel: dense/hybrid Stage-2-Embedding von B (bge-base-en-v1.5, 768d, generisch)
# auf D (SFR-Embedding-Code-400M_R, 1024d, code-spezialisiert) umstellen und
# "ab V11" gegen die bge-base-Baseline vergleichen. Statt jede Variante manuell
# zu duplizieren, klonen wir alle V11+-Conditions automatisch und tauschen nur
# das Embedding-Label. Neue IDs erhalten das Suffix "_SFR" → eigene ES-Indizes,
# keine Kollision mit den bge-base-Läufen.
#
# Hinweis: Conditions ohne Embedding (reines BM25) werden übersprungen — dort
# ändert ein Embedding-Tausch nichts.
import dataclasses as _dataclasses

# Quellen für die Embedding-Tausch-Varianten. Abgedeckt:
#   Flat Dense:  V3 (function), V6 (fixed), V7 (class/file), V8 (ast), V9 (heuristic)
#   Flat Hybrid: V5a/b/c (function, α-Sweep), V10a/b/c (ast, α-Sweep)
#   Hierarchisch: alles ab V11 (Stage-2 dense/hybrid → Embedding relevant)
# Conditions ohne Embedding (reines BM25) werden in _gen_code_variants übersprungen.
_CODE_SOURCE_IDS = [
    "V3", "V6", "V7", "V8", "V9",
    "V5a", "V5b", "V5c",
    "V10a", "V10b", "V10c",
    "V11a", "V11b", "V11c",
    "V12a", "V12b", "V12c",
    "V13",
    "V14a", "V14b",
    "V15a", "V15b",
    "V16a", "V16b", "V16c",
    "V17a", "V17b", "V17c",
    "V18a", "V18b", "V18c", "V18d", "V18d_a02", "V18d_a08",
    "V19a", "V19b",
]

_base_map = {c.condition_id: c for c in CONDITIONS}


def _gen_code_variants(label: str, suffix: str, tag: str) -> List[ConditionConfig]:
    """Klone aller V11+-Conditions mit getauschtem Embedding-Label.

    Conditions ohne Embedding (reines BM25) werden übersprungen — dort ändert
    ein Embedding-Tausch nichts. Neue IDs erhalten `suffix` → eigene ES-Indizes.
    """
    out: List[ConditionConfig] = []
    for sid in _CODE_SOURCE_IDS:
        base = _base_map.get(sid)
        if base is None or base.embedding_model_label is None:
            continue
        out.append(_dataclasses.replace(
            base,
            condition_id=f"{sid}{suffix}",
            embedding_model_label=label,
            description=f"[{tag}] {base.description}",
        ))
    return out


# trust_remote_code-Modelle (brechen auf sehr neuen transformers — s. o.):
_sfr_conditions = _gen_code_variants("D", "_SFR", "SFR-Code-400M")
_jina_conditions = _gen_code_variants("E", "_JINA", "Jina-Code")
# NATIVE Modelle (kein remote code → laufen auf py3.13 / neuem transformers):
_qwen3_conditions = _gen_code_variants("F", "_QWEN3", "Qwen3-0.6B")
_codesearch_conditions = _gen_code_variants("G", "_CODESEARCH", "CodeSearch-DistilRoBERTa")
_qwen34b_conditions = _gen_code_variants("H", "_QWEN34B", "Qwen3-4B")

CONDITIONS.extend(_sfr_conditions)
CONDITIONS.extend(_jina_conditions)
CONDITIONS.extend(_qwen3_conditions)
CONDITIONS.extend(_codesearch_conditions)
CONDITIONS.extend(_qwen34b_conditions)

# Bequeme Listen der generierten IDs (für Run-Skripte).
SFR_CONDITION_IDS: List[str] = [c.condition_id for c in _sfr_conditions]
JINA_CONDITION_IDS: List[str] = [c.condition_id for c in _jina_conditions]
QWEN3_CONDITION_IDS: List[str] = [c.condition_id for c in _qwen3_conditions]
CODESEARCH_CONDITION_IDS: List[str] = [c.condition_id for c in _codesearch_conditions]
QWEN3_4B_CONDITION_IDS: List[str] = [c.condition_id for c in _qwen34b_conditions]
CODE_EMBED_CONDITION_IDS: List[str] = (
    SFR_CONDITION_IDS + JINA_CONDITION_IDS + QWEN3_CONDITION_IDS
    + CODESEARCH_CONDITION_IDS + QWEN3_4B_CONDITION_IDS
)

# V20: Stage-1 Top-N-Sweep (V16c-Design + Qwen3-4B). Eigene IDs, kein Auto-Clone
# (nutzen Embedding H direkt). Zwei Grobstufen-Familien über denselben N-Sweep:
#   sparse coarse (a–d), N = 20/40/80/150  vs.  hybrid coarse (e–i),
#   N = 20/40/80/150/500 (V20i: N≈500 ≈ alle Dateien, keine Stage-1-Filterung).
V20_SPARSE_IDS: List[str] = ["V20a", "V20b", "V20c", "V20d"]   # sparse Grobstufe
V20_HYBRID_IDS: List[str] = ["V20e", "V20f", "V20g", "V20h", "V20i"]   # hybrid Grobstufe
V20_CONDITION_IDS: List[str] = V20_SPARSE_IDS + V20_HYBRID_IDS

CONDITIONS_MAP: Dict[str, ConditionConfig] = {c.condition_id: c for c in CONDITIONS}


# ──────────────────────────────────────────────────────────────
# Query-Modus-Ablation
# ──────────────────────────────────────────────────────────────
# Kontrollierte Matrix Retriever × query_mode × Embedding-Modell. Jede Zelle
# ist eine bereits existierende Condition; alle Zellen einer (Modell, Retriever)-
# Zeile unterscheiden sich AUSSCHLIESSLICH im query_mode — garantiert durch
# tests/test_query_mode_ablation.py. Damit ist der reine Expansionseffekt
# pro Modell isoliert. Finaler Befund (pandas, kontrolliert):
# full_expanded schlägt full UND llm_expanded für bge (hybrid +8,3pp R@10/
# +4,9pp MRR) UND Qwen3-0.6B (hybrid +7,0pp/+6,7pp), über 3 Repos konsistent
# positiv → QUERY_MODE_RECOMMENDED_DEFAULT unten.
# Konsumiert von scripts/run_query_mode_ablation.py.
#
# Modi: "full" (Baseline), "llm_expanded" (Expansion ERSETZT Query),
#       "full_expanded" (Expansion an vollen Issue-Text ANGEHÄNGT).
# Model-Keys folgen scripts/run_code_embed_v11plus.py (bge = Basis B).
QUERY_MODE_ABLATION_MATRIX: Dict[str, Dict[str, Dict[str, str]]] = {
    "bge": {
        "dense":  {"full": "V8",   "llm_expanded": "V18a", "full_expanded": "V18c"},
        "hybrid": {"full": "V10b", "llm_expanded": "V18b", "full_expanded": "V18d"},
    },
    "qwen3": {
        "dense":  {"full": "V8_QWEN3",   "llm_expanded": "V18a_QWEN3",
                   "full_expanded": "V18c_QWEN3"},
        "hybrid": {"full": "V10b_QWEN3", "llm_expanded": "V18b_QWEN3",
                   "full_expanded": "V18d_QWEN3"},
    },
    "qwen3-4b": {
        "dense":  {"full": "V8_QWEN34B",   "llm_expanded": "V18a_QWEN34B",
                   "full_expanded": "V18c_QWEN34B"},
        "hybrid": {"full": "V10b_QWEN34B", "llm_expanded": "V18b_QWEN34B",
                   "full_expanded": "V18d_QWEN34B"},
    },
}

QUERY_MODE_ABLATION_CONDITION_IDS: List[str] = [
    cid
    for _retrievers in QUERY_MODE_ABLATION_MATRIX.values()
    for _modes in _retrievers.values()
    for cid in _modes.values()
]

# Finale Default-Empfehlung je Modell für die flachen
# llm-Conditions (dense/hybrid), aus dem abgeschlossenen Kombi-Lauf
# (Baseline-Expansion × full_expanded, siehe
# benchmark/results/query_mode_ablation/lean/variant_vs_baseline_lean.md und
# docs/query_mode_default_empfehlung.md). qwen3-4b ist hier nicht erneut
# isoliert bestätigt und bewusst NICHT gelistet.
QUERY_MODE_RECOMMENDED_DEFAULT: Dict[str, str] = {
    "bge": "full_expanded",
    "qwen3": "full_expanded",
}


# ──────────────────────────────────────────────────────────────
# K-Values & Metrics
# ──────────────────────────────────────────────────────────────

K_VALUES = [1, 5, 10]
# Retrieval runs per repo: only k=max(K_VALUES) is retrieved;
# k=1 and k=5 metrics are derived from the same top-k results.
TOTAL_RETRIEVAL_RUNS = len(CONDITIONS)                    # 14 per repo
TOTAL_RESULT_SETS = len(CONDITIONS) * len(K_VALUES)       # 42 per repo (RunResults)

# Primary metrics: MRR@10, Recall@10
# Secondary: Recall@1, Recall@5 (derived from k=10 retrieval)

# ──────────────────────────────────────────────────────────────
# Elasticsearch
# ──────────────────────────────────────────────────────────────

ES_URL = "http://localhost:9200"
ES_INDEX_PREFIX = "benchmark"


# ──────────────────────────────────────────────────────────────
# Repository Portfolio (info.txt §5.1 + §9)
# ──────────────────────────────────────────────────────────────

@dataclass
class RepoConfig:
    """Configuration for a target repository."""
    name: str
    owner: str
    url: str
    domain: str
    phase: int
    commit_hash: Optional[str] = None
    source_dirs: List[str] = field(default_factory=lambda: ["src"])
    description: str = ""


REPOSITORIES: List[RepoConfig] = [
    RepoConfig(
        name="requests", owner="psf",
        url="https://github.com/psf/requests",
        domain="HTTP Library", phase=1,
        source_dirs=["src/requests", "requests"],
        description="Utility, synchron, flache Hierarchie",
    ),
    RepoConfig(
        name="flask", owner="pallets",
        url="https://github.com/pallets/flask",
        domain="Web Framework", phase=1,
        source_dirs=["src/flask"],
        description="Modular, WSGI-basiert",
    ),
    RepoConfig(
        name="click", owner="pallets",
        url="https://github.com/pallets/click",
        domain="CLI Tooling", phase=2,
        source_dirs=["src/click"],
        description="Dekorator-basiert",
    ),
    RepoConfig(
        name="fastapi", owner="fastapi",
        url="https://github.com/fastapi/fastapi",
        domain="Modern Web", phase=2,
        source_dirs=["fastapi"],
        description="Asynchron, Type-Hints",
    ),
    RepoConfig(
        name="pandas", owner="pandas-dev",
        url="https://github.com/pandas-dev/pandas",
        domain="Data Science", phase=2,
        source_dirs=["pandas"],
        description="High-Performance, Mix (Py/C)",
    ),
]

REPOS_MAP: Dict[str, RepoConfig] = {r.name: r for r in REPOSITORIES}


def get_all_run_configs():
    """Generate all (condition × k) combinations."""
    return [
        {"condition": c, "k": k, "run_label": f"{c.condition_id}_k{k}"}
        for c in CONDITIONS for k in K_VALUES
    ]


if __name__ == "__main__":
    print(f"FLBench v1 – Kernset-Matrix")
    print(f"{'='*60}")
    print(f"\n{len(CONDITIONS)} Varianten × retrieve k={max(K_VALUES)} = {TOTAL_RETRIEVAL_RUNS} Retrievals/Repo ({TOTAL_RESULT_SETS} RunResults)\n")
    print("Embedding-Modelle:")
    for label, m in EMBEDDING_MODELS.items():
        print(f"  {label}: {m.name} ({m.dims}d) – {m.description}")
    print(f"\nVarianten:")
    for c in CONDITIONS:
        print(f"  {c}")
    print(f"\nk = {K_VALUES}")
    print(f"\nRepos ({len(REPOSITORIES)}):")
    for r in REPOSITORIES:
        print(f"  P{r.phase}: {r.owner}/{r.name} ({r.domain})")
