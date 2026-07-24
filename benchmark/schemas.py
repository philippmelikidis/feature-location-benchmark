#!/usr/bin/env python3
"""
schemas.py – Pydantic models for the benchmark pipeline.

Follows the Sample Structure from spec Section 2.2.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class TestCaseType(str, Enum):
    """Test Case Types as defined in spec Section 3."""
    TC1 = "TC1"  # Lexikalisch nahe (Query enthält Identifier)
    TC2 = "TC2"  # Semantisch (keine direkten Namen)
    TC3 = "TC3"  # Strukturabhängig (Repo-Architektur-Verständnis)


class CodeEra(str, Enum):
    """Era classification for human-written vs. potentially AI-assisted code."""
    PRE_AI = "pre_ai"    # Before 2022: predominantly human-written code
    POST_AI = "post_ai"  # 2022+: potentially AI-assisted (Copilot, ChatGPT era)
    UNKNOWN = "unknown"  # No date available


# Cutoff date: GitHub Copilot general availability was June 2022,
# ChatGPT launched Nov 2022. We use 2022-01-01 as conservative cutoff.
AI_ERA_CUTOFF = "2022-01-01"


# ──────────────────────────────────────────────────────────────
# Ground Truth & Dataset
# ──────────────────────────────────────────────────────────────

class GroundTruthTarget(BaseModel):
    """A single ground-truth target (file/function that should be found)."""
    file_path: str
    function_name: Optional[str] = None
    span: Optional[List[int]] = None  # [start_line, end_line]


class GroundTruth(BaseModel):
    """Ground truth for a single sample."""
    targets: List[GroundTruthTarget]


class SampleMetadata(BaseModel):
    """Metadata for a benchmark sample."""
    test_case_type: TestCaseType
    issue_number: Optional[int] = None
    issue_url: Optional[str] = None
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    difficulty: Optional[str] = None  # "easy", "medium", "hard"
    issue_created_at: Optional[str] = None  # ISO-8601 timestamp
    pr_created_at: Optional[str] = None     # ISO-8601 timestamp
    pr_merged_at: Optional[str] = None      # ISO-8601 timestamp
    era: Optional[CodeEra] = None           # pre_ai / post_ai / unknown


class Sample(BaseModel):
    """A single benchmark sample (spec Section 2.2)."""
    sample_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    repo_id: str
    commit_hash: str
    query: str  # The feature request / issue description
    ground_truth: GroundTruth
    metadata: SampleMetadata


class BenchmarkDataset(BaseModel):
    """The complete benchmark dataset."""
    version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    description: str = "FLBench Feature Location Benchmark v1"
    samples: List[Sample]

    @property
    def total_samples(self) -> int:
        return len(self.samples)

    @property
    def samples_by_repo(self) -> Dict[str, List[Sample]]:
        result: Dict[str, List[Sample]] = {}
        for s in self.samples:
            result.setdefault(s.repo_id, []).append(s)
        return result

    @property
    def samples_by_tc(self) -> Dict[str, List[Sample]]:
        result: Dict[str, List[Sample]] = {}
        for s in self.samples:
            result.setdefault(s.metadata.test_case_type, []).append(s)
        return result

    @property
    def samples_by_era(self) -> Dict[str, List[Sample]]:
        result: Dict[str, List[Sample]] = {}
        for s in self.samples:
            era = s.metadata.era.value if s.metadata.era else "unknown"
            result.setdefault(era, []).append(s)
        return result

    def filter_by_era(self, era: str, cutoff: Optional[str] = None,
                      date_field: str = "issue_created_at") -> "BenchmarkDataset":
        """Return a new dataset containing only samples of the given era.

        cutoff: Optionaler ISO-Datums-Cutoff (z. B. "2022-11-30",
        ChatGPT-Launch = primärer Cutoff des Research-Teams). Wenn gesetzt,
        wird die Era DYNAMISCH reklassifiziert statt das beim Build
        eingebrannte Label (AI_ERA_CUTOFF=2022-01-01) zu verwenden.
        Samples ohne Datum zählen dann als "unknown".

        date_field: "issue_created_at" (Default; charakterisiert die QUERY-
        Seite — wann das Issue formuliert wurde) oder "pr_merged_at"
        (charakterisiert die CODE-Seite — wann der Fix gemergt wurde; für die
        KI-generierter-Code-Hypothese der passendere Proxy, Einwand aus dem Team:
        ein Pre-AI-Issue kann Post-AI mit KI-Unterstützung gefixt worden
        sein). Nur im Cutoff-Modus wirksam.
        """
        if date_field not in ("issue_created_at", "pr_merged_at"):
            raise ValueError(f"Unbekanntes date_field: {date_field!r}")
        if cutoff:
            def _era_of(s: "Sample") -> str:
                if date_field == "pr_merged_at":
                    date = s.metadata.pr_merged_at or ""
                else:
                    date = (s.metadata.issue_created_at
                            or s.metadata.pr_created_at or "")
                if not date:
                    return "unknown"
                return "pre_ai" if date < cutoff else "post_ai"
            filtered = [s for s in self.samples if _era_of(s) == era]
            tag = f"era={era}@{cutoff}({date_field})"
        else:
            filtered = [
                s for s in self.samples
                if (s.metadata.era and s.metadata.era.value == era)
                or (era == "unknown" and not s.metadata.era)
            ]
            tag = f"era={era}"
        return BenchmarkDataset(
            version=self.version,
            created_at=self.created_at,
            description=f"{self.description} [{tag}]",
            samples=filtered,
        )


# ──────────────────────────────────────────────────────────────
# Chunks (output of chunking strategies)
# ──────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """A code chunk produced by a chunking strategy."""
    chunk_id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str = "code"  # "function", "class", "file", "block"
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    token_count: Optional[int] = None


# ──────────────────────────────────────────────────────────────
# Retrieval Results
# ──────────────────────────────────────────────────────────────

class RetrievedItem(BaseModel):
    """A single retrieved chunk with its score."""
    chunk_id: str
    file_path: str
    score: float
    rank: int
    function_name: Optional[str] = None


class RetrievalResult(BaseModel):
    """Result of retrieval for a single query."""
    sample_id: str
    condition_id: str
    k: int
    retrieved: List[RetrievedItem]
    retrieval_time_ms: float = 0.0


# ──────────────────────────────────────────────────────────────
# Metrics & Run Results
# ──────────────────────────────────────────────────────────────

class SampleMetrics(BaseModel):
    """Metrics for a single sample."""
    sample_id: str
    recall_at_k: float
    mrr_at_k: float
    first_hit_rank: Optional[int] = None  # Rank of first ground-truth hit
    tc_type: Optional[str] = None         # TC1/TC2/TC3
    repo_id: Optional[str] = None         # Repository identifier
    # ── Hierarchical-retrieval diagnostics (V11). None for non-V11 runs. ──
    # stage1_hit: True iff at least one GT file was in the Stage-1 candidate
    #   set. Lets the report distinguish "Stage-1 missed it" (coarse bug)
    #   from "Stage-1 found it but Stage-2 didn't rank it" (fine bug).
    # stage1_n_files: how many candidate files Stage-1 returned (= top_n_files
    #   in the typical case).
    stage1_hit: Optional[bool] = None
    stage1_n_files: Optional[int] = None
    # ── Stage-2 diagnostics (V12): how Stage 2 used the candidate set ──
    stage2_from_candidates: Optional[int] = None  # final results from Stage-1 files
    stage2_from_fallback: Optional[int] = None    # final results NOT in Stage-1 files


class RunMetrics(BaseModel):
    """Aggregated metrics for a run (condition × k)."""
    condition_id: str
    k: int
    recall_at_k: float  # Mean over all samples
    mrr_at_k: float  # Mean over all samples
    num_samples: int
    per_sample: List[SampleMetrics] = []


class RunResult(BaseModel):
    """Complete result of a benchmark run."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    condition_id: str
    k: int
    dataset_version: str
    repo_id: str
    chunking_strategy: str
    retriever_type: str
    embedding_model: Optional[str]
    metrics: RunMetrics
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    retrieval_results: List[RetrievalResult] = []


class BenchmarkReport(BaseModel):
    """Full benchmark report with all runs."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    dataset_version: str
    runs: List[RunResult]

    @property
    def summary_table(self) -> List[Dict[str, Any]]:
        """Generate summary table rows."""
        rows = []
        for run in self.runs:
            rows.append({
                "condition_id": run.condition_id,
                "k": run.k,
                "repo": run.repo_id,
                "retriever": run.retriever_type,
                "chunking": run.chunking_strategy,
                "embedding": run.embedding_model or "–",
                "recall@k": run.metrics.recall_at_k,
                "mrr@k": run.metrics.mrr_at_k,
                "n_samples": run.metrics.num_samples,
            })
        return rows
