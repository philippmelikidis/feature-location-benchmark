#!/usr/bin/env python3
"""
dataset_builder.py – Build Benchmark Dataset from Extracted Samples

Converts raw GitHub-extracted samples into the benchmark dataset format
(spec Section 2.2). Includes TC1/TC2/TC3 classification.
"""

import glob
import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Union  # noqa: F811

from benchmark.schemas import (
    Sample, GroundTruth, GroundTruthTarget,
    SampleMetadata, TestCaseType, BenchmarkDataset,
    CodeEra, AI_ERA_CUTOFF,
)

# Maintenance-/Meta-Issues ohne einzelnes Code-Ziel —
# identisch zum Extractor-Filter, greift auch bei alten raw-Dateien.
META_TITLE_PATTERN = re.compile(
    r"^(DOC|TRACKER|STY|BUILD|CI|TST|WEB|DEPS|RLS|CLN|META|Release Plan)\b[:\s]",
    re.IGNORECASE,
)


def deterministic_sample_id(repo_name: str, issue_number) -> str:
    """Stabile 8-Hex-ID aus (repo, issue) — identisch über Rebuilds hinweg.

    Ersetzt uuid4: damit bleiben sample_id-basierte Joins (Ergebnisse,
    LLM-Expansions) über Dataset-Rebuilds gültig.
    """
    return hashlib.md5(f"{repo_name}#{issue_number}".encode()).hexdigest()[:8]


class _RepoFileIndex:
    """Alle .py-Pfade eines Checkouts, für Suffix-Matching wie in metrics.

    Wichtig: Die Target-Validierung muss dieselbe Semantik haben wie
    metrics._file_matches (Suffix-Match) — sonst werden Samples getötet,
    deren GT-Pfad nur ein anderes Repo-Root-Prefix hat (z. B. requests'
    Umzug auf src/-Layout: GT "requests/models.py" existiert bei HEAD
    als "src/requests/models.py" und MATCHT in der Metrik).
    """

    def __init__(self, repo_root: Path):
        self.files = [
            p.relative_to(repo_root).as_posix()
            for p in repo_root.rglob("*.py")
        ]

    def target_alive(self, target_path: str) -> bool:
        t = target_path.replace("\\", "/").strip("/").removeprefix("./")
        if not t:
            return False
        for f in self.files:
            if f == t or f.endswith("/" + t) or t.endswith("/" + f):
                return True
        return False


# Common Python identifiers/patterns that indicate TC1 (lexical match)
CODE_PATTERNS = [
    re.compile(r"\b[A-Z][a-z]+[A-Z]\w+\b"),      # CamelCase class names
    re.compile(r"\b\w+_\w+\b"),                    # snake_case names
    re.compile(r"\b(?:def|class|import|from)\s+\w+"),  # Code keywords
    re.compile(r"`\w+`"),                          # Backtick-quoted code
    re.compile(r"\.\w+\("),                        # Method calls
    re.compile(r"(?:Error|Exception|Warning)\b"),   # Exception names
    re.compile(r"(?:GET|POST|PUT|DELETE|PATCH)\b"), # HTTP methods
]


def classify_era(issue_created_at: str, pr_created_at: str = "") -> CodeEra:
    """
    Classify a sample into pre-AI or post-AI era based on creation dates.

    Uses the issue creation date primarily; falls back to PR creation date.
    Cutoff: 2022-01-01 (before GitHub Copilot GA and ChatGPT).
    """
    date_str = issue_created_at or pr_created_at
    if not date_str:
        return CodeEra.UNKNOWN

    # ISO-8601 strings are lexicographically comparable
    if date_str < AI_ERA_CUTOFF:
        return CodeEra.PRE_AI
    return CodeEra.POST_AI


def classify_test_case(query: str, changed_files: List[str]) -> TestCaseType:
    """
    Heuristically classify a query into TC1, TC2, or TC3.

    TC1 (Lexical): Query contains code identifiers → BM25 should find it.
    TC2 (Semantic): Query describes behavior without naming code elements.
    TC3 (Structural): Query requires understanding repo architecture.
    """
    # Count code-like patterns in the query
    code_pattern_hits = 0
    for pattern in CODE_PATTERNS:
        matches = pattern.findall(query)
        code_pattern_hits += len(matches)

    # TC1: Query clearly contains code identifiers
    if code_pattern_hits >= 3:
        return TestCaseType.TC1

    # TC3: References to multiple directories/modules or architectural concepts
    arch_keywords = [
        "archit", "modul", "refactor", "reorganiz", "restructur",
        "depend", "circular", "coupling", "layer", "middleware",
        "plugin", "hook", "extension", "blueprint",
    ]
    arch_hits = sum(1 for kw in arch_keywords if kw in query.lower())
    if arch_hits >= 2 or len(changed_files) >= 5:
        return TestCaseType.TC3

    # TC2: Semantic / natural language description
    return TestCaseType.TC2


def build_dataset(
    raw_samples_path: Union[str, List[str]],
    output_path: str = "benchmark/data/benchmark_dataset.json",
    dataset_version: str = "1.0.0",
    era_filter: Optional[str] = None,
    build_era_splits: bool = False,
    skip_meta_issues: bool = True,
    validate_targets_dir: Optional[str] = "benchmark/repos",
) -> BenchmarkDataset:
    """
    Build a benchmark dataset from raw extracted samples.

    Dedupliziert auf Issue-Ebene, vergibt deterministische
    sample_ids, filtert Meta-Issues und validiert Target-Existenz gegen
    den Repo-Checkout (tote Samples = garantierte Misses werden gedroppt).

    Args:
        raw_samples_path: Path, glob pattern, or list of paths to raw JSON files.
        output_path: Output path for the benchmark dataset.
        dataset_version: Version string.
        era_filter: If set, only include samples of this era ("pre_ai" or "post_ai").
        build_era_splits: If True, also write separate dataset files per era.
        skip_meta_issues: Meta-Issues (DOC:/TRACKER:/…) überspringen.
        validate_targets_dir: Verzeichnis mit Repo-Checkouts (benchmark/repos).
            Targets, deren Datei dort fehlt, werden entfernt; Samples ohne
            verbleibendes Target gedroppt. None/nicht vorhandener Checkout
            → Validierung wird für dieses Repo übersprungen (mit Warnung).

    Returns:
        BenchmarkDataset object.
    """
    # Resolve input files (support glob patterns and lists)
    if isinstance(raw_samples_path, list):
        input_files = raw_samples_path
    else:
        input_files = sorted(glob.glob(raw_samples_path))

    if not input_files:
        print(f"Keine Input-Dateien gefunden: {raw_samples_path}")
        return None

    raw_data = []
    for fp in input_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  {Path(fp).name}: {len(data)} Samples")
        raw_data.extend(data)

    print(f"\nBuilding dataset from {len(raw_data)} raw samples ({len(input_files)} files)...")

    samples = []
    skipped = 0
    skipped_dupes = 0
    skipped_meta = 0
    skipped_dead = 0
    dropped_targets = 0
    seen_issues = set()
    unvalidated_repos = set()
    file_indexes: Dict[str, "_RepoFileIndex"] = {}

    for raw in raw_data:
        # Dedup auf Issue-Ebene: alte raws enthalten pro
        # verlinktem PR ein Sample — erstes gewinnt (deterministisch,
        # da Input-Dateien sortiert eingelesen werden).
        issue_key = (raw["repo_name"], raw["issue_number"])
        if issue_key in seen_issues:
            skipped_dupes += 1
            continue
        seen_issues.add(issue_key)

        # Meta-Issues ohne einzelnes Code-Ziel filtern
        if skip_meta_issues and META_TITLE_PATTERN.match((raw["issue_title"] or "").strip()):
            skipped_meta += 1
            continue

        # Build ground truth targets
        targets = []
        for file_path in raw["changed_files"]:
            targets.append(GroundTruthTarget(
                file_path=file_path,
                function_name=None,  # Will be refined later via AST analysis
                span=None,
            ))

        n_targets_raw = len(targets)

        # Target-Existenz gegen HEAD-Checkout validieren:
        # der Benchmark indexiert HEAD — Targets, die dort nicht mehr
        # existieren, sind garantierte Misses und verzerren die Scores.
        # Suffix-Semantik wie metrics._file_matches (s. _RepoFileIndex).
        if validate_targets_dir:
            short = raw["repo_name"].split("/")[-1]
            repo_root = Path(validate_targets_dir) / short
            if repo_root.exists():
                if short not in file_indexes:
                    file_indexes[short] = _RepoFileIndex(repo_root)
                idx = file_indexes[short]
                existing = [t for t in targets if idx.target_alive(t.file_path)]
                dropped_targets += len(targets) - len(existing)
                targets = existing
            else:
                unvalidated_repos.add(short)

        if not targets:
            if n_targets_raw > 0:
                skipped_dead += 1  # alle Targets tot (Datei @HEAD weg)
            else:
                skipped += 1       # schon roh ohne Targets
            continue

        if len(targets) > 3:
            # Spec says 1–3 targets per query; take first 3
            targets = targets[:3]

        # Build query from issue title + body
        query = raw["issue_title"]
        body = raw.get("issue_body", "")
        if body:
            # Use first 500 chars of body as additional context
            query = f"{query}\n\n{body[:500]}"

        # Classify test case type
        tc_type = classify_test_case(query, raw["changed_files"])

        # Classify era (pre-AI vs post-AI)
        issue_created_at = raw.get("issue_created_at", "")
        pr_created_at = raw.get("pr_created_at", "")
        pr_merged_at = raw.get("pr_merged_at", "")
        era = classify_era(issue_created_at, pr_created_at)

        # Apply era filter if set
        if era_filter:
            if era.value != era_filter:
                skipped += 1
                continue

        sample = Sample(
            sample_id=deterministic_sample_id(raw["repo_name"], raw["issue_number"]),
            repo_id=raw["repo_name"],
            commit_hash=raw["commit_hash"],
            query=query,
            ground_truth=GroundTruth(targets=targets),
            metadata=SampleMetadata(
                test_case_type=tc_type,
                issue_number=raw["issue_number"],
                issue_url=raw.get("issue_url"),
                pr_number=raw["pr_number"],
                pr_url=raw.get("pr_url"),
                issue_created_at=issue_created_at or None,
                pr_created_at=pr_created_at or None,
                pr_merged_at=pr_merged_at or None,
                era=era,
            ),
        )

        samples.append(sample)

    dataset = BenchmarkDataset(
        version=dataset_version,
        description=f"FLBench Feature Location Benchmark v1 – {len(samples)} samples",
        samples=samples,
    )

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json(indent=2))

    # Print stats
    print(f"\nDataset Statistics:")
    print(f"   Total samples: {len(samples)}")
    print(f"   Skipped (no targets): {skipped}")
    print(f"   Skipped (Duplikate, gleiche Issue): {skipped_dupes}")
    print(f"   Skipped (Meta-Issues DOC:/TRACKER:/…): {skipped_meta}")
    print(f"   Skipped (alle Targets tot @HEAD): {skipped_dead}")
    print(f"   Entfernte Einzel-Targets (tot @HEAD): {dropped_targets}")
    if unvalidated_repos:
        print(f"    Nicht validiert (kein Checkout unter {validate_targets_dir}): "
              f"{', '.join(sorted(unvalidated_repos))}")

    by_repo = dataset.samples_by_repo
    print(f"\n   Per Repository:")
    for repo, repo_samples in by_repo.items():
        print(f"     {repo}: {len(repo_samples)} samples")

    by_tc = dataset.samples_by_tc
    print(f"\n   Per Test Case Type:")
    for tc, tc_samples in by_tc.items():
        print(f"     {tc}: {len(tc_samples)} samples")

    by_era = dataset.samples_by_era
    print(f"\n   Per Era (pre-AI vs post-AI):")
    for era_name, era_samples in by_era.items():
        print(f"     {era_name}: {len(era_samples)} samples")

    if era_filter:
        print(f"\n    Era-Filter aktiv: nur '{era_filter}' Samples")

    print(f"\nDataset gespeichert: {output_path}")

    # Optionally build separate datasets per era
    if build_era_splits:
        base_path = Path(output_path)
        for era_name in ("pre_ai", "post_ai"):
            era_dataset = dataset.filter_by_era(era_name)
            if era_dataset.samples:
                era_path = base_path.parent / f"{base_path.stem}_{era_name}{base_path.suffix}"
                with open(era_path, "w", encoding="utf-8") as f:
                    f.write(era_dataset.model_dump_json(indent=2))
                print(f"Era-Split '{era_name}': {len(era_dataset.samples)} Samples -> {era_path}")

    return dataset


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build benchmark dataset")
    parser.add_argument("--input", type=str, default="benchmark/data/raw_*.json",
                        help="Path or glob pattern to raw samples JSON (e.g. 'benchmark/data/raw_*.json')")
    parser.add_argument("--output", type=str, default="benchmark/data/benchmark_dataset.json",
                        help="Output dataset path")
    parser.add_argument("--version", type=str, default="1.0.0",
                        help="Dataset version")
    parser.add_argument("--era", type=str, default=None, choices=["pre_ai", "post_ai"],
                        help="Only include samples from this era")
    parser.add_argument("--era-splits", action="store_true",
                        help="Also generate separate dataset files per era (pre_ai, post_ai)")
    parser.add_argument("--include-meta", action="store_true",
                        help="Meta-Issues (DOC:/TRACKER:/…) NICHT filtern")
    parser.add_argument("--repos-dir", type=str, default="benchmark/repos",
                        help="Repo-Checkouts für Target-Validierung")
    parser.add_argument("--no-validate-targets", action="store_true",
                        help="Target-Existenz-Validierung überspringen")

    args = parser.parse_args()

    build_dataset(
        args.input, args.output, args.version,
        era_filter=args.era,
        build_era_splits=args.era_splits,
        skip_meta_issues=not args.include_meta,
        validate_targets_dir=None if args.no_validate_targets else args.repos_dir,
    )
