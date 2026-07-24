#!/usr/bin/env python3
"""
aggregate.py – Merge per-repo benchmark result JSONs into a single report.

Used by the matrix `benchmark_all.yml` workflow: each matrix leg uploads
its own `benchmark_results_<ts>.json` artifact, and this module pulls
them back together to produce a unified Macro/Micro/TC report across
all repos.

Usage:
    python -m benchmark.aggregate --inputs artifacts/ --output-dir out/

It walks `inputs` recursively, picks up any `benchmark_results_*.json`,
concatenates their `runs` arrays, and feeds the merged report through
the existing `reporting.generate_report`.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.schemas import BenchmarkReport, RunResult
from benchmark.reporting import generate_report


def find_result_jsons(inputs_dir: Path) -> List[Path]:
    """Return all benchmark_results_*.json files under inputs_dir.

    Excludes _PARTIAL files unless no full results exist for that leg —
    we keep at least one file per artifact directory so a wholly-failed
    repo still appears in the merged report.
    """
    files: List[Path] = []
    for sub in sorted(inputs_dir.iterdir()):
        if not sub.is_dir():
            continue
        # Per-leg artifact dir: prefer non-PARTIAL files, fall back to PARTIAL.
        full = sorted(sub.rglob("benchmark_results_2*.json"))
        partial = sorted(sub.rglob("benchmark_results_*_PARTIAL.json"))
        non_partial = [f for f in full if "_PARTIAL" not in f.name]
        if non_partial:
            files.append(non_partial[-1])  # newest full
        elif partial:
            files.append(partial[-1])
            print(f"  [PARTIAL] using partial result from {sub.name}")
    return files


def load_runs(files: List[Path]) -> List[RunResult]:
    runs: List[RunResult] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for run_data in data.get("runs", []):
            try:
                runs.append(RunResult(**run_data))
            except Exception as e:
                print(f"  WARN: skipping malformed run in {f.name}: {e}")
    return runs


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-repo benchmark results")
    parser.add_argument("--inputs", required=True,
                        help="Directory containing per-repo artifact subdirs")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write the merged report + JSON")
    args = parser.parse_args()

    inputs_dir = Path(args.inputs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not inputs_dir.exists():
        print(f"::warning::inputs dir {inputs_dir} does not exist — nothing to aggregate")
        # Write a stub so the upload-artifact step has something to grab.
        (output_dir / "EMPTY.txt").write_text(
            f"No artifacts found at {inputs_dir} on {datetime.now().isoformat()}\n"
        )
        return

    print(f"\n[AGGREGATE] scanning {inputs_dir}...")
    files = find_result_jsons(inputs_dir)
    if not files:
        print("::warning::no result JSONs found in artifacts")
        (output_dir / "EMPTY.txt").write_text(
            f"No benchmark_results_*.json found under {inputs_dir}\n"
        )
        return

    print(f"  found {len(files)} result file(s):")
    for f in files:
        print(f"    {f}")

    runs = load_runs(files)
    print(f"\n  loaded {len(runs)} runs across {len({r.repo_id for r in runs})} repos")

    # Pick the most common dataset_version (should all be the same — but
    # if a leg accidentally ran with a different version, surface it).
    versions = {r.dataset_version for r in runs}
    if len(versions) > 1:
        print(f"  WARN: mixed dataset versions: {versions} — using {sorted(versions)[-1]}")
    dataset_version = sorted(versions)[-1] if versions else "unknown"

    report = BenchmarkReport(dataset_version=dataset_version, runs=runs)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    json_out = output_dir / f"benchmark_results_aggregated_{ts}.json"
    md_out = output_dir / f"benchmark_report_aggregated_{ts}.md"
    latest_json = output_dir / "benchmark_results_aggregated_latest.json"
    latest_md = output_dir / "benchmark_report_aggregated_latest.md"

    with open(json_out, "w", encoding="utf-8") as fh:
        fh.write(report.model_dump_json(indent=2))
    with open(latest_json, "w", encoding="utf-8") as fh:
        fh.write(report.model_dump_json(indent=2))
    print(f"\n[SAVE] {json_out}")

    generate_report(report, str(md_out))
    generate_report(report, str(latest_md))
    print(f"[SAVE] {md_out}")


if __name__ == "__main__":
    main()
