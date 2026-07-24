#!/usr/bin/env python3
"""
backfill_dates.py – Ergänze bestehende raw_*.json um Datums-Informationen

Liest bestehende Raw-Sample-Dateien und holt via GitHub API die
created_at / merged_at Timestamps nach. So können auch bereits extrahierte
Daten in pre_ai / post_ai aufgeteilt werden.

Usage:
    python -m benchmark.ground_truth.backfill_dates --input benchmark/data/raw_flask.json
    python -m benchmark.ground_truth.backfill_dates --input "benchmark/data/raw_*.json"
"""

import glob
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests as http_requests
except ImportError:
    http_requests = None

GITHUB_API = "https://api.github.com"


def backfill_dates(input_path: str, dry_run: bool = False):
    """Backfill created_at dates for existing raw sample files."""
    if http_requests is None:
        print("'requests' not installed")
        sys.exit(1)

    token = os.getenv("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    files = sorted(glob.glob(input_path))
    if not files:
        print(f"Keine Dateien gefunden: {input_path}")
        return

    for fp in files:
        print(f"\n{fp}")
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        updated = 0
        already_has = 0
        errors = 0

        for sample in data:
            # Skip if already has dates
            if sample.get("issue_created_at"):
                already_has += 1
                continue

            repo_name = sample["repo_name"]  # e.g. "pallets/flask"
            issue_number = sample["issue_number"]
            pr_number = sample["pr_number"]

            if dry_run:
                print(f"  [DRY] Would fetch: {repo_name}#I{issue_number} + PR#{pr_number}")
                continue

            # Fetch issue
            url = f"{GITHUB_API}/repos/{repo_name}/issues/{issue_number}"
            try:
                resp = http_requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200:
                    issue_data = resp.json()
                    sample["issue_created_at"] = issue_data.get("created_at", "")
                else:
                    print(f"   Issue #{issue_number}: HTTP {resp.status_code}")
                    errors += 1
                    continue
            except Exception as e:
                print(f"   Issue #{issue_number}: {e}")
                errors += 1
                continue

            # Fetch PR
            url = f"{GITHUB_API}/repos/{repo_name}/pulls/{pr_number}"
            try:
                resp = http_requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200:
                    pr_data = resp.json()
                    sample["pr_created_at"] = pr_data.get("created_at", "")
                    sample["pr_merged_at"] = pr_data.get("merged_at", "")
                else:
                    print(f"   PR #{pr_number}: HTTP {resp.status_code}")
            except Exception as e:
                print(f"   PR #{pr_number}: {e}")

            updated += 1

            # Rate limit handling
            remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
            if remaining < 10:
                reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                wait = max(0, reset_time - int(time.time())) + 1
                print(f"  ⏳ Rate limit niedrig ({remaining}), warte {wait}s...")
                time.sleep(min(wait, 60))

        print(f"  {updated} aktualisiert, {already_has} hatten schon Datum, {errors} Fehler")

        if not dry_run and updated > 0:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  Gespeichert: {fp}")

        # Print era distribution
        from benchmark.ground_truth.dataset_builder import classify_era
        pre = sum(1 for s in data if classify_era(s.get("issue_created_at", "")).value == "pre_ai")
        post = sum(1 for s in data if classify_era(s.get("issue_created_at", "")).value == "post_ai")
        unknown = sum(1 for s in data if classify_era(s.get("issue_created_at", "")).value == "unknown")
        print(f"  pre_ai={pre} | post_ai={post} | unknown={unknown}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill dates in raw sample files")
    parser.add_argument("--input", type=str, default="benchmark/data/raw_*.json",
                        help="Path or glob pattern to raw JSON files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would be fetched")

    args = parser.parse_args()
    backfill_dates(args.input, dry_run=args.dry_run)
