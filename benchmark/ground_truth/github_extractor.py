#!/usr/bin/env python3
"""
github_extractor.py – Extract Ground Truth from GitHub Issues + PRs

Connects to the GitHub REST API to:
1. Fetch closed issues with linked PRs (via "Fixes #N" / "Closes #N")
2. Extract changed .py files from PR diffs
3. Build (query → ground_truth) pairs for the benchmark dataset

Requires: GITHUB_TOKEN environment variable for higher rate limits.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    import requests as http_requests
except ImportError:
    http_requests = None


@dataclass
class ExtractedSample:
    """A raw sample extracted from GitHub."""
    repo_name: str
    issue_number: int
    issue_title: str
    issue_body: str
    pr_number: int
    pr_title: str
    commit_hash: str
    changed_files: List[str]  # .py files changed in the PR
    issue_url: str
    pr_url: str
    issue_created_at: str = ""   # ISO-8601 timestamp
    pr_created_at: str = ""      # ISO-8601 timestamp
    pr_merged_at: str = ""       # ISO-8601 timestamp


GITHUB_API = "https://api.github.com"
GITHUB_GRAPHQL = "https://api.github.com/graphql"

# GraphQL: Issues mit verlinkten Fix-PRs in EINEM Query — nutzt GitHubs
# autoritative Verknüpfung (closedByPullRequestsReferences = PRs, die das
# Issue tatsächlich geschlossen haben; Development-Section/auto-close).
# Deutlich höhere Ausbeute + Präzision als Text-Pattern-Matching:
# der REST-Pfad fand bei requests 7/400, der Pool per linked:pr ist 196.
_GRAPHQL_SEARCH = """
query($q: String!, $first: Int!, $after: String) {
  search(query: $q, type: ISSUE, first: $first, after: $after) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on Issue {
        number title body createdAt url
        labels(first: 10) { nodes { name } }
        closedByPullRequestsReferences(first: 3, includeClosedPrs: false) {
          nodes {
            number title url createdAt mergedAt
            mergeCommit { oid }
            files(first: 100) { nodes { path } }
          }
        }
      }
    }
  }
}
"""

# Patterns that link issues to PRs in commit messages / PR bodies
FIX_PATTERNS = [
    re.compile(r"(?:fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)\s+#(\d+)", re.IGNORECASE),
]

# Maintenance-/Meta-Issue-Titel ohne einzelnes Code-Ziel.
# Diese Klasse verzerrt die Auswertung (siehe Precision-Analyse §5.3 und
# V12b→V16c-Stage-1-Regressionen: 7/8 waren pandas-Maintenance-Issues).
META_TITLE_PATTERN = re.compile(
    r"^(DOC|TRACKER|STY|BUILD|CI|TST|WEB|DEPS|RLS|CLN|META|Release Plan)\b[:\s]",
    re.IGNORECASE,
)


def _pr_explicitly_fixes(pr: Dict, issue_number: int) -> bool:
    """True, wenn PR-Body/Titel das Issue EXPLIZIT per fixes/closes referenziert.

    Verhindert False-Positive-Ground-Truth durch bloße Erwähnungen
    („related to #N", Cross-Referenzen aus Diskussionen).
    """
    text = (pr.get("body") or "") + " " + (pr.get("title") or "")
    for pattern in FIX_PATTERNS:
        if str(issue_number) in pattern.findall(text):
            return True
    return False


class GitHubExtractor:
    """Extracts benchmark ground truth from GitHub repositories."""

    def __init__(self, token: Optional[str] = None):
        if http_requests is None:
            raise ImportError("requests not installed")

        self.token = token or os.getenv("GITHUB_TOKEN", "")
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
        self.rate_limit_remaining = 5000

    def _get(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make a GitHub API GET request with rate limit handling."""
        try:
            resp = http_requests.get(url, headers=self.headers, params=params, timeout=30)

            # Track rate limit
            self.rate_limit_remaining = int(resp.headers.get("X-RateLimit-Remaining", 0))
            if self.rate_limit_remaining < 10:
                reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                wait = max(0, reset_time - int(time.time())) + 1
                print(f"  ⏳ Rate limit niedrig ({self.rate_limit_remaining}), warte {wait}s...")
                time.sleep(min(wait, 60))

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 403:
                print(f"   Rate limit erreicht. Token setzen: export GITHUB_TOKEN=...")
                return None
            else:
                return None
        except Exception as e:
            print(f"   API-Fehler: {e}")
            return None

    def extract_samples(
        self,
        owner: str,
        repo: str,
        max_issues: int = 100,
        min_body_words: int = 30,
        source_dirs: List[str] = None,
        before_date: Optional[str] = None,
        after_date: Optional[str] = None,
        exclude_labels: Tuple[str, ...] = ("invalid", "duplicate", "wontfix"),
        exclude_meta_titles: bool = True,
    ) -> List[ExtractedSample]:
        """
        Extract benchmark samples from a GitHub repository.

        Ein Sample pro Issue (Dedup), nur PRs mit explizitem
        fixes/closes-Bezug, Meta-Issues (DOC:/TRACKER:/STY:/…) gefiltert.

        Args:
            owner: Repository owner (e.g., "pallets")
            repo: Repository name (e.g., "flask")
            max_issues: Maximum number of issues to process
            min_body_words: Minimum words in issue body (spec: >30)
            source_dirs: Only include files in these directories
            before_date: Only include issues created before this date (ISO-8601, e.g. "2022-01-01")
            after_date: Only include issues created after this date (ISO-8601, e.g. "2022-01-01")
            exclude_labels: Issues mit diesen Labels überspringen. Bewusst OHNE
                "bug" — Bug-Lokalisierung ist ein Kernanwendungsfall, und der
                alte bug-Filter war ohnehin inkonsistent wirksam.
            exclude_meta_titles: Maintenance-/Meta-Issues (DOC:/TRACKER:/STY:/
                BUILD:/…) überspringen — sie haben kein einzelnes Code-Ziel
                und verzerren die Auswertung (siehe Precision-Analyse §5.3).

        Returns:
            List of ExtractedSample objects (max. eines pro Issue).
        """
        print(f"\nExtrahiere Ground Truth: {owner}/{repo}")
        print(f"   Token: {'gesetzt' if self.token else 'nicht gesetzt (60 req/h)'}")
        if before_date:
            print(f"   Filter: nur Issues erstellt vor {before_date}")
        if after_date:
            print(f"   Filter: nur Issues erstellt nach {after_date}")

        # Step 1: Get closed issues
        issues = self._get_closed_issues(owner, repo, max_issues)
        print(f"   {len(issues)} geschlossene Issues gefunden")

        samples = []
        seen_issues = set()
        skipped_by_date = 0
        skipped_meta = 0
        for issue in issues:
            # Filter: body must have enough content
            body = issue.get("body", "") or ""
            if len(body.split()) < min_body_words:
                continue

            # Filter: excluded labels (Default bewusst ohne "bug", s. Docstring)
            labels = [l.get("name", "").lower() for l in issue.get("labels", [])]
            if any(l in exclude_labels for l in labels):
                continue

            # Filter: Meta-/Maintenance-Issues ohne einzelnes Code-Ziel
            title = issue.get("title", "") or ""
            if exclude_meta_titles and META_TITLE_PATTERN.match(title.strip()):
                skipped_meta += 1
                continue

            # Filter: date range
            issue_created = issue.get("created_at", "")
            if before_date and issue_created and issue_created >= before_date:
                skipped_by_date += 1
                continue
            if after_date and issue_created and issue_created < after_date:
                skipped_by_date += 1
                continue

            issue_number = issue["number"]

            # Dedup: genau EIN Sample pro Issue
            if issue_number in seen_issues:
                continue

            # Step 2: Find linked fix-PRs (nur explizite fixes/closes-Referenz)
            linked_prs = self._find_linked_prs(owner, repo, issue_number)
            # Deterministisch: frühester gemergter Fix-PR zuerst
            linked_prs.sort(key=lambda p: p.get("merged_at") or "")

            for pr_info in linked_prs:
                pr_number = pr_info["number"]

                # Step 3: Get changed files from PR
                changed_files = self._get_pr_files(owner, repo, pr_number)

                # Filter: only .py files in source directories
                py_files = self._filter_files(changed_files, source_dirs)

                if not py_files:
                    continue

                # Get merge commit
                commit_hash = pr_info.get("merge_commit_sha", "")
                if not commit_hash:
                    continue

                samples.append(ExtractedSample(
                    repo_name=f"{owner}/{repo}",
                    issue_number=issue_number,
                    issue_title=title,
                    issue_body=body,
                    pr_number=pr_number,
                    pr_title=pr_info.get("title", ""),
                    commit_hash=commit_hash,
                    changed_files=py_files,
                    issue_url=issue.get("html_url", ""),
                    pr_url=pr_info.get("html_url", ""),
                    issue_created_at=issue_created,
                    pr_created_at=pr_info.get("created_at", ""),
                    pr_merged_at=pr_info.get("merged_at", ""),
                ))
                seen_issues.add(issue_number)
                break  # ein Sample pro Issue — erster verwertbarer Fix-PR

        if skipped_by_date:
            print(f"   {skipped_by_date} Issues übersprungen (Datumsfilter)")
        if skipped_meta:
            print(f"   {skipped_meta} Meta-Issues übersprungen (DOC:/TRACKER:/…)")
        print(f"   {len(samples)} Samples extrahiert ({len(seen_issues)} eindeutige Issues)")
        return samples

    def _graphql(self, query: str, variables: Dict) -> Optional[Dict]:
        """GraphQL-Request (braucht Token)."""
        try:
            resp = http_requests.post(
                GITHUB_GRAPHQL,
                json={"query": query, "variables": variables},
                headers=self.headers,
                timeout=60,
            )
            if resp.status_code != 200:
                print(f"   GraphQL HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            payload = resp.json()
            if payload.get("errors"):
                print(f"   GraphQL errors: {payload['errors'][:1]}")
                return None
            return payload.get("data")
        except Exception as e:
            print(f"   GraphQL-Fehler: {e}")
            return None

    def extract_samples_graphql(
        self,
        owner: str,
        repo: str,
        max_issues: int = 600,
        min_body_words: int = 30,
        source_dirs: List[str] = None,
        before_date: Optional[str] = None,
        after_date: Optional[str] = None,
        exclude_labels: Tuple[str, ...] = ("invalid", "duplicate", "wontfix"),
        exclude_meta_titles: bool = True,
    ) -> List[ExtractedSample]:
        """GraphQL-Extraktion über `linked:pr`-Suche + closedByPullRequestsReferences.

        Ein Sample pro Issue; als Fix-PR gilt der früheste gemergte PR aus
        GitHubs autoritativer Verknüpfung, der qualifizierende .py-Dateien
        ändert. Search-API liefert max. 1000 Issues pro Query (neueste zuerst).
        """
        q = f"repo:{owner}/{repo} is:issue is:closed linked:pr sort:updated-desc"
        print(f"\nExtrahiere Ground Truth (GraphQL): {owner}/{repo}")
        print(f"   Query: {q}")

        samples = []
        scanned = 0
        skipped_meta = 0
        skipped_by_date = 0
        after = None
        page_size = 50

        while scanned < max_issues:
            data = self._graphql(_GRAPHQL_SEARCH, {
                "q": q, "first": min(page_size, max_issues - scanned), "after": after,
            })
            if not data:
                break
            search = data["search"]
            if scanned == 0:
                print(f"   Pool: {search['issueCount']} Issues mit verlinktem PR")

            for issue in search["nodes"]:
                if not issue:  # non-Issue Knoten
                    continue
                scanned += 1

                body = issue.get("body") or ""
                if len(body.split()) < min_body_words:
                    continue

                labels = [l["name"].lower() for l in issue["labels"]["nodes"]]
                if any(l in exclude_labels for l in labels):
                    continue

                title = issue.get("title") or ""
                if exclude_meta_titles and META_TITLE_PATTERN.match(title.strip()):
                    skipped_meta += 1
                    continue

                created = issue.get("createdAt", "")
                if before_date and created and created >= before_date:
                    skipped_by_date += 1
                    continue
                if after_date and created and created < after_date:
                    skipped_by_date += 1
                    continue

                prs = [p for p in issue["closedByPullRequestsReferences"]["nodes"]
                       if p and p.get("mergedAt")]
                prs.sort(key=lambda p: p.get("mergedAt") or "")

                for pr in prs:
                    files = [f["path"] for f in pr["files"]["nodes"]]
                    py_files = self._filter_files(files, source_dirs)
                    commit = (pr.get("mergeCommit") or {}).get("oid", "")
                    if not py_files or not commit:
                        continue
                    samples.append(ExtractedSample(
                        repo_name=f"{owner}/{repo}",
                        issue_number=issue["number"],
                        issue_title=title,
                        issue_body=body,
                        pr_number=pr["number"],
                        pr_title=pr.get("title", ""),
                        commit_hash=commit,
                        changed_files=py_files,
                        issue_url=issue.get("url", ""),
                        pr_url=pr.get("url", ""),
                        issue_created_at=created,
                        pr_created_at=pr.get("createdAt", ""),
                        pr_merged_at=pr.get("mergedAt", ""),
                    ))
                    break  # ein Sample pro Issue

            if not search["pageInfo"]["hasNextPage"]:
                break
            after = search["pageInfo"]["endCursor"]

        if skipped_by_date:
            print(f"   {skipped_by_date} Issues übersprungen (Datumsfilter)")
        if skipped_meta:
            print(f"   {skipped_meta} Meta-Issues übersprungen")
        print(f"   {len(samples)} Samples aus {scanned} gescannten Issues")
        return samples

    def _get_closed_issues(self, owner: str, repo: str, max_issues: int) -> List[Dict]:
        """Fetch closed issues (paginated)."""
        issues = []
        page = 1
        per_page = 100

        while len(issues) < max_issues:
            url = f"{GITHUB_API}/repos/{owner}/{repo}/issues"
            params = {
                "state": "closed",
                "per_page": per_page,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            }
            data = self._get(url, params)
            if not data:
                break

            # Filter out pull requests (GitHub API returns PRs as issues too)
            real_issues = [i for i in data if "pull_request" not in i]
            issues.extend(real_issues)

            if len(data) < per_page:
                break
            page += 1

        return issues[:max_issues]

    def _find_linked_prs(self, owner: str, repo: str, issue_number: int) -> List[Dict]:
        """Find PRs that fix/close a given issue."""
        linked = []

        # Method 1: Search for PRs referencing the issue
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls"
        params = {"state": "closed", "per_page": 30, "sort": "updated", "direction": "desc"}
        prs = self._get(url, params)

        if prs:
            for pr in prs:
                if not pr.get("merged_at"):
                    continue
                if _pr_explicitly_fixes(pr, issue_number):
                    linked.append(pr)

        # Method 2: Timeline events (findet auch ältere PRs, kostet mehr Calls).
        # Cross-Referenzen allein reichen NICHT mehr — der PR muss
        # das Issue explizit per fixes/closes referenzieren, sonst entstehen
        # False-Positive-Targets aus bloßen Erwähnungen.
        if not linked:
            url = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}/timeline"
            headers = {**self.headers, "Accept": "application/vnd.github.mockingbird-preview+json"}
            try:
                resp = http_requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200:
                    events = resp.json()
                    for event in events:
                        if event.get("event") == "cross-referenced":
                            source = event.get("source", {}).get("issue", {})
                            if source.get("pull_request") and source.get("state") == "closed":
                                # Get full PR data
                                pr_url = source["pull_request"].get("url", "")
                                if pr_url:
                                    pr_data = self._get(pr_url)
                                    if (pr_data and pr_data.get("merged_at")
                                            and _pr_explicitly_fixes(pr_data, issue_number)):
                                        linked.append(pr_data)
            except Exception:
                pass

        return linked

    def _get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[str]:
        """Get list of files changed in a PR."""
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/files"
        params = {"per_page": 100}
        data = self._get(url, params)

        if not data:
            return []

        return [f["filename"] for f in data if f.get("filename")]

    def _filter_files(
        self, files: List[str], source_dirs: Optional[List[str]] = None
    ) -> List[str]:
        """Filter for .py files in source directories (no tests)."""
        filtered = []
        for f in files:
            # Only .py files
            if not f.endswith(".py"):
                continue

            # Exclude test files
            parts = f.split("/")
            if any(p in ("tests", "test", "testing") for p in parts):
                continue
            if any(p.startswith("test_") or p.endswith("_test.py") for p in parts):
                continue

            # Exclude common non-source dirs
            if any(p in ("docs", "examples", "benchmarks", "scripts", "tools") for p in parts):
                continue

            # If source_dirs specified, check prefix
            if source_dirs:
                in_source = any(f.startswith(d) for d in source_dirs)
                if not in_source:
                    # Also try without the first directory component
                    in_source = any(part == d.split("/")[-1] for part in parts for d in source_dirs)
                if not in_source:
                    continue

            filtered.append(f)

        return filtered


def save_samples(samples: List[ExtractedSample], output_path: str):
    """Save extracted samples to JSON."""
    data = [
        {
            "repo_name": s.repo_name,
            "issue_number": s.issue_number,
            "issue_title": s.issue_title,
            "issue_body": s.issue_body,
            "pr_number": s.pr_number,
            "pr_title": s.pr_title,
            "commit_hash": s.commit_hash,
            "changed_files": s.changed_files,
            "issue_url": s.issue_url,
            "pr_url": s.pr_url,
            "issue_created_at": s.issue_created_at,
            "pr_created_at": s.pr_created_at,
            "pr_merged_at": s.pr_merged_at,
        }
        for s in samples
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n{len(data)} Samples gespeichert: {output_path}")


# ──────────────────────────────────────────────────────────────
# CLI interface
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract ground truth from GitHub")
    parser.add_argument("--repo", type=str, default="pallets/flask",
                        help="owner/repo (default: pallets/flask)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max issues to process")
    parser.add_argument("--min-words", type=int, default=30,
                        help="Minimum words in issue body")
    parser.add_argument("--output", type=str, default="benchmark/data/raw_samples.json",
                        help="Output JSON path")
    parser.add_argument("--before", type=str, default=None,
                        help="Only issues created before this date (ISO-8601, e.g. 2022-01-01)")
    parser.add_argument("--after", type=str, default=None,
                        help="Only issues created after this date (ISO-8601, e.g. 2022-01-01)")
    parser.add_argument("--include-meta", action="store_true",
                        help="Meta-Issues (DOC:/TRACKER:/STY:/…) NICHT filtern")
    parser.add_argument("--source-dirs", nargs="*", default=None,
                        help="Nur Dateien unter diesen Verzeichnissen (z.B. src/flask)")
    parser.add_argument("--rest", action="store_true",
                        help="REST-Pfad erzwingen (Default: GraphQL, wenn Token gesetzt)")

    args = parser.parse_args()
    owner, repo = args.repo.split("/")

    extractor = GitHubExtractor()
    common = dict(
        owner=owner,
        repo=repo,
        max_issues=args.limit,
        min_body_words=args.min_words,
        source_dirs=args.source_dirs,
        before_date=args.before,
        after_date=args.after,
        exclude_meta_titles=not args.include_meta,
    )
    if extractor.token and not args.rest:
        samples = extractor.extract_samples_graphql(**common)
    else:
        if not extractor.token:
            print(" Kein Token → REST-Fallback (deutlich geringere Ausbeute)")
        samples = extractor.extract_samples(**common)

    if samples:
        save_samples(samples, args.output)
    else:
        print("Keine Samples gefunden.")
