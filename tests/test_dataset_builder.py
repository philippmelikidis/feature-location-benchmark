#!/usr/bin/env python3
"""
test_dataset_builder.py – Guards für die GT-Pipeline-Härtung.

Sichert die vier Fixes ab:
  1. Issue-Level-Dedup (v1 hatte 19 Duplikat-Samples, click #1476 4×)
  2. Deterministische sample_ids (Rebuild-stabile Joins)
  3. Meta-Issue-Filter (DOC:/TRACKER:/STY:/… — 69 Samples in v1)
  4. Target-Existenz-Validierung gegen HEAD-Checkout (16 tote Samples in v1,
     requests 40%)
Plus: expliziter fixes/closes-Check des Extractors (False-Positive-Links).
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.ground_truth.dataset_builder import (  # noqa: E402
    build_dataset, deterministic_sample_id, META_TITLE_PATTERN,
)
from benchmark.ground_truth.github_extractor import _pr_explicitly_fixes  # noqa: E402


def _raw(issue_number, title="Fix the frobnicator when x is None and more words "
         "to pass any future body checks in the builder pipeline",
         files=None, repo="acme/demo", pr_number=None):
    return {
        "repo_name": repo,
        "issue_number": issue_number,
        "issue_title": title,
        "issue_body": "word " * 40,
        "pr_number": pr_number or issue_number * 10,
        "pr_title": "fix",
        "commit_hash": "abc123",
        "changed_files": files or ["src/demo/core.py"],
        "issue_url": "",
        "pr_url": "",
        "issue_created_at": "2023-05-01T00:00:00Z",
        "pr_created_at": "2023-05-02T00:00:00Z",
        "pr_merged_at": "2023-05-03T00:00:00Z",
    }


def _build(raws, tmp, **kwargs):
    raw_path = Path(tmp) / "raw_demo.json"
    raw_path.write_text(json.dumps(raws))
    out_path = Path(tmp) / "dataset.json"
    ds = build_dataset(str(raw_path), str(out_path), "test", **kwargs)
    return ds


def test_dedup_same_issue_multiple_prs():
    with tempfile.TemporaryDirectory() as tmp:
        raws = [
            _raw(1476, pr_number=100),
            _raw(1476, pr_number=101),   # zweiter verlinkter PR → Duplikat
            _raw(1476, pr_number=102),
            _raw(2939, pr_number=200),
        ]
        ds = _build(raws, tmp, validate_targets_dir=None)
        issues = [s.metadata.issue_number for s in ds.samples]
        assert sorted(issues) == [1476, 2939], f"Dedup fehlgeschlagen: {issues}"


def test_sample_ids_deterministic_and_stable():
    a = deterministic_sample_id("acme/demo", 1476)
    b = deterministic_sample_id("acme/demo", 1476)
    c = deterministic_sample_id("acme/demo", 1477)
    assert a == b and a != c and len(a) == 8
    with tempfile.TemporaryDirectory() as tmp:
        ds1 = _build([_raw(1)], tmp, validate_targets_dir=None)
    with tempfile.TemporaryDirectory() as tmp:
        ds2 = _build([_raw(1)], tmp, validate_targets_dir=None)
    assert ds1.samples[0].sample_id == ds2.samples[0].sample_id, (
        "sample_id nicht rebuild-stabil"
    )


def test_meta_issues_filtered():
    assert META_TITLE_PATTERN.match("DOC: Fix docstring validation")
    assert META_TITLE_PATTERN.match("TRACKER: add support for Python 3.14")
    assert META_TITLE_PATTERN.match("STY: Enforce Ruff rule B905")
    assert not META_TITLE_PATTERN.match("BUG: DataFrame.groupby broken")
    assert not META_TITLE_PATTERN.match("Documented behaviour is wrong")  # kein Prefix

    with tempfile.TemporaryDirectory() as tmp:
        raws = [
            _raw(1, title="DOC: Fix docstring validation errors for groupby"),
            _raw(2, title="BUG: groupby drops NaN keys unexpectedly in 2.2"),
        ]
        ds = _build(raws, tmp, validate_targets_dir=None)
        assert [s.metadata.issue_number for s in ds.samples] == [2]
        ds_all = _build(raws, tmp, validate_targets_dir=None, skip_meta_issues=False)
        assert len(ds_all.samples) == 2, "--include-meta muss Filter abschalten"


def test_dead_target_validation():
    with tempfile.TemporaryDirectory() as tmp:
        # Fake-Checkout: nur core.py existiert
        repo_root = Path(tmp) / "repos" / "demo" / "src" / "demo"
        repo_root.mkdir(parents=True)
        (repo_root / "core.py").write_text("def frobnicate(): pass\n")

        raws = [
            _raw(1, files=["src/demo/core.py"]),                       # lebt
            _raw(2, files=["src/demo/removed.py"]),                    # tot
            _raw(3, files=["src/demo/removed.py", "src/demo/core.py"]),  # teil-tot
            # Suffix-Fall (Repo-Umzug auf src/-Layout, wie psf/requests):
            # GT-Pfad ohne src/-Prefix MUSS als lebendig gelten, weil
            # metrics._file_matches per Suffix matcht.
            _raw(4, files=["demo/core.py"]),
        ]
        ds = _build(raws, tmp, validate_targets_dir=str(Path(tmp) / "repos"))
        by_issue = {s.metadata.issue_number: s for s in ds.samples}
        assert set(by_issue) == {1, 3, 4}, "totes Sample (2) muss gedroppt sein, Suffix-Sample (4) leben"
        assert [t.file_path for t in by_issue[3].ground_truth.targets] == \
            ["src/demo/core.py"], "totes Einzel-Target muss entfernt sein"

        # Ohne Checkout: keine Validierung, nichts gedroppt (mit Warnung)
        ds_all = _build(raws, tmp, validate_targets_dir=str(Path(tmp) / "nope"))
        assert len(ds_all.samples) == 4, "ohne Checkout darf nichts gedroppt werden"


def test_pr_explicit_fix_reference():
    fixes = {"body": "This PR fixes #123 properly.", "title": "Repair frobnicator"}
    mention = {"body": "Related to #123, see discussion.", "title": "Refactor"}
    title_fix = {"body": "", "title": "Closes #123: repair frobnicator"}
    assert _pr_explicitly_fixes(fixes, 123)
    assert not _pr_explicitly_fixes(mention, 123), (
        "bloße Erwähnung darf NICHT als Fix-Link zählen (False-Positive-GT)"
    )
    assert _pr_explicitly_fixes(title_fix, 123)
    assert not _pr_explicitly_fixes(fixes, 124), "falsche Issue-Nr darf nicht matchen"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  {name}")
            except AssertionError as e:
                fails += 1
                print(f"  {name}: {e}")
    sys.exit(1 if fails else 0)
