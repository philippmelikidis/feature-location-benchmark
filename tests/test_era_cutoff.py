#!/usr/bin/env python3
"""
test_era_cutoff.py – Guards für die dynamische Era-Reklassifikation.

Das Research-Team nutzt als primären Cutoff 2022-11-30 (ChatGPT-Launch);
die eingebauten Dataset-Labels basieren auf AI_ERA_CUTOFF=2022-01-01.
filter_by_era(era, cutoff=…) muss dynamisch aus issue_created_at
reklassifizieren, damit beide Cutoffs ohne Dataset-Rebuild laufen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.schemas import (  # noqa: E402
    BenchmarkDataset, Sample, GroundTruth, GroundTruthTarget,
    SampleMetadata, TestCaseType, CodeEra,
)


def _sample(sid, created_at, era=CodeEra.POST_AI, merged_at=None):
    return Sample(
        sample_id=sid,
        repo_id="acme/demo",
        commit_hash="abc",
        query=f"Issue {sid}\n\nsome body",
        ground_truth=GroundTruth(targets=[GroundTruthTarget(file_path="a.py")]),
        metadata=SampleMetadata(
            test_case_type=TestCaseType.TC1,
            issue_number=1,
            issue_created_at=created_at,
            pr_merged_at=merged_at,
            era=era,
        ),
    )


def _ds():
    # Drei Zonen relativ zu den beiden Cutoffs:
    #   2021-06: pre bei beiden | 2022-06: post@2022-01-01, pre@2022-11-30
    #   2023-06: post bei beiden | ohne Datum: unknown bei Cutoff-Modus
    return BenchmarkDataset(version="t", description="t", samples=[
        _sample("old", "2021-06-01T00:00:00Z", era=CodeEra.PRE_AI),
        _sample("mid", "2022-06-01T00:00:00Z", era=CodeEra.POST_AI),
        _sample("new", "2023-06-01T00:00:00Z", era=CodeEra.POST_AI),
        _sample("nodate", "", era=CodeEra.POST_AI),
    ])


def test_default_uses_stored_labels():
    ds = _ds()
    assert {s.sample_id for s in ds.filter_by_era("pre_ai").samples} == {"old"}
    assert {s.sample_id for s in ds.filter_by_era("post_ai").samples} == {"mid", "new", "nodate"}


def test_cutoff_2022_11_30_reclassifies_transition_year():
    ds = _ds()
    pre = {s.sample_id for s in ds.filter_by_era("pre_ai", cutoff="2022-11-30").samples}
    post = {s.sample_id for s in ds.filter_by_era("post_ai", cutoff="2022-11-30").samples}
    # "mid" (Juni 2022) wechselt mit dem ChatGPT-Cutoff in die Pre-Ära —
    # das eingebaute post_ai-Label wird korrekt ÜBERSTIMMT.
    assert pre == {"old", "mid"}
    assert post == {"new"}


def test_cutoff_2022_01_01_matches_stored_labels():
    # Sensitivitäts-Cutoff = eingebauter Cutoff → gleiche Aufteilung
    # (für Samples MIT Datum; nodate fällt im Cutoff-Modus raus)
    ds = _ds()
    pre = {s.sample_id for s in ds.filter_by_era("pre_ai", cutoff="2022-01-01").samples}
    post = {s.sample_id for s in ds.filter_by_era("post_ai", cutoff="2022-01-01").samples}
    assert pre == {"old"}
    assert post == {"mid", "new"}


def test_date_field_pr_merged_at():
    """Team-Einwand: Für die KI-Code-Hypothese zählt der Fix-Zeitpunkt.
    Kernszenario: Issue VOR der KI-Ära erstellt, Fix DANACH gemergt —
    nach issue_created_at pre, nach pr_merged_at post."""
    ds = BenchmarkDataset(version="t", description="t", samples=[
        # Issue 2021, Fix 2023 → das Streit-Szenario
        _sample("old_issue_new_fix", "2021-06-01T00:00:00Z",
                merged_at="2023-03-01T00:00:00Z", era=CodeEra.PRE_AI),
        # Issue und Fix beide pre
        _sample("all_pre", "2021-01-01T00:00:00Z",
                merged_at="2021-02-01T00:00:00Z", era=CodeEra.PRE_AI),
        # Fix-Datum fehlt → im pr_merged_at-Modus unknown
        _sample("no_merge_date", "2021-01-01T00:00:00Z", era=CodeEra.PRE_AI),
    ])
    cutoff = "2022-11-30"

    by_issue_pre = {s.sample_id for s in ds.filter_by_era(
        "pre_ai", cutoff=cutoff).samples}
    assert by_issue_pre == {"old_issue_new_fix", "all_pre", "no_merge_date"}

    by_merge_pre = {s.sample_id for s in ds.filter_by_era(
        "pre_ai", cutoff=cutoff, date_field="pr_merged_at").samples}
    by_merge_post = {s.sample_id for s in ds.filter_by_era(
        "post_ai", cutoff=cutoff, date_field="pr_merged_at").samples}
    assert by_merge_pre == {"all_pre"}
    assert by_merge_post == {"old_issue_new_fix"}, (
        "Pre-AI-Issue mit Post-AI-Fix muss nach pr_merged_at post_ai sein"
    )
    assert "no_merge_date" not in by_merge_pre | by_merge_post


def test_date_field_validation():
    ds = _ds()
    try:
        ds.filter_by_era("pre_ai", cutoff="2022-11-30", date_field="banana")
        raise AssertionError("unbekanntes date_field muss ValueError werfen")
    except ValueError:
        pass


def test_cutoff_excludes_samples_without_date():
    ds = _ds()
    for era in ("pre_ai", "post_ai"):
        ids = {s.sample_id for s in ds.filter_by_era(era, cutoff="2022-11-30").samples}
        assert "nodate" not in ids, "ohne Datum darf im Cutoff-Modus nichts zugeordnet werden"


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
