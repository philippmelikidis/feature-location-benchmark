#!/usr/bin/env python3
"""
test_query_mode_ablation_variant.py – Guard für den Expansion-Varianten-
Kombi-Lauf: --expansion-variant Output-Isolation + der
Variante-vs-Baseline-Report in scripts/run_query_mode_ablation.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import run_query_mode_ablation as rqma  # noqa: E402


def test_output_dir_for_variant_baseline_stays_at_root():
    base = Path("benchmark/results/query_mode_ablation")
    assert rqma._output_dir_for_variant(base, "baseline") == base
    assert rqma._output_dir_for_variant(base, "") == base
    assert rqma._output_dir_for_variant(base, None) == base


def test_output_dir_for_variant_non_baseline_gets_subdir():
    base = Path("benchmark/results/query_mode_ablation")
    assert rqma._output_dir_for_variant(base, "lean") == base / "lean"
    assert rqma._output_dir_for_variant(base, " lean ") == base / "lean"


def test_expansion_variant_flag_defaults_to_baseline():
    parser_args = rqma.build_arg_parser().parse_args([])
    assert parser_args.expansion_variant == "baseline"


def test_expansion_variant_flag_accepts_variant_name():
    parser_args = rqma.build_arg_parser().parse_args(["--expansion-variant", "lean"])
    assert parser_args.expansion_variant == "lean"


def _fake_map(cid_to_recall_mrr, repo="pandas", k=10):
    """{cid: (recall, mrr)} -> results_map wie _build_results_map() es baut."""
    return {(cid, repo, k): {"recall": r, "mrr": m, "n": 20}
            for cid, (r, m) in cid_to_recall_mrr.items()}


def test_generate_variant_report_computes_deltas_at_fixed_query_mode():
    # V18a/V18b = bge dense/hybrid llm_expanded; V18c/V18d = bge dense/hybrid
    # full_expanded (siehe QUERY_MODE_ABLATION_MATRIX).
    baseline_map = _fake_map({
        "V18a": (0.40, 0.30), "V18b": (0.45, 0.35),
        "V18c": (0.50, 0.40), "V18d": (0.55, 0.45),
    })
    variant_map = _fake_map({
        "V18a": (0.38, 0.29), "V18b": (0.44, 0.34),
        "V18c": (0.58, 0.44), "V18d": (0.60, 0.49),
    })
    report = rqma.generate_variant_report(
        models=["bge"], repos=["pandas"], k_values=[10],
        variant_name="lean",
        baseline_map=baseline_map, variant_map=variant_map,
    )
    assert "Baseline-Expansion" in report
    assert "full_expanded" in report and "llm_expanded" in report
    # full_expanded/hybrid (V18d): 0.60 - 0.55 = +0.05 delta must show up
    assert "+0.0500" in report
    # llm_expanded/dense (V18a): 0.38 - 0.40 = -0.02 delta must show up
    assert "-0.0200" in report


def test_generate_variant_report_handles_missing_cells():
    report = rqma.generate_variant_report(
        models=["bge"], repos=["pandas"], k_values=[10],
        variant_name="lean",
        baseline_map={}, variant_map={},
    )
    # Keine Baseline-Daten -> Zeilen mit "–", kein Crash.
    assert "| bge |" in report
    assert "–" in report


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
