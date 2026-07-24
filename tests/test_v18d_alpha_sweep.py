#!/usr/bin/env python3
"""
test_v18d_alpha_sweep.py – Guard für den V18d-Alpha-Sweep (V20-v2-Nachlauf):
V18d_a02/a08 (und ihre Qwen3-4B-Klone) duerfen sich von V18d nur in
condition_id/hybrid_alpha/description unterscheiden.
"""

import sys
import dataclasses
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.config import CONDITIONS_MAP  # noqa: E402

_ALLOWED_DIFF_FIELDS = {"condition_id", "hybrid_alpha", "description"}


def _assert_alpha_variant(base_cid, variant_cid, expected_alpha):
    base = CONDITIONS_MAP[base_cid]
    variant = CONDITIONS_MAP[variant_cid]
    diffs = {
        f.name for f in dataclasses.fields(base)
        if getattr(base, f.name) != getattr(variant, f.name)
    }
    assert diffs <= _ALLOWED_DIFF_FIELDS, (
        f"{variant_cid} weicht von {base_cid} unerwartet ab in: {diffs - _ALLOWED_DIFF_FIELDS}"
    )
    assert variant.hybrid_alpha == expected_alpha, (
        f"{variant_cid}: hybrid_alpha={variant.hybrid_alpha}, erwartet {expected_alpha}"
    )


def test_v18d_a02_a08_exist():
    for cid in ["V18d_a02", "V18d_a08", "V18d_a02_QWEN34B", "V18d_a08_QWEN34B"]:
        assert cid in CONDITIONS_MAP, f"{cid} fehlt in CONDITIONS_MAP"


def test_v18d_a02_matches_v18d_except_alpha():
    _assert_alpha_variant("V18d", "V18d_a02", 0.2)


def test_v18d_a08_matches_v18d_except_alpha():
    _assert_alpha_variant("V18d", "V18d_a08", 0.8)


def test_v18d_a02_qwen34b_matches_v18d_qwen34b_except_alpha():
    _assert_alpha_variant("V18d_QWEN34B", "V18d_a02_QWEN34B", 0.2)


def test_v18d_a08_qwen34b_matches_v18d_qwen34b_except_alpha():
    _assert_alpha_variant("V18d_QWEN34B", "V18d_a08_QWEN34B", 0.8)


def test_alpha_sweep_all_use_full_expanded_query_mode():
    # Der ganze Sinn des Sweeps: dieselbe (full_expanded) Query, nur alpha variiert.
    for cid in ["V18d_a02", "V18d_a08", "V18d_a02_QWEN34B", "V18d_a08_QWEN34B"]:
        assert CONDITIONS_MAP[cid].query_mode == "full_expanded", (
            f"{cid}: query_mode={CONDITIONS_MAP[cid].query_mode}, erwartet full_expanded"
        )


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
