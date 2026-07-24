#!/usr/bin/env python3
"""
test_v20_conditions.py – Guard für die V20i-Condition (V20-Hybrid-Coarse-v2):
top_n_files gross genug, um "alle Dateien" zu approximieren, und konsistent
mit dem V20e-h-Hybrid-Coarse-Design (nur top_n_files unterscheidet sich).
"""

import sys
import dataclasses
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.config import (  # noqa: E402
    CONDITIONS_MAP, V20_HYBRID_IDS, V20_CONDITION_IDS, RetrieverType,
)


def test_v20i_exists_in_hybrid_and_condition_ids():
    assert "V20i" in V20_HYBRID_IDS
    assert "V20i" in V20_CONDITION_IDS
    assert "V20i" in CONDITIONS_MAP


def test_v20i_top_n_files_exceeds_pandas_vdoc_file_count():
    # Beobachteter VDoc-Dateicount für pandas: 281. 500 ist eine komfortable Marge fuer "effektiv alle
    # Dateien, keine Stage-1-Filterung".
    assert CONDITIONS_MAP["V20i"].top_n_files >= 500


def test_v20i_matches_hybrid_coarse_design_except_top_n():
    """V20i darf sich von V20h (letzte bestehende Hybrid-Coarse-Zelle) NUR in
    top_n_files, condition_id und description unterscheiden — sonst ist es
    kein sauberer N-Sweep-Punkt mehr."""
    allowed_diff = {"condition_id", "top_n_files", "description"}
    v20h = CONDITIONS_MAP["V20h"]
    v20i = CONDITIONS_MAP["V20i"]
    diffs = {
        f.name for f in dataclasses.fields(v20h)
        if getattr(v20h, f.name) != getattr(v20i, f.name)
    }
    assert diffs <= allowed_diff, f"V20i weicht unerwartet ab in: {diffs - allowed_diff}"
    assert v20i.coarse_retriever_type == RetrieverType.HYBRID
    assert v20i.embedding_model_label == "H"


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
