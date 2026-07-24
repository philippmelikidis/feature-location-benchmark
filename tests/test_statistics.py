#!/usr/bin/env python3
"""
test_statistics.py – Guards für die Statistik-Helpers.

Prüft die Bootstrap-CIs und gepaarten Tests gegen bekannte Sollwerte:
exakter McNemar-p für kleine b/c ist analytisch nachrechenbar, Bootstrap
muss deterministisch (Seed) und richtungslogisch korrekt sein.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Chunk-Import in metrics stubben (kein tree_sitter nötig)
_chunking_base = types.ModuleType("benchmark.chunking.base")
_chunking_base.Chunk = MagicMock
_chunking_base.BaseChunker = MagicMock
sys.modules.setdefault("benchmark.chunking.base", _chunking_base)
sys.modules.setdefault("benchmark.chunking", types.ModuleType("benchmark.chunking"))

from benchmark.metrics import (  # noqa: E402
    bootstrap_ci, mcnemar_exact, paired_bootstrap_diff,
)


def test_bootstrap_ci_basics():
    # leer / n=1 / konstant
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([0.7]) == (0.7, 0.7, 0.7)
    m, lo, hi = bootstrap_ci([0.5] * 100)
    assert m == lo == hi == 0.5, "konstante Daten → CI ohne Breite"

    # Mitte + Ordnung + Mittelwert im CI
    vals = [0.0] * 50 + [1.0] * 50
    m, lo, hi = bootstrap_ci(vals)
    assert abs(m - 0.5) < 1e-9
    assert lo <= m <= hi and lo < hi
    # grobe Plausibilität: bei n=100, p=0.5 ist das 95%-CI etwa ±0.10
    assert 0.35 < lo < 0.47 and 0.53 < hi < 0.65


def test_bootstrap_ci_deterministic():
    # Garantie: gleicher Seed → identisches Ergebnis (Reproduzierbarkeit).
    # (Verschiedene Seeds DÜRFEN zufällig gleiche Perzentile treffen.)
    vals = [i / 10 for i in range(10)]
    assert bootstrap_ci(vals, seed=42) == bootstrap_ci(vals, seed=42)
    assert bootstrap_ci(vals, seed=7) == bootstrap_ci(vals, seed=7)


def test_mcnemar_exact_known_values():
    # b=8, c=0 → p = 2 * P(X<=0) = 2 * 0.5^8 = 0.0078125 (exakt)
    a = [True] * 8 + [True] * 10 + [False] * 10
    b = [False] * 8 + [True] * 10 + [False] * 10
    r = mcnemar_exact(a, b)
    assert (r["b"], r["c"]) == (8, 0)
    assert abs(r["p_value"] - 0.0078125) < 1e-9

    # symmetrisch b=c → p=1 (kein Unterschied)
    a = [True, False] * 5
    b = [False, True] * 5
    r = mcnemar_exact(a, b)
    assert r["b"] == r["c"] == 5
    assert r["p_value"] == 1.0

    # keine diskordanten Paare → p=1
    same = [True, True, False]
    assert mcnemar_exact(same, same)["p_value"] == 1.0


def test_mcnemar_requires_pairing():
    try:
        mcnemar_exact([True], [True, False])
        raise AssertionError("ungepaarte Listen müssen ValueError werfen")
    except ValueError:
        pass


def test_paired_bootstrap_diff_direction_and_significance():
    # klarer Effekt: a durchgängig +0.3 mit etwas Rauschen → signifikant, CI > 0
    base = [0.1, 0.2, 0.3, 0.4, 0.5] * 20
    a = [x + 0.3 + (0.01 if i % 2 else -0.01) for i, x in enumerate(base)]
    r = paired_bootstrap_diff(a, base)
    assert r["mean_diff"] > 0.29
    assert r["ci_lo"] > 0, "CI muss vollständig über 0 liegen"
    assert r["p_value"] < 0.01

    # kein Effekt: symmetrisches Rauschen um 0 → nicht signifikant
    noise = [(-1) ** i * 0.05 for i in range(100)]
    b = [0.5 + d for d in noise]
    r = paired_bootstrap_diff(b, [0.5] * 100)
    assert r["p_value"] > 0.05
    assert r["ci_lo"] < 0 < r["ci_hi"]


def test_paired_bootstrap_deterministic_and_degenerate():
    a = [0.9, 0.8, 0.7, 0.6]
    b = [0.5, 0.6, 0.4, 0.3]
    assert paired_bootstrap_diff(a, b, seed=42) == paired_bootstrap_diff(a, b, seed=42)
    # identische Listen → Differenzen konstant 0 → p=1, CI=Punkt
    r = paired_bootstrap_diff(a, a)
    assert r["mean_diff"] == 0.0 and r["p_value"] == 1.0
    assert r["ci_lo"] == r["ci_hi"] == 0.0


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
