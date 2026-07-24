#!/usr/bin/env python3
"""
test_query_mode_ablation.py – Guard für die Query-Modus-Ablation.

Der ursprüngliche V8↔V18-Vergleich war confounded: Ergebnisse verschiedener
Läufe/Modelle wurden gemischt. Die Ablation-Matrix in config.py behebt das,
indem alle Zellen einer (Modell, Retriever)-Zeile aus Conditions bestehen, die
sich AUSSCHLIESSLICH im query_mode unterscheiden. Diese Tests stellen sicher,
dass das so bleibt:

  1. Alle Matrix-Zellen existieren als Conditions.
  2. Jede Nicht-Baseline-Zelle unterscheidet sich von der "full"-Baseline nur
     in condition_id / query_mode / description.
  3. Die query_mode-Werte entsprechen dem Matrix-Schlüssel.
  4. Jede Modell-Zeile nutzt durchgängig dasselbe Embedding-Label.
"""

import sys
import dataclasses
from pathlib import Path

# Insert project root so benchmark.* is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.config import (  # noqa: E402
    CONDITIONS_MAP,
    QUERY_MODE_ABLATION_MATRIX,
    QUERY_MODE_ABLATION_CONDITION_IDS,
    QUERY_MODE_RECOMMENDED_DEFAULT,
    RetrieverType,
)

# Felder, in denen sich ein Ablation-Paar unterscheiden DARF.
_ALLOWED_DIFF_FIELDS = {"condition_id", "query_mode", "description"}

# Erwartetes Embedding-Label je Modell-Key (siehe EMBEDDING_MODELS).
_EXPECTED_LABELS = {"bge": "B", "qwen3": "F", "qwen3-4b": "H"}

# Modi der Matrix: "full" ist die Baseline, gegen die alle anderen verglichen werden.
_MODES = {"full", "llm_expanded", "full_expanded"}


def _iter_cells():
    """Yield (model_key, retriever_key, mode, cid) für jede Matrix-Zelle."""
    for model_key, retrievers in QUERY_MODE_ABLATION_MATRIX.items():
        for retriever_key, modes in retrievers.items():
            for mode, cid in modes.items():
                yield model_key, retriever_key, mode, cid


def _iter_pairs():
    """Yield (model_key, retriever_key, mode, full_cid, cid) für jede
    Nicht-Baseline-Zelle gepaart mit ihrer 'full'-Baseline."""
    for model_key, retrievers in QUERY_MODE_ABLATION_MATRIX.items():
        for retriever_key, modes in retrievers.items():
            full_cid = modes["full"]
            for mode, cid in modes.items():
                if mode == "full":
                    continue
                yield model_key, retriever_key, mode, full_cid, cid


def test_all_matrix_conditions_exist():
    missing = [c for c in QUERY_MODE_ABLATION_CONDITION_IDS if c not in CONDITIONS_MAP]
    assert not missing, f"Ablation-Conditions fehlen in CONDITIONS_MAP: {missing}"


def test_matrix_is_complete():
    # 3 Modelle × 2 Retriever × 3 Modi = 18 eindeutige Zellen
    assert len(QUERY_MODE_ABLATION_CONDITION_IDS) == 18
    assert len(set(QUERY_MODE_ABLATION_CONDITION_IDS)) == 18
    assert set(QUERY_MODE_ABLATION_MATRIX) == set(_EXPECTED_LABELS)
    for _, retrievers in QUERY_MODE_ABLATION_MATRIX.items():
        assert set(retrievers) == {"dense", "hybrid"}
        for modes in retrievers.values():
            assert set(modes) == _MODES


def test_pairs_differ_only_in_query_mode():
    """Kern-Guard: identischer Retriever, nur der query_mode variiert."""
    for model_key, retriever_key, mode, full_cid, cid in _iter_pairs():
        full_c = CONDITIONS_MAP[full_cid]
        cell_c = CONDITIONS_MAP[cid]
        diffs = [
            f.name
            for f in dataclasses.fields(full_c)
            if getattr(full_c, f.name) != getattr(cell_c, f.name)
        ]
        illegal = set(diffs) - _ALLOWED_DIFF_FIELDS
        assert not illegal, (
            f"[{model_key}/{retriever_key}/{mode}] {full_cid} vs. {cid} "
            f"unterscheiden sich auch in {sorted(illegal)} — Confound! "
            f"Erlaubt sind nur {sorted(_ALLOWED_DIFF_FIELDS)}."
        )
        # query_mode MUSS sich unterscheiden, sonst misst das Paar nichts.
        assert "query_mode" in diffs, (
            f"[{model_key}/{retriever_key}/{mode}] {full_cid} und {cid} haben "
            f"denselben query_mode — Zelle misst keinen Expansionseffekt."
        )


def test_query_modes_match_matrix_key():
    for model_key, retriever_key, mode, cid in _iter_cells():
        actual = CONDITIONS_MAP[cid].query_mode
        assert actual == mode, (
            f"{cid}: query_mode={actual!r}, erwartet {mode!r} "
            f"(Matrix-Zelle {model_key}/{retriever_key}/{mode})"
        )


def test_model_rows_use_expected_embedding_label():
    for model_key, retriever_key, mode, cid in _iter_cells():
        expected = _EXPECTED_LABELS[model_key]
        label = CONDITIONS_MAP[cid].embedding_model_label
        assert label == expected, (
            f"{cid}: Embedding-Label {label!r}, erwartet {expected!r} "
            f"für Modell-Key {model_key!r}"
        )


def test_retriever_types_match_matrix_key():
    expected_types = {"dense": RetrieverType.DENSE, "hybrid": RetrieverType.HYBRID}
    for model_key, retriever_key, mode, cid in _iter_cells():
        rt = CONDITIONS_MAP[cid].retriever_type
        assert rt == expected_types[retriever_key], (
            f"{cid}: retriever_type={rt}, erwartet {expected_types[retriever_key]} "
            f"für Matrix-Zeile {retriever_key!r}"
        )


def test_cells_are_flat_retrievers():
    """query_mode wirkt nur auf flache Retriever (runner.py) — hierarchische
    nutzen stage1/stage2_query_mode. Die Matrix darf daher nur flache enthalten."""
    for _, _, _, cid in _iter_cells():
        assert not CONDITIONS_MAP[cid].is_hierarchical, (
            f"{cid} ist hierarchisch — query_mode würde dort ignoriert."
        )


def test_recommended_default_keys_are_known_models():
    assert set(QUERY_MODE_RECOMMENDED_DEFAULT) <= set(QUERY_MODE_ABLATION_MATRIX)
    assert QUERY_MODE_RECOMMENDED_DEFAULT, "Empfehlung darf nicht leer sein"


def test_recommended_default_values_are_valid_modes():
    for model_key, mode in QUERY_MODE_RECOMMENDED_DEFAULT.items():
        assert mode in _MODES, f"{model_key}: unbekannter Modus {mode!r}"
        # Der empfohlene Modus muss für jede Retriever-Zeile des Modells
        # tatsächlich als Condition existieren.
        for retriever_key, modes in QUERY_MODE_ABLATION_MATRIX[model_key].items():
            assert mode in modes, (
                f"{model_key}/{retriever_key}: empfohlener Modus {mode!r} "
                f"fehlt in der Matrix-Zeile"
            )


def test_full_expanded_recommended_for_bge_and_qwen3_06b():
    # Befund aus dem Kombi-Lauf: full_expanded ist auf pandas bester Modus für
    # bge UND Qwen3-0.6B (bge hybrid +8,3pp R@10/+4,9pp MRR; 0.6B hybrid
    # +7,0pp/+6,7pp), über 3 Repos konsistent positiv.
    assert QUERY_MODE_RECOMMENDED_DEFAULT.get("bge") == "full_expanded"
    assert QUERY_MODE_RECOMMENDED_DEFAULT.get("qwen3") == "full_expanded"


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
