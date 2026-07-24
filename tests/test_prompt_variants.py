#!/usr/bin/env python3
"""
test_prompt_variants.py – Guards für die Prompt-Design-Iteration.

Sichert ab:
  1. PROMPT_VARIANTS enthält die Baseline + mindestens 3 echte Varianten.
  2. Jede Variante hat beide Pflicht-Platzhalter und ist formatierbar
     (keine kaputten {…}-Literale, wichtig fürs Few-Shot-JSON-Beispiel).
  3. Baseline ist byte-identisch zum historischen EXPANSION_PROMPT
     (Reproduzierbarkeit bestehender Expansionen).
  4. variant_output_file: Baseline behält den alten Dateinamen, Varianten
     bekommen das __<variante>-Suffix.
  5. expansions_file_for_repo respektiert die EXPANSION_VARIANT-Env
     (Downstream-Benchmarks gegen Varianten-Expansions).
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Stub heavy deps so the retriever module imports without ES/torch ──
for _mod in ("tree_sitter", "tree_sitter_python", "sentence_transformers",
             "elasticsearch", "numpy", "torch"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

_chunking_base = types.ModuleType("benchmark.chunking.base")
_chunking_base.Chunk = MagicMock
_chunking_base.BaseChunker = MagicMock
sys.modules["benchmark.chunking.base"] = _chunking_base
sys.modules["benchmark.chunking"] = types.ModuleType("benchmark.chunking")

_ret_base = types.ModuleType("benchmark.retrievers.base")
_ret_base.BaseRetriever = object
sys.modules["benchmark.retrievers.base"] = _ret_base

_es_mod = types.ModuleType("benchmark.retrievers.es_retriever")
_es_mod.ElasticsearchRetriever = MagicMock
sys.modules["benchmark.retrievers.es_retriever"] = _es_mod

import importlib.util as _ilu


def _load(rel: str, name: str):
    path = Path(__file__).resolve().parent.parent / rel
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    # Relative imports (.base, .es_retriever) auf die Stubs umbiegen:
    mod.__package__ = "benchmark.retrievers"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pre = _load("scripts/precompute_llm_expansions.py", "pre_llm_exp")
v16 = _load("benchmark/retrievers/hierarchical_v16_retriever.py",
            "benchmark.retrievers.hierarchical_v16_retriever")


def test_registry_has_baseline_and_three_variants():
    assert "baseline" in pre.PROMPT_VARIANTS
    non_baseline = [k for k in pre.PROMPT_VARIANTS if k != "baseline"]
    assert len(non_baseline) >= 3, (
        f"Gefordert sind ≥3 Varianten, gefunden: {non_baseline}"
    )
    # Varianten müssen sich wirklich von der Baseline unterscheiden
    for k in non_baseline:
        assert pre.PROMPT_VARIANTS[k] != pre.PROMPT_VARIANTS["baseline"], (
            f"Variante {k!r} ist identisch zur Baseline"
        )


def test_baseline_unchanged():
    assert pre.PROMPT_VARIANTS["baseline"] == pre.EXPANSION_PROMPT


def test_variants_formatable_with_required_placeholders():
    for name, tpl in pre.PROMPT_VARIANTS.items():
        assert "{repo_name}" in tpl, f"{name}: Platzhalter repo_name fehlt"
        assert "{issue_text}" in tpl, f"{name}: Platzhalter issue_text fehlt"
        # .format() darf nicht an unescapten {…}-Literalen scheitern
        rendered = tpl.format(repo_name="flask", issue_text="Example issue")
        assert "flask" in rendered and "Example issue" in rendered


def test_variant_output_file_naming():
    base = pre.variant_output_file("flask", "baseline")
    lean = pre.variant_output_file("flask", "lean")
    assert base.name == "llm_expansions_flask.json"
    assert lean.name == "llm_expansions_flask__lean.json"


def test_expansions_file_env_override():
    old = os.environ.pop("EXPANSION_VARIANT", None)
    try:
        default = v16.expansions_file_for_repo("pandas")
        assert default.endswith("llm_expansions_pandas.json")

        os.environ["EXPANSION_VARIANT"] = "lean"
        assert v16.expansions_file_for_repo("pandas").endswith(
            "llm_expansions_pandas__lean.json")
        assert v16.expansions_file_for_repo("pallets/flask").endswith(
            "llm_expansions_flask__lean.json")
        # Fallback (kein repo_id) bekommt ebenfalls das Suffix
        assert v16.expansions_file_for_repo(None).endswith(
            "llm_expansions_pandas__lean.json")

        os.environ["EXPANSION_VARIANT"] = "baseline"
        assert v16.expansions_file_for_repo("pandas").endswith(
            "llm_expansions_pandas.json")
    finally:
        if old is None:
            os.environ.pop("EXPANSION_VARIANT", None)
        else:
            os.environ["EXPANSION_VARIANT"] = old


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
