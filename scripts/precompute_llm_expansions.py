#!/usr/bin/env python3
"""
precompute_llm_expansions.py – Alle Queries vorab durch LM Studio expandieren.

Generiert für jede der 95 pandas-Queries einmal die hypothetischen
Code-Snippets und speichert sie in einer JSON-Datei. Der V16-Retriever
liest dann nur noch aus dieser Datei — kein LLM-Call während des Benchmarks.

Features:
  - Resume: bereits expandierte Queries werden übersprungen
  - Fortschrittsanzeige pro Query
  - Einzelne Query testen mit --test-query <id>
  - Prompt-Varianten: --prompt-variant wählt eine Variante aus
    PROMPT_VARIANTS; jede Variante schreibt eine eigene JSON-Datei
    (llm_expansions_<repo>__<variante>.json), Baseline bleibt beim alten Namen.

Aufruf:
  python scripts/precompute_llm_expansions.py
  python scripts/precompute_llm_expansions.py --llm-url http://localhost:1234/v1
  python scripts/precompute_llm_expansions.py --timeout 120
  python scripts/precompute_llm_expansions.py --test-query 0   # erste Query testen
  python scripts/precompute_llm_expansions.py --repo flask \
      --prompt-variant strict-functions --llm-url http://localhost:11434/v1 \
      --model qwen2.5-coder:7b
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

OUTPUT_FILE = ROOT / "benchmark" / "data" / "llm_expansions_pandas.json"
DATASET_FILE = ROOT / "benchmark" / "data" / "benchmark_dataset.json"
REPO_NAME = "pandas"


# ──────────────────────────────────────────────────────────────
# LLM Prompt (identisch mit hierarchical_v16_retriever.py)
# ──────────────────────────────────────────────────────────────

EXPANSION_PROMPT = """\
You are a code search assistant. Given a GitHub issue (bug report or feature request) for the Python library "{repo_name}", generate a list of hypothetical code identifiers that are likely involved in fixing this issue.

Output ONLY a JSON object with these fields:
- "functions": list of function/method names (e.g. ["DataFrame.groupby", "reindex_axis", "_get_result"])
- "classes": list of class names (e.g. ["GroupBy", "DataFrame"])
- "files": list of likely file paths or module names (e.g. ["core/groupby/groupby.py", "core/frame.py"])
- "imports": list of relevant imports or API calls (e.g. ["import numpy as np", "from pandas.core.dtypes.common import is_numeric_dtype"])
- "keywords": list of 3-5 domain-specific keywords from the code (not natural language)

Be specific to the {repo_name} codebase. Focus on Python identifiers, not English descriptions.

Issue:
{issue_text}

JSON:"""


# ──────────────────────────────────────────────────────────────
# Prompt-Varianten
# ──────────────────────────────────────────────────────────────
# Datengetrieben aus der Precision-Analyse (V16 §5.3, alle 5 Repos):
#   functions 0.50 (schwächste Kategorie — LLM erfindet plausible Namen),
#   keywords 0.65, files 0.84, classes 0.86, imports 0.99.
# Jede Variante adressiert genau einen Befund; Baseline bleibt unverändert,
# damit bestehende Expansionen/Vergleiche reproduzierbar bleiben.

# V1: functions-Präzision erzwingen — nur Namen, die sehr wahrscheinlich
#     existieren; im Zweifel weglassen (Precision > Recall der Terme).
_PROMPT_STRICT_FUNCTIONS = """\
You are a code search assistant. Given a GitHub issue (bug report or feature request) for the Python library "{repo_name}", generate a list of code identifiers that are likely involved in fixing this issue.

STRICT RULE for function names: only include a function/method name if you are CONFIDENT it actually exists in the {repo_name} codebase (public API names, names quoted in the issue, or names you have seen in {repo_name} source). If unsure, OMIT it — fewer, correct names are better than many guesses. Never invent plausible-sounding helper names.

Output ONLY a JSON object with these fields:
- "functions": list of function/method names that really exist (may be short or empty)
- "classes": list of class names (e.g. ["GroupBy", "DataFrame"])
- "files": list of likely file paths or module names (e.g. ["core/groupby/groupby.py"])
- "imports": list of relevant imports or API calls
- "keywords": list of 3-5 domain-specific keywords from the code (not natural language)

Be specific to the {repo_name} codebase. Focus on Python identifiers, not English descriptions.

Issue:
{issue_text}

JSON:"""

# V2: nur treffsichere Kategorien, harte Mengen-Caps — kompaktes,
#     BM25-freundliches Signal statt breitem Termteppich.
_PROMPT_LEAN = """\
You are a code search assistant. Given a GitHub issue (bug report or feature request) for the Python library "{repo_name}", name the few places in the codebase most likely involved in fixing it.

Be minimal and precise. Output ONLY a JSON object with these fields:
- "classes": at most 3 class names that really exist in {repo_name}
- "files": at most 3 likely file paths or module names
- "imports": at most 3 relevant imports or API calls
- "functions": at most 2 function/method names, ONLY if you are confident they exist
- "keywords": []

Do not pad the lists. An empty list is better than a guess. Focus on Python identifiers, not English descriptions.

Issue:
{issue_text}

JSON:"""

# V3: Few-Shot-Grounding — ein durchgerechnetes Beispiel verankert Format
#     und Spezifität (echte Identifier statt generischer Namen).
_PROMPT_FEWSHOT = """\
You are a code search assistant. Given a GitHub issue (bug report or feature request) for the Python library "{repo_name}", generate a list of code identifiers that are likely involved in fixing this issue. Only name symbols that plausibly exist in the codebase — never invent generic helper names.

Example (library "flask"):
Issue: "url_for generates wrong URL when SERVER_NAME contains a port"
JSON: {{"functions": ["url_for", "create_url_adapter"], "classes": ["Flask"], "files": ["src/flask/app.py", "src/flask/helpers.py"], "imports": ["from werkzeug.routing import Map"], "keywords": ["SERVER_NAME", "url_adapter", "subdomain"]}}

Now do the same for this issue. Output ONLY a JSON object with the fields "functions", "classes", "files", "imports", "keywords" (3-5 domain-specific code keywords, not natural language). Be specific to the {repo_name} codebase.

Issue:
{issue_text}

JSON:"""

PROMPT_VARIANTS = {
    "baseline": EXPANSION_PROMPT,
    "strict-functions": _PROMPT_STRICT_FUNCTIONS,
    "lean": _PROMPT_LEAN,
    "fewshot": _PROMPT_FEWSHOT,
}


def variant_output_file(repo_name: str, variant: str) -> Path:
    """Output-Datei je (Repo, Variante).

    Baseline behält den historischen Namen llm_expansions_<repo>.json;
    Varianten bekommen ein __<variante>-Suffix, damit Retriever/Analyse
    sie getrennt laden können (siehe EXPANSION_VARIANT-Env im Retriever).
    """
    if variant == "baseline":
        return ROOT / "benchmark" / "data" / f"llm_expansions_{repo_name}.json"
    return ROOT / "benchmark" / "data" / f"llm_expansions_{repo_name}__{variant}.json"


def parse_expansion(llm_output: str) -> dict:
    """Parse LLM JSON output into structured dict."""
    json_str = llm_output
    if "```" in json_str:
        parts = json_str.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                json_str = part
                break

    start = json_str.find("{")
    end = json_str.rfind("}") + 1
    if start >= 0 and end > start:
        json_str = json_str[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"raw": llm_output}


def flatten_expansion(data: dict) -> str:
    """Flatten parsed expansion dict into a BM25-friendly string."""
    if "raw" in data:
        return data["raw"].replace('"', " ").replace("{", " ").replace("}", " ")

    terms = []
    for key in ["functions", "classes", "files", "imports", "keywords"]:
        items = data.get(key, [])
        if isinstance(items, list):
            for item in items:
                cleaned = str(item).replace('"', "").replace("'", "")
                terms.append(cleaned)
    return " ".join(terms)


def call_llm(prompt: str, api_url: str, timeout: float, max_retries: int = 3,
             model: str = "local-model") -> str:
    """Call LM Studio and return the response text.

    Each call uses a fresh HTTP connection (no keep-alive) and sends only
    a single user message — no chat history. This prevents LM Studio from
    trying to continue a previous conversation or running out of KV cache.

    Handles Qwen3-style thinking models: these spend tokens on internal
    reasoning (reasoning_content) before producing the actual answer (content).
    We set max_tokens high enough and also try to extract from reasoning_content
    if content is empty.

    Retries on timeout AND on empty responses.
    """
    import requests

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,  # Qwen3 needs room for thinking + answer
        "stream": False,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # Fresh connection each time — no session reuse.
            resp = requests.post(
                f"{api_url}/chat/completions",
                json=body,
                timeout=timeout,
                headers={"Connection": "close"},
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "").strip()

            # Qwen3 thinking models: the actual JSON might be in
            # reasoning_content if the model ran out of max_tokens
            # before writing the answer.
            if not content:
                reasoning = message.get("reasoning_content", "")
                if reasoning:
                    # Try to extract JSON from the reasoning
                    json_start = reasoning.rfind("{")
                    json_end = reasoning.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        candidate = reasoning[json_start:json_end]
                        try:
                            import json as _json
                            _json.loads(candidate)
                            content = candidate
                            print(f" (extracted from reasoning)", end="")
                        except _json.JSONDecodeError:
                            pass

            # Still empty? Retry.
            if not content:
                finish = choice.get("finish_reason", "unknown")
                if attempt < max_retries:
                    wait = 5 * attempt
                    print(f"\n      empty response (finish_reason={finish}), "
                          f"retry {attempt+1}/{max_retries} in {wait}s...", end="", flush=True)
                    time.sleep(wait)
                    continue
                else:
                    raise ValueError(
                        f"LLM returned empty response after {max_retries} attempts "
                        f"(finish_reason={finish}). Model may need more max_tokens "
                        f"or try disabling thinking mode in LM Studio."
                    )

            return content

        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"\n      timeout, retry {attempt+1}/{max_retries} in {wait}s...", end="", flush=True)
                time.sleep(wait)
        except ValueError as e:
            raise e
        except Exception as e:
            raise e

    raise last_error or ValueError("LLM call failed")


def test_connection(api_url: str) -> bool:
    """Quick check that LM Studio is running."""
    import requests
    try:
        resp = requests.get(f"{api_url}/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            print(f"LM Studio erreichbar, Modell: {models[0].get('id', '?')}")
            return True
        else:
            print("LM Studio läuft, aber kein Modell geladen!")
            return False
    except Exception as e:
        print(f"LM Studio nicht erreichbar: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute LLM query expansions")
    parser.add_argument("--llm-url", default="http://localhost:1234/v1",
                        help="OpenAI-kompatible API URL (LM Studio: :1234/v1, Ollama: :11434/v1)")
    parser.add_argument("--model", default="local-model",
                        help="Modellname im Request. LM Studio ignoriert ihn; "
                             "Ollama braucht den echten Namen, z.B. qwen2.5-coder:7b")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Timeout pro Query in Sekunden (default: 120)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max Retries bei Timeout/leerer Antwort (default: 3)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Pause in Sekunden zwischen Queries (default: 2)")
    parser.add_argument("--retry-empty", action="store_true",
                        help="Bereits vorhandene 0-terms Einträge nochmal versuchen")
    parser.add_argument("--test-query", type=int, default=None,
                        help="Nur eine Query testen (Index, z.B. 0)")
    parser.add_argument("--repo", type=str, default=REPO_NAME,
                        help="Repo, dessen Queries expandiert werden "
                             "(pandas|requests|flask|click|fastapi). Default: pandas")
    parser.add_argument("--output", type=str, default=None,
                        help="Output-Datei (default: benchmark/data/llm_expansions_<repo>.json)")
    parser.add_argument("--prompt-variant", default="baseline",
                        choices=sorted(PROMPT_VARIANTS),
                        help="Prompt-Variante. Nicht-Baseline schreibt "
                             "nach llm_expansions_<repo>__<variante>.json")
    parser.add_argument("--dataset", type=str, default=str(DATASET_FILE),
                        help="Dataset-JSON (z.B. benchmark_dataset_v2.json)")
    args = parser.parse_args()

    repo_name = args.repo
    prompt_template = PROMPT_VARIANTS[args.prompt_variant]
    output_file = (
        Path(args.output) if args.output
        else variant_output_file(repo_name, args.prompt_variant)
    )
    print(f"\nPrompt-Variante: {args.prompt_variant}")

    # Test connection
    print(f"\nLM Studio: {args.llm_url}")
    if not test_connection(args.llm_url):
        sys.exit(1)

    # Load dataset
    dataset_file = Path(args.dataset)
    print(f"\nLade Dataset: {dataset_file}")
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = [s for s in dataset["samples"] if repo_name in s["repo_id"]]
    print(f"   {len(samples)} {repo_name}-Queries gefunden")

    # Single query test mode
    if args.test_query is not None:
        idx = args.test_query
        if idx >= len(samples):
            print(f"Index {idx} zu groß (max {len(samples)-1})")
            sys.exit(1)

        sample = samples[idx]
        title = sample["query"].split("\n")[0].strip()
        print(f"\nTeste Query {idx}: {title[:80]}")
        print(f"   sample_id: {sample['sample_id']}")

        prompt = prompt_template.format(
            repo_name=repo_name,
            issue_text=sample["query"][:2000],
        )

        t0 = time.time()
        raw = call_llm(prompt, args.llm_url, args.timeout, args.max_retries,
                       model=args.model)
        elapsed = time.time() - t0

        parsed = parse_expansion(raw)
        flat = flatten_expansion(parsed)

        print(f"\n   ⏱  {elapsed:.1f}s")
        print(f"\n   Raw LLM Output:")
        print(f"   {raw[:500]}")
        print(f"\n   Parsed:")
        print(f"   {json.dumps(parsed, indent=2)[:500]}")
        print(f"\n   Flattened (BM25 query):")
        print(f"   {flat[:200]}")
        print(f"\n   Combined (title + terms):")
        print(f"   {title} {flat}"[:200])
        return

    # Load existing expansions (for resume).
    # Resume matcht zusätzlich über den Titel: Dataset-v2 vergibt neue
    # (deterministische) sample_ids, aber der Retriever-Lookup läuft über
    # den Titel — vorhandene Expansionen bleiben gültig und werden nicht
    # neu berechnet.
    existing = {}
    done_titles = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        all_done_ids = set(existing.get("expansions", {}).keys())
        done_titles = {
            (e.get("title") or "").strip().lower()
            for e in existing.get("expansions", {}).values()
            if e.get("flat_terms", "").strip()
        }

        if args.retry_empty:
            # Re-try entries that had 0 terms or errors
            retry_ids = set()
            for sid, entry in existing.get("expansions", {}).items():
                flat = entry.get("flat_terms", "")
                has_error = entry.get("error")
                if not flat.strip() or has_error:
                    retry_ids.add(sid)
            done_ids = all_done_ids - retry_ids
            print(f"   {len(all_done_ids)} gesamt, {len(retry_ids)} mit 0 terms → retry")
        else:
            done_ids = all_done_ids
            print(f"   {len(done_ids)} bereits expandiert (Resume-Modus)")
    else:
        done_ids = set()

    # Initialize output structure
    if "expansions" not in existing:
        existing = {
            "repo": repo_name,
            "prompt_variant": args.prompt_variant,
            "created_at": datetime.now().isoformat(),
            "llm_url": args.llm_url,
            "num_samples": len(samples),
            "expansions": {},
        }

    # Process all queries
    print(f"\nStarte Expansion ({len(samples) - len(done_ids)} verbleibend)...\n")

    total_time = 0
    failures = 0

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        title = sample["query"].split("\n")[0].strip()

        # Skip already done (per sample_id oder — nach Dataset-Rebuild — per Titel)
        if sid in done_ids:
            print(f"  [{i+1:3d}/{len(samples)}] ⏭  {sid} (bereits vorhanden)")
            continue
        if title.strip().lower() in done_titles:
            print(f"  [{i+1:3d}/{len(samples)}] ⏭  {sid} (Titel bereits expandiert)")
            continue

        print(f"  [{i+1:3d}/{len(samples)}] {sid}: {title[:60]}...", end="", flush=True)

        prompt = prompt_template.format(
            repo_name=repo_name,
            issue_text=sample["query"][:2000],
        )

        try:
            t0 = time.time()
            raw = call_llm(prompt, args.llm_url, args.timeout, args.max_retries, model=args.model)
            elapsed = time.time() - t0
            total_time += elapsed

            parsed = parse_expansion(raw)
            flat = flatten_expansion(parsed)

            existing["expansions"][sid] = {
                "title": title,
                "raw_llm_output": raw,
                "parsed": parsed,
                "flat_terms": flat,
                "combined_query": f"{title} {flat}",
                "expansion_time_s": round(elapsed, 1),
            }

            n_terms = len(flat.split())
            print(f"  {elapsed:.1f}s, {n_terms} terms")

        except Exception as e:
            failures += 1
            print(f"  {e}")
            existing["expansions"][sid] = {
                "title": title,
                "raw_llm_output": None,
                "parsed": None,
                "flat_terms": "",
                "combined_query": title,  # fallback: title only
                "expansion_time_s": 0,
                "error": str(e),
            }

        # Save after each query (crash-safe)
        existing["last_updated"] = datetime.now().isoformat()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        # Pause between queries so LM Studio can clear its KV cache
        if args.delay > 0 and i < len(samples) - 1:
            time.sleep(args.delay)

    # Summary
    done = len(existing["expansions"])
    avg_time = total_time / max(done - len(done_ids), 1)
    print(f"\n{'='*60}")
    print(f"  Fertig: {done}/{len(samples)} Queries expandiert")
    print(f"  Fehler: {failures}")
    print(f"  ⏱  Ø {avg_time:.1f}s pro Query, gesamt {total_time:.0f}s")
    print(f"  Gespeichert: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
