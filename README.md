# Feature Location Benchmark

A benchmark that measures how well different retrieval methods find the correct source file for a given GitHub issue.

The practical problem: given a bug report or feature request, a developer (or an AI coding tool) first has to find the right place in the codebase. This project automates that step and evaluates it rigorously, using 918 real issues from five well-known Python projects where the ground truth is known from the actual fix commits.

## What is compared

| Approach | Idea |
|---|---|
| BM25 (sparse) | Classic keyword search over code chunks |
| Dense retrieval | Semantic search with embedding models (bge, Qwen3) |
| Hybrid | Weighted combination of BM25 and dense scores |
| Hierarchical | Two stages: first select candidate files, then rank code chunks within them |
| LLM query expansion | A local LLM reads the issue and guesses likely function, class and file names, which are added to the query |

Beyond method comparison, the benchmark answers two research questions:

1. **Does LLM query expansion actually help, and for which embedding models?** Answered with a controlled ablation where only the query construction changes.
2. **Is code from before the AI era (pre ChatGPT) easier to retrieve than code written after?** Answered with a time-based split of all queries, configurable by cutoff date and by date field (issue creation vs. fix merge date).

## Key results (dataset v2, 918 queries, k=10)

| Method | Recall@10 (macro) | MRR@10 (macro) |
|---|---|---|
| Hierarchical without expansion (V12b) | 0.698 | 0.562 |
| Flat hybrid (V10b) | 0.734 | 0.571 |
| **Hierarchical + LLM expansion (V16c)** | **0.751** | **0.587** |

Selected findings:

- **LLM expansion is the real lever, not the two-stage architecture.** Without expansion, the hierarchical setup falls behind plain flat search. The advantage of V16c over V12b is statistically significant (paired McNemar test, p < 0.0001, n = 918).
- **Whether expansion helps depends on the embedding model.** It helps bge consistently. For Qwen3 it hurts when the expansion replaces the issue text, but helps when appended to it. This "append instead of replace" mode (`full_expanded`) came out of the ablation and became the recommended default.
- **Term precision is a misleading metric.** A prompt variant that produced more accurate code terms made actual retrieval worse, because fewer terms mean fewer chances to hit the right file. Coverage beats precision.
- **Data quality changes conclusions.** Validating the ground truth revealed that the first dataset version contained only 149 unique issues out of 206 entries (duplicates, umbrella issues, deleted target files). After rebuilding the pipeline and re-extracting via the GitHub GraphQL API, the cleaned dataset grew to 918 valid queries, and all methods scored higher than previously measured.
- **Pre-AI code is easier to retrieve at Recall@10** (about 6 to 8 percentage points across repositories), but the effect size depends on how "era" is defined. Classifying by fix merge date instead of issue creation date moves 21% of the samples and shrinks the gap. Both options are built in.

Full result write-ups are in [docs/](docs/) (German).

## How it works

```
GitHub issues + fix commits          Repositories at HEAD
        |                                    |
        v                                    v
  Ground-truth pipeline              Chunking (function, class,
  (GraphQL extraction,               fixed-size, AST, heuristic,
   dedup, validation)                virtual documents)
        |                                    |
        v                                    v
   benchmark_dataset_v2.json  ---->  Elasticsearch index
                                     (text + dense vectors)
                                             |
                                             v
                              Retrieval (BM25 / dense / hybrid /
                              hierarchical / + LLM expansion)
                                             |
                                             v
                              Recall@k, MRR@k, bootstrap CIs,
                              paired significance tests, reports
```

- **Ground truth:** an issue counts as solvable if GitHub's own issue-to-pull-request link identifies the merged fix, and the changed Python files still exist in the checkout. One entry per issue, deterministic IDs, validation report on every build.
- **LLM expansion:** precomputed once per query with a local model (qwen2.5-coder:7b via Ollama), stored as JSON. No LLM calls at benchmark time, so retrieval latency and LLM latency stay separate.
- **Statistics:** every report includes 95% bootstrap confidence intervals. Method comparisons on the same queries use paired tests (exact McNemar for recall hits, paired bootstrap for MRR).

## Quickstart

Requirements: Python 3.11+, a running Elasticsearch 8.x on `localhost:9200`, and about 2 GB disk for models.

```bash
pip install -r requirements_benchmark.txt

# Small smoke run: one method, one repository
python -m benchmark.runner --conditions V10b --repos flask \
    --dataset benchmark/data/benchmark_dataset_v2.json

# The headline comparison
python -m benchmark.runner --conditions V12b V16c V10b \
    --dataset benchmark/data/benchmark_dataset_v2.json

# Controlled ablation of the query construction
python scripts/run_query_mode_ablation.py --models bge --repos flask

# Significance test between two methods
python scripts/analyze_significance.py \
    --results-a benchmark/results --condition-a V16c \
    --results-b benchmark/results --condition-b V12b
```

The repositories under test are cloned automatically on first run. Precomputed LLM expansions for all 918 queries ship with the repo, so no local LLM is needed to reproduce the main results. To regenerate them: `python scripts/precompute_llm_expansions.py --repo flask --llm-url http://localhost:11434/v1 --model qwen2.5-coder:7b`.

## Tests

```bash
for t in tests/test_*.py; do python "$t"; done
```

11 test suites cover the metric functions, the statistics helpers, the ground-truth pipeline (dedup, validation, era classification) and structural guards that keep the ablation comparisons clean (for example: paired conditions may differ in nothing but the query mode). CI runs the same suites plus a lint pass on every pull request.

## Project structure

```
benchmark/
  chunking/        6 chunking strategies (function-level to AST-based)
  retrievers/      BM25, dense, hybrid, 4 hierarchical variants
  ground_truth/    GitHub extraction, dataset builder, validation
  reranker/        cross-encoder and LLM reranking (experimental)
  config.py        all benchmark conditions (V1..V20) as data
  runner.py        orchestration: chunk, index, retrieve, evaluate
  metrics.py       Recall@k, MRR@k, bootstrap CI, McNemar
  reporting.py     Markdown reports with confidence intervals
scripts/           ablation runner, significance tests, expansion precompute
  experiments/     historical per-experiment runners (V11..V20)
tests/             11 test suites, runnable without Elasticsearch or GPU
docs/              result write-ups per experiment (German)
```

## Background

This code was built as part of a university year project (Reutlingen University) evaluating retrieval strategies for AI coding assistants, in a team of two engineers plus a research group. My own focus areas: the ground-truth pipeline and dataset validation, the query-mode ablation framework, the statistics layer, the era analysis and the LLM expansion tooling. This repository is a cleaned snapshot: the original project lives in a private university repository, so its commit history (which contains internal ticket references) is not included here. The experiment series V1 to V20 in code and docs still reflects the actual course of the project, including negative results, which are documented with the same care as the positive ones.
