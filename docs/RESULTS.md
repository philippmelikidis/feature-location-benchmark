# Results summary (English)

This is a condensed English summary of all experiment write-ups in this folder. The detailed per-experiment documents are in German; each section links to its source.

## 1. Which retrieval method wins?

Evaluated on dataset v2 (918 real GitHub issues, five Python repositories), k = 10, macro average (each repository weighted equally):

| Method | Recall@10 | MRR@10 |
|---|---|---|
| Hierarchical two-stage, no LLM (V12b) | 0.698 | 0.562 |
| Flat hybrid search (V10b) | 0.734 | 0.571 |
| Hierarchical + LLM query expansion (V16c) | **0.751** | **0.587** |

The V16c advantage over V12b is statistically significant (paired exact McNemar test on recall hits: p < 0.0001; paired bootstrap on MRR: p = 0.001). Notably, the two-stage architecture **without** the LLM expansion performs worse than plain flat search. The expansion is the lever, not the architecture. Details: [dataset_v2_ergebnisse.md](dataset_v2_ergebnisse.md), [significance_ergebnisse.md](significance_ergebnisse.md).

## 2. Does LLM query expansion help? (controlled ablation)

Setup: identical retriever and embedding model, only the query construction varies. Three modes: the raw issue text (`full`), only the LLM-generated code terms (`llm_expanded`), or the issue text with the terms appended (`full_expanded`).

Findings on pandas (95 queries, later confirmed in direction on three repositories):

- For the general-purpose embedding model **bge**, expansion helps in both forms (up to +8.3 points Recall@10).
- For the code-strong model **Qwen3-0.6B**, replacing the issue text with terms hurts (-8.8 points), but appending them helps (+7.0 points). The model needs the issue context; the terms are useful only as an addition.
- For the large **Qwen3-4B**, expansion raises recall (+8 to +9 points) but worsens ranking (MRR -4 points).

Recommendation: `full_expanded` as the default query mode for flat retrieval. Details: [query_mode_default_empfehlung.md](query_mode_default_empfehlung.md).

## 3. Can a better prompt improve the expansion?

Three alternative prompts for the term-generating LLM were tested. The precision-optimized variant (`lean`) produced terms that were correct more often, but made actual retrieval worse (up to -10.7 points Recall@10 in the hierarchical setup). Fewer terms mean fewer chances to hit the right file: coverage beats precision. The standard prompt remains the default. This is a useful negative result: optimizing a proxy metric (term precision) would have degraded the real target metric. Details: [prompt_variants_ergebnisse.md](prompt_variants_ergebnisse.md).

## 4. Data quality: from 206 entries to 918 valid queries

Auditing the first dataset revealed that of 206 entries only 149 were unique issues. The rest were duplicates (one pandas issue appeared 17 times), umbrella issues without a single target file, and entries whose target files no longer exist in the repository. After rebuilding the extraction on GitHub's authoritative issue-to-pull-request links (GraphQL), deduplicating, and validating target files against the checkout, the cleaned dataset contains 918 queries. All methods score higher on the clean data; the old dataset had systematically underestimated them. The headline finding (section 1) replicates on the clean data, which is the strongest validity check in the project. Details: [dataset_v2_ergebnisse.md](dataset_v2_ergebnisse.md).

## 5. Is pre-AI code easier to retrieve?

Splitting the 918 queries by time (before/after the ChatGPT launch on 2022-11-30) shows a consistent advantage for pre-AI code at Recall@10: 6 to 8 percentage points macro, in four of five repositories. However, the effect depends on how "era" is assigned: classifying by the fix merge date instead of the issue creation date moves 21% of the samples across the boundary, shrinks the Recall@10 gap, and flips Recall@1 in favor of post-AI code. Era comparisons are unpaired and confounded with issue age and issue mix, so these numbers describe a difference and do not prove a cause. Details: [era_split_v2_ergebnisse.md](era_split_v2_ergebnisse.md).

## 6. Statistics used throughout

- 95% bootstrap confidence intervals (percentile method, 1000 to 2000 resamples, fixed seed) on every reported Recall and MRR value.
- Paired tests for method comparisons on the same queries: exact McNemar for recall hits, paired bootstrap for MRR differences.
- A practical consequence documented in the write-ups: with 50 queries per repository, the confidence interval on Recall@10 is roughly plus or minus 14 points. Several early per-repository "findings" did not survive this check and are labeled as tendencies.
