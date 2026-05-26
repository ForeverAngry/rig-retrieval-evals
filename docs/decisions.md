# Architectural Decisions

This document captures the rationale, formulas, and references for the metric definitions and evaluation strategy implemented in `rig-retrieval-evals`.

## Supported Metrics (v0.1)

Traditional Information Retrieval (IR) metrics based on graded relevance (or binary relevance). We follow the standard formulas from TREC `trec_eval` and BEIR.

### Recall @ K
Fractions of relevant documents retrieved in the top K.
- **Formula**: `|retrieved[1..K] ∩ relevant| / |relevant|`
- **Notes**: Vacuously `1.0` if `|relevant| == 0`. Best for evaluating if the knowledge base contains the answer at all.

### Precision @ K
Fraction of the top K retrieved documents that are relevant.
- **Formula**: `|retrieved[1..K] ∩ relevant| / K`
- **Notes**: Returns `0.0` if `K == 0`. Useful for understanding how much "noise" is in the context window.

### Hit Rate @ K
Binary check if *any* relevant document appears in the top K.
- **Formula**: `1.0` if `|retrieved[1..K] ∩ relevant| > 0` else `0.0`.
- **Notes**: Useful minimum-viability metric for single-hop QA (if you only need one correct chunk to answer).

### MRR (Mean Reciprocal Rank)
Reciprocal of the rank of the *first* relevant document.
- **Formula**: `1.0 / rank` of the first relevant hit, or `0.0` if none.
- **Notes**: Emphasizes finding the answer as early as possible.

### MAP @ K (Mean Average Precision)
Approximation of the area under the Precision-Recall curve up to rank K.
- **Formula**: `(1 / |relevant|) * Σ_{i=1..k} (Precision@i * rel(i))` (where `rel(i)` is 1 if relevant, else 0).
- **Notes**: Highly sensitive to the rank of *all* relevant documents.

### nDCG @ K (Normalized Discounted Cumulative Gain)
Measures ranking quality, penalizing relevant documents that appear lower in the list, while gracefully handling different *grades* of relevance.
- **Formula**: `DCG@K / IDCG@K`
- **DCG@K**: `Σ_{i=1..k} ( (2^grade_i - 1) / log2(i + 1) )`
- **IDCG@K**: The DCG of the ideal (perfectly sorted) ranking of the available graded labels.
- **Notes**: BEIR/TREC standard for graded relevance.

## RAGAS-style LLM Judges (v0.2)

Based on the [RAGAS framework](https://arxiv.org/abs/2309.15217), we evaluate RAG generation and retrieval quality using LLM judges. These metrics avoid the need for strict string-matching against gold labels and instead rely on bounded LLM extraction.

### Faithfulness
Measures the factual consistency of the generated answer against the retrieved context.
- **Workflow**:
  1. **Claim Extraction**: Extract a set of atomic statements (claims) from the generated answer.
  2. **Attribution**: For each claim, evaluate if it can be logically inferred from the retrieved context. (Pass/Fail).
- **Formula**: `|Attributed Claims| / |Total Claims|`
- **Output**: Continuous `[0.0, 1.0]`. Returns `1.0` if there are zero claims.

### Context Precision
Measures whether the relevant context chunks were ranked high in the retrieved set.
- **Workflow**:
  1. **Chunk Evaluation**: For each retrieved chunk `c_i`, ask the LLM if it contains the information required to answer the query (Relevant: 1, Irrelevant: 0).
  2. **Rank-weighted Scoring**: Compute Precision@k at every relevant chunk's rank.
- **Formula**: `Σ (Precision@k * rel(k)) / Total Relevant Chunks in k`
- **Output**: Continuous `[0.0, 1.0]`. This acts like MAP but uses an LLM instead of human gold labels.

### Context Recall
Measures the extent to which the retrieved context covers the information present in a *reference answer*.
- **Workflow**:
  1. **Claim Extraction**: Extract atomic statements from the *reference answer*.
  2. **Attribution**: For each claim, determine if the *retrieved context* provides sufficient evidence to support it. (Pass/Fail).
- **Formula**: `|Supported Claims| / |Total Reference Claims|`
- **Output**: Continuous `[0.0, 1.0]`. Perfect recall means the context contained everything necessary to reconstruct the gold answer.

### Answer Relevance
Measures how directly the answer addresses the user's initial query, penalizing incomplete or tangentially related responses.
- **Workflow**:
  1. **Reverse Query Generation**: Use the answer to generate `N` (default: 3) hypothetical questions that this answer perfectly satisfies.
  2. **Embedding Similarity**: Compute the cosine similarity between the original user query and the `N` generated questions.
- **Formula**: `Mean(CosineSim(orig_q, gen_q_i))` over `i=1..N`.
- **Output**: Continuous `[0.0, 1.0]`.

## Knowledge Gain & Novelty (v0.3)
To measure whether ingesting a *new* document actually improves the knowledge base:
- **Gain(d, K)**: `α * ΔRecall@k + β * ΔContextRecall + γ * novelty(d, K)`
- **Novelty(d, K)**: `1 - max(cosine(embed(chunks(d)), embed(chunks(K))))`

## References

1. **BEIR**: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models [Thakur et al., 2021]
2. **RAGAS**: Automated Evaluation of Retrieval Augmented Generation [Es et al., 2023]
3. **MTEB**: Massive Text Embedding Benchmark [Muennighoff et al., 2022]
4. **TREC eval**: Standard IR metric computation (https://github.com/usnistgov/trec_eval)

## Stress eval finding: memvid-core WAL bug

The `eval_memvid_attack` example (added 2025-01) was designed as a real-corpus
stress test for `MemvidStore`, using MITRE ATT&CK Enterprise STIX 2.1 (~2.7k
indexable nodes covering techniques, groups, malware, tools, mitigations,
data-components, data-sources, detection-strategies, and tactics — directly
mapping onto the user's "network / device / asset / AD / services / processes"
domains).

When pointed at `rig-memvid` 0.2.0 (which pins `memvid-core` 2.0.139) the
example fails deterministically with:

```
memvid error: Embedded WAL is corrupted at offset <N>: wal record checksum mismatch
```

### Reproducer (independently verified)

The bug was independently reproduced in a stand-alone Cargo project that
depends only on `rig-memvid` 0.2.0 (default features + `lex`) and replays the
same ATT&CK STIX bundle. It fails deterministically on the **16th** put
(MITRE mitigation `M1045 "Code Signing"`, ~2.5 KB) after 15 successful
mitigation inserts (~30 KB cumulative payload). The error is constant:

```
memvid error: Embedded WAL is corrupted at offset 78667: wal record checksum mismatch
```

Notes from bisecting:

- A naive synthetic probe (30 × ~3.8 KB docs of *highly repetitive* lorem-ipsum
  text) ingests cleanly. The trigger is **content-dependent**, not raw-size
  dependent — most likely related to the lex/tantivy posting size when text
  is varied enough to produce many unique tokens.
- An earlier hypothesis that the bug was a "ring-buffer wrap at 64 KB"
  was **wrong**: the WAL header's default `wal_size` is 4 MB, and the
  failure offset (78,667 B) is *inside* that region, not at a wrap. The bug
  is therefore in `memvid-core`'s WAL append / grow path, not in any
  fixed-size-ring logic.
- The bug surfaces with `MemvidStore::builder().enable_lex().open_or_create()`
  + per-put `put_text` + a single `commit()` at the end — i.e. the canonical
  rig-memvid usage pattern. `rig-memvid`'s own test suite passes only because
  it ingests ≤ 3 small docs.

### Disposition

- `eval_memvid_attack` is shipped as-is; it exits with code `3` and prints the
  offending error verbatim when the bug is hit. This keeps the stress eval
  faithful — we will not synthesise fake numbers around a broken backend.
- The example will start producing real metrics once `rig-memvid` updates
  to a `memvid-core` version with the WAL fix.
- The minimal upstream repro (single binary, ~70 lines, ATT&CK JSON only)
  is captured in the conversation transcript and is ready to file against
  `memvid-core`.
