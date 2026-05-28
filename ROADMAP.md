# rig-retrieval-evals Roadmap

This roadmap is the crate-local operating plan for `rig-retrieval-evals`. The cross-crate coordination summary lives in [`rig-ecosystem/docs/roadmap.md`](../rig-ecosystem/docs/roadmap.md). The fuller planning record for phased evaluation work remains in [`../rig-ecosystem/docs/evals-rag-plan.md`](../rig-ecosystem/docs/evals-rag-plan.md).

## Role

`rig-retrieval-evals` measures retrieval and knowledge-base quality for any Rig `VectorStoreIndex`. It evaluates stores and RAG context, not general agent behavior or product dashboards.

## Landed

- BEIR-compatible JSONL `Qrels` loader.
- `RetrievalMetric` trait with `RecallAtK`, `PrecisionAtK`, `HitRateAtK`, `Mrr`, `MapAtK`, and `NdcgAtK`.
- Async `RetrievalHarness` over `VectorStoreIndexDyn` with bounded concurrency.
- `MetricReport` and `MultiReport` with JSON, Markdown, aggregation, and baseline diffing.
- Off-by-default `ragas` feature with RAGAS-style judges, `RagasHarness`, bounded-concurrency judge calls, XML-fenced prompts, abstention scores, and judge fingerprint diff protection.
- Off-by-default `ingestion` feature with zero-waste ingestion deltas, deterministic IoC extraction, proposition extraction, redundancy checks, and model-independent contract tests for LLM-backed proposition extractors.
- Off-by-default `ingestion-graph` sub-feature with knowledge-graph triples, in-memory and `petgraph` baselines, and model-independent contract tests for LLM-backed triple extractors.
- Chunk-stat ingestion linting (`ChunkLintConfig`, `ChunkStats`,
  `ChunkLintReport`, `ChunkLintWarning`, `lint_chunks`) covering
  token-length distributions, metadata coverage, language/encoding sanity,
  opt-in MinHash-style near-duplicate detection, and per-chunk warnings
  ([src/ingestion/lint.rs](src/ingestion/lint.rs)).
- Repeated-trial reliability reporting with thresholded pass@k and pass^k over
  shared `MetricReport`s.
- Default-feature `staleness` module (`CorpusVersions`, `StalenessAnnotation`,
  `StaleHit` / `StalenessReport`, `ConflictGroup` / `ConflictReport`,
  `detect_stale_hits`, `detect_conflicts`) for flagging top-k hits that are
  superseded by a newer doc id and `version_key` collisions inside the same
  window. Covers the supersession contract that `rig-memvid`'s
  `MemoryContextPack` produces at the host side, without taking a dependency
  on it. Committed fixture under [tests/data/tiny_corpus_versions.jsonl](tests/data/tiny_corpus_versions.jsonl).

## Prototype Grade

- Retrieval metrics are usable; RAGAS is merged but release pending.
- Zero-waste ingestion tracks are merged and covered by deterministic tests; chunk-stat linting ships token-length, metadata-coverage, language/encoding, and opt-in MinHash-style near-duplicate checks today.
- `examples/eval_memvid.rs` now runs `rig-memvid::MemvidStore` and `MemoryCardContext` through `RetrievalHarness` / `EvalShadowStore` against committed raw-frame and structured-card fixtures.
- `EvalShadowStore` is available behind `shadow` for pre/post scoring over two `VectorStoreIndexDyn` snapshots.
- Knowledge-gain scoring is available behind `knowledge-gain`, including candidate-document ranking and host-supplied novelty scores; `embedding-novelty` adds a provider-neutral adapter over host-supplied Rig embedding models.
- Bootstrap confidence intervals (`MetricCi`, `MetricReport::bootstrap_ci` / `with_bootstrap_ci`, deterministic SplitMix64) and non-zero CI exits (`ReportDiff::is_clean` / `exit_code`) ship in the default `retrieval` build and round-trip through `MultiReport::diff`.
- `observe` feature emits `MultiReport` / `ReportDiff` as `rig-tap`-compatible JSON envelopes (`eval.retrieval_report`, `eval.regression_diff`) on the `rig_tap` tracing target without depending on `rig-tap`.
- `FreshnessReport` / `FreshnessQueryRollup` roll stale-hit and conflict
  detector outputs into `MultiReport`, preserving dataset-level counts/rates
  and per-query rates. `MultiReport::with_freshness_metrics` appends
  score-like freshness metric rows (`freshness.stale_free_rate@k`,
  `freshness.conflict_free_rate@k`) so freshness regressions use the same
  baseline-diff and `RegressionGate` path as IR metrics.

## Next Work

1. Cut the RAGAS and ingestion release after final validation and README/changelog sync.
2. Add provider-specific novelty examples only where they belong: downstream demos or docs that already own credentials and model setup.

## Maturity Bar

- A KB regression can be reproduced in CI against committed qrels fixtures.
- Retrieval and RAGAS reports include stable metadata and refuse unsafe comparisons across judge fingerprints.
- Ingestion quality failures identify chunking, duplication, metadata, retrieval, or judging as the likely layer.
- Knowledge gain produces a single report-level score and ranked candidate documents with optional host-supplied or adapter-computed novelty signals.

## Non-Goals

- Do not become a trace-level agent evaluation harness until the platform runner exists.
- Do not own product dashboards, online shadow scoring, tenancy slicing, or safety/adversarial suites.
- Do not route models; measure quality and report signals for other layers to consume.
