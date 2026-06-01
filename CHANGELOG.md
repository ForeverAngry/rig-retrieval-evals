# Changelog

<!-- markdownlint-disable MD024 -->

All notable changes to `rig-retrieval-evals` are documented here. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this crate
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.3](https://github.com/ForeverAngry/rig-retrieval-evals/compare/v0.3.2...v0.3.3) - 2026-06-01

### Documentation

- Align README Status with shipped 0.3.2 ([#4](https://github.com/ForeverAngry/rig-retrieval-evals/pull/4))

## [0.3.2](https://github.com/ForeverAngry/rig-retrieval-evals/compare/v0.3.1...v0.3.2) - 2026-06-01

### Tests

- Cover ingestion graph knowledge gain interaction ([#7](https://github.com/ForeverAngry/rig-retrieval-evals/pull/7))

## [0.3.1](https://github.com/ForeverAngry/rig-retrieval-evals/compare/v0.3.0...v0.3.1) - 2026-05-28

### Documentation

- Align docs/decisions.md heading + AGENTS feature list with v0.2.0+ ([#2](https://github.com/ForeverAngry/rig-retrieval-evals/pull/2))

## [0.3.0](https://github.com/ForeverAngry/rig-retrieval-evals/compare/v0.2.0...v0.3.0) - 2026-05-28

### Build

- Drop path override on rig-memvid dev-dep so release-plz can resolve crates.io fallback

### CI

- Add release-plz workflow and config

### Added

- Bootstrap confidence intervals on `MetricReport::mean`: new `MetricCi`
  type plus `MetricReport::bootstrap_ci(iterations, level, seed)` and
  builder `MetricReport::with_bootstrap_ci(...)`. Uses a deterministic
  SplitMix64 stream (no `rand` dependency) so the same seed reproduces
  the same interval. `MetricCi` is `#[serde(default,
  skip_serializing_if = "Option::is_none")]` so existing JSON reports
  stay schema-compatible. `MetricDelta` carries `current_ci` /
  `baseline_ci` alongside the existing mean delta.
- Non-zero CI exit support on `ReportDiff`: `is_clean(&gate) -> bool`
  and `exit_code(&gate) -> i32` (0 = pass, 1 = regression). Lets eval
  binaries gate CI with `std::process::exit(diff.exit_code(&gate))`
  without rebuilding the regression-walking logic at every call site.
- `observe` feature: emit `MultiReport` and `ReportDiff` as
  `rig-tap`-compatible JSON envelopes on the `rig_tap` tracing target
  without depending on `rig-tap`. Introduces `EvalEnvelope` + `EvalKind`
  (`eval.retrieval_report`, `eval.regression_diff`),
  `report_envelopes`, `diff_envelopes`, and tracing helpers
  `emit_report` / `emit_diff`. Each event carries the full JSON envelope
  plus stable scalar `rig_tap.kind` / `rig_tap.metric` /
  `rig_tap.regressed` / `rig_tap.conversation_id` attributes so existing
  OpenTelemetry collectors route eval reports the same way they route
  prompt and tool events.
- `FreshnessReport` and `FreshnessQueryRollup` for rolling
  `StalenessReport` / `ConflictReport` outputs into `MultiReport`.
  `MultiReport::with_freshness` attaches the dataset/per-query rollup, while
  `with_freshness_metrics` also appends score-like
  `freshness.stale_free_rate@k` and `freshness.conflict_free_rate@k` metric
  rows so freshness regressions trip the existing `RegressionGate` / diff
  path.
- Opt-in MinHash-style chunk near-duplicate linting via
  `NearDuplicateLintConfig`, `NearDuplicatePair`,
  `ChunkStats::near_duplicate_pairs`, and
  `ChunkLintWarning::NearDuplicateChunks`. The default remains disabled so
  existing ingestion gates keep their current warning set until hosts opt in.

## [0.2.0] - 2026-05-24

First tagged release after the initial unreleased line. Folds in optional
RAGAS judges, the zero-waste ingestion pipeline (IoC + propositions +
graph), shadow scoring, knowledge-gain scoring, embedding-novelty
adapter, memory / models / agents behavior harnesses, the deterministic
skills harness, RAGAS-as-skill grader, reliability (pass@k / pass^k),
regression-gated baseline diffs, chunk-stat ingestion linting, and the
default-feature staleness / conflict detector.

### Added

- New `staleness` module (default feature) for retrieval-time freshness
  evaluation independent of the qrels grade. Ships `StalenessAnnotation`,
  `CorpusVersions` (JSONL loader keyed by `doc_id`), `StaleHit` /
  `StalenessReport`, `ConflictGroup` / `ConflictReport`, and two pure
  detectors: `detect_stale_hits` flags top-k hits that are superseded by
  a newer doc (explicit `superseded_by` or strictly-newer
  `effective_timestamp` inside the same `version_key`); `detect_conflicts`
  flags `version_key` collisions where the retriever surfaced multiple
  generations of the same fact in one window. Re-exported from the crate
  root. Committed fixture at `tests/data/tiny_corpus_versions.jsonl` and
  integration coverage in `tests/staleness.rs`.
- Ingestion chunk linting now includes encoding and optional language checks:
  `EncodingLintWarning` reports control characters and BOMs, while
  `LanguageLintConfig` uses `whatlang` to flag unknown-language rates and
  detected languages outside a caller-supplied allow-list.
- Add ingestion-local lexical novelty helpers:
  `jaccard_knowledge_gain` and `corpus_jaccard_knowledge_gain`, plus
  `IngestionDelta::knowledge_gain` / `with_knowledge_gain` for carrying the
  resulting `[0, 1]` score alongside accepted/dropped ingestion outputs.

### Changed

- `EvalShadowStore::run` now executes the baseline and candidate harness
  runs concurrently via `futures::try_join!` (they share no state) and is
  instrumented with a `tracing` span (`k`, `concurrency`, `queries`,
  `metrics`). Sequential semantics are preserved for the resulting
  `ShadowEvalReport`.
- `EmbeddingNoveltyAdapter` honours `EmbeddingModel::MAX_DOCUMENTS` by
  flattening candidate chunks once and batching the underlying embed
  calls (reference chunks are batched the same way). A new
  `with_concurrency(usize)` knob bounds in-flight batches via
  `buffered`; the default is `1` so behaviour stays serial unless opted
  in. Both `score_candidates` and the internal `embed_batched` carry
  `tracing` spans for observability.
- **Breaking (pre-publish):** `CandidateDocumentGain` now exposes the
  raw `relevance_gain` (`Σ query_gain × grade`) and `novelty` (the
  host-supplied novelty score) separately from the weighted summands
  `weighted_relevance_gain` and `weighted_novelty_gain`. The previous
  `novelty_gain` field has been removed in favour of
  `weighted_novelty_gain`. `to_markdown` table headers update
  accordingly (`weighted_relevance | weighted_novelty | novelty`). The
  composite `score` formula is unchanged.

- Bumped `rig-core` dependency from `0.36.0` to `0.37.0` and aliased the
  package as `rig` (`rig = { package = "rig-core", version = "0.37.0",
  default-features = false }`) to absorb upstream's library-name change.
  Source `use rig::…` paths are unchanged. `Chat::chat` history-append
  semantics in 0.37 are not exercised by this crate (no call sites). Full
  `just check` matrix is clean under the bump.

### Added

- `memory`, `models`, and `agents` feature flags with runner-driven behavior
  harnesses. `MemoryHarness` grades captured recall evidence across write /
  query / reload scenarios; `ModelBehaviorHarness` grades provider-neutral
  output behavior such as required terms, forbidden terms, JSON validity, and
  output-token budgets; `AgentHarness` grades final output, expected tool
  calls, and turn budgets. All three emit typed reports with per-task scores
  and conversion into the shared `MetricReport` layer. No new dependencies.
- `knowledge-gain` feature with `KnowledgeGainConfig` and
  `KnowledgeGainReport`, a model-free scoring surface that aggregates weighted
  candidate-minus-baseline retrieval deltas from a `ReportDiff` into one score
  plus per-metric, per-query, and candidate-document movers. Candidate ranking
  uses qrels-backed query gains and optional host-supplied novelty scores. The
  `eval_memvid` example now prints knowledge-gain summaries and ranked
  candidate documents for raw-frame and structured-card shadow paths.
- `embedding-novelty` feature with `EmbeddingNoveltyAdapter`, a provider-neutral
  adapter over a host-supplied `rig::embeddings::EmbeddingModel`. It embeds
  candidate chunks and reference KB chunks, computes `1 - max cosine` novelty,
  and returns `CandidateDocumentGainInput` values that plug into
  `KnowledgeGainReport::with_candidate_documents`. Provider client setup,
  model choice, chunking policy, and credential handling remain host-owned.
- `shadow` feature with `EvalShadowStore` and `ShadowEvalReport` for pre/post
  retrieval scoring over two `VectorStoreIndexDyn` snapshots. It runs the same
  qrels and metrics through baseline and candidate stores, then returns the two
  `MultiReport`s plus their candidate-minus-baseline `ReportDiff`. The runner
  is intentionally non-mutating; hosts own backend-specific ingest and snapshot
  preparation.
- `eval_memvid` example (requires `memvid-example`) with committed corpus,
  memory-card, and qrels fixtures. It seeds a temporary
  `rig-memvid::MemvidStore`, evaluates raw frames through `MemvidStore`,
  evaluates structured/domain-memory facts through `MemoryCardContext`, and
  uses `EvalShadowStore` to print empty-baseline vs seeded-candidate deltas for
  both paths. The dependency remains example-only; the library surface still
  evaluates stores through `VectorStoreIndexDyn`.
- `SkillTaskSet` JSONL I/O (requires `skills`): `load_jsonl`,
  `load_jsonl_with_id`, `from_jsonl_str`, `to_jsonl_string`, and
  `save_jsonl` let skill suites live as versioned data files instead of
  Rust literals. Parse errors reuse the existing line-numbered
  `DatasetParse` error shape.
- `skills_basic` example (requires `skills`) showing the full local flow:
  load a JSONL task suite, run a deterministic `AgentRunner`, mix
  contains / tool-call / trigger graders, and emit a JSON report. No
  provider keys or network calls required.
- `RetrievalGroundednessGrader` (requires `skills`): async grader that
  re-queries any [`VectorStoreIndexDyn`] with a `(task, transcript)`
  extract (default: `transcript.final_output`), retrieves the top-`k`
  documents, and scores the answer against the concatenated context.
  Default scorer is token-recall (no LLM in the loop); both the query
  derivation, scorer, and document extractor are pluggable closures.
  Closes the loop between the skill harness and this crate's primary
  surface — the retriever under evaluation.
- `AsyncGrader` trait (in `skills`) for graders that need to `.await`
  during scoring. Every existing deterministic [`Grader`] auto-implements
  it via a blanket impl, so existing code keeps working with a one-line
  type swap (`Vec<Box<dyn Grader>>` → `Vec<Box<dyn AsyncGrader>>`). The
  harness now drives `AsyncGrader` directly so LLM-rubric judges can sit
  alongside deterministic checks in the same registry.
- `RagasJudgeGrader` (requires `skills` + `ragas`): wraps any
  `RagasMetric` (`FaithfulnessMetric`, `AnswerRelevanceMetric`,
  `ContextPrecisionMetric`, `ContextRecallMetric`, or a custom impl) as
  an `AsyncGrader`. Pass threshold is configurable; raw judge score is
  preserved in the outcome `notes` for audit. The default mapping uses
  `task.prompt → query` and `transcript.final_output → answer`; override
  via `with_inputs_fn` for context-aware judges.
- New optional `skills` feature: deterministic skill / agent evaluation
  harness alongside the existing retrieval one. Adds
  `rig_retrieval_evals::skills::{SkillHarness, SkillTask, SkillTaskSet,
  AgentRunner, Transcript, ToolCall, Usage, Grader, GraderOutcome,
  ContainsGrader, ToolCallGrader, TranscriptBudget, TriggerGrader,
  SkillEvalReport, TrialRow}`. The harness runs `(tasks × trials)`,
  applies deterministic graders to captured transcripts, and reuses the
  existing `MetricReport` / `ReliabilityReport` (pass@k / pass^k)
  aggregation. LLM-rubric judging is intentionally out of scope for
  this feature — users wanting rubric scoring can pair the existing
  `ragas` feature with a custom `Grader`. Zero new dependencies.
- `ReliabilityReport` / `QueryReliability` aggregate repeated
  `MetricReport`s for the same metric into thresholded pass/fail reliability
  estimates: mean pass rate, pass@k (at least one success in k attempts),
  and pass^k (all k attempts succeed), with per-query breakdowns and
  validation that trial reports share the same metric and query set.
- `ReportDiff` now carries per-query winners/losers/unchanged counts and
  a sorted `query_changes: Vec<QueryDelta>` listing the largest movers per
  metric, computed by intersecting the two reports' `per_query` vectors on
  `query_id`. Queries missing from either side are skipped. The new
  `RegressionGate` lets callers configure per-metric tolerated drops in
  mean score; `ReportDiff::regressions(&gate)` returns the metrics whose
  delta breaches the gate. `ReportDiff::to_json` complements the existing
  `to_markdown`, and the Markdown table grew `win`/`lose`/`same` columns.
  Backward compatible: new `MetricDelta` fields use `#[serde(default)]`.
- `tests/ingestion_ab.rs` — fixture-based ingestion A/B test that runs
  the harness against two `MockStore` configurations (improvement vs
  regression) and asserts the diff/gate verdicts. No LLM, no network.
- `LlmTripleExtractor` and `LlmPropositionExtractor` wrappers enabling live extraction via `rig-core`.
- Deterministic `ingestion_llm.rs` contract tests validating LLM adapters with a fake `rig::CompletionModel`, independent of model/vendor behavior.
- `live_ollama_ingestion.rs` smoke test for manual validation against a local tool-capable Ollama model.
- Track 2 (knowledge-graph distillation) on the `ingestion` feature:
  `Triple` (with normalised predicate — lowercased, whitespace collapsed
  to underscores), `TripleExtractor` + `GraphBaseline` traits, the
  deterministic `StubTripleExtractor`, and `InMemoryGraphBaseline`
  (`HashSet`-backed; always available). `DistillationPipeline::with_graph`
  layers the track via a second type-state axis (`NoGraphTrack` /
  `ActiveGraphTrack`) so unconfigured pipelines pay no runtime cost and
  Tracks 2 and 3 compose independently. `IngestionDelta` grew a
  `triples: Vec<Triple>` field, `DroppedItem::Triple(Triple)` and
  `DroppedReason::DuplicateEdge` were added (all `non_exhaustive`,
  additive).
- `ingestion-graph` sub-feature (depends on `ingestion`): pulls
  [`petgraph`](https://docs.rs/petgraph) and ships `PetgraphBaseline`,
  a `Graph<String, String>`-backed `GraphBaseline` for hosts that want
  richer downstream queries (path / predicate adjacency) against the same
  store the pipeline uses for dedup.
- Track 3 (propositional distillation) on the `ingestion` feature:
  `Proposition`, `PropositionExtractor` + `RedundancyCheck` traits, the
  deterministic `StubPropositionExtractor` (sentence splitter), and
  `VectorStoreRedundancyCheck` which drives any `VectorStoreIndexDyn`.
  `DistillationPipeline::with_propositions` layers the track via a
  type-state (`NoPropositionTrack` / `ActivePropositionTrack`) so
  unconfigured pipelines pay no runtime cost. `IngestionDelta` grew a
  `propositions: Vec<Proposition>` field and `DroppedReason::Redundant
  { similarity }` was added (both `non_exhaustive`, additive).
- `ingestion` feature (off by default): zero-waste ingestion pipeline that
  emits structured `IngestionDelta`s (net-new items + dropped items with
  reasons) instead of committing chunks. Ships Track 1 (IoC filter):
  `Document` / `Section`, `IocExtractor` + `IocBaseline` traits, the
  deterministic `RegexIocExtractor` (CVE / IPv4 / IPv6 / MD5 / SHA-1 /
  SHA-256 / domain / URL / Windows registry key) and an
  `InMemoryIocBaseline`. `DistillationPipeline` orchestrates the track and
  is generic over extractor/baseline so callers can swap implementations
  without trait objects. Tracks 2 (knowledge graph) and 3 (propositions)
  layer onto the same orchestrator in follow-up PRs.
- Add crate-local `ROADMAP.md` documenting maturity status, next work, and
  non-goals for retrieval and RAG evaluation.
- `ragas` feature (off by default): LLM-based RAGAS-style judges
  (`FaithfulnessMetric`, `ContextPrecisionMetric`, `ContextRecallMetric`,
  `AnswerRelevanceMetric`) wired through a single `RagasMetric` trait and an
  object-safe `DynRagasMetric` shim.
- `RagasInputs` / `RagasScore` envelopes; `RagasScore::not_measurable`
  lets a judge abstain instead of fabricating a score.
- `RagasHarness` async driver with `futures::buffered` concurrency. The
  harness composes every judge's `fingerprint_component` into the
  `MultiReport::judge_fingerprint` so diffs across judge revisions refuse
  to compare.
- Typed `Error::Extraction` (`#[from] rig::extractor::ExtractionError`) and
  `Error::Embedding` (`#[from] rig::embeddings::EmbeddingError`) variants
  under the `ragas` feature, replacing the prior `Error::Llm(String)`.

### Changed

- Per-claim and per-chunk LLM calls inside every RAGAS judge now run with
  bounded concurrency via `futures::stream::buffered`.
- All RAGAS prompts now fence their inputs in XML-style tags and instruct
  the judge to treat fenced contents as data — narrower prompt-injection
  surface.

## [0.1.0] - TBD

### Added

- BEIR-compatible JSONL qrels loader (`Qrels::load_jsonl`).
- `RetrievalMetric` trait + concrete implementations: `RecallAtK`,
  `PrecisionAtK`, `HitRateAtK`, `Mrr`, `MapAtK`, `NdcgAtK`.
- Async `RetrievalHarness` over any `rig::vector_store::VectorStoreIndexDyn`,
  with bounded concurrency and per-query tracing spans.
- `MetricReport` / `MultiReport` with mean, stddev, P50/P95, min/max.
- JSON + Markdown report serialization and `MultiReport::diff` with
  `judge_fingerprint` mismatch detection.

[Unreleased]: https://github.com/ForeverAngry/rig-retrieval-evals/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ForeverAngry/rig-retrieval-evals/releases/tag/v0.1.0
