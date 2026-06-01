# rig-retrieval-evals

[![Crates.io](https://img.shields.io/crates/v/rig-retrieval-evals.svg)](https://crates.io/crates/rig-retrieval-evals)
[![Docs.rs](https://docs.rs/rig-retrieval-evals/badge.svg)](https://docs.rs/rig-retrieval-evals)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

> Retrieval and knowledge-base evaluation harness for
> [Rig](https://crates.io/crates/rig-core) agents.

`rig-retrieval-evals` measures **knowledge-base quality**, not just answer quality.
Point it at any `VectorStoreIndex` (`rig`'s in-memory store, `rig-memvid`,
`rig-lancedb`, …), give it a labeled qrels file, and get a report you can
diff between runs to catch regressions before they ship.

## Status

Crate version: `0.3.2`. Rust edition: 2024. MSRV: 1.89. Runtime-agnostic
library; `tokio` is only a dev-dependency for tests and examples.

The default build ships **retrieval-quality** evaluation plus stale /
conflict detection. Optional RAGAS judges, zero-waste ingestion tracks,
shadow scoring, and model-free knowledge-gain scoring live behind
feature flags.

| Capability | Default | Feature | Validation |
| --- | :---: | --- | --- |
| BEIR-style qrels loader | ✅ | `retrieval` | Unit + harness integration tests |
| Recall / Precision / MRR / MAP / nDCG / HitRate | ✅ | `retrieval` | Metric unit tests |
| Async `RetrievalHarness` over any `VectorStoreIndexDyn` | ✅ | `retrieval` | `tests/harness.rs` |
| JSON / Markdown reports + baseline diff | ✅ | `retrieval` | Report unit tests + harness test |
| Repeated-trial pass@k / pass^k reliability reports | ✅ | `retrieval` | Report unit tests |
| Stale-content + `version_key` conflict detection | ✅ | `retrieval` | `tests/staleness.rs` |
| Freshness rollups in `MultiReport` + regression gates | ✅ | `retrieval` | Report unit tests |
| Pre/post shadow-store scoring | — | `shadow` | `tests/shadow.rs` |
| Model-free knowledge-gain scoring | — | `knowledge-gain` | Unit tests + `eval_memvid` |
| Candidate-document gain ranking + host novelty | — | `knowledge-gain` | Unit tests + `eval_memvid` |
| Generic embedding novelty adapter | — | `embedding-novelty` | Deterministic fake-model unit test |
| Memory behavior harness | — | `memory` | `tests/memory_harness.rs` |
| Model behavior harness | — | `models` | `tests/models_harness.rs` |
| Agent behavior harness | — | `agents` | `tests/agents_harness.rs` |
| RAGAS-style LLM judges (faithfulness, context recall, …) | — | `ragas` | Unit tests with deterministic judge fixtures |
| Zero-waste IoC ingestion | — | `ingestion` | `tests/ingestion_ioc.rs` |
| Proposition distillation + redundancy checks | — | `ingestion` | `tests/ingestion_propositions.rs` |
| Knowledge-graph triples + graph baseline | — | `ingestion-graph` | `tests/ingestion_graph.rs` |
| LLM-backed ingestion extractors | — | `ingestion` | Model-independent fake-provider contract tests + optional live Ollama smoke |
| Provider-specific novelty setup | — | host-owned | Not implemented here |

The crate-local maturity plan lives in [ROADMAP.md](ROADMAP.md). The fuller
phased planning record, including out-of-scope items and reopen triggers, lives
in
[`rig-ecosystem/docs/evals-rag-plan.md`](https://github.com/ForeverAngry/rig-ecosystem/blob/main/docs/evals-rag-plan.md).
Cross-crate coordination lives in
[`rig-ecosystem/docs/roadmap.md`](https://github.com/ForeverAngry/rig-ecosystem/blob/main/docs/roadmap.md).

## Feature flags

| Feature | Default | Enables |
| --- | --- | --- |
| `retrieval` | yes | Pure-Rust retrieval metrics, qrels loading, harness, reports, and diffs. |
| `ragas` | no | LLM-backed RAGAS-style judges and `RagasHarness`. |
| `ingestion` | no | Zero-waste ingestion Track 1 (IoCs), Track 3 (propositions), chunk linting with encoding/language/near-duplicate checks, lexical knowledge gain, and LLM extractor adapters. |
| `ingestion-graph` | no | Track 2 knowledge-graph triples plus `petgraph`-backed baseline. Implies `ingestion`. |
| `embedding-novelty` | no | `EmbeddingNoveltyAdapter` over a host-provided `rig::embeddings::EmbeddingModel`. Implies `knowledge-gain`. |
| `knowledge-gain` | no | `KnowledgeGainReport` for weighted candidate-minus-baseline scoring, candidate-document ranking, and host-supplied novelty from a `ReportDiff`. Implies `shadow`. |
| `memvid-example` | no | Builds the example-only `eval_memvid` harness against `rig-memvid`; implies `knowledge-gain`. The library still depends only on `VectorStoreIndexDyn`. |
| `shadow` | no | `EvalShadowStore` for pre/post retrieval scoring over two `VectorStoreIndexDyn` snapshots. |
| `memory` | no | Backend-neutral memory behavior harness over host-provided runners and captured recall observations. |
| `models` | no | Provider-neutral model behavior harness for output terms, JSON validity, and token-budget checks. |
| `agents` | no | Agent behavior harness for final-output assertions, expected tools, and turn-budget checks. |

## Quick start

```rust,no_run
use anyhow::Result;
use rig::vector_store::VectorStoreIndexDyn;
use rig_retrieval_evals::{
    NdcgAtK, Qrels, RecallAtK, RetrievalHarness, RetrievalMetric,
};

# async fn run(store: impl VectorStoreIndexDyn + 'static) -> Result<()> {
let qrels = Qrels::load_jsonl("tests/data/tiny_qrels.jsonl")?;

let metrics: Vec<Box<dyn RetrievalMetric>> = vec![
    Box::new(RecallAtK::new(10)),
    Box::new(NdcgAtK::new(10)),
];

let report = RetrievalHarness::new(&store, 10)
    .with_concurrency(4)
    .run(&qrels, &metrics)
    .await?;

println!("{}", report.to_markdown());
std::fs::write("report.json", report.to_json()?)?;
# Ok(()) }
```

### Diffing against a baseline

```rust,no_run
# use rig_retrieval_evals::MultiReport;
# fn read(p: &str) -> anyhow::Result<MultiReport> { unimplemented!() }
# fn demo() -> anyhow::Result<()> {
let current  = read("report.json")?;
let baseline = read("baseline.json")?;
let diff = current.diff(&baseline)?;
println!("{}", diff.to_markdown());
# Ok(()) }
```

The diff refuses to compare reports whose `judge_fingerprint` differs, so
swapping an LLM judge never silently moves your score.

### Freshness rollups

The `staleness` module can flag stale top-k hits and `version_key` conflicts
per query. Convert those detector outputs into a `FreshnessReport`, then attach
it to a `MultiReport`. Use `with_freshness_metrics` when freshness should also
participate in the existing `RegressionGate` path:

```rust,no_run
# use rig_retrieval_evals::{
#     ConflictReport, FreshnessReport, MultiReport, RegressionGate,
#     StalenessReport,
# };
# fn demo(stale: Vec<StalenessReport>, conflicts: Vec<ConflictReport>) -> rig_retrieval_evals::Result<()> {
let freshness = FreshnessReport::from_query_reports(10, &stale, &conflicts)?;
let report = MultiReport::new(vec![]).with_freshness_metrics(freshness);

let gate = RegressionGate::new()
    .with_threshold("freshness.stale_free_rate@10", 0.01)
    .with_threshold("freshness.conflict_free_rate@10", 0.01);
# let baseline = report.clone();
let diff = report.diff(&baseline)?;
assert_eq!(diff.exit_code(&gate), 0);
# Ok(()) }
```

The attached `FreshnessReport` retains dataset-level rates (`stale_rate`,
`conflict_rate`, query rates, total counts) and per-query stale/conflict rates.
The generated metric rows are score-like (`*_free_rate@k`, higher is better),
so they work with the same baseline-diff gate semantics as recall, nDCG, or
MRR.

### Shadow scoring

The `shadow` feature packages the common pre/post pattern: run the same qrels
and metrics against a baseline retriever and a candidate retriever, then diff
candidate against baseline.

```rust,no_run
# use rig::vector_store::VectorStoreIndexDyn;
# use rig_retrieval_evals::{EvalShadowStore, Qrels, RecallAtK, RetrievalMetric};
# async fn demo(before: &dyn VectorStoreIndexDyn, after: &dyn VectorStoreIndexDyn, qrels: &Qrels) -> rig_retrieval_evals::Result<()> {
let metrics: Vec<Box<dyn RetrievalMetric>> = vec![Box::new(RecallAtK::new(5))];
let shadow = EvalShadowStore::new(before, after, 5)
    .with_concurrency(2)
    .run(qrels, &metrics)
    .await?;

println!("{}", shadow.diff.to_markdown());
# Ok(()) }
```

The stores are snapshots supplied by the caller; `EvalShadowStore` does not
mutate either one. That keeps ingest policy and backend-specific cloning in the
host while giving every retriever the same report/diff surface.

### Knowledge gain

The `knowledge-gain` feature turns a shadow `ReportDiff` into a single weighted
score plus per-metric, per-query, and candidate-document movers. The ranking is
deliberately model-free: it measures qrels-backed retrieval improvement and can
blend in host-supplied novelty scores without requiring this crate to own an
embedding model.

```rust,no_run
# use rig_retrieval_evals::{CandidateDocumentGainInput, KnowledgeGainConfig, KnowledgeGainReport, Qrels, ReportDiff};
# fn demo(diff: &ReportDiff, qrels: &Qrels) {
let config = KnowledgeGainConfig::new()
    .with_metric_weight("recall@5", 2.0)
    .with_metric_weight("ndcg@5", 1.0)
    .with_novelty_weight(0.25);
let candidates = [CandidateDocumentGainInput::new("doc-7").with_novelty(0.6)];
let gain = KnowledgeGainReport::from_diff(diff, &config)
    .with_candidate_documents(qrels, &candidates, &config);
println!("{}", gain.to_markdown());
# }
```

The `embedding-novelty` feature adds a narrow adapter for hosts that already
have a Rig embedding model. It does not construct provider clients or choose
models. The host supplies candidate chunks and reference KB chunks; the adapter
returns `CandidateDocumentGainInput` values with novelty filled in. The
adapter flattens candidate chunks across all candidates into a single embed
pass batched by `M::MAX_DOCUMENTS`; pass `.with_concurrency(n)` to fan out
batches in parallel via `buffered(n)`.

```rust,no_run
# use rig::embeddings::EmbeddingModel;
# use rig_retrieval_evals::{CandidateNoveltyInput, EmbeddingNoveltyAdapter};
# async fn demo<M: EmbeddingModel>(model: M) -> rig_retrieval_evals::Result<()> {
let adapter = EmbeddingNoveltyAdapter::new(model).with_concurrency(4);
let candidates = [CandidateNoveltyInput::new(
    "doc-7",
    ["new memory fact".to_string()],
)];
let reference = vec!["existing memory fact".to_string()];
let scored = adapter.score_candidates(&candidates, &reference).await?;
# let _ = scored;
# Ok(()) }
```

### Memvid example

The repository includes committed tiny corpus, memory-card, and qrels fixtures
that run the generic retrieval harness against a temporary `rig-memvid` archive:

```sh
cargo run --example eval_memvid --features memvid-example
```

The example prints current `MultiReport`s, pre/post shadow deltas, and
knowledge-gain summaries with ranked candidate documents for two paths. The
first evaluates raw frame retrieval through `MemvidStore`; the second evaluates
structured/domain-memory facts through `MemoryCardContext`. Logical fixture ids
are remapped into the id space returned by each retriever, keeping the crate's
public API generic over `VectorStoreIndexDyn` while proving both Memvid
integration paths end to end.

### Behavior harnesses

The `memory`, `models`, and `agents` features add small runner-driven harnesses
for surfaces that are broader than pure retrieval. They do not construct
provider clients or own runtime policy; hosts implement the runner trait for a
real agent/model/memory backend, and the harness grades the captured
observation into the same `MetricReport` layer used by retrieval.

```rust,no_run
# use std::future::Future;
# use std::pin::Pin;
# use rig_retrieval_evals::{ModelBehaviorHarness, ModelBehaviorTask, ModelBehaviorTaskSet, ModelObservation, ModelRunner, Result};
# struct Runner;
# impl ModelRunner for Runner {
#     fn run<'a>(&'a self, _: &'a ModelBehaviorTask) -> Pin<Box<dyn Future<Output = Result<ModelObservation>> + Send + 'a>> {
#         Box::pin(async { Ok(ModelObservation { output: "{\"answer\":\"ok\"}".to_string(), output_tokens: Some(6), ..Default::default() }) })
#     }
# }
# async fn demo() -> Result<()> {
let mut suite = ModelBehaviorTaskSet::new("json-smoke.v1");
suite.push(
    ModelBehaviorTask::new("q1", "answer as JSON")
        .requiring_json()
        .with_max_output_tokens(32),
);

let report = ModelBehaviorHarness::new(Runner).run(&suite).await?;
println!("mean={:.3}", report.mean_score);
# Ok(()) }
```

### Repeated-trial reliability

When a retriever, RAG pipeline, or judge is stochastic, run the same suite
multiple times and aggregate the resulting `MetricReport`s into pass@k and
pass^k estimates:

```rust,no_run
# use rig_retrieval_evals::{MetricReport, ReliabilityReport};
# fn demo(trials: Vec<MetricReport>) -> anyhow::Result<()> {
let reliability = ReliabilityReport::from_metric_reports(
    "recall@10",
    1.0, // score threshold counted as a pass
    3,   // k attempts
    &trials,
)?;

println!("pass@3={:.3}", reliability.pass_at_k);
println!("pass^3={:.3}", reliability.pass_all_k);
# Ok(()) }
```

`pass@k` estimates whether at least one of `k` attempts succeeds; `pass^k`
estimates whether all `k` attempts succeed. The same helper works for pure
retrieval reports and `ragas` judge reports because it operates on the shared
report layer.

## Optional ingestion checks

The ingestion feature family moves quality control upstream of vector-store
commit. Instead of storing every chunk and hoping retrieval compensates later,
the pipeline emits an `IngestionDelta` containing net-new IoCs, propositions,
and graph triples plus structured drop reasons for duplicates or redundant
facts. The same feature includes `lint_chunks` for pre-embedding corpus shape
checks; it now flags empty/tiny/giant/duplicate chunks, missing IDs, control
characters, byte-order marks, optional `whatlang` language allow-list
violations through `LanguageLintConfig`, and opt-in MinHash-style near
duplicates through `NearDuplicateLintConfig`.

For cheap ingestion-delta scoring, `jaccard_knowledge_gain` and
`corpus_jaccard_knowledge_gain` compute lexical novelty over normalized token
sets. The resulting score can be carried on `IngestionDelta::knowledge_gain`
with `with_knowledge_gain`.

Deterministic extractors and baselines are the CI path. `LlmTripleExtractor`
and `LlmPropositionExtractor` adapt Rig's structured `Extractor` for hosts that
want model-backed extraction; their contract tests use a fake `CompletionModel`
so validation does not depend on a specific provider or local model. The ignored
`live_ollama_ingestion` test remains a manual smoke test for tool-capable local
models.

## Dataset format

`qrels.jsonl`, one query per line, BEIR-compatible semantics:

```jsonl
{"query_id":"q1","query":"who wrote 1984?","relevant_docs":{"doc-orwell":2,"doc-1984":1}}
{"query_id":"q2","query":"…","relevant_docs":{"doc-7":1},"reference_answer":"…"}
```

Grades are integers in `1..=N`; documents not listed are non-relevant
(grade 0). The optional `reference_answer` field is used by answer-level judges
when the `ragas` feature is enabled.

## License

Dual-licensed under either:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
