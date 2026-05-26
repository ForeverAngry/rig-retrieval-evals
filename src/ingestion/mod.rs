//! Zero-waste ingestion pipeline.
//!
//! `rig-retrieval-evals` traditionally measures retrieval quality after documents
//! land in a store. The ingestion pipeline moves that gate upstream: only
//! commit deltas (net-new IoCs, graph edges, propositions) so the vector
//! store never accumulates redundant chunks.
//!
//! The base `ingestion` feature ships:
//!
//! - [`Document`] / [`Section`] — caller-supplied parsed input.
//! - [`IngestionDelta`] + [`Dropped`] / [`DroppedReason`] — inspectable
//!   pipeline output. Every discarded item carries a structured reason.
//! - Track 1 (IoCs): [`IocExtractor`], [`IocBaseline`], [`RegexIocExtractor`],
//!   [`InMemoryIocBaseline`].
//! - Track 2 (knowledge graph, opt-in): [`TripleExtractor`],
//!   [`GraphBaseline`], [`StubTripleExtractor`], [`InMemoryGraphBaseline`].
//!   A `petgraph`-backed [`PetgraphBaseline`] is available behind the
//!   `ingestion-graph` sub-feature.
//! - Track 3 (propositions, opt-in): [`PropositionExtractor`],
//!   [`RedundancyCheck`], [`StubPropositionExtractor`],
//!   [`VectorStoreRedundancyCheck`].
//! - [`DistillationPipeline`] — orchestrator. Runs Track 1 always; layer
//!   Track 2 with [`DistillationPipeline::with_graph`] and Track 3 with
//!   [`DistillationPipeline::with_propositions`].
//!
//! ## Design notes
//!
//! - The crate is store-agnostic: the pipeline returns deltas; the caller
//!   owns commits to their IoC store / graph DB / vector store.
//! - The crate is runtime-agnostic. Track concurrency (when more than one
//!   track ships) uses `futures::join!`, not `tokio::join!`.
//! - Stub extractors live in the library (not `#[cfg(test)]`) so hosts can
//!   use them as deterministic CI gates for their own pipelines.
//!
//! ## Example
//!
//! ```no_run
//! # use rig_retrieval_evals::{
//! #     DistillationPipeline, Document, InMemoryIocBaseline, RegexIocExtractor,
//! # };
//! # async fn demo() -> Result<(), rig_retrieval_evals::Error> {
//! let pipeline = DistillationPipeline::new(
//!     RegexIocExtractor::new()?,
//!     InMemoryIocBaseline::new(),
//! );
//! let doc = Document::new(
//!     "report-1",
//!     "APT-28 exploited CVE-2024-12345 from 192.0.2.10.",
//! );
//! let delta = pipeline.ingest(&doc).await?;
//! assert!(!delta.iocs.is_empty());
//! # Ok(())
//! # }
//! ```

mod graph;
mod ioc;
mod knowledge_gain;
pub mod lint;
mod pipeline;
mod proposition;
mod report;
mod types;

#[cfg(feature = "ingestion-graph")]
pub use graph::PetgraphBaseline;
pub use graph::{
    GraphBaseline, InMemoryGraphBaseline, StubTripleExtractor, Triple, TripleExtractor,
};
pub use ioc::{InMemoryIocBaseline, Ioc, IocBaseline, IocExtractor, IocKind, RegexIocExtractor};
pub use knowledge_gain::{corpus_jaccard_knowledge_gain, jaccard_knowledge_gain};
pub use lint::{
    Chunk, ChunkLintConfig, ChunkLintReport, ChunkLintWarning, ChunkStats, EncodingLintWarning,
    LanguageCount, LanguageLintConfig, lint_chunks, lint_chunks_strict,
};
pub use pipeline::{
    ActiveGraphTrack, ActivePropositionTrack, DistillationPipeline, GraphTrack, NoGraphTrack,
    NoPropositionTrack, PropositionTrack,
};
pub use proposition::{
    Proposition, PropositionExtractor, RedundancyCheck, RedundancyVerdict,
    StubPropositionExtractor, VectorStoreRedundancyCheck,
};
pub use report::{DropTotals, IngestionReport, TrackTotals};
pub use types::{
    Document, Dropped, DroppedItem, DroppedReason, IngestionDelta, Section, SectionKind,
};

pub mod llm;
pub use llm::{LlmPropositionExtractor, LlmTripleExtractor};
