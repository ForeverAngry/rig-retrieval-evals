//! [`DistillationPipeline`] — orchestrates ingestion tracks.
//!
//! Track 1 (IoCs) is always active. Track 2 (knowledge-graph triples) and
//! Track 3 (propositions) are opt-in via
//! [`DistillationPipeline::with_graph`] and
//! [`DistillationPipeline::with_propositions`], each implemented as a
//! type-state so an unconfigured pipeline costs zero at runtime and no
//! trait objects are allocated.

use std::future::Future;

use crate::error::Result;

use super::graph::{GraphBaseline, TripleExtractor};
use super::ioc::{IocBaseline, IocExtractor};
use super::proposition::{PropositionExtractor, RedundancyCheck};
use super::types::{Document, Dropped, DroppedItem, DroppedReason, IngestionDelta};

/// Pluggable Track 3 implementation. The default
/// ([`NoPropositionTrack`]) is a no-op; calling
/// [`DistillationPipeline::with_propositions`] swaps in an active stage.
///
/// Hosts compose Track 3 through the
/// [`PropositionExtractor`] + [`RedundancyCheck`] traits and should not
/// implement `PropositionTrack` directly.
pub trait PropositionTrack: Send + Sync {
    /// Run the propositions stage against `doc`, populating `delta`.
    fn run<'a>(
        &'a self,
        doc: &'a Document,
        delta: &'a mut IngestionDelta,
    ) -> impl Future<Output = Result<()>> + Send + 'a;
}

/// Track-3-disabled marker. Default for [`DistillationPipeline`] —
/// pipelines built with [`DistillationPipeline::new`] use this and only
/// run Track 1.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoPropositionTrack;

impl PropositionTrack for NoPropositionTrack {
    async fn run<'a>(&'a self, _doc: &'a Document, _delta: &'a mut IngestionDelta) -> Result<()> {
        Ok(())
    }
}

/// Track-3-enabled stage. Pairs a [`PropositionExtractor`] with a
/// [`RedundancyCheck`]. Built by
/// [`DistillationPipeline::with_propositions`].
#[derive(Debug)]
pub struct ActivePropositionTrack<P, R> {
    extractor: P,
    redundancy: R,
}

impl<P, R> ActivePropositionTrack<P, R> {
    /// Borrow the configured extractor. Primarily useful for tests.
    pub fn extractor(&self) -> &P {
        &self.extractor
    }

    /// Borrow the configured redundancy check.
    pub fn redundancy(&self) -> &R {
        &self.redundancy
    }
}

impl<P, R> PropositionTrack for ActivePropositionTrack<P, R>
where
    P: PropositionExtractor,
    R: RedundancyCheck,
{
    async fn run<'a>(&'a self, doc: &'a Document, delta: &'a mut IngestionDelta) -> Result<()> {
        let candidates = self.extractor.extract(doc).await?;
        tracing::debug!(
            target: "rig_retrieval_evals::ingestion::pipeline",
            doc_id = %doc.id,
            candidate_count = candidates.len(),
            "track 3: candidate propositions extracted"
        );
        for prop in candidates {
            let verdict = self.redundancy.check(&prop).await?;
            if verdict.is_redundant {
                delta.dropped.push(Dropped {
                    item: DroppedItem::Proposition(prop),
                    reason: DroppedReason::Redundant {
                        similarity: verdict.similarity,
                    },
                });
            } else {
                delta.propositions.push(prop);
            }
        }
        tracing::debug!(
            target: "rig_retrieval_evals::ingestion::pipeline",
            doc_id = %doc.id,
            new = delta.propositions.len(),
            "track 3: delta updated"
        );
        Ok(())
    }
}

/// Drives every configured ingestion track and returns the resulting
/// [`IngestionDelta`].
///
/// Construct with [`DistillationPipeline::new`] for Track-1-only; layer
/// Track 2 with [`DistillationPipeline::with_graph`] and Track 3 with
/// [`DistillationPipeline::with_propositions`]. The pipeline is generic
/// over track implementations so callers swap in domain-specific
/// extractors and baselines without trait objects.
///
/// The pipeline is `Send + Sync` whenever its tracks are, so it can be
/// shared across tasks under an `Arc`.
#[derive(Debug)]
pub struct DistillationPipeline<E, B, T = NoPropositionTrack, G = NoGraphTrack> {
    extractor: E,
    baseline: B,
    propositions: T,
    graph: G,
}

impl<E, B> DistillationPipeline<E, B, NoPropositionTrack, NoGraphTrack> {
    /// Build a Track-1-only pipeline from an IoC extractor and baseline.
    pub fn new(extractor: E, baseline: B) -> Self {
        Self {
            extractor,
            baseline,
            propositions: NoPropositionTrack,
            graph: NoGraphTrack,
        }
    }
}

impl<E, B, T, G> DistillationPipeline<E, B, T, G> {
    /// Layer Track 3 (propositional distillation) onto the pipeline.
    /// Returns a new pipeline value with the propositions stage enabled;
    /// existing Track 1 and Track 2 configuration is preserved.
    pub fn with_propositions<P, R>(
        self,
        extractor: P,
        redundancy: R,
    ) -> DistillationPipeline<E, B, ActivePropositionTrack<P, R>, G>
    where
        P: PropositionExtractor,
        R: RedundancyCheck,
    {
        DistillationPipeline {
            extractor: self.extractor,
            baseline: self.baseline,
            propositions: ActivePropositionTrack {
                extractor,
                redundancy,
            },
            graph: self.graph,
        }
    }

    /// Layer Track 2 (knowledge-graph distillation) onto the pipeline.
    /// Returns a new pipeline value with the graph stage enabled;
    /// existing Track 1 and Track 3 configuration is preserved.
    pub fn with_graph<X, GB>(
        self,
        extractor: X,
        baseline: GB,
    ) -> DistillationPipeline<E, B, T, ActiveGraphTrack<X, GB>>
    where
        X: TripleExtractor,
        GB: GraphBaseline,
    {
        DistillationPipeline {
            extractor: self.extractor,
            baseline: self.baseline,
            propositions: self.propositions,
            graph: ActiveGraphTrack {
                extractor,
                baseline,
            },
        }
    }

    /// Borrow the configured Track 1 extractor. Primarily useful for tests.
    pub fn extractor(&self) -> &E {
        &self.extractor
    }

    /// Borrow the configured Track 1 baseline. Primarily useful for tests
    /// and to let callers commit net-new IoCs back into the same store.
    pub fn baseline(&self) -> &B {
        &self.baseline
    }

    /// Borrow the configured Track 3 stage. Useful for tests that need to
    /// inspect the active extractor / redundancy check.
    pub fn propositions(&self) -> &T {
        &self.propositions
    }

    /// Borrow the configured Track 2 stage. Useful for tests that need
    /// to commit net-new triples back into the same baseline.
    pub fn graph(&self) -> &G {
        &self.graph
    }
}

impl<E, B, T, G> DistillationPipeline<E, B, T, G>
where
    E: IocExtractor + Send + Sync,
    B: IocBaseline,
    T: PropositionTrack,
    G: GraphTrack,
{
    /// Run every configured track against `doc` and return the
    /// [`IngestionDelta`]. Tracks execute in declaration order; later PRs
    /// may parallelise independent tracks via `futures::join!`.
    pub async fn ingest(&self, doc: &Document) -> Result<IngestionDelta> {
        let mut delta = IngestionDelta::new();

        // Track 1: deterministic IoC set difference.
        let candidates = self.extractor.extract(doc);
        tracing::debug!(
            target: "rig_retrieval_evals::ingestion::pipeline",
            doc_id = %doc.id,
            candidate_count = candidates.len(),
            "track 1: candidate IoCs extracted"
        );
        for ioc in candidates {
            if self.baseline.contains(&ioc).await? {
                delta.dropped.push(Dropped {
                    item: DroppedItem::Ioc(ioc),
                    reason: DroppedReason::DuplicateIoc,
                });
            } else {
                delta.iocs.push(ioc);
            }
        }
        tracing::debug!(
            target: "rig_retrieval_evals::ingestion::pipeline",
            doc_id = %doc.id,
            new = delta.iocs.len(),
            dropped = delta.dropped.len(),
            "track 1: delta computed"
        );

        // Track 2: knowledge-graph edge set difference (no-op when disabled).
        self.graph.run(doc, &mut delta).await?;

        // Track 3: propositional distillation (no-op when disabled).
        self.propositions.run(doc, &mut delta).await?;

        Ok(delta)
    }
}

/// Pluggable Track 2 implementation. The default ([`NoGraphTrack`]) is
/// a no-op; calling [`DistillationPipeline::with_graph`] swaps in an
/// active stage.
///
/// Hosts compose Track 2 through the [`TripleExtractor`] +
/// [`GraphBaseline`] traits and should not implement `GraphTrack`
/// directly.
pub trait GraphTrack: Send + Sync {
    /// Run the graph stage against `doc`, populating `delta`.
    fn run<'a>(
        &'a self,
        doc: &'a Document,
        delta: &'a mut IngestionDelta,
    ) -> impl Future<Output = Result<()>> + Send + 'a;
}

/// Track-2-disabled marker. Default for [`DistillationPipeline`] —
/// pipelines built with [`DistillationPipeline::new`] use this and skip
/// Track 2.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoGraphTrack;

impl GraphTrack for NoGraphTrack {
    async fn run<'a>(&'a self, _doc: &'a Document, _delta: &'a mut IngestionDelta) -> Result<()> {
        Ok(())
    }
}

/// Track-2-enabled stage. Pairs a [`TripleExtractor`] with a
/// [`GraphBaseline`]. Built by [`DistillationPipeline::with_graph`].
#[derive(Debug)]
pub struct ActiveGraphTrack<X, B> {
    extractor: X,
    baseline: B,
}

impl<X, B> ActiveGraphTrack<X, B> {
    /// Borrow the configured extractor. Primarily useful for tests.
    pub fn extractor(&self) -> &X {
        &self.extractor
    }

    /// Borrow the configured graph baseline.
    pub fn baseline(&self) -> &B {
        &self.baseline
    }
}

impl<X, B> GraphTrack for ActiveGraphTrack<X, B>
where
    X: TripleExtractor,
    B: GraphBaseline,
{
    async fn run<'a>(&'a self, doc: &'a Document, delta: &'a mut IngestionDelta) -> Result<()> {
        let candidates = self.extractor.extract(doc).await?;
        tracing::debug!(
            target: "rig_retrieval_evals::ingestion::pipeline",
            doc_id = %doc.id,
            candidate_count = candidates.len(),
            "track 2: candidate triples extracted"
        );
        for triple in candidates {
            if self.baseline.contains(&triple).await? {
                delta.dropped.push(Dropped {
                    item: DroppedItem::Triple(triple),
                    reason: DroppedReason::DuplicateEdge,
                });
            } else {
                delta.triples.push(triple);
            }
        }
        tracing::debug!(
            target: "rig_retrieval_evals::ingestion::pipeline",
            doc_id = %doc.id,
            new = delta.triples.len(),
            "track 2: delta updated"
        );
        Ok(())
    }
}
