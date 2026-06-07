//! Async driver that executes a [`Qrels`] against a
//! [`VectorStoreIndexDyn`] and aggregates per-query metric scores.
//!
//! ```no_run
//! use rig_retrieval_evals::{
//!     dataset::Qrels,
//!     harness::RetrievalHarness,
//!     retrieval::{NdcgAtK, RecallAtK, RetrievalMetric},
//! };
//!
//! # async fn run<I>(store: I) -> Result<(), rig_retrieval_evals::Error>
//! # where
//! #   I: rig::vector_store::VectorStoreIndexDyn + 'static,
//! # {
//! let qrels = Qrels::load_jsonl("tests/data/tiny_qrels.jsonl")?;
//! let metrics: Vec<Box<dyn RetrievalMetric>> = vec![
//!     Box::new(RecallAtK::new(10)),
//!     Box::new(NdcgAtK::new(10)),
//! ];
//! let report = RetrievalHarness::new(&store, 10)
//!     .with_concurrency(4)
//!     .run(&qrels, &metrics)
//!     .await?;
//! println!("{}", report.to_markdown());
//! # Ok(()) }
//! ```

use rig::vector_store::VectorStoreIndexDyn;
use tracing::instrument;

use crate::dataset::{Qrels, RetrievedSet};
use crate::error::{Error, Result};
use crate::report::MultiReport;
use crate::retrieval::RetrievalMetric;
use crate::retriever::{VectorStoreRetriever, retrieve_all, score_retriever};

/// Async driver that retrieves top-k hits per gold query and scores them
/// with a set of [`RetrievalMetric`]s.
///
/// The harness is generic over any [`VectorStoreIndexDyn`] so the same code
/// drives `rig`'s in-memory store, `rig-memvid`, `rig-lancedb`, or anything
/// else that implements the trait. To score a backend that is *not* a vector
/// store (a lexical / BM25 engine, a hybrid reranker, a remote search API),
/// implement [`crate::retriever::Retriever`] and call
/// [`crate::retriever::score_retriever`] directly.
pub struct RetrievalHarness<'s> {
    store: &'s dyn VectorStoreIndexDyn,
    k: usize,
    concurrency: usize,
    bootstrap: Option<(usize, f64, u64)>,
}

impl<'s> RetrievalHarness<'s> {
    /// Build a harness against `store` that retrieves the top `k` hits per
    /// query.
    ///
    /// Returns `Err(Error::Config)` if `k == 0`.
    pub fn new(store: &'s dyn VectorStoreIndexDyn, k: usize) -> Self {
        Self {
            store,
            k,
            concurrency: 1,
            bootstrap: None,
        }
    }

    /// Set the maximum number of concurrent in-flight retrievals. Defaults
    /// to `1` (sequential). Values of `0` are clamped to `1`.
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Attach a deterministic percentile-bootstrap confidence interval to
    /// every metric in the produced [`MultiReport`].
    ///
    /// `iterations` resamples are drawn per metric using a SplitMix64 stream
    /// seeded by `seed`, and the two-sided interval is reported at `level`
    /// (e.g. `0.95`). The same inputs always yield the same intervals, so CI
    /// gates and reproducibility tests need no extra fixtures. Passing
    /// `iterations == 0` disables the CI again.
    #[must_use]
    pub fn with_bootstrap(mut self, iterations: usize, level: f64, seed: u64) -> Self {
        self.bootstrap = if iterations == 0 {
            None
        } else {
            Some((iterations, level, seed))
        };
        self
    }

    /// Configured top-k.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Run every gold query in `qrels` through the store, then score each
    /// retrieval against every metric in `metrics`. Returns a
    /// [`MultiReport`] keyed by metric name.
    #[instrument(skip_all, fields(evals.k = self.k, evals.queries = qrels.len(), evals.metrics = metrics.len()))]
    pub async fn run(
        &self,
        qrels: &Qrels,
        metrics: &[Box<dyn RetrievalMetric>],
    ) -> Result<MultiReport> {
        if self.k == 0 {
            return Err(Error::Config("top-k must be > 0".into()));
        }

        let retriever = VectorStoreRetriever::new(self.store, "vector-store");
        let report = score_retriever(&retriever, qrels, self.k, metrics, self.concurrency).await?;
        Ok(match self.bootstrap {
            Some((iterations, level, seed)) => report.with_bootstrap(iterations, level, seed),
            None => report,
        })
    }

    /// Retrieve top-k hits for every gold query, returning one
    /// [`RetrievedSet`] per query in input order. Errors from individual
    /// retrievals short-circuit the run.
    pub async fn retrieve_all(&self, qrels: &Qrels) -> Result<Vec<RetrievedSet>> {
        let retriever = VectorStoreRetriever::new(self.store, "vector-store");
        retrieve_all(&retriever, qrels, self.k, self.concurrency).await
    }
}
