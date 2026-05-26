//! Pre/post retrieval scoring over two store snapshots.
//!
//! `EvalShadowStore` is intentionally small: callers provide a baseline
//! retriever and a candidate/current retriever, both exposed as
//! [`rig::vector_store::VectorStoreIndexDyn`]. The shadow runner executes the
//! same qrels and metrics against both snapshots and returns the two reports
//! plus their [`crate::ReportDiff`].
//!
//! ```no_run
//! use rig::vector_store::VectorStoreIndexDyn;
//! use rig_retrieval_evals::{EvalShadowStore, Qrels, RecallAtK, RetrievalMetric};
//!
//! # async fn run(
//! #     before: &dyn VectorStoreIndexDyn,
//! #     after: &dyn VectorStoreIndexDyn,
//! #     qrels: &Qrels,
//! # ) -> Result<(), rig_retrieval_evals::Error> {
//! let metrics: Vec<Box<dyn RetrievalMetric>> = vec![Box::new(RecallAtK::new(5))];
//! let report = EvalShadowStore::new(before, after, 5)
//!     .with_concurrency(2)
//!     .run(qrels, &metrics)
//!     .await?;
//!
//! println!("{}", report.diff.to_markdown());
//! # Ok(()) }
//! ```

use rig::vector_store::VectorStoreIndexDyn;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::dataset::Qrels;
use crate::error::Result;
use crate::harness::RetrievalHarness;
use crate::report::{MultiReport, ReportDiff};
use crate::retrieval::RetrievalMetric;

/// Compare retrieval quality between a baseline store and a candidate store.
///
/// This type does not mutate either store. Hosts are responsible for preparing
/// the two snapshots, for example by cloning a KB, ingesting candidate material
/// into the clone, then passing `(before, after)` into this runner.
pub struct EvalShadowStore<'s> {
    baseline: &'s dyn VectorStoreIndexDyn,
    candidate: &'s dyn VectorStoreIndexDyn,
    k: usize,
    concurrency: usize,
}

impl<'s> EvalShadowStore<'s> {
    /// Build a pre/post evaluator that retrieves the top `k` hits from each
    /// store for every query.
    #[must_use]
    pub fn new(
        baseline: &'s dyn VectorStoreIndexDyn,
        candidate: &'s dyn VectorStoreIndexDyn,
        k: usize,
    ) -> Self {
        Self {
            baseline,
            candidate,
            k,
            concurrency: 1,
        }
    }

    /// Set the maximum number of concurrent in-flight retrievals for each
    /// snapshot run. Values of `0` are clamped to `1`.
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Configured top-k.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Run baseline and candidate retrieval evals, then diff candidate against
    /// baseline. Both snapshot runs execute concurrently because they share no
    /// state.
    #[instrument(level = "debug", skip(self, qrels, metrics), fields(k = self.k, concurrency = self.concurrency, queries = qrels.queries.len(), metrics = metrics.len()))]
    pub async fn run(
        &self,
        qrels: &Qrels,
        metrics: &[Box<dyn RetrievalMetric>],
    ) -> Result<ShadowEvalReport> {
        let baseline_harness =
            RetrievalHarness::new(self.baseline, self.k).with_concurrency(self.concurrency);
        let candidate_harness =
            RetrievalHarness::new(self.candidate, self.k).with_concurrency(self.concurrency);
        let baseline_run = baseline_harness.run(qrels, metrics);
        let current_run = candidate_harness.run(qrels, metrics);
        let (baseline, current) = futures::try_join!(baseline_run, current_run)?;
        let diff = current.diff(&baseline)?;
        Ok(ShadowEvalReport {
            baseline,
            current,
            diff,
        })
    }
}

/// Baseline/current reports plus their candidate-minus-baseline diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowEvalReport {
    /// Metrics collected before candidate ingestion or retrieval changes.
    pub baseline: MultiReport,
    /// Metrics collected after candidate ingestion or retrieval changes.
    pub current: MultiReport,
    /// `current.diff(&baseline)`; positive deltas mean the candidate improved
    /// the metric.
    pub diff: ReportDiff,
}

impl ShadowEvalReport {
    /// Attach matching dataset and store identifiers to both baseline and
    /// current reports. The diff is unchanged because it was already computed
    /// from those metric rows.
    #[must_use]
    pub fn with_metadata(mut self, dataset: impl Into<String>, store: impl Into<String>) -> Self {
        let dataset = dataset.into();
        let store = store.into();
        self.baseline = self
            .baseline
            .with_dataset(dataset.clone())
            .with_store(store.clone());
        self.current = self.current.with_dataset(dataset).with_store(store);
        self
    }

    /// Render the three report sections as Markdown.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        format!(
            "## Baseline\n{}\n## Current\n{}\n## Delta\n{}",
            self.baseline.to_markdown(),
            self.current.to_markdown(),
            self.diff.to_markdown()
        )
    }
}
