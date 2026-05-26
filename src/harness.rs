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

use std::collections::HashMap;

use futures::stream::{self, StreamExt};
use rig::vector_store::{VectorSearchRequest, VectorStoreIndexDyn, request::Filter};
use tracing::{debug, instrument, warn};

use crate::dataset::{GoldQuery, Qrels, RetrievedDoc, RetrievedSet};
use crate::error::{Error, Result};
use crate::report::{MetricReport, MultiReport};
use crate::retrieval::RetrievalMetric;

/// Async driver that retrieves top-k hits per gold query and scores them
/// with a set of [`RetrievalMetric`]s.
///
/// The harness is generic over any [`VectorStoreIndexDyn`] so the same code
/// drives `rig`'s in-memory store, `rig-memvid`, `rig-lancedb`, or anything
/// else that implements the trait.
pub struct RetrievalHarness<'s> {
    store: &'s dyn VectorStoreIndexDyn,
    k: usize,
    concurrency: usize,
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
        }
    }

    /// Set the maximum number of concurrent in-flight retrievals. Defaults
    /// to `1` (sequential). Values of `0` are clamped to `1`.
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

        let retrievals = self.retrieve_all(qrels).await?;
        debug!(
            retrieved = retrievals.len(),
            "scoring retrievals against metrics"
        );

        let by_query: HashMap<&str, &RetrievedSet> = retrievals
            .iter()
            .map(|r| (r.query_id.as_str(), r))
            .collect();

        let mut reports: Vec<MetricReport> = Vec::with_capacity(metrics.len());
        for metric in metrics {
            let name = metric.name();
            let mut per_query = Vec::with_capacity(qrels.queries.len());
            for q in &qrels.queries {
                let Some(retrieved) = by_query.get(q.query_id.as_str()) else {
                    warn!(query_id = %q.query_id, "no retrieval recorded; skipping");
                    continue;
                };
                let score = metric.score(q, retrieved);
                per_query.push((q.query_id.clone(), score));
            }
            reports.push(MetricReport::from_per_query(name, per_query));
        }

        Ok(MultiReport::new(reports))
    }

    /// Retrieve top-k hits for every gold query, returning one
    /// [`RetrievedSet`] per query in input order. Errors from individual
    /// retrievals short-circuit the run.
    pub async fn retrieve_all(&self, qrels: &Qrels) -> Result<Vec<RetrievedSet>> {
        let k = self.k;
        let store = self.store;
        let results: Vec<Result<RetrievedSet>> =
            stream::iter(qrels.queries.iter().map(|q| run_one(store, q, k)))
                .buffered(self.concurrency)
                .collect()
                .await;
        results.into_iter().collect()
    }
}

async fn run_one(
    store: &dyn VectorStoreIndexDyn,
    gold: &GoldQuery,
    k: usize,
) -> Result<RetrievedSet> {
    let req: VectorSearchRequest<Filter<serde_json::Value>> = VectorSearchRequest::builder()
        .query(gold.query.clone())
        .samples(k as u64)
        .build();
    let hits = store.top_n_ids(req).await?;
    let ranked = hits
        .into_iter()
        .map(|(score, doc_id)| RetrievedDoc { doc_id, score })
        .collect();
    Ok(RetrievedSet {
        query_id: gold.query_id.clone(),
        ranked,
    })
}
