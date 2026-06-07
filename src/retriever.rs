//! A retriever abstraction over anything that maps a query to a ranked list
//! of documents — not just vector stores.
//!
//! [`crate::harness::RetrievalHarness`] drives a
//! [`rig::vector_store::VectorStoreIndexDyn`] directly, which is the common
//! case. But lexical engines (ripgrep / BM25), hybrid rerankers, and remote
//! search APIs do not implement that trait, yet should be scored with the
//! exact same IR metrics. The [`Retriever`] trait is that seam: implement it
//! for any backend and hand it to [`score_retriever`] to get a
//! [`MultiReport`] keyed by metric.
//!
//! Vector stores plug in for free via [`VectorStoreRetriever`].
//!
//! ```no_run
//! use rig_retrieval_evals::{
//!     dataset::Qrels,
//!     retrieval::{NdcgAtK, RecallAtK, RetrievalMetric},
//!     retriever::{score_retriever, VectorStoreRetriever},
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
//! let retriever = VectorStoreRetriever::new(&store, "memvid");
//! let report = score_retriever(&retriever, &qrels, 10, &metrics, 4).await?;
//! println!("{}", report.to_markdown());
//! # Ok(()) }
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use futures::stream::{self, StreamExt};
use rig::vector_store::{VectorSearchRequest, VectorStoreIndexDyn, request::Filter};

use crate::dataset::{Qrels, RetrievedDoc, RetrievedSet};
use crate::error::{Error, Result};
use crate::report::{MetricReport, MultiReport};
use crate::retrieval::RetrievalMetric;

/// Boxed, `Send` future returned by [`Retriever::retrieve`].
pub type RetrieveFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<RetrievedDoc>>> + Send + 'a>>;

/// Maps a natural-language query to a ranked list of documents.
///
/// This is the runtime-agnostic generalization of
/// [`rig::vector_store::VectorStoreIndexDyn`]: a vector store is one
/// implementation (via [`VectorStoreRetriever`]), but lexical search engines,
/// hybrid rerankers, and remote APIs can implement it too and be scored with
/// the same [`RetrievalMetric`]s.
pub trait Retriever: Send + Sync {
    /// Short identifier used as the report's store label.
    fn name(&self) -> &str;

    /// Return up to `k` documents ranked best-first for `query`.
    fn retrieve<'a>(&'a self, query: &'a str, k: usize) -> RetrieveFuture<'a>;
}

/// Adapter exposing any [`VectorStoreIndexDyn`] as a [`Retriever`].
///
/// The wrapped store's `top_n_ids` results are mapped one-to-one to
/// [`RetrievedDoc`]s, so scoring is identical to
/// [`crate::harness::RetrievalHarness`] — this adapter is in fact what the
/// harness uses internally.
pub struct VectorStoreRetriever<'s> {
    store: &'s dyn VectorStoreIndexDyn,
    name: String,
}

impl<'s> VectorStoreRetriever<'s> {
    /// Wrap `store`, labelling its reports with `name`.
    pub fn new(store: &'s dyn VectorStoreIndexDyn, name: impl Into<String>) -> Self {
        Self {
            store,
            name: name.into(),
        }
    }
}

impl Retriever for VectorStoreRetriever<'_> {
    fn name(&self) -> &str {
        &self.name
    }

    fn retrieve<'a>(&'a self, query: &'a str, k: usize) -> RetrieveFuture<'a> {
        Box::pin(async move {
            let req: VectorSearchRequest<Filter<serde_json::Value>> =
                VectorSearchRequest::builder()
                    .query(query.to_string())
                    .samples(k as u64)
                    .build();
            let hits = self.store.top_n_ids(req).await?;
            Ok(hits
                .into_iter()
                .map(|(score, doc_id)| RetrievedDoc { doc_id, score })
                .collect())
        })
    }
}

/// Retrieve top-`k` hits for every gold query, returning one
/// [`RetrievedSet`] per query in input order. Errors from individual
/// retrievals short-circuit the run. `concurrency` is clamped to at least 1.
pub async fn retrieve_all(
    retriever: &dyn Retriever,
    qrels: &Qrels,
    k: usize,
    concurrency: usize,
) -> Result<Vec<RetrievedSet>> {
    let results: Vec<Result<RetrievedSet>> =
        stream::iter(qrels.queries.iter().map(|q| async move {
            let ranked = retriever.retrieve(&q.query, k).await?;
            Ok(RetrievedSet {
                query_id: q.query_id.clone(),
                ranked,
            })
        }))
        .buffered(concurrency.max(1))
        .collect()
        .await;
    results.into_iter().collect()
}

/// Score `retriever` against `qrels`, producing a [`MultiReport`] with one
/// [`MetricReport`] per metric.
///
/// Returns `Err(Error::Config)` if `k == 0`. `concurrency` is the maximum
/// number of in-flight retrievals and is clamped to at least 1.
pub async fn score_retriever(
    retriever: &dyn Retriever,
    qrels: &Qrels,
    k: usize,
    metrics: &[Box<dyn RetrievalMetric>],
    concurrency: usize,
) -> Result<MultiReport> {
    if k == 0 {
        return Err(Error::Config("top-k must be > 0".into()));
    }

    let retrievals = retrieve_all(retriever, qrels, k, concurrency).await?;
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
                continue;
            };
            per_query.push((q.query_id.clone(), metric.score(q, retrieved)));
        }
        reports.push(MetricReport::from_per_query(name, per_query));
    }

    Ok(MultiReport::new(reports))
}
