//! Async driver that runs a set of [`DynRagasMetric`] judges over a list of
//! [`RagasInputs`] samples and bundles the result into a
//! [`crate::report::MultiReport`].
//!
//! Why a separate harness?  The retrieval harness in [`crate::harness`]
//! drives `VectorStoreIndexDyn` and only knows about doc-id gold labels.
//! RAGAS judges need the *generated answer text* and the *retrieved chunk
//! text*, neither of which the retrieval store exposes — those come from
//! a caller-supplied pipeline. Keeping the two drivers separate avoids
//! conflating the contracts.
//!
//! ```no_run
//! # async fn demo<M, E>(model: M, embedder: E) -> Result<(), rig_retrieval_evals::Error>
//! # where
//! #   M: rig::completion::CompletionModel + Clone + Send + Sync + 'static,
//! #   E: rig::embeddings::EmbeddingModel + Clone + Send + Sync + 'static,
//! # {
//! use rig_retrieval_evals::ragas::{
//!     AnswerRelevanceMetric, DynRagasMetric, FaithfulnessMetric, RagasHarness, RagasInputs,
//! };
//!
//! let samples = vec![RagasInputs::new(
//!     "q1",
//!     "Who wrote the Rig framework?",
//!     "Rig is maintained by 0xPlaygrounds.",
//!     vec!["Rig is a Rust agent framework by 0xPlaygrounds.".to_string()],
//! )];
//!
//! let metrics: Vec<Box<dyn DynRagasMetric>> = vec![
//!     Box::new(FaithfulnessMetric::new(model.clone(), "ollama:qwen3.5:9b@v1")),
//!     Box::new(AnswerRelevanceMetric::new(model, embedder, 3, "ollama:qwen3.5:9b@v1")?),
//! ];
//!
//! let report = RagasHarness::new().with_concurrency(2).run(&samples, &metrics).await?;
//! println!("{}", report.to_markdown());
//! # Ok(()) }
//! ```

use futures::stream::{self, StreamExt};
use tracing::{instrument, warn};

use crate::error::Result;
use crate::ragas::{DynRagasMetric, RagasInputs};
use crate::report::{MetricReport, MultiReport};

/// Async driver for a heterogeneous set of [`DynRagasMetric`] judges.
#[derive(Debug, Default)]
pub struct RagasHarness {
    sample_concurrency: usize,
    dataset_id: Option<String>,
}

impl RagasHarness {
    /// Construct a sequential harness (concurrency = 1).
    #[must_use]
    pub fn new() -> Self {
        Self {
            sample_concurrency: 1,
            dataset_id: None,
        }
    }

    /// Maximum number of in-flight per-sample judge invocations across all
    /// metrics. Defaults to `1`. Values of `0` are clamped to `1`.
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.sample_concurrency = concurrency.max(1);
        self
    }

    /// Attach a dataset identifier surfaced in the final [`MultiReport`].
    #[must_use]
    pub fn with_dataset(mut self, id: impl Into<String>) -> Self {
        self.dataset_id = Some(id.into());
        self
    }

    /// Run every metric against every sample and aggregate per-metric
    /// statistics into a [`MultiReport`]. Samples that yield
    /// [`RagasScore::not_measurable`][nm] are recorded in the per-metric
    /// rationales but excluded from the aggregate average.
    ///
    /// The returned report carries a `judge_fingerprint` composed of every
    /// metric's [`DynRagasMetric::fingerprint_component`], so diffs against
    /// a baseline produced by a different judge configuration will
    /// (correctly) refuse to compare.
    ///
    /// [nm]: crate::ragas::RagasScore::not_measurable
    #[instrument(skip_all, fields(
        evals.samples = samples.len(),
        evals.metrics = metrics.len(),
        evals.concurrency = self.sample_concurrency,
    ))]
    pub async fn run(
        &self,
        samples: &[RagasInputs],
        metrics: &[Box<dyn DynRagasMetric>],
    ) -> Result<MultiReport> {
        let mut reports: Vec<MetricReport> = Vec::with_capacity(metrics.len());
        let mut fingerprint_parts: Vec<String> = Vec::with_capacity(metrics.len());

        for metric in metrics {
            fingerprint_parts.push(metric.fingerprint_component());

            // Fan out per-sample scoring with bounded concurrency.
            let scored: Vec<Result<(String, Option<f64>)>> =
                stream::iter(samples.iter().map(|s| async move {
                    let outcome = metric.score(s).await?;
                    if let Some(reason) = outcome.rationales.first()
                        && outcome.value.is_none()
                    {
                        warn!(
                            metric = metric.name(),
                            query_id = %s.query_id,
                            reason = %reason,
                            "sample not measurable",
                        );
                    }
                    Ok::<(String, Option<f64>), crate::error::Error>((
                        s.query_id.clone(),
                        outcome.value,
                    ))
                }))
                .buffered(self.sample_concurrency)
                .collect()
                .await;

            let mut per_query: Vec<(String, f64)> = Vec::with_capacity(samples.len());
            for r in scored {
                let (id, maybe) = r?;
                if let Some(v) = maybe {
                    per_query.push((id, v));
                }
            }
            reports.push(MetricReport::from_per_query(
                metric.name().to_string(),
                per_query,
            ));
        }

        fingerprint_parts.sort();
        let fingerprint = fingerprint_parts.join("|");

        let mut report = MultiReport::new(reports).with_judge_fingerprint(fingerprint);
        if let Some(id) = &self.dataset_id {
            report = report.with_dataset(id.clone());
        }
        Ok(report)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::ragas::{DynRagasMetric, RagasMetric, RagasScore};
    use std::future::Future;
    use std::sync::Mutex;

    /// Deterministic stub metric: returns a pre-canned score per query id.
    struct StubMetric {
        name: &'static str,
        fp: String,
        scores: Mutex<std::collections::HashMap<String, RagasScore>>,
    }

    impl StubMetric {
        fn new(name: &'static str, fp: &str, scores: Vec<(&str, RagasScore)>) -> Self {
            let map = scores
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            Self {
                name,
                fp: fp.to_string(),
                scores: Mutex::new(map),
            }
        }
    }

    impl RagasMetric for StubMetric {
        fn name(&self) -> &'static str {
            self.name
        }
        fn fingerprint_component(&self) -> String {
            self.fp.clone()
        }
        fn score(&self, inputs: &RagasInputs) -> impl Future<Output = Result<RagasScore>> + Send {
            let res = {
                let guard = self.scores.lock().unwrap();
                guard
                    .get(&inputs.query_id)
                    .cloned()
                    .unwrap_or_else(|| RagasScore::not_measurable("no fixture for query"))
            };
            async move { Ok(res) }
        }
    }

    fn sample(id: &str) -> RagasInputs {
        RagasInputs::new(id, "q", "a", vec!["c".into()])
    }

    #[tokio::test]
    async fn harness_aggregates_measured_samples_only() {
        let metric: Box<dyn DynRagasMetric> = Box::new(StubMetric::new(
            "faithfulness",
            "stub:v1",
            vec![
                ("q1", RagasScore::measured(1.0)),
                ("q2", RagasScore::measured(0.5)),
                ("q3", RagasScore::not_measurable("skip")),
            ],
        ));
        let samples = vec![sample("q1"), sample("q2"), sample("q3")];
        let report = RagasHarness::new()
            .with_concurrency(2)
            .run(&samples, std::slice::from_ref(&metric))
            .await
            .unwrap();
        assert_eq!(report.metrics.len(), 1);
        let m = &report.metrics[0];
        assert_eq!(m.metric, "faithfulness");
        assert_eq!(m.n, 2);
        assert!((m.mean - 0.75).abs() < 1e-9);
        assert_eq!(
            report.judge_fingerprint.as_deref(),
            Some("stub:v1"),
            "fingerprint should be the sorted join of components",
        );
    }

    #[tokio::test]
    async fn harness_combines_fingerprints_across_metrics() {
        let a: Box<dyn DynRagasMetric> = Box::new(StubMetric::new(
            "a",
            "B:fp",
            vec![("q1", RagasScore::measured(1.0))],
        ));
        let b: Box<dyn DynRagasMetric> = Box::new(StubMetric::new(
            "b",
            "A:fp",
            vec![("q1", RagasScore::measured(0.0))],
        ));
        let report = RagasHarness::new()
            .run(&[sample("q1")], &[a, b])
            .await
            .unwrap();
        // Components are sorted before joining → "A:fp|B:fp".
        assert_eq!(report.judge_fingerprint.as_deref(), Some("A:fp|B:fp"));
    }

    #[tokio::test]
    async fn harness_handles_empty_metric_set() {
        let report = RagasHarness::new().run(&[sample("q1")], &[]).await.unwrap();
        assert!(report.metrics.is_empty());
        assert_eq!(report.judge_fingerprint.as_deref(), Some(""));
    }
}
