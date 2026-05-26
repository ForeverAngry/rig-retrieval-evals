//! Knowledge-gain scoring from pre/post retrieval deltas.
//!
//! The knowledge-gain surface is intentionally host-owned: it consumes a
//! [`crate::ReportDiff`] produced by pre/post retrieval scoring and turns metric
//! deltas into a single weighted score plus per-query movers. Hosts may also
//! pass candidate document ids with optional novelty scores to rank which new
//! documents explain the gain. Positive values mean the candidate store
//! improved retrieval against the supplied qrels.
//!
//! ```
//! use rig_retrieval_evals::{KnowledgeGainConfig, KnowledgeGainReport, MetricDelta, ReportDiff};
//!
//! let diff = ReportDiff {
//!     rows: vec![MetricDelta {
//!         metric: "recall@5".into(),
//!         current_mean: 1.0,
//!         baseline_mean: Some(0.25),
//!         delta: Some(0.75),
//!         winners: 1,
//!         losers: 0,
//!         unchanged: 0,
//!         query_changes: Vec::new(),
//!     }],
//! };
//! let gain = KnowledgeGainReport::from_diff(&diff, &KnowledgeGainConfig::default());
//! assert_eq!(gain.score, 0.75);
//! ```

use std::collections::BTreeMap;

#[cfg(feature = "embedding-novelty")]
use futures::stream::{self, StreamExt, TryStreamExt};
#[cfg(feature = "embedding-novelty")]
use rig::embeddings::EmbeddingModel;
use serde::{Deserialize, Serialize};
#[cfg(feature = "embedding-novelty")]
use tracing::instrument;

use crate::dataset::Qrels;
#[cfg(feature = "embedding-novelty")]
use crate::error::{Error, Result};
use crate::report::{QueryDelta, ReportDiff};

/// Configuration for aggregating retrieval deltas into one knowledge-gain score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGainConfig {
    /// Optional per-metric weights. When empty, every comparable metric in the
    /// diff contributes with weight `1.0`. When non-empty, only metrics listed
    /// here contribute.
    pub metric_weights: BTreeMap<String, f64>,
    /// Weight applied to qrels-backed per-document relevance gain.
    #[serde(default = "default_document_relevance_weight")]
    pub document_relevance_weight: f64,
    /// Weight applied to optional host-supplied novelty scores.
    #[serde(default)]
    pub novelty_weight: f64,
}

impl Default for KnowledgeGainConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeGainConfig {
    /// Build a config that averages every comparable metric equally.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metric_weights: BTreeMap::new(),
            document_relevance_weight: 1.0,
            novelty_weight: 0.0,
        }
    }

    /// Add or replace a metric weight. Negative and non-finite values are
    /// clamped to `0.0` so a configured metric can be effectively ignored
    /// without making aggregate scores surprising.
    #[must_use]
    pub fn with_metric_weight(mut self, metric: impl Into<String>, weight: f64) -> Self {
        self.metric_weights
            .insert(metric.into(), clean_weight(weight));
        self
    }

    /// Set the weight for qrels-backed document relevance when ranking
    /// candidate documents.
    #[must_use]
    pub fn with_document_relevance_weight(mut self, weight: f64) -> Self {
        self.document_relevance_weight = clean_weight(weight);
        self
    }

    /// Set the weight for optional host-supplied document novelty when ranking
    /// candidate documents.
    #[must_use]
    pub fn with_novelty_weight(mut self, weight: f64) -> Self {
        self.novelty_weight = clean_weight(weight);
        self
    }

    fn weight_for(&self, metric: &str) -> Option<f64> {
        if self.metric_weights.is_empty() {
            Some(1.0)
        } else {
            self.metric_weights.get(metric).copied().map(clean_weight)
        }
    }
}

fn default_document_relevance_weight() -> f64 {
    1.0
}

fn clean_weight(weight: f64) -> f64 {
    if weight.is_finite() {
        weight.max(0.0)
    } else {
        0.0
    }
}

/// Knowledge-gain contribution for one metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricGain {
    /// Metric identifier, e.g. `recall@5`.
    pub metric: String,
    /// Weight applied to this metric.
    pub weight: f64,
    /// Mean score delta from `current - baseline`.
    pub delta: f64,
    /// Weighted contribution (`weight * delta`).
    pub contribution: f64,
    /// Number of queries improved for this metric.
    pub winners: usize,
    /// Number of queries regressed for this metric.
    pub losers: usize,
    /// Number of unchanged queries for this metric.
    pub unchanged: usize,
}

/// Weighted knowledge-gain contribution for one query across included metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryGain {
    /// Gold-query identifier.
    pub query_id: String,
    /// Weighted mean delta across metrics that reported this query.
    pub score: f64,
    /// Sum of metric weights represented in `score`.
    pub weight: f64,
}

/// Candidate document input for knowledge-gain ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateDocumentGainInput {
    /// Backend document id, in the same id space as [`Qrels`] relevance labels.
    pub doc_id: String,
    /// Optional host-supplied novelty score. The crate does not compute
    /// embeddings itself; hosts can plug in any novelty signal and pass the
    /// normalized result here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub novelty: Option<f64>,
}

/// Candidate text chunks used by [`EmbeddingNoveltyAdapter`].
#[cfg(feature = "embedding-novelty")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateNoveltyInput {
    /// Backend document id to carry into [`CandidateDocumentGainInput`].
    pub doc_id: String,
    /// Text chunks representing this candidate document.
    pub chunks: Vec<String>,
}

#[cfg(feature = "embedding-novelty")]
impl CandidateNoveltyInput {
    /// Build a candidate novelty input from a document id and text chunks.
    #[must_use]
    pub fn new(doc_id: impl Into<String>, chunks: impl IntoIterator<Item = String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            chunks: chunks.into_iter().collect(),
        }
    }
}

/// Generic adapter that computes candidate novelty with a Rig embedding model.
///
/// The adapter owns no provider setup and makes no assumptions about where the
/// model came from. Hosts pass candidate chunks plus reference KB chunks, and
/// receive [`CandidateDocumentGainInput`] values whose novelty can be blended
/// into [`KnowledgeGainReport::with_candidate_documents`].
///
/// Embedding calls are batched to `M::MAX_DOCUMENTS` and may be issued
/// concurrently via [`EmbeddingNoveltyAdapter::with_concurrency`]. The model
/// trait already requires `Sync`, so concurrent dispatch is safe.
#[cfg(feature = "embedding-novelty")]
pub struct EmbeddingNoveltyAdapter<M> {
    model: M,
    concurrency: usize,
}

#[cfg(feature = "embedding-novelty")]
impl<M> EmbeddingNoveltyAdapter<M>
where
    M: EmbeddingModel,
{
    /// Build an adapter around a host-provided embedding model.
    #[must_use]
    pub fn new(model: M) -> Self {
        Self {
            model,
            concurrency: 1,
        }
    }

    /// Set the maximum number of in-flight embedding batches. Defaults to `1`
    /// (sequential). Values of `0` are clamped to `1`.
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Compute novelty-ranked candidate inputs from candidate/reference text.
    ///
    /// Novelty is `1.0 - max_cosine(candidate_chunk, reference_chunk)`,
    /// averaged across candidate chunks and clamped to `[0.0, 1.0]`. Empty
    /// candidate chunks receive `0.0`; candidates compared against an empty
    /// reference corpus receive `1.0`. Negative cosine values (rare with
    /// normalized embedding models) are treated as zero similarity, i.e. the
    /// chunk is considered fully novel.
    ///
    /// All reference chunks and all candidate chunks are batched into single
    /// flat embedding calls respecting `M::MAX_DOCUMENTS`, so this is `O(B)`
    /// provider round-trips total rather than `O(candidates)`.
    #[instrument(level = "debug", skip(self, candidates, reference_chunks), fields(candidates = candidates.len(), reference_chunks = reference_chunks.len(), concurrency = self.concurrency, max_batch = M::MAX_DOCUMENTS))]
    pub async fn score_candidates(
        &self,
        candidates: &[CandidateNoveltyInput],
        reference_chunks: &[String],
    ) -> Result<Vec<CandidateDocumentGainInput>> {
        let reference_embeddings = self.embed_batched(reference_chunks.to_vec()).await?;

        let mut offsets: Vec<(usize, usize)> = Vec::with_capacity(candidates.len());
        let mut flat_chunks: Vec<String> = Vec::new();
        for candidate in candidates {
            let start = flat_chunks.len();
            flat_chunks.extend(candidate.chunks.iter().cloned());
            let end = flat_chunks.len();
            offsets.push((start, end));
        }
        let candidate_embeddings = self.embed_batched(flat_chunks).await?;

        let mut scored = Vec::with_capacity(candidates.len());
        for (candidate, (start, end)) in candidates.iter().zip(offsets) {
            let slice = candidate_embeddings.get(start..end).unwrap_or(&[]);
            let novelty = compute_novelty(slice, &reference_embeddings);
            scored.push(
                CandidateDocumentGainInput::new(candidate.doc_id.clone()).with_novelty(novelty),
            );
        }
        Ok(scored)
    }

    /// Embed `texts` in batches of `M::MAX_DOCUMENTS`, dispatching up to
    /// `self.concurrency` batches at a time and preserving input order.
    #[instrument(level = "trace", skip(self, texts), fields(total = texts.len(), concurrency = self.concurrency, max_batch = M::MAX_DOCUMENTS))]
    async fn embed_batched(&self, texts: Vec<String>) -> Result<Vec<Vec<f64>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batch_size = M::MAX_DOCUMENTS.max(1);
        let total = texts.len();
        let mut batches: Vec<Vec<String>> = Vec::new();
        let mut iter = texts.into_iter();
        loop {
            let batch: Vec<String> = iter.by_ref().take(batch_size).collect();
            if batch.is_empty() {
                break;
            }
            batches.push(batch);
        }

        let concurrency = self.concurrency.max(1);
        let model = &self.model;
        let batch_results: Vec<Vec<Vec<f64>>> =
            stream::iter(batches.into_iter().map(|batch| async move {
                let embeddings = model.embed_texts(batch).await?;
                Ok::<Vec<Vec<f64>>, Error>(
                    embeddings
                        .into_iter()
                        .map(|embedding| embedding.vec)
                        .collect(),
                )
            }))
            .buffered(concurrency)
            .try_collect()
            .await?;

        let mut out: Vec<Vec<f64>> = Vec::with_capacity(total);
        for batch in batch_results {
            out.extend(batch);
        }
        Ok(out)
    }
}

#[cfg(feature = "embedding-novelty")]
fn compute_novelty(candidate_embeddings: &[Vec<f64>], reference_embeddings: &[Vec<f64>]) -> f64 {
    if candidate_embeddings.is_empty() {
        return 0.0;
    }
    if reference_embeddings.is_empty() {
        return 1.0;
    }
    let mut total = 0.0;
    for candidate_embedding in candidate_embeddings {
        let max_similarity = reference_embeddings
            .iter()
            .map(|reference_embedding| cosine(candidate_embedding, reference_embedding))
            .fold(0.0_f64, f64::max);
        total += 1.0 - max_similarity.clamp(0.0, 1.0);
    }
    total / candidate_embeddings.len() as f64
}

impl CandidateDocumentGainInput {
    /// Build a candidate document input without a novelty score.
    #[must_use]
    pub fn new(doc_id: impl Into<String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            novelty: None,
        }
    }

    /// Attach a novelty score. Negative and non-finite values are clamped to
    /// `0.0`; values above `1.0` are clamped to `1.0`.
    #[must_use]
    pub fn with_novelty(mut self, novelty: f64) -> Self {
        self.novelty = Some(clean_novelty(novelty));
        self
    }
}

/// A query-level contribution to a candidate document's score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateQueryGain {
    /// Gold-query identifier.
    pub query_id: String,
    /// Qrels relevance grade for this document and query.
    pub grade: u8,
    /// Weighted query gain from [`KnowledgeGainReport::queries`].
    pub query_gain: f64,
    /// Contribution from this query before optional novelty is applied.
    pub contribution: f64,
}

/// Ranked candidate document knowledge-gain score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateDocumentGain {
    /// Backend document id.
    pub doc_id: String,
    /// Final weighted score used for ranking
    /// (`weighted_relevance_gain + weighted_novelty_gain`).
    pub score: f64,
    /// Raw qrels-backed relevance contribution before applying
    /// [`KnowledgeGainConfig::document_relevance_weight`].
    pub relevance_gain: f64,
    /// `relevance_gain * document_relevance_weight`.
    pub weighted_relevance_gain: f64,
    /// `novelty * novelty_weight` (zero when `novelty` is `None`).
    pub weighted_novelty_gain: f64,
    /// Raw novelty score supplied by the host, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub novelty: Option<f64>,
    /// Query-level reasons for this document's score.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub query_gains: Vec<CandidateQueryGain>,
}

/// Aggregated model-free knowledge-gain score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGainReport {
    /// Weighted mean metric delta. Positive means the candidate improved the KB.
    pub score: f64,
    /// Total metric weight included in `score`.
    pub total_weight: f64,
    /// Per-metric contributions in diff order.
    pub metrics: Vec<MetricGain>,
    /// Per-query weighted movers, sorted by absolute score descending.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub queries: Vec<QueryGain>,
    /// Ranked candidate documents, populated by
    /// [`KnowledgeGainReport::with_candidate_documents`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidate_documents: Vec<CandidateDocumentGain>,
}

impl KnowledgeGainReport {
    /// Build a knowledge-gain report from a candidate-minus-baseline diff.
    #[must_use]
    pub fn from_diff(diff: &ReportDiff, config: &KnowledgeGainConfig) -> Self {
        let mut total_weight = 0.0;
        let mut total_contribution = 0.0;
        let mut metrics = Vec::new();
        let mut query_accumulator: BTreeMap<String, (f64, f64)> = BTreeMap::new();

        for row in &diff.rows {
            let Some(delta) = row.delta else {
                continue;
            };
            let Some(weight) = config.weight_for(&row.metric) else {
                continue;
            };
            if weight == 0.0 {
                continue;
            }

            let contribution = weight * delta;
            total_weight += weight;
            total_contribution += contribution;
            metrics.push(MetricGain {
                metric: row.metric.clone(),
                weight,
                delta,
                contribution,
                winners: row.winners,
                losers: row.losers,
                unchanged: row.unchanged,
            });
            accumulate_query_gains(&mut query_accumulator, &row.query_changes, weight);
        }

        let score = if total_weight > 0.0 {
            total_contribution / total_weight
        } else {
            0.0
        };
        let mut queries = query_accumulator
            .into_iter()
            .map(|(query_id, (contribution, weight))| QueryGain {
                query_id,
                score: if weight > 0.0 {
                    contribution / weight
                } else {
                    0.0
                },
                weight,
            })
            .collect::<Vec<_>>();
        queries.sort_by(|left, right| {
            right
                .score
                .abs()
                .total_cmp(&left.score.abs())
                .then_with(|| left.query_id.cmp(&right.query_id))
        });

        Self {
            score,
            total_weight,
            metrics,
            queries,
            candidate_documents: Vec::new(),
        }
    }

    /// Rank candidate documents using qrels-backed query movers and optional
    /// host-supplied novelty scores.
    ///
    /// A candidate document receives relevance gain when it is labeled relevant
    /// for a query that improved in this report. Novelty is never inferred by
    /// this crate; pass precomputed novelty scores in `candidates` and tune
    /// [`KnowledgeGainConfig::novelty_weight`] to include them in ranking.
    #[must_use]
    pub fn with_candidate_documents(
        mut self,
        qrels: &Qrels,
        candidates: &[CandidateDocumentGainInput],
        config: &KnowledgeGainConfig,
    ) -> Self {
        self.candidate_documents = self.rank_candidate_documents(qrels, candidates, config);
        self
    }

    /// Return ranked candidate documents without mutating this report.
    #[must_use]
    pub fn rank_candidate_documents(
        &self,
        qrels: &Qrels,
        candidates: &[CandidateDocumentGainInput],
        config: &KnowledgeGainConfig,
    ) -> Vec<CandidateDocumentGain> {
        let query_scores = self
            .queries
            .iter()
            .map(|query| (query.query_id.as_str(), query.score))
            .collect::<BTreeMap<_, _>>();
        let mut ranked = candidates
            .iter()
            .map(|candidate| rank_candidate_document(candidate, qrels, &query_scores, config))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
        });
        ranked
    }

    /// Render the aggregate and metric contributions as Markdown.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("knowledge_gain={:.4}\n\n", self.score));
        out.push_str("| metric | weight | delta | contribution | win | lose | same |\n");
        out.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
        for metric in &self.metrics {
            out.push_str(&format!(
                "| {} | {:.4} | {:+.4} | {:+.4} | {} | {} | {} |\n",
                metric.metric,
                metric.weight,
                metric.delta,
                metric.contribution,
                metric.winners,
                metric.losers,
                metric.unchanged
            ));
        }
        if !self.queries.is_empty() {
            out.push_str("\n| query | gain | weight |\n");
            out.push_str("|---|---:|---:|\n");
            for query in &self.queries {
                out.push_str(&format!(
                    "| {} | {:+.4} | {:.4} |\n",
                    query.query_id, query.score, query.weight
                ));
            }
        }
        if !self.candidate_documents.is_empty() {
            out.push_str(
                "\n| candidate_doc | score | weighted_relevance | weighted_novelty | novelty |\n",
            );
            out.push_str("|---|---:|---:|---:|---:|\n");
            for candidate in &self.candidate_documents {
                let novelty = candidate
                    .novelty
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "-".to_string());
                out.push_str(&format!(
                    "| {} | {:+.4} | {:+.4} | {:+.4} | {} |\n",
                    candidate.doc_id,
                    candidate.score,
                    candidate.weighted_relevance_gain,
                    candidate.weighted_novelty_gain,
                    novelty
                ));
            }
        }
        out
    }
}

fn rank_candidate_document(
    candidate: &CandidateDocumentGainInput,
    qrels: &Qrels,
    query_scores: &BTreeMap<&str, f64>,
    config: &KnowledgeGainConfig,
) -> CandidateDocumentGain {
    let mut relevance_gain = 0.0;
    let mut query_gains = Vec::new();
    for query in &qrels.queries {
        let grade = query.grade(&candidate.doc_id);
        if grade == 0 {
            continue;
        }
        let Some(query_gain) = query_scores.get(query.query_id.as_str()).copied() else {
            continue;
        };
        let contribution = query_gain * f64::from(grade);
        relevance_gain += contribution;
        query_gains.push(CandidateQueryGain {
            query_id: query.query_id.clone(),
            grade,
            query_gain,
            contribution,
        });
    }
    let weighted_relevance_gain = relevance_gain * config.document_relevance_weight;
    let novelty = candidate.novelty.map(clean_novelty);
    let weighted_novelty_gain = novelty.unwrap_or(0.0) * config.novelty_weight;
    CandidateDocumentGain {
        doc_id: candidate.doc_id.clone(),
        score: weighted_relevance_gain + weighted_novelty_gain,
        relevance_gain,
        weighted_relevance_gain,
        weighted_novelty_gain,
        novelty,
        query_gains,
    }
}

fn clean_novelty(novelty: f64) -> f64 {
    if novelty.is_finite() {
        novelty.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(feature = "embedding-novelty")]
fn cosine(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left_value, right_value) in left.iter().zip(right) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

fn accumulate_query_gains(
    accumulator: &mut BTreeMap<String, (f64, f64)>,
    changes: &[QueryDelta],
    weight: f64,
) {
    for change in changes {
        let entry = accumulator
            .entry(change.query_id.clone())
            .or_insert((0.0, 0.0));
        entry.0 += change.delta * weight;
        entry.1 += weight;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::dataset::GoldQuery;
    use crate::report::{MetricDelta, QueryDelta};

    #[cfg(feature = "embedding-novelty")]
    use rig::embeddings::{Embedding, EmbeddingError};

    #[test]
    fn default_config_averages_all_comparable_metrics() {
        let diff = ReportDiff {
            rows: vec![
                row("recall@3", 0.5, vec![query("q1", 0.5)]),
                row("mrr", 1.0, vec![query("q1", 1.0)]),
            ],
        };
        let gain = KnowledgeGainReport::from_diff(&diff, &KnowledgeGainConfig::default());
        assert!((gain.score - 0.75).abs() < 1e-9);
        assert_eq!(gain.metrics.len(), 2);
        assert_eq!(gain.queries[0].query_id, "q1");
        assert!((gain.queries[0].score - 0.75).abs() < 1e-9);
    }

    #[test]
    fn weighted_config_includes_only_named_metrics() {
        let diff = ReportDiff {
            rows: vec![
                row("recall@3", 0.5, vec![query("q1", 0.5)]),
                row("mrr", 1.0, vec![query("q1", 1.0)]),
            ],
        };
        let config = KnowledgeGainConfig::new().with_metric_weight("recall@3", 2.0);
        let gain = KnowledgeGainReport::from_diff(&diff, &config);
        assert_eq!(gain.metrics.len(), 1);
        assert!((gain.score - 0.5).abs() < 1e-9);
        assert!((gain.total_weight - 2.0).abs() < 1e-9);
    }

    #[test]
    fn candidate_document_ranking_uses_query_gain_and_novelty() {
        let diff = ReportDiff {
            rows: vec![row(
                "recall@3",
                1.0,
                vec![query("q1", 1.0), query("q2", 0.25)],
            )],
        };
        let qrels = qrels();
        let config = KnowledgeGainConfig::new()
            .with_metric_weight("recall@3", 1.0)
            .with_document_relevance_weight(1.0)
            .with_novelty_weight(0.5);
        let gain = KnowledgeGainReport::from_diff(&diff, &config).with_candidate_documents(
            &qrels,
            &[
                CandidateDocumentGainInput::new("d1").with_novelty(0.2),
                CandidateDocumentGainInput::new("d2").with_novelty(1.0),
                CandidateDocumentGainInput::new("d3").with_novelty(0.1),
            ],
            &config,
        );

        let d1 = &gain.candidate_documents[0];
        assert_eq!(d1.doc_id, "d1");
        // q1 gain = 1.0, grade(d1)=2 → relevance_gain raw = 2.0
        assert!((d1.relevance_gain - 2.0).abs() < 1e-9);
        assert!((d1.weighted_relevance_gain - 2.0).abs() < 1e-9);
        assert!((d1.weighted_novelty_gain - 0.1).abs() < 1e-9);
        assert!((d1.score - 2.1).abs() < 1e-9);

        let d2 = &gain.candidate_documents[1];
        assert_eq!(d2.doc_id, "d2");
        assert!((d2.relevance_gain - 0.25).abs() < 1e-9);
        assert!((d2.weighted_relevance_gain - 0.25).abs() < 1e-9);
        assert!((d2.weighted_novelty_gain - 0.5).abs() < 1e-9);
        assert!((d2.score - 0.75).abs() < 1e-9);
    }

    #[test]
    fn negative_diff_produces_negative_knowledge_gain() {
        let diff = ReportDiff {
            rows: vec![
                row("recall@3", -0.5, vec![query("q1", -0.5)]),
                row("mrr", -0.25, vec![query("q1", -0.25)]),
            ],
        };
        let gain = KnowledgeGainReport::from_diff(&diff, &KnowledgeGainConfig::default());
        assert!(gain.score < 0.0);
        assert!((gain.score + 0.375).abs() < 1e-9);
        assert!(gain.queries[0].score < 0.0);
    }

    #[cfg(feature = "embedding-novelty")]
    #[tokio::test]
    async fn embedding_novelty_adapter_scores_candidate_chunks() {
        let adapter = EmbeddingNoveltyAdapter::new(FakeEmbeddingModel::<32>::new());
        let scored = adapter
            .score_candidates(
                &[
                    CandidateNoveltyInput::new("same", vec!["alpha".to_string()]),
                    CandidateNoveltyInput::new("new", vec!["beta".to_string()]),
                ],
                &["alpha".to_string()],
            )
            .await
            .unwrap();

        let same = scored
            .iter()
            .find(|candidate| candidate.doc_id == "same")
            .unwrap();
        let new_doc = scored
            .iter()
            .find(|candidate| candidate.doc_id == "new")
            .unwrap();
        assert_eq!(same.novelty, Some(0.0));
        assert_eq!(new_doc.novelty, Some(1.0));
    }

    #[cfg(feature = "embedding-novelty")]
    #[tokio::test]
    async fn embedding_novelty_adapter_handles_partial_similarity_and_empties() {
        let adapter = EmbeddingNoveltyAdapter::new(FakeEmbeddingModel::<32>::new());
        let scored = adapter
            .score_candidates(
                &[
                    // cos((1,0),(0.6,0.8)) = 0.6 → novelty 0.4
                    CandidateNoveltyInput::new("partial", vec!["mid".to_string()]),
                    // empty candidate chunks → 0.0
                    CandidateNoveltyInput::new("empty_candidate", Vec::<String>::new()),
                    // two chunks (cosine 1.0 and 0.0) → mean novelty 0.5
                    CandidateNoveltyInput::new(
                        "multi",
                        vec!["alpha".to_string(), "beta".to_string()],
                    ),
                ],
                &["alpha".to_string()],
            )
            .await
            .unwrap();

        let by_id = |id: &str| -> CandidateDocumentGainInput {
            scored
                .iter()
                .find(|candidate| candidate.doc_id == id)
                .cloned()
                .unwrap()
        };
        let partial = by_id("partial").novelty.unwrap();
        assert!((partial - 0.4).abs() < 1e-9, "got {partial}");
        assert_eq!(by_id("empty_candidate").novelty, Some(0.0));
        let multi = by_id("multi").novelty.unwrap();
        assert!((multi - 0.5).abs() < 1e-9, "got {multi}");
    }

    #[cfg(feature = "embedding-novelty")]
    #[tokio::test]
    async fn embedding_novelty_adapter_empty_reference_yields_full_novelty() {
        let adapter = EmbeddingNoveltyAdapter::new(FakeEmbeddingModel::<32>::new());
        let scored = adapter
            .score_candidates(
                &[CandidateNoveltyInput::new(
                    "isolated",
                    vec!["alpha".to_string()],
                )],
                &[],
            )
            .await
            .unwrap();
        assert_eq!(scored[0].novelty, Some(1.0));
    }

    #[cfg(feature = "embedding-novelty")]
    #[tokio::test]
    async fn embedding_novelty_adapter_respects_max_documents() {
        // MAX_DOCUMENTS = 2, so 5 reference + 4 candidate chunks must split into
        // ceil(5/2) + ceil(4/2) = 3 + 2 = 5 batches.
        let model = FakeEmbeddingModel::<2>::new();
        let counter = model.call_counter();
        let adapter = EmbeddingNoveltyAdapter::new(model).with_concurrency(2);

        let reference: Vec<String> = (0..5).map(|i| format!("ref{i}")).collect();
        let candidate_chunks: Vec<String> = (0..4).map(|i| format!("c{i}")).collect();
        let candidates = vec![CandidateNoveltyInput::new("doc", candidate_chunks)];

        let _ = adapter
            .score_candidates(&candidates, &reference)
            .await
            .unwrap();

        let calls = counter.lock().unwrap();
        assert_eq!(*calls, 5, "expected 5 batched calls, observed {}", *calls);
    }

    #[cfg(feature = "embedding-novelty")]
    #[tokio::test]
    async fn embedding_novelty_adapter_propagates_provider_errors() {
        let adapter =
            EmbeddingNoveltyAdapter::new(FakeEmbeddingModel::<32>::new().failing("provider boom"));
        let result = adapter
            .score_candidates(
                &[CandidateNoveltyInput::new("doc", vec!["alpha".to_string()])],
                &["alpha".to_string()],
            )
            .await;
        let err = result.unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("embedding"),
            "expected typed embedding error, got: {message}"
        );
    }

    fn row(metric: &str, delta: f64, query_changes: Vec<QueryDelta>) -> MetricDelta {
        MetricDelta {
            metric: metric.to_string(),
            current_mean: delta,
            baseline_mean: Some(0.0),
            delta: Some(delta),
            winners: 1,
            losers: 0,
            unchanged: 0,
            query_changes,
            current_ci: None,
            baseline_ci: None,
        }
    }

    fn query(query_id: &str, delta: f64) -> QueryDelta {
        QueryDelta {
            query_id: query_id.to_string(),
            current: delta,
            baseline: 0.0,
            delta,
        }
    }

    fn qrels() -> Qrels {
        Qrels {
            queries: vec![
                GoldQuery {
                    query_id: "q1".to_string(),
                    query: "one".to_string(),
                    relevant_docs: BTreeMap::from([("d1".to_string(), 2u8)])
                        .into_iter()
                        .collect(),
                    reference_answer: None,
                },
                GoldQuery {
                    query_id: "q2".to_string(),
                    query: "two".to_string(),
                    relevant_docs: BTreeMap::from([("d2".to_string(), 1u8)])
                        .into_iter()
                        .collect(),
                    reference_answer: None,
                },
            ],
        }
    }

    #[cfg(feature = "embedding-novelty")]
    use std::sync::{Arc, Mutex};

    #[cfg(feature = "embedding-novelty")]
    #[derive(Clone)]
    struct FakeEmbeddingModel<const MAX: usize> {
        calls: Arc<Mutex<usize>>,
        failure: Option<String>,
    }

    #[cfg(feature = "embedding-novelty")]
    impl<const MAX: usize> FakeEmbeddingModel<MAX> {
        fn new() -> Self {
            Self {
                calls: Arc::new(Mutex::new(0)),
                failure: None,
            }
        }

        fn failing(mut self, message: impl Into<String>) -> Self {
            self.failure = Some(message.into());
            self
        }

        fn call_counter(&self) -> Arc<Mutex<usize>> {
            Arc::clone(&self.calls)
        }
    }

    #[cfg(feature = "embedding-novelty")]
    impl<const MAX: usize> EmbeddingModel for FakeEmbeddingModel<MAX> {
        const MAX_DOCUMENTS: usize = MAX;

        type Client = ();

        fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
            Self::new()
        }

        fn ndims(&self) -> usize {
            2
        }

        async fn embed_texts(
            &self,
            texts: impl IntoIterator<Item = String> + rig::wasm_compat::WasmCompatSend,
        ) -> std::result::Result<Vec<Embedding>, EmbeddingError> {
            let texts: Vec<String> = texts.into_iter().collect();
            if let Some(message) = self.failure.as_ref() {
                return Err(EmbeddingError::ProviderError(message.clone()));
            }
            // Enforce the per-type batch limit so tests can verify the adapter
            // respects MAX_DOCUMENTS-style chunking.
            if texts.len() > MAX {
                return Err(EmbeddingError::ProviderError(format!(
                    "fake model batch overflow: {} > {MAX}",
                    texts.len()
                )));
            }
            {
                let mut calls = self.calls.lock().unwrap();
                *calls = calls.saturating_add(1);
            }
            Ok(texts
                .into_iter()
                .map(|document| Embedding {
                    vec: fake_vector(&document),
                    document,
                })
                .collect())
        }
    }

    #[cfg(feature = "embedding-novelty")]
    fn fake_vector(text: &str) -> Vec<f64> {
        match text {
            "alpha" => vec![1.0, 0.0],
            "beta" => vec![0.0, 1.0],
            "mid" => vec![0.6, 0.8],
            _ => vec![0.0, 0.0],
        }
    }
}
