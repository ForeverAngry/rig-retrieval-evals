//! Aggregation, serialization, and baseline diffing of per-query metric
//! scores produced by [`crate::harness::RetrievalHarness`].
//!
//! Two layers:
//!
//! - [`MetricReport`] — aggregates a single metric across all queries (mean,
//!   stddev, P50/P95, min/max, per-query scores).
//! - [`ReliabilityReport`] — aggregates repeated trials for one metric into
//!   pass@k / pass^k reliability estimates.
//! - [`MultiReport`]  — bundles several [`MetricReport`]s with optional
//!   metadata (dataset id, store kind, judge fingerprint) so reports can be
//!   diffed across runs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::staleness::{ConflictReport, StalenessReport};

/// Aggregated statistics for a single metric across a query set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricReport {
    /// Metric identifier (e.g. `"recall@10"`).
    pub metric: String,
    /// Number of queries scored.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation (N-1). `0.0` for `n < 2`.
    pub stddev: f64,
    /// Minimum observed score.
    pub min: f64,
    /// Maximum observed score.
    pub max: f64,
    /// 50th percentile (median) via linear interpolation.
    pub p50: f64,
    /// 95th percentile via linear interpolation.
    pub p95: f64,
    /// Per-query `(query_id, score)` pairs, in input order.
    pub per_query: Vec<(String, f64)>,
    /// Optional bootstrap confidence interval for [`MetricReport::mean`].
    /// Populated by [`MetricReport::with_bootstrap_ci`]. Serialized as
    /// `"ci"` when present, omitted when `None` so existing reports stay
    /// schema-compatible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ci: Option<MetricCi>,
}

/// Bootstrap confidence interval for [`MetricReport::mean`].
///
/// Produced by [`MetricReport::bootstrap_ci`] using a deterministic
/// percentile bootstrap: resample the per-query scores `iterations` times
/// with replacement, take the empirical mean of each resample, and report
/// the two-sided quantile interval for the requested `level`. The same
/// `seed` yields the same interval on the same input, so CI gates and
/// reproducibility tests don't need extra fixtures.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct MetricCi {
    /// Lower bound of the confidence interval (inclusive).
    pub lower: f64,
    /// Upper bound of the confidence interval (inclusive).
    pub upper: f64,
    /// Two-sided coverage probability the interval was computed for,
    /// e.g. `0.95` for a 95 % CI.
    pub level: f64,
    /// Number of bootstrap resamples drawn.
    pub iterations: usize,
}

impl MetricReport {
    /// Build a [`MetricReport`] from per-query `(query_id, score)` pairs.
    ///
    /// Scores are aggregated in-place; the original ordering is preserved
    /// in [`MetricReport::per_query`] for diff and audit use cases.
    pub fn from_per_query(metric: String, per_query: Vec<(String, f64)>) -> Self {
        let n = per_query.len();
        if n == 0 {
            return Self {
                metric,
                n: 0,
                mean: 0.0,
                stddev: 0.0,
                min: 0.0,
                max: 0.0,
                p50: 0.0,
                p95: 0.0,
                per_query,
                ci: None,
            };
        }
        let scores: Vec<f64> = per_query.iter().map(|(_, s)| *s).collect();
        let sum: f64 = scores.iter().sum();
        let mean = sum / n as f64;
        let var = if n > 1 {
            scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
        } else {
            0.0
        };
        let stddev = var.sqrt();

        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);
        let p50 = percentile(&sorted, 0.50);
        let p95 = percentile(&sorted, 0.95);

        Self {
            metric,
            n,
            mean,
            stddev,
            min,
            max,
            p50,
            p95,
            per_query,
            ci: None,
        }
    }

    /// Compute a percentile-bootstrap confidence interval for
    /// [`MetricReport::mean`] without mutating `self`.
    ///
    /// Returns `None` when there are no per-query scores or when
    /// `iterations` / `level` are out of range (`iterations == 0`, or
    /// `level` not strictly inside `(0.0, 1.0)`). Otherwise draws
    /// `iterations` resamples of size `n` with replacement using a
    /// deterministic SplitMix64 stream seeded by `seed`, and returns the
    /// two-sided percentile interval at the requested `level`.
    #[must_use]
    pub fn bootstrap_ci(&self, iterations: usize, level: f64, seed: u64) -> Option<MetricCi> {
        if self.per_query.is_empty() || iterations == 0 {
            return None;
        }
        if !(level > 0.0 && level < 1.0) {
            return None;
        }
        let scores: Vec<f64> = self.per_query.iter().map(|(_, s)| *s).collect();
        let n = scores.len();
        let mut state = seed;
        let mut resample_means: Vec<f64> = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let mut sum = 0.0;
            for _ in 0..n {
                let r = splitmix64(&mut state);
                let idx = (r as usize) % n;
                sum += scores.get(idx).copied().unwrap_or(0.0);
            }
            resample_means.push(sum / n as f64);
        }
        resample_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let alpha = (1.0 - level) / 2.0;
        let lower = percentile(&resample_means, alpha);
        let upper = percentile(&resample_means, 1.0 - alpha);
        Some(MetricCi {
            lower,
            upper,
            level,
            iterations,
        })
    }

    /// Compute and attach a percentile-bootstrap confidence interval to
    /// [`MetricReport::ci`]. See [`MetricReport::bootstrap_ci`] for the
    /// algorithm and return-value semantics. Returns `self` so the call
    /// chains directly off [`MetricReport::from_per_query`].
    #[must_use]
    pub fn with_bootstrap_ci(mut self, iterations: usize, level: f64, seed: u64) -> Self {
        self.ci = self.bootstrap_ci(iterations, level, seed);
        self
    }
}

/// Deterministic SplitMix64 PRNG. Stable, dependency-free, and good
/// enough for bootstrap resampling — we are not generating cryptographic
/// material here.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted.first().copied().unwrap_or(0.0);
    }
    let rank = q * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let lo_v = sorted.get(lo).copied().unwrap_or(0.0);
    let hi_v = sorted.get(hi).copied().unwrap_or(lo_v);
    let frac = rank - lo as f64;
    lo_v + (hi_v - lo_v) * frac
}

/// Reliability summary for one query across repeated trials of the same
/// metric.
///
/// A trial counts as successful when its score is greater than or equal to
/// [`ReliabilityReport::threshold`]. The per-query pass@k estimate is the
/// probability that at least one of `k` sampled trials succeeds; pass^k is
/// the probability that all `k` sampled trials succeed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryReliability {
    /// Gold-query identifier.
    pub query_id: String,
    /// Number of repeated trials observed for this query.
    pub trials: usize,
    /// Number of trials whose score met or exceeded the threshold.
    pub successes: usize,
    /// `successes / trials`.
    pub pass_rate: f64,
    /// Empirical pass@k estimate for this query.
    pub pass_at_k: f64,
    /// Empirical pass^k estimate for this query.
    pub pass_all_k: f64,
}

/// Repeated-trial reliability report for a single metric.
///
/// This report turns a set of repeated [`MetricReport`]s for the same metric
/// into reliability estimates. Scores are thresholded into pass/fail outcomes
/// first, then pass@k and pass^k are estimated per query and averaged.
///
/// ```
/// use rig_retrieval_evals::{MetricReport, ReliabilityReport};
///
/// let trial_a = MetricReport::from_per_query(
///     "recall@10".into(),
///     vec![("q1".into(), 1.0), ("q2".into(), 0.0)],
/// );
/// let trial_b = MetricReport::from_per_query(
///     "recall@10".into(),
///     vec![("q1".into(), 1.0), ("q2".into(), 1.0)],
/// );
///
/// let reliability = ReliabilityReport::from_metric_reports(
///     "recall@10",
///     1.0,
///     2,
///     &[trial_a, trial_b],
/// )?;
/// assert_eq!(reliability.n_queries, 2);
/// assert_eq!(reliability.trials_per_query, 2);
/// # Ok::<(), rig_retrieval_evals::Error>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityReport {
    /// Metric identifier shared by every trial report.
    pub metric: String,
    /// Score threshold used to convert each trial into pass/fail.
    pub threshold: f64,
    /// Number of attempts sampled in pass@k / pass^k estimates.
    pub k: usize,
    /// Number of queries included in the reliability estimate.
    pub n_queries: usize,
    /// Number of trials observed for each query.
    pub trials_per_query: usize,
    /// Mean per-query pass rate.
    pub mean_pass_rate: f64,
    /// Mean per-query pass@k.
    pub pass_at_k: f64,
    /// Mean per-query pass^k.
    pub pass_all_k: f64,
    /// Per-query reliability rows, in the first trial report's query order.
    pub per_query: Vec<QueryReliability>,
}

/// Per-query freshness rollup derived from stale-hit and conflict detectors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FreshnessQueryRollup {
    /// Query id shared by the stale and conflict reports.
    pub query_id: String,
    /// Number of top-k positions considered for this query.
    pub considered: usize,
    /// Number of stale hits inside the considered window.
    pub stale_hits: usize,
    /// Fraction of considered positions that were stale.
    pub stale_rate: f64,
    /// Number of conflict groups inside the considered window.
    pub conflict_groups: usize,
    /// Number of documents participating in conflict groups.
    pub conflicting_doc_count: usize,
    /// Fraction of considered positions that participated in a conflict group.
    pub conflict_rate: f64,
}

/// Dataset-level freshness rollup for stale hits and version-key conflicts.
///
/// `FreshnessReport` is intentionally separate from IR metrics so callers can
/// inspect stale/conflict details as freshness signals. Use
/// [`MultiReport::with_freshness_metrics`] when those signals should also be
/// converted into score-like metric rows that participate in
/// [`RegressionGate`] checks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FreshnessReport {
    /// Top-k window used to produce the underlying stale/conflict reports.
    pub k: usize,
    /// Number of queries rolled up.
    pub query_count: usize,
    /// Total top-k positions considered across all queries.
    pub total_considered: usize,
    /// Number of stale hits across all queries.
    pub stale_hit_count: usize,
    /// Number of queries with at least one stale hit.
    pub stale_query_count: usize,
    /// Dataset-level stale-hit rate: `stale_hit_count / total_considered`.
    pub stale_rate: f64,
    /// Fraction of queries that had at least one stale hit.
    pub stale_query_rate: f64,
    /// Number of conflict groups across all queries.
    pub conflict_group_count: usize,
    /// Number of documents participating in conflict groups across all queries.
    pub conflicting_doc_count: usize,
    /// Number of queries with at least one conflict group.
    pub conflict_query_count: usize,
    /// Dataset-level conflict rate: `conflicting_doc_count / total_considered`.
    pub conflict_rate: f64,
    /// Fraction of queries that had at least one conflict group.
    pub conflict_query_rate: f64,
    /// Per-query stale/conflict rollups, preserving stale-report input order.
    pub per_query: Vec<FreshnessQueryRollup>,
}

impl FreshnessReport {
    /// Build a dataset-level freshness rollup from per-query detector outputs.
    ///
    /// `staleness` and `conflicts` must cover the same query ids in the same
    /// order and must use the same `considered` count per query.
    pub fn from_query_reports(
        k: usize,
        staleness: &[StalenessReport],
        conflicts: &[ConflictReport],
    ) -> Result<Self> {
        if staleness.len() != conflicts.len() {
            return Err(Error::BaselineMismatch(format!(
                "freshness report count mismatch: stale={} conflict={}",
                staleness.len(),
                conflicts.len()
            )));
        }

        let mut per_query = Vec::with_capacity(staleness.len());
        for (stale, conflict) in staleness.iter().zip(conflicts) {
            if stale.query_id != conflict.query_id {
                return Err(Error::BaselineMismatch(format!(
                    "freshness query mismatch: stale={} conflict={}",
                    stale.query_id, conflict.query_id
                )));
            }
            if stale.considered != conflict.considered {
                return Err(Error::BaselineMismatch(format!(
                    "freshness considered mismatch for {}: stale={} conflict={}",
                    stale.query_id, stale.considered, conflict.considered
                )));
            }
            per_query.push(FreshnessQueryRollup {
                query_id: stale.query_id.clone(),
                considered: stale.considered,
                stale_hits: stale.stale_hits.len(),
                stale_rate: stale.stale_rate(),
                conflict_groups: conflict.groups.len(),
                conflicting_doc_count: conflict.conflicting_doc_count,
                conflict_rate: conflict.conflict_rate(),
            });
        }

        let query_count = per_query.len();
        let total_considered = per_query.iter().map(|row| row.considered).sum();
        let stale_hit_count = per_query.iter().map(|row| row.stale_hits).sum();
        let stale_query_count = per_query.iter().filter(|row| row.stale_hits > 0).count();
        let conflict_group_count = per_query.iter().map(|row| row.conflict_groups).sum();
        let conflicting_doc_count = per_query.iter().map(|row| row.conflicting_doc_count).sum();
        let conflict_query_count = per_query
            .iter()
            .filter(|row| row.conflict_groups > 0)
            .count();

        Ok(Self {
            k,
            query_count,
            total_considered,
            stale_hit_count,
            stale_query_count,
            stale_rate: ratio(stale_hit_count, total_considered),
            stale_query_rate: ratio(stale_query_count, query_count),
            conflict_group_count,
            conflicting_doc_count,
            conflict_query_count,
            conflict_rate: ratio(conflicting_doc_count, total_considered),
            conflict_query_rate: ratio(conflict_query_count, query_count),
            per_query,
        })
    }

    /// Convert freshness failures into score-like metric reports.
    ///
    /// Higher is better for these metrics, so the existing
    /// [`RegressionGate`] can flag freshness regressions without a new gate
    /// direction system.
    #[must_use]
    pub fn metric_reports(&self) -> Vec<MetricReport> {
        let stale_free = self
            .per_query
            .iter()
            .map(|row| (row.query_id.clone(), 1.0 - row.stale_rate))
            .collect();
        let conflict_free = self
            .per_query
            .iter()
            .map(|row| (row.query_id.clone(), 1.0 - row.conflict_rate))
            .collect();
        vec![
            MetricReport::from_per_query(
                format!("freshness.stale_free_rate@{}", self.k),
                stale_free,
            ),
            MetricReport::from_per_query(
                format!("freshness.conflict_free_rate@{}", self.k),
                conflict_free,
            ),
        ]
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

impl ReliabilityReport {
    /// Build a repeated-trial reliability report from multiple
    /// [`MetricReport`]s for the same metric.
    ///
    /// Every report must contain the same query ids exactly once. `k` must be
    /// in `1..=reports.len()`. Scores must be finite.
    pub fn from_metric_reports(
        metric: impl Into<String>,
        threshold: f64,
        k: usize,
        reports: &[MetricReport],
    ) -> Result<Self> {
        let metric = metric.into();
        if reports.is_empty() {
            return Err(Error::Config(
                "at least one trial report is required".into(),
            ));
        }
        if k == 0 {
            return Err(Error::Config("pass@k requires k > 0".into()));
        }
        if !threshold.is_finite() {
            return Err(Error::Config("reliability threshold must be finite".into()));
        }
        if k > reports.len() {
            return Err(Error::Config(format!(
                "pass@k k={} exceeds trial count {}",
                k,
                reports.len()
            )));
        }

        for report in reports {
            if report.metric != metric {
                return Err(Error::BaselineMismatch(format!(
                    "metric mismatch: expected {metric}, got {}",
                    report.metric
                )));
            }
        }

        let first = reports
            .first()
            .ok_or_else(|| Error::Config("at least one trial report is required".into()))?;
        let mut query_order = Vec::with_capacity(first.per_query.len());
        let mut seen = std::collections::BTreeSet::new();
        for (query_id, score) in &first.per_query {
            if !score.is_finite() {
                return Err(Error::Config(format!(
                    "non-finite score for query {query_id}"
                )));
            }
            if !seen.insert(query_id.as_str()) {
                return Err(Error::BaselineMismatch(format!(
                    "duplicate query id in trial report: {query_id}"
                )));
            }
            query_order.push(query_id.clone());
        }

        let mut scores_by_query: BTreeMap<String, Vec<f64>> = query_order
            .iter()
            .map(|query_id| (query_id.clone(), Vec::with_capacity(reports.len())))
            .collect();

        for report in reports {
            let mut report_scores = BTreeMap::new();
            for (query_id, score) in &report.per_query {
                if !score.is_finite() {
                    return Err(Error::Config(format!(
                        "non-finite score for query {query_id}"
                    )));
                }
                if report_scores.insert(query_id.as_str(), *score).is_some() {
                    return Err(Error::BaselineMismatch(format!(
                        "duplicate query id in trial report: {query_id}"
                    )));
                }
            }
            if report_scores.len() != query_order.len() {
                return Err(Error::BaselineMismatch(format!(
                    "trial report has {} queries; expected {}",
                    report_scores.len(),
                    query_order.len()
                )));
            }
            for query_id in &query_order {
                let Some(score) = report_scores.get(query_id.as_str()).copied() else {
                    return Err(Error::BaselineMismatch(format!(
                        "trial report missing query id {query_id}"
                    )));
                };
                let Some(scores) = scores_by_query.get_mut(query_id) else {
                    return Err(Error::BaselineMismatch(format!(
                        "unexpected query id {query_id}"
                    )));
                };
                scores.push(score);
            }
        }

        let mut per_query = Vec::with_capacity(query_order.len());
        for query_id in query_order {
            let Some(scores) = scores_by_query.remove(&query_id) else {
                return Err(Error::BaselineMismatch(format!(
                    "missing scores for query id {query_id}"
                )));
            };
            per_query.push(query_reliability(query_id, &scores, threshold, k));
        }

        let n_queries = per_query.len();
        let trials_per_query = reports.len();
        let mean_pass_rate = mean_by(&per_query, |q| q.pass_rate);
        let pass_at_k = mean_by(&per_query, |q| q.pass_at_k);
        let pass_all_k = mean_by(&per_query, |q| q.pass_all_k);

        Ok(Self {
            metric,
            threshold,
            k,
            n_queries,
            trials_per_query,
            mean_pass_rate,
            pass_at_k,
            pass_all_k,
            per_query,
        })
    }
}

fn query_reliability(
    query_id: String,
    scores: &[f64],
    threshold: f64,
    k: usize,
) -> QueryReliability {
    let trials = scores.len();
    let successes = scores.iter().filter(|score| **score >= threshold).count();
    let pass_rate = if trials == 0 {
        0.0
    } else {
        successes as f64 / trials as f64
    };
    QueryReliability {
        query_id,
        trials,
        successes,
        pass_rate,
        pass_at_k: pass_at_k_estimate(trials, successes, k),
        pass_all_k: pass_all_k_estimate(trials, successes, k),
    }
}

fn mean_by(rows: &[QueryReliability], f: impl Fn(&QueryReliability) -> f64) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    rows.iter().map(f).sum::<f64>() / rows.len() as f64
}

fn pass_at_k_estimate(trials: usize, successes: usize, k: usize) -> f64 {
    if k == 0 || trials == 0 || successes == 0 {
        return 0.0;
    }
    if k > trials || trials - successes < k {
        return 1.0;
    }
    let fail_all = (0..k).fold(1.0, |acc, offset| {
        acc * ((trials - successes - offset) as f64 / (trials - offset) as f64)
    });
    1.0 - fail_all
}

fn pass_all_k_estimate(trials: usize, successes: usize, k: usize) -> f64 {
    if k == 0 || trials == 0 || successes < k || k > trials {
        return 0.0;
    }
    (0..k).fold(1.0, |acc, offset| {
        acc * ((successes - offset) as f64 / (trials - offset) as f64)
    })
}

/// A bundle of [`MetricReport`]s with optional run metadata, suitable for
/// JSON persistence and baseline comparison.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MultiReport {
    /// Free-form dataset identifier (e.g. `"beir/nq"` or `"internal/v3"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,
    /// Free-form store identifier (e.g. `"memvid:livetest.mv2"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store_kind: Option<String>,
    /// Opaque fingerprint of any LLM judges used. Reports with mismatched
    /// fingerprints refuse to diff to prevent silent comparison drift.
    /// Reserved for the upcoming `ragas` feature; pure retrieval runs leave
    /// this empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_fingerprint: Option<String>,
    /// One report per metric, in the order metrics were declared.
    pub metrics: Vec<MetricReport>,
    /// Optional stale/conflict freshness rollup for the same dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freshness: Option<FreshnessReport>,
}

impl MultiReport {
    /// Construct a [`MultiReport`] from a metric report vector. Other
    /// metadata is filled in via the `with_*` builders.
    #[must_use]
    pub fn new(metrics: Vec<MetricReport>) -> Self {
        Self {
            metrics,
            ..Default::default()
        }
    }

    /// Attach a dataset identifier.
    #[must_use]
    pub fn with_dataset(mut self, id: impl Into<String>) -> Self {
        self.dataset_id = Some(id.into());
        self
    }

    /// Attach a store kind identifier.
    #[must_use]
    pub fn with_store(mut self, kind: impl Into<String>) -> Self {
        self.store_kind = Some(kind.into());
        self
    }

    /// Attach a judge fingerprint (reserved for `ragas`).
    #[must_use]
    pub fn with_judge_fingerprint(mut self, fp: impl Into<String>) -> Self {
        self.judge_fingerprint = Some(fp.into());
        self
    }

    /// Attach a freshness rollup without modifying metric rows.
    #[must_use]
    pub fn with_freshness(mut self, freshness: FreshnessReport) -> Self {
        self.freshness = Some(freshness);
        self
    }

    /// Attach a freshness rollup and append score-like freshness metrics.
    ///
    /// Appended metric names are `freshness.stale_free_rate@k` and
    /// `freshness.conflict_free_rate@k`. Because higher is better, these rows
    /// can be gated with [`RegressionGate`] just like recall, nDCG, or MRR.
    #[must_use]
    pub fn with_freshness_metrics(mut self, freshness: FreshnessReport) -> Self {
        self.metrics.extend(freshness.metric_reports());
        self.freshness = Some(freshness);
        self
    }

    /// Serialize as pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Render a compact Markdown summary table.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("| metric | n | mean | stddev | p50 | p95 | min | max |\n");
        out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
        for m in &self.metrics {
            out.push_str(&format!(
                "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                m.metric, m.n, m.mean, m.stddev, m.p50, m.p95, m.min, m.max
            ));
        }
        out
    }

    /// Diff this report against a baseline. Returns a [`ReportDiff`] with
    /// per-metric Δ-mean and per-query winners/losers. Fails if the two
    /// reports were produced with different judge fingerprints (silent
    /// comparison drift).
    ///
    /// Per-query deltas are computed by intersecting the two reports'
    /// `per_query` vectors on `query_id`. Queries missing from either side
    /// are skipped (they cannot be compared). `winners`, `losers`, and
    /// `unchanged` use an absolute threshold of `1e-9` to filter floating
    /// point noise; callers needing different sensitivity should inspect
    /// [`MetricDelta::query_changes`] directly.
    pub fn diff(&self, baseline: &MultiReport) -> Result<ReportDiff> {
        if self.judge_fingerprint != baseline.judge_fingerprint {
            return Err(Error::BaselineMismatch(format!(
                "judge fingerprint mismatch: current={:?} baseline={:?}",
                self.judge_fingerprint, baseline.judge_fingerprint
            )));
        }
        let base_by_name: BTreeMap<&str, &MetricReport> = baseline
            .metrics
            .iter()
            .map(|m| (m.metric.as_str(), m))
            .collect();
        let mut rows = Vec::with_capacity(self.metrics.len());
        for m in &self.metrics {
            let base = base_by_name.get(m.metric.as_str()).copied();
            let baseline_mean = base.map(|b| b.mean);
            let (query_changes, winners, losers, unchanged) = match base {
                Some(b) => compute_query_changes(&m.per_query, &b.per_query),
                None => (Vec::new(), 0, 0, 0),
            };
            rows.push(MetricDelta {
                metric: m.metric.clone(),
                current_mean: m.mean,
                baseline_mean,
                delta: baseline_mean.map(|b| m.mean - b),
                winners,
                losers,
                unchanged,
                query_changes,
                current_ci: m.ci,
                baseline_ci: base.and_then(|b| b.ci),
            });
        }
        Ok(ReportDiff { rows })
    }
}

/// Floating-point noise floor used when bucketing per-query deltas into
/// winners / losers / unchanged. Deltas with `|delta| <= EPSILON` count as
/// unchanged.
const EPSILON: f64 = 1e-9;

/// Intersect per-query scores and return `(changes, winners, losers, unchanged)`.
/// `changes` is sorted by `|delta|` descending so the largest movers are
/// surfaced first.
fn compute_query_changes(
    current: &[(String, f64)],
    baseline: &[(String, f64)],
) -> (Vec<QueryDelta>, usize, usize, usize) {
    let base_by_query: BTreeMap<&str, f64> =
        baseline.iter().map(|(q, s)| (q.as_str(), *s)).collect();
    let mut changes = Vec::new();
    let mut winners = 0usize;
    let mut losers = 0usize;
    let mut unchanged = 0usize;
    for (query_id, cur_score) in current {
        let Some(base_score) = base_by_query.get(query_id.as_str()).copied() else {
            continue;
        };
        let delta = cur_score - base_score;
        if delta > EPSILON {
            winners += 1;
        } else if delta < -EPSILON {
            losers += 1;
        } else {
            unchanged += 1;
        }
        changes.push(QueryDelta {
            query_id: query_id.clone(),
            current: *cur_score,
            baseline: base_score,
            delta,
        });
    }
    changes.sort_by(|a, b| {
        b.delta
            .abs()
            .partial_cmp(&a.delta.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    (changes, winners, losers, unchanged)
}

/// Per-metric delta produced by [`MultiReport::diff`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDelta {
    /// Metric identifier.
    pub metric: String,
    /// Mean from the current report.
    pub current_mean: f64,
    /// Mean from the baseline report, if the metric was present.
    pub baseline_mean: Option<f64>,
    /// `current_mean - baseline_mean`, if comparable.
    pub delta: Option<f64>,
    /// Number of queries whose score improved relative to the baseline.
    #[serde(default)]
    pub winners: usize,
    /// Number of queries whose score regressed relative to the baseline.
    #[serde(default)]
    pub losers: usize,
    /// Number of queries whose score was unchanged (within floating-point
    /// noise) relative to the baseline.
    #[serde(default)]
    pub unchanged: usize,
    /// Per-query deltas for queries present in both reports, sorted by
    /// `|delta|` descending. Empty if the metric was missing from the
    /// baseline.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub query_changes: Vec<QueryDelta>,
    /// Bootstrap CI on the current report's mean, if it was computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_ci: Option<MetricCi>,
    /// Bootstrap CI on the baseline report's mean, if it was computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_ci: Option<MetricCi>,
}

/// Per-query score change for a single metric, produced by
/// [`MultiReport::diff`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryDelta {
    /// Gold-query identifier.
    pub query_id: String,
    /// Score on the current report.
    pub current: f64,
    /// Score on the baseline report.
    pub baseline: f64,
    /// `current - baseline`.
    pub delta: f64,
}

/// Result of [`MultiReport::diff`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportDiff {
    /// One row per metric in the current report.
    pub rows: Vec<MetricDelta>,
}

impl ReportDiff {
    /// Render the diff as a Markdown table including per-metric mean delta
    /// and per-query winner/loser/unchanged counts. Per-query movers are
    /// not inlined; inspect [`MetricDelta::query_changes`] for that detail.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("| metric | current | baseline | Δ | win | lose | same |\n");
        out.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
        for r in &self.rows {
            let baseline = r
                .baseline_mean
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "—".to_string());
            let delta = r
                .delta
                .map(|v| format!("{v:+.4}"))
                .unwrap_or_else(|| "—".to_string());
            out.push_str(&format!(
                "| {} | {:.4} | {} | {} | {} | {} | {} |\n",
                r.metric, r.current_mean, baseline, delta, r.winners, r.losers, r.unchanged
            ));
        }
        out
    }

    /// Serialize as pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Evaluate the diff against a [`RegressionGate`]. Returns the subset
    /// of [`MetricDelta`] rows whose mean delta is more negative than the
    /// configured threshold for that metric. Metrics not listed in the
    /// gate are ignored.
    #[must_use]
    pub fn regressions(&self, gate: &RegressionGate) -> Vec<MetricDelta> {
        self.rows
            .iter()
            .filter(|r| match (gate.threshold(&r.metric), r.delta) {
                (Some(threshold), Some(delta)) => delta < -threshold,
                _ => false,
            })
            .cloned()
            .collect()
    }

    /// True when [`ReportDiff::regressions`] returns no rows for `gate`.
    /// Convenience accessor for CI scripts that just want a yes / no.
    #[must_use]
    pub fn is_clean(&self, gate: &RegressionGate) -> bool {
        self.regressions(gate).is_empty()
    }

    /// Process exit code suitable for `std::process::exit` in a CI eval
    /// binary: `0` when the diff passes `gate`, `1` when one or more
    /// metrics regress beyond their tolerated drop. Mirrors the
    /// long-standing UNIX convention of `0 = success, non-zero = failure`
    /// and is the single bit consumers should branch on.
    #[must_use]
    pub fn exit_code(&self, gate: &RegressionGate) -> i32 {
        if self.is_clean(gate) { 0 } else { 1 }
    }
}

/// Threshold-based regression gate over a [`ReportDiff`].
///
/// Each entry maps a metric name to the **minimum tolerated drop** in mean
/// score: a metric regresses when its `delta` is more negative than
/// `-threshold`. Thresholds are non-negative; negative values are clamped
/// to zero on insert.
///
/// ```
/// use rig_retrieval_evals::RegressionGate;
///
/// let gate = RegressionGate::new()
///     .with_threshold("recall@10", 0.02)
///     .with_threshold("ndcg@10", 0.01);
/// assert_eq!(gate.threshold("recall@10"), Some(0.02));
/// assert_eq!(gate.threshold("mrr"), None);
/// ```
#[derive(Debug, Clone, Default)]
pub struct RegressionGate {
    thresholds: BTreeMap<String, f64>,
}

impl RegressionGate {
    /// Build an empty gate. Metrics added via
    /// [`RegressionGate::with_threshold`] participate in regression checks;
    /// any others are ignored.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `threshold` as the maximum tolerated drop in mean score
    /// for `metric`. Negative values are clamped to `0.0`.
    #[must_use]
    pub fn with_threshold(mut self, metric: impl Into<String>, threshold: f64) -> Self {
        self.thresholds.insert(metric.into(), threshold.max(0.0));
        self
    }

    /// Threshold registered for `metric`, if any.
    #[must_use]
    pub fn threshold(&self, metric: &str) -> Option<f64> {
        self.thresholds.get(metric).copied()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::staleness::{ConflictGroup, ConflictReport, StaleHit, StalenessReport};

    #[test]
    fn metric_report_aggregates() {
        let r = MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 0.0), ("q2".into(), 0.5), ("q3".into(), 1.0)],
        );
        assert_eq!(r.n, 3);
        assert!((r.mean - 0.5).abs() < 1e-9);
        assert!((r.min - 0.0).abs() < 1e-9);
        assert!((r.max - 1.0).abs() < 1e-9);
        assert!((r.p50 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn empty_report_is_zero() {
        let r = MetricReport::from_per_query("m".into(), vec![]);
        assert_eq!(r.n, 0);
        assert_eq!(r.mean, 0.0);
    }

    #[test]
    fn diff_flags_fingerprint_mismatch() {
        let a = MultiReport::new(vec![]).with_judge_fingerprint("a");
        let b = MultiReport::new(vec![]).with_judge_fingerprint("b");
        assert!(a.diff(&b).is_err());
    }

    #[test]
    fn diff_computes_per_metric_delta() {
        let cur = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 0.8)],
        )]);
        let base = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 0.6)],
        )]);
        let diff = cur.diff(&base).unwrap();
        assert_eq!(diff.rows.len(), 1);
        let row = &diff.rows[0];
        assert!((row.delta.unwrap_or(0.0) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn diff_buckets_per_query_winners_losers_and_unchanged() {
        let cur = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![
                ("q1".into(), 1.0), // winner: 0.5 -> 1.0
                ("q2".into(), 0.0), // loser:  0.5 -> 0.0
                ("q3".into(), 0.5), // unchanged
                ("q4".into(), 0.9), // current-only, skipped
            ],
        )]);
        let base = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![
                ("q1".into(), 0.5),
                ("q2".into(), 0.5),
                ("q3".into(), 0.5),
                ("q5".into(), 1.0), // baseline-only, skipped
            ],
        )]);
        let diff = cur.diff(&base).unwrap();
        let row = &diff.rows[0];
        assert_eq!(row.winners, 1);
        assert_eq!(row.losers, 1);
        assert_eq!(row.unchanged, 1);
        // q4 / q5 are skipped because they are not in both reports.
        assert_eq!(row.query_changes.len(), 3);
        // Sorted by |delta| desc: q1 and q2 tie at 0.5, q3 at 0.0.
        assert_eq!(row.query_changes[2].query_id, "q3");
        assert!((row.query_changes[2].delta).abs() < 1e-9);
    }

    #[test]
    fn diff_query_changes_empty_when_baseline_missing_metric() {
        let cur = MultiReport::new(vec![MetricReport::from_per_query(
            "ndcg@10".into(),
            vec![("q1".into(), 0.9)],
        )]);
        let base = MultiReport::new(vec![]);
        let diff = cur.diff(&base).unwrap();
        let row = &diff.rows[0];
        assert!(row.delta.is_none());
        assert_eq!(row.winners, 0);
        assert_eq!(row.losers, 0);
        assert_eq!(row.unchanged, 0);
        assert!(row.query_changes.is_empty());
    }

    #[test]
    fn regression_gate_flags_only_metrics_below_threshold() {
        // recall@10 drops 0.10 (regression), ndcg@10 drops 0.005 (within
        // tolerance), mrr is not in the gate (ignored).
        let cur = MultiReport::new(vec![
            MetricReport::from_per_query("recall@10".into(), vec![("q1".into(), 0.50)]),
            MetricReport::from_per_query("ndcg@10".into(), vec![("q1".into(), 0.595)]),
            MetricReport::from_per_query("mrr".into(), vec![("q1".into(), 0.10)]),
        ]);
        let base = MultiReport::new(vec![
            MetricReport::from_per_query("recall@10".into(), vec![("q1".into(), 0.60)]),
            MetricReport::from_per_query("ndcg@10".into(), vec![("q1".into(), 0.60)]),
            MetricReport::from_per_query("mrr".into(), vec![("q1".into(), 0.90)]),
        ]);
        let diff = cur.diff(&base).unwrap();
        let gate = RegressionGate::new()
            .with_threshold("recall@10", 0.02)
            .with_threshold("ndcg@10", 0.02);
        let regressed = diff.regressions(&gate);
        assert_eq!(regressed.len(), 1);
        assert_eq!(regressed[0].metric, "recall@10");
    }

    #[test]
    fn regression_gate_clamps_negative_thresholds() {
        let gate = RegressionGate::new().with_threshold("recall@10", -0.5);
        assert_eq!(gate.threshold("recall@10"), Some(0.0));
    }

    #[test]
    fn report_diff_to_json_round_trips() {
        let cur = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 1.0), ("q2".into(), 0.0)],
        )]);
        let base = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 0.5), ("q2".into(), 0.5)],
        )]);
        let diff = cur.diff(&base).unwrap();
        let json = diff.to_json().unwrap();
        let parsed: ReportDiff = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.rows.len(), 1);
        assert_eq!(parsed.rows[0].winners, 1);
        assert_eq!(parsed.rows[0].losers, 1);
        assert_eq!(parsed.rows[0].query_changes.len(), 2);
    }

    #[test]
    fn reliability_report_computes_pass_at_k_and_pass_all_k() {
        let reports = vec![
            MetricReport::from_per_query(
                "recall@10".into(),
                vec![("q1".into(), 1.0), ("q2".into(), 0.0)],
            ),
            MetricReport::from_per_query(
                "recall@10".into(),
                vec![("q1".into(), 0.0), ("q2".into(), 1.0)],
            ),
            MetricReport::from_per_query(
                "recall@10".into(),
                vec![("q1".into(), 1.0), ("q2".into(), 0.0)],
            ),
        ];

        let reliability =
            ReliabilityReport::from_metric_reports("recall@10", 1.0, 2, &reports).unwrap();

        assert_eq!(reliability.n_queries, 2);
        assert_eq!(reliability.trials_per_query, 3);
        assert!((reliability.mean_pass_rate - 0.5).abs() < 1e-9);
        // q1: 2/3 successes => pass@2 = 1.0, pass^2 = 1/3.
        // q2: 1/3 successes => pass@2 = 2/3, pass^2 = 0.0.
        assert!((reliability.pass_at_k - (5.0 / 6.0)).abs() < 1e-9);
        assert!((reliability.pass_all_k - (1.0 / 6.0)).abs() < 1e-9);
        assert_eq!(reliability.per_query[0].query_id, "q1");
        assert_eq!(reliability.per_query[0].successes, 2);
    }

    #[test]
    fn reliability_report_requires_matching_metrics() {
        let reports = vec![
            MetricReport::from_per_query("recall@10".into(), vec![("q1".into(), 1.0)]),
            MetricReport::from_per_query("ndcg@10".into(), vec![("q1".into(), 1.0)]),
        ];

        let err =
            ReliabilityReport::from_metric_reports("recall@10", 1.0, 1, &reports).unwrap_err();

        match err {
            Error::BaselineMismatch(message) => assert!(message.contains("metric mismatch")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn reliability_report_requires_same_queries() {
        let reports = vec![
            MetricReport::from_per_query("recall@10".into(), vec![("q1".into(), 1.0)]),
            MetricReport::from_per_query("recall@10".into(), vec![("q2".into(), 1.0)]),
        ];

        let err =
            ReliabilityReport::from_metric_reports("recall@10", 1.0, 1, &reports).unwrap_err();

        match err {
            Error::BaselineMismatch(message) => assert!(message.contains("missing query id")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn reliability_report_rejects_k_larger_than_trials() {
        let reports = vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 1.0)],
        )];

        let err =
            ReliabilityReport::from_metric_reports("recall@10", 1.0, 2, &reports).unwrap_err();

        match err {
            Error::Config(message) => assert!(message.contains("exceeds trial count")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn reliability_report_serializes() {
        let reports = vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 1.0)],
        )];

        let reliability =
            ReliabilityReport::from_metric_reports("recall@10", 1.0, 1, &reports).unwrap();
        let json = serde_json::to_string(&reliability).unwrap();
        let parsed: ReliabilityReport = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.metric, "recall@10");
        assert_eq!(parsed.per_query.len(), 1);
        assert!((parsed.pass_at_k - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bootstrap_ci_brackets_the_mean_for_well_separated_scores() {
        let r = MetricReport::from_per_query(
            "recall@10".into(),
            (0..50)
                .map(|i| (format!("q{i}"), if i % 2 == 0 { 0.8 } else { 0.6 }))
                .collect(),
        )
        .with_bootstrap_ci(1000, 0.95, 0xC0FFEE);
        let ci = r.ci.unwrap();
        assert!(ci.lower < r.mean);
        assert!(ci.upper > r.mean);
        assert!(ci.lower >= 0.6 - 1e-9);
        assert!(ci.upper <= 0.8 + 1e-9);
        assert_eq!(ci.iterations, 1000);
        assert!((ci.level - 0.95).abs() < 1e-9);
    }

    #[test]
    fn bootstrap_ci_is_deterministic_for_a_fixed_seed() {
        let scores: Vec<(String, f64)> = (0..30)
            .map(|i| (format!("q{i}"), (i % 5) as f64 / 4.0))
            .collect();
        let a = MetricReport::from_per_query("m".into(), scores.clone())
            .with_bootstrap_ci(500, 0.9, 42)
            .ci
            .unwrap();
        let b = MetricReport::from_per_query("m".into(), scores)
            .with_bootstrap_ci(500, 0.9, 42)
            .ci
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn bootstrap_ci_rejects_invalid_input() {
        let r = MetricReport::from_per_query("m".into(), vec![("q1".into(), 0.5)]);
        assert!(r.bootstrap_ci(0, 0.95, 1).is_none());
        assert!(r.bootstrap_ci(100, 0.0, 1).is_none());
        assert!(r.bootstrap_ci(100, 1.0, 1).is_none());
        let empty = MetricReport::from_per_query("m".into(), vec![]);
        assert!(empty.bootstrap_ci(100, 0.95, 1).is_none());
    }

    #[test]
    fn report_diff_exit_code_signals_regression() {
        let baseline = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 0.9), ("q2".into(), 0.9)],
        )]);
        let candidate = MultiReport::new(vec![MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 0.4), ("q2".into(), 0.4)],
        )]);
        let diff = candidate.diff(&baseline).unwrap();
        let gate = RegressionGate::new().with_threshold("recall@10", 0.05);
        assert!(!diff.is_clean(&gate));
        assert_eq!(diff.exit_code(&gate), 1);

        let lax_gate = RegressionGate::new().with_threshold("recall@10", 1.0);
        assert!(diff.is_clean(&lax_gate));
        assert_eq!(diff.exit_code(&lax_gate), 0);
    }

    #[test]
    fn report_diff_carries_bootstrap_ci_on_both_sides() {
        let scores: Vec<(String, f64)> = (0..20).map(|i| (format!("q{i}"), 0.5)).collect();
        let candidate = MultiReport::new(vec![
            MetricReport::from_per_query("recall@10".into(), scores.clone())
                .with_bootstrap_ci(200, 0.95, 7),
        ]);
        let baseline = MultiReport::new(vec![
            MetricReport::from_per_query("recall@10".into(), scores)
                .with_bootstrap_ci(200, 0.95, 7),
        ]);
        let diff = candidate.diff(&baseline).unwrap();
        let row = diff.rows.first().unwrap();
        assert!(row.current_ci.is_some());
        assert!(row.baseline_ci.is_some());
    }

    #[test]
    fn freshness_report_rolls_up_query_rates_and_appends_gateable_metrics() {
        let staleness = vec![
            StalenessReport {
                query_id: "q1".into(),
                stale_hits: vec![StaleHit {
                    doc_id: "old".into(),
                    rank: 0,
                    superseded_by: "new".into(),
                }],
                considered: 2,
            },
            StalenessReport {
                query_id: "q2".into(),
                stale_hits: vec![],
                considered: 2,
            },
        ];
        let conflicts = vec![
            ConflictReport {
                query_id: "q1".into(),
                groups: vec![ConflictGroup {
                    version_key: "alice:address".into(),
                    doc_ids: vec!["old".into(), "new".into()],
                }],
                conflicting_doc_count: 2,
                considered: 2,
            },
            ConflictReport {
                query_id: "q2".into(),
                groups: vec![],
                conflicting_doc_count: 0,
                considered: 2,
            },
        ];

        let freshness = FreshnessReport::from_query_reports(2, &staleness, &conflicts).unwrap();

        assert_eq!(freshness.query_count, 2);
        assert_eq!(freshness.total_considered, 4);
        assert_eq!(freshness.stale_hit_count, 1);
        assert_eq!(freshness.stale_query_count, 1);
        assert!((freshness.stale_rate - 0.25).abs() < 1e-9);
        assert!((freshness.stale_query_rate - 0.5).abs() < 1e-9);
        assert_eq!(freshness.conflict_group_count, 1);
        assert_eq!(freshness.conflicting_doc_count, 2);
        assert!((freshness.conflict_rate - 0.5).abs() < 1e-9);
        assert_eq!(freshness.per_query[0].stale_rate, 0.5);
        assert_eq!(freshness.per_query[0].conflict_rate, 1.0);

        let report = MultiReport::new(vec![]).with_freshness_metrics(freshness);
        assert!(report.freshness.is_some());
        assert_eq!(report.metrics.len(), 2);
        assert_eq!(report.metrics[0].metric, "freshness.stale_free_rate@2");
        assert!((report.metrics[0].mean - 0.75).abs() < 1e-9);
        assert_eq!(report.metrics[1].metric, "freshness.conflict_free_rate@2");
        assert!((report.metrics[1].mean - 0.5).abs() < 1e-9);
    }

    #[test]
    fn freshness_metrics_participate_in_existing_regression_gate() {
        let baseline_freshness = FreshnessReport::from_query_reports(
            2,
            &[StalenessReport {
                query_id: "q1".into(),
                stale_hits: vec![],
                considered: 2,
            }],
            &[ConflictReport {
                query_id: "q1".into(),
                groups: vec![],
                conflicting_doc_count: 0,
                considered: 2,
            }],
        )
        .unwrap();
        let candidate_freshness = FreshnessReport::from_query_reports(
            2,
            &[StalenessReport {
                query_id: "q1".into(),
                stale_hits: vec![StaleHit {
                    doc_id: "old".into(),
                    rank: 0,
                    superseded_by: "new".into(),
                }],
                considered: 2,
            }],
            &[ConflictReport {
                query_id: "q1".into(),
                groups: vec![],
                conflicting_doc_count: 0,
                considered: 2,
            }],
        )
        .unwrap();

        let baseline = MultiReport::new(vec![]).with_freshness_metrics(baseline_freshness);
        let candidate = MultiReport::new(vec![]).with_freshness_metrics(candidate_freshness);
        let diff = candidate.diff(&baseline).unwrap();
        let gate = RegressionGate::new().with_threshold("freshness.stale_free_rate@2", 0.1);

        let regressions = diff.regressions(&gate);
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].metric, "freshness.stale_free_rate@2");
    }

    #[test]
    fn freshness_report_requires_matching_query_reports() {
        let err = FreshnessReport::from_query_reports(
            5,
            &[StalenessReport {
                query_id: "q1".into(),
                stale_hits: vec![],
                considered: 1,
            }],
            &[ConflictReport {
                query_id: "q2".into(),
                groups: vec![],
                conflicting_doc_count: 0,
                considered: 1,
            }],
        )
        .unwrap_err();

        match err {
            Error::BaselineMismatch(message) => {
                assert!(message.contains("freshness query mismatch"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
