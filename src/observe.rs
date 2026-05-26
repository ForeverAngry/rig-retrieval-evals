//! `rig-tap`-compatible event envelopes for retrieval evaluation reports.
//!
//! This module is gated behind the `observe` feature. It mirrors the
//! [`rig-tap`](https://crates.io/crates/rig-tap) v1 wire shape — a flat
//! JSON envelope with `version`, `occurred_at_millis`, `tick`,
//! `conversation_id`, and a flattened `kind` payload — **without**
//! depending on `rig-tap`. Hosts that already ship a `rig-tap`-aware
//! collector see eval reports on the same `rig_tap` tracing target,
//! routed and indexed via the same `rig_tap.kind` / `rig_tap.metric`
//! scalar attributes used by every other producer in the ecosystem.
//!
//! Two custom kinds are introduced:
//!
//! - `eval.retrieval_report` — one envelope per [`MetricReport`] inside
//!   a [`MultiReport`]. Carries `metric`, `n`, `mean`, and (when
//!   present) the bootstrap CI bounds.
//! - `eval.regression_diff` — one envelope per [`MetricDelta`] inside a
//!   [`ReportDiff`]. Carries `metric`, `current_mean`, `baseline_mean`,
//!   `delta`, and the gate verdict `regressed`.
//!
//! These names sit outside the typed [`rig-tap`](https://crates.io/crates/rig-tap)
//! `EventKind` enum on purpose: a strict serde decoder pinned to
//! `rig-tap` will reject them, while collectors that read JSON
//! (OpenTelemetry, Phoenix, log shippers) accept them transparently.
//! When [`rig-tap`](https://crates.io/crates/rig-tap) promotes
//! `eval.report` into its schema, this module can switch to typed
//! emission without a wire-format change.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::report::{MetricCi, MetricDelta, MetricReport, MultiReport, RegressionGate, ReportDiff};

/// Wire-format version. Matches `rig-tap`'s `SCHEMA_VERSION` so
/// collectors can treat both producers identically.
pub const SCHEMA_VERSION: u32 = 1;

static TICK: AtomicU64 = AtomicU64::new(0);

fn next_tick() -> u64 {
    TICK.fetch_add(1, Ordering::Relaxed)
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Flat envelope that serializes to the same JSON shape as
/// `rig_tap::ObservabilityEvent`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalEnvelope {
    /// Schema version (matches `rig-tap`'s `SCHEMA_VERSION`).
    pub version: u32,
    /// Wall-clock timestamp in milliseconds since the Unix epoch.
    pub occurred_at_millis: u64,
    /// Monotonic per-process counter. Use to order events without clock skew.
    pub tick: u64,
    /// Caller-supplied correlation id. For eval runs this is typically
    /// the run id or git SHA being evaluated.
    pub conversation_id: String,
    /// Payload tagged on the wire as `"kind": "eval.<name>"`. Flattened
    /// into the parent object, identical to `rig-tap`'s flatten layout.
    #[serde(flatten)]
    pub kind: EvalKind,
}

/// Eval-specific payload variants. Tagged on the wire as
/// `"kind": "eval.<dotted.name>"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[non_exhaustive]
pub enum EvalKind {
    /// One [`MetricReport`] inside a [`MultiReport`].
    #[serde(rename = "eval.retrieval_report")]
    RetrievalReport {
        /// Metric label, e.g. `"recall@10"`, `"ndcg@10"`.
        metric: String,
        /// Number of queries that contributed to the report.
        n: usize,
        /// Mean score across all per-query scores.
        mean: f64,
        /// Lower bound of the bootstrap confidence interval, when computed.
        #[serde(skip_serializing_if = "Option::is_none")]
        ci_lower: Option<f64>,
        /// Upper bound of the bootstrap confidence interval, when computed.
        #[serde(skip_serializing_if = "Option::is_none")]
        ci_upper: Option<f64>,
        /// Two-sided coverage probability for `ci_lower` / `ci_upper`.
        #[serde(skip_serializing_if = "Option::is_none")]
        ci_level: Option<f64>,
    },
    /// One [`MetricDelta`] row from a [`ReportDiff`].
    #[serde(rename = "eval.regression_diff")]
    RegressionDiff {
        /// Metric label.
        metric: String,
        /// Mean for the candidate report.
        current_mean: f64,
        /// Mean for the baseline report, when the metric was present
        /// on both sides.
        #[serde(skip_serializing_if = "Option::is_none")]
        baseline_mean: Option<f64>,
        /// `current_mean - baseline_mean`, when defined.
        #[serde(skip_serializing_if = "Option::is_none")]
        delta: Option<f64>,
        /// `true` when the [`RegressionGate`] flagged this row as a
        /// regression. `false` for both "passed" and "no baseline".
        regressed: bool,
    },
}

/// Convert a [`MultiReport`] into one [`EvalEnvelope`] per metric.
///
/// `conversation_id` is forwarded into every envelope; collectors use it
/// to correlate the rows of one eval run.
#[must_use]
pub fn report_envelopes(report: &MultiReport, conversation_id: &str) -> Vec<EvalEnvelope> {
    report
        .metrics
        .iter()
        .map(|m| envelope(conversation_id, retrieval_kind(m)))
        .collect()
}

/// Convert a [`ReportDiff`] into one [`EvalEnvelope`] per row.
///
/// `gate` decides the `regressed` flag: rows whose delta is below the
/// gate threshold for their metric are marked `regressed = true`.
#[must_use]
pub fn diff_envelopes(
    diff: &ReportDiff,
    gate: &RegressionGate,
    conversation_id: &str,
) -> Vec<EvalEnvelope> {
    diff.rows
        .iter()
        .map(|row| envelope(conversation_id, regression_kind(row, gate)))
        .collect()
}

/// Emit every envelope produced by [`report_envelopes`] through
/// `tracing::info!` on the `rig_tap` target.
///
/// Each event carries the full JSON envelope under the `event` field
/// plus stable `rig_tap.kind` / `rig_tap.metric` scalar attributes so
/// OpenTelemetry collectors can route without parsing the JSON string.
pub fn emit_report(report: &MultiReport, conversation_id: &str) {
    for env in report_envelopes(report, conversation_id) {
        emit(&env);
    }
}

/// Emit every envelope produced by [`diff_envelopes`] through
/// `tracing::info!` on the `rig_tap` target.
pub fn emit_diff(diff: &ReportDiff, gate: &RegressionGate, conversation_id: &str) {
    for env in diff_envelopes(diff, gate, conversation_id) {
        emit(&env);
    }
}

fn emit(env: &EvalEnvelope) {
    let json = serde_json::to_string(env).unwrap_or_else(|_| String::from("{}"));
    match &env.kind {
        EvalKind::RetrievalReport { metric, .. } => {
            tracing::info!(
                target: "rig_tap",
                event = %json,
                rig_tap.kind = "eval.retrieval_report",
                rig_tap.metric = %metric,
                rig_tap.conversation_id = %env.conversation_id,
            );
        }
        EvalKind::RegressionDiff {
            metric, regressed, ..
        } => {
            tracing::info!(
                target: "rig_tap",
                event = %json,
                rig_tap.kind = "eval.regression_diff",
                rig_tap.metric = %metric,
                rig_tap.regressed = *regressed,
                rig_tap.conversation_id = %env.conversation_id,
            );
        }
    }
}

fn envelope(conversation_id: &str, kind: EvalKind) -> EvalEnvelope {
    EvalEnvelope {
        version: SCHEMA_VERSION,
        occurred_at_millis: now_millis(),
        tick: next_tick(),
        conversation_id: conversation_id.to_string(),
        kind,
    }
}

fn retrieval_kind(m: &MetricReport) -> EvalKind {
    let (ci_lower, ci_upper, ci_level) = match m.ci {
        Some(MetricCi {
            lower,
            upper,
            level,
            ..
        }) => (Some(lower), Some(upper), Some(level)),
        None => (None, None, None),
    };
    EvalKind::RetrievalReport {
        metric: m.metric.clone(),
        n: m.n,
        mean: m.mean,
        ci_lower,
        ci_upper,
        ci_level,
    }
}

fn regression_kind(row: &MetricDelta, gate: &RegressionGate) -> EvalKind {
    let regressed = match (gate.threshold(&row.metric), row.delta) {
        (Some(threshold), Some(delta)) => delta < -threshold,
        _ => false,
    };
    EvalKind::RegressionDiff {
        metric: row.metric.clone(),
        current_mean: row.current_mean,
        baseline_mean: row.baseline_mean,
        delta: row.delta,
        regressed,
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use crate::report::MetricReport;

    fn report() -> MultiReport {
        let m = MetricReport::from_per_query(
            "recall@10".into(),
            vec![("q1".into(), 1.0), ("q2".into(), 0.5)],
        )
        .with_bootstrap_ci(200, 0.9, 1);
        MultiReport::new(vec![m])
    }

    #[test]
    fn report_envelopes_carry_metric_and_ci() {
        let envs = report_envelopes(&report(), "run-abc");
        assert_eq!(envs.len(), 1);
        let env = &envs[0];
        assert_eq!(env.version, SCHEMA_VERSION);
        assert_eq!(env.conversation_id, "run-abc");
        match &env.kind {
            EvalKind::RetrievalReport {
                metric,
                n,
                mean,
                ci_lower,
                ci_upper,
                ci_level,
            } => {
                assert_eq!(metric, "recall@10");
                assert_eq!(*n, 2);
                assert!((mean - 0.75).abs() < 1e-9);
                assert!(ci_lower.is_some());
                assert!(ci_upper.is_some());
                assert_eq!(*ci_level, Some(0.9));
            }
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    #[test]
    fn report_envelope_omits_ci_when_absent() {
        let no_ci = MultiReport::new(vec![MetricReport::from_per_query(
            "ndcg@10".into(),
            vec![("q1".into(), 0.8)],
        )]);
        let envs = report_envelopes(&no_ci, "run-1");
        let json = serde_json::to_string(&envs[0]).unwrap();
        assert!(!json.contains("ci_lower"));
        assert!(!json.contains("ci_upper"));
        assert!(!json.contains("ci_level"));
    }

    #[test]
    fn diff_envelope_marks_regression_against_gate() {
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
        let envs = diff_envelopes(&diff, &gate, "run-2");
        match &envs[0].kind {
            EvalKind::RegressionDiff {
                regressed, delta, ..
            } => {
                assert!(*regressed);
                assert!(delta.unwrap() < 0.0);
            }
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    #[test]
    fn envelope_serializes_with_flat_kind_tag() {
        let envs = report_envelopes(&report(), "run-flat");
        let json = serde_json::to_string(&envs[0]).unwrap();
        assert!(json.contains("\"kind\":\"eval.retrieval_report\""));
        assert!(json.contains("\"conversation_id\":\"run-flat\""));
        assert!(json.contains("\"version\":1"));
    }

    #[test]
    fn ticks_are_monotonic_within_a_batch() {
        let envs = report_envelopes(&report(), "run-tick");
        // Single-row report; emit two more to compare ticks.
        let next = report_envelopes(&report(), "run-tick");
        assert!(next[0].tick > envs[0].tick);
    }
}
