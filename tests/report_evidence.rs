//! Fixture documenting the `ReportDiff` evaluator-evidence contract.
//!
//! The boundary is intentionally small: a [`MultiReport::diff`] result
//! serialises to JSON via [`ReportDiff::to_json`], reloads via
//! `serde_json::from_str`, and round-trips a [`RegressionGate`] verdict
//! identically against either copy.
//!
//! Recommended metric names and threshold conventions used by
//! downstream consumers are codified here as test data so a schema drift
//! surfaces as a test failure.

#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use rig_retrieval_evals::{MetricReport, MultiReport, RegressionGate, ReportDiff};

fn report(metric: &str, per_query: &[(&str, f64)]) -> MultiReport {
    MultiReport::new(vec![MetricReport::from_per_query(
        metric.to_string(),
        per_query
            .iter()
            .map(|(q, s)| ((*q).to_string(), *s))
            .collect(),
    )])
}

/// A `ReportDiff` survives a JSON round-trip with a stable
/// `RegressionGate` verdict. This is the shape host evaluators can store
/// when promoting / rejecting candidates.
#[test]
fn report_diff_json_round_trips_with_stable_gate_verdict() {
    let baseline = report("recall@10", &[("q1", 0.6), ("q2", 0.7)]);
    let candidate = report("recall@10", &[("q1", 0.4), ("q2", 0.4)]);

    let diff = candidate.diff(&baseline).unwrap();
    let gate = RegressionGate::new().with_threshold("recall@10", 0.05);

    let regressed_live = diff.regressions(&gate);
    assert_eq!(regressed_live.len(), 1);
    assert_eq!(regressed_live[0].metric, "recall@10");

    let json = diff.to_json().unwrap();
    let reloaded: ReportDiff = serde_json::from_str(&json).unwrap();

    let regressed_reloaded = reloaded.regressions(&gate);
    assert_eq!(regressed_reloaded.len(), 1);
    assert_eq!(regressed_reloaded[0].metric, regressed_live[0].metric);
    assert!((regressed_reloaded[0].delta.unwrap() - regressed_live[0].delta.unwrap()).abs() < 1e-9);
}

/// A non-regressing diff produces an empty regression set both live
/// and after JSON reload — the contract host evaluators rely on for the
/// "promote" branch.
#[test]
fn report_diff_promotion_path_round_trips_clean() {
    let baseline = report("recall@10", &[("q1", 0.5), ("q2", 0.5)]);
    let candidate = report("recall@10", &[("q1", 1.0), ("q2", 0.6)]);

    let diff = candidate.diff(&baseline).unwrap();
    let gate = RegressionGate::new().with_threshold("recall@10", 0.05);

    assert!(diff.regressions(&gate).is_empty());

    let reloaded: ReportDiff = serde_json::from_str(&diff.to_json().unwrap()).unwrap();
    assert!(reloaded.regressions(&gate).is_empty());
}

/// Multi-metric diffs preserve per-metric rows across the round-trip
/// so consumers can index by metric name. Recommended names for
/// retrieval evaluators: `recall@K`, `ndcg@K`, `mrr`, `precision@K`,
/// `hit_rate@K`, `map@K`. Thresholds in the gate are non-negative
/// fractional drops (e.g. `0.02` tolerates a 2-point mean drop).
#[test]
fn report_diff_preserves_multi_metric_rows() {
    let baseline = MultiReport::new(vec![
        MetricReport::from_per_query("recall@10".into(), vec![("q1".into(), 0.60)]),
        MetricReport::from_per_query("ndcg@10".into(), vec![("q1".into(), 0.50)]),
        MetricReport::from_per_query("mrr".into(), vec![("q1".into(), 0.40)]),
    ]);
    let candidate = MultiReport::new(vec![
        MetricReport::from_per_query("recall@10".into(), vec![("q1".into(), 0.50)]),
        MetricReport::from_per_query("ndcg@10".into(), vec![("q1".into(), 0.55)]),
        MetricReport::from_per_query("mrr".into(), vec![("q1".into(), 0.40)]),
    ]);

    let diff = candidate.diff(&baseline).unwrap();
    let json = diff.to_json().unwrap();
    let reloaded: ReportDiff = serde_json::from_str(&json).unwrap();

    let names: Vec<&str> = reloaded.rows.iter().map(|r| r.metric.as_str()).collect();
    assert_eq!(names, vec!["recall@10", "ndcg@10", "mrr"]);

    // recall@10 regresses, ndcg@10 improves, mrr unchanged.
    let gate = RegressionGate::new()
        .with_threshold("recall@10", 0.02)
        .with_threshold("ndcg@10", 0.02)
        .with_threshold("mrr", 0.02);
    let regressed = reloaded.regressions(&gate);
    assert_eq!(regressed.len(), 1);
    assert_eq!(regressed[0].metric, "recall@10");
}
