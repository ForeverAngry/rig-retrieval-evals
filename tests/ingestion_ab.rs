//! Fixture-based A/B ingestion test: simulate two ingestion configurations
//! (baseline vs candidate) by indexing different document subsets in a
//! token-overlap mock store, run the same `Qrels` through both, then assert
//! that [`MultiReport::diff`] surfaces per-query winners/losers and that the
//! [`RegressionGate`] catches a real regression.
//!
//! This is the pure-Rust analogue of "did re-chunking / re-extracting my
//! corpus help or regress retrieval?" — the question the harness exists to
//! answer.

#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::HashMap;

use rig::vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex, request::Filter};
use rig::wasm_compat::WasmCompatSend;
use rig_retrieval_evals::{
    NdcgAtK, Qrels, RecallAtK, RegressionGate, RetrievalHarness, RetrievalMetric,
};
use serde::Deserialize;

/// Token-overlap mock store, parameterized by the document set it indexes.
/// Two instances with different document sets stand in for "before" and
/// "after" an ingestion change.
struct MockStore {
    docs: HashMap<String, Vec<String>>,
}

impl MockStore {
    fn new(entries: Vec<(&str, &str)>) -> Self {
        let docs = entries
            .into_iter()
            .map(|(id, text)| (id.to_string(), tokens(text)))
            .collect();
        Self { docs }
    }

    fn rank(&self, query: &str, k: usize) -> Vec<(f64, String)> {
        let q_tokens: Vec<String> = tokens(query);
        let mut scored: Vec<(f64, String)> = self
            .docs
            .iter()
            .map(|(id, doc_tokens)| {
                let overlap = doc_tokens.iter().filter(|t| q_tokens.contains(t)).count();
                (overlap as f64, id.clone())
            })
            .filter(|(s, _)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        scored.truncate(k);
        scored
    }
}

fn tokens(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

impl VectorStoreIndex for MockStore {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'a> Deserialize<'a> + WasmCompatSend,
    {
        let ranked = self.rank(req.query(), req.samples() as usize);
        let mut out = Vec::with_capacity(ranked.len());
        for (score, id) in ranked {
            let value = serde_json::json!({ "id": id });
            let doc: T = serde_json::from_value(value).map_err(VectorStoreError::JsonError)?;
            out.push((score, id, doc));
        }
        Ok(out)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(self.rank(req.query(), req.samples() as usize))
    }
}

fn metrics() -> Vec<Box<dyn RetrievalMetric>> {
    vec![Box::new(RecallAtK::new(5)), Box::new(NdcgAtK::new(5))]
}

/// Baseline corpus: q1's strongest relevant doc (`doc-orwell`) is missing.
fn baseline_store() -> MockStore {
    MockStore::new(vec![
        ("doc-1984", "1984 is a dystopian novel by Orwell"),
        ("doc-paris", "Paris is the capital of France"),
        ("doc-light", "The speed of light in vacuum is 299792458 m/s"),
        (
            "doc-physics",
            "Physics describes the speed of light and gravity",
        ),
        ("doc-noise", "Unrelated chatter about cooking pasta"),
    ])
}

/// Candidate ingestion adds `doc-orwell` (strongest q1 hit). Expected
/// outcome: q1 improves, q2/q3 unchanged, no regressions.
fn improved_store() -> MockStore {
    MockStore::new(vec![
        ("doc-orwell", "George Orwell wrote 1984 and Animal Farm"),
        ("doc-1984", "1984 is a dystopian novel by Orwell"),
        ("doc-paris", "Paris is the capital of France"),
        ("doc-light", "The speed of light in vacuum is 299792458 m/s"),
        (
            "doc-physics",
            "Physics describes the speed of light and gravity",
        ),
        ("doc-noise", "Unrelated chatter about cooking pasta"),
    ])
}

/// Candidate ingestion regresses: `doc-light` and `doc-physics` were dropped
/// from the index, so q3 ("speed of light") finds nothing relevant.
fn regressed_store() -> MockStore {
    MockStore::new(vec![
        ("doc-orwell", "George Orwell wrote 1984 and Animal Farm"),
        ("doc-1984", "1984 is a dystopian novel by Orwell"),
        ("doc-paris", "Paris is the capital of France"),
        ("doc-noise", "Unrelated chatter about cooking pasta"),
    ])
}

#[tokio::test]
async fn improvement_diff_shows_winners_and_no_regressions() {
    let qrels = Qrels::load_jsonl("tests/data/tiny_qrels.jsonl").unwrap();

    let base_report = RetrievalHarness::new(&baseline_store(), 5)
        .run(&qrels, &metrics())
        .await
        .unwrap();
    let cur_report = RetrievalHarness::new(&improved_store(), 5)
        .run(&qrels, &metrics())
        .await
        .unwrap();

    let diff = cur_report.diff(&base_report).unwrap();
    let recall = diff.rows.iter().find(|r| r.metric == "recall@5").unwrap();

    // Mean improves, at least q1 wins, no query regresses.
    assert!(
        recall.delta.unwrap_or(0.0) > 0.0,
        "expected recall@5 to improve, got {:?}",
        recall.delta
    );
    assert!(
        recall.winners >= 1,
        "expected at least one query winner, got {}",
        recall.winners
    );
    assert_eq!(
        recall.losers, 0,
        "no query should regress in the improvement scenario"
    );
    // q1 should appear with a positive delta and lead the sorted movers.
    let top = recall.query_changes.first().unwrap();
    assert_eq!(top.query_id, "q1");
    assert!(top.delta > 0.0);

    // The gate is silent when nothing regresses.
    let gate = RegressionGate::new()
        .with_threshold("recall@5", 0.02)
        .with_threshold("ndcg@5", 0.02);
    assert!(
        diff.regressions(&gate).is_empty(),
        "regression gate fired on a strict improvement: {:?}",
        diff.regressions(&gate)
    );

    // Markdown summary surfaces winner/loser columns.
    let md = diff.to_markdown();
    assert!(md.contains("win"));
    assert!(md.contains("lose"));
}

#[tokio::test]
async fn regression_diff_flags_dropped_docs() {
    let qrels = Qrels::load_jsonl("tests/data/tiny_qrels.jsonl").unwrap();

    let base_report = RetrievalHarness::new(&improved_store(), 5)
        .run(&qrels, &metrics())
        .await
        .unwrap();
    let cur_report = RetrievalHarness::new(&regressed_store(), 5)
        .run(&qrels, &metrics())
        .await
        .unwrap();

    let diff = cur_report.diff(&base_report).unwrap();
    let recall = diff.rows.iter().find(|r| r.metric == "recall@5").unwrap();

    assert!(
        recall.delta.unwrap_or(0.0) < 0.0,
        "expected recall@5 to regress, got {:?}",
        recall.delta
    );
    assert!(
        recall.losers >= 1,
        "expected at least one query loser, got {}",
        recall.losers
    );
    // q3 ("speed of light") is the dropped query and should top the movers.
    let top = recall.query_changes.first().unwrap();
    assert_eq!(top.query_id, "q3");
    assert!(top.delta < 0.0);

    // A 2% gate flags this regression for both configured metrics.
    let gate = RegressionGate::new()
        .with_threshold("recall@5", 0.02)
        .with_threshold("ndcg@5", 0.02);
    let regressions = diff.regressions(&gate);
    assert!(
        !regressions.is_empty(),
        "regression gate missed a real drop: diff={}",
        diff.to_markdown()
    );
    assert!(regressions.iter().any(|r| r.metric == "recall@5"));
}
