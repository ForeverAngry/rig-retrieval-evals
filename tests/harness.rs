//! End-to-end test of the retrieval harness against a hand-rolled mock
//! `VectorStoreIndex`. Validates that the harness drives an arbitrary store,
//! retrievals flow into the metric pipeline, and the multi-report exposes
//! plausible aggregate values.

#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::HashMap;

use rig::vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex, request::Filter};
use rig::wasm_compat::WasmCompatSend;
use rig_retrieval_evals::{
    HitRateAtK, MapAtK, Mrr, NdcgAtK, PrecisionAtK, Qrels, RecallAtK, RetrievalHarness,
    RetrievalMetric,
};
use serde::Deserialize;

/// In-memory lexical mock: documents are indexed by token; queries score
/// candidates by token-overlap count. Deterministic, no embeddings.
struct MockStore {
    /// doc_id -> tokens
    docs: HashMap<String, Vec<String>>,
}

impl MockStore {
    fn new() -> Self {
        let mut docs = HashMap::new();
        docs.insert(
            "doc-orwell".to_string(),
            tokens("George Orwell wrote 1984 and Animal Farm"),
        );
        docs.insert(
            "doc-1984".to_string(),
            tokens("1984 is a dystopian novel by Orwell"),
        );
        docs.insert(
            "doc-paris".to_string(),
            tokens("Paris is the capital of France"),
        );
        docs.insert(
            "doc-light".to_string(),
            tokens("The speed of light in vacuum is 299792458 m/s"),
        );
        docs.insert(
            "doc-physics".to_string(),
            tokens("Physics describes the speed of light and gravity"),
        );
        docs.insert(
            "doc-noise".to_string(),
            tokens("Unrelated chatter about cooking pasta"),
        );
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
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
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

#[tokio::test]
async fn harness_drives_mock_store_end_to_end() {
    let qrels = Qrels::load_jsonl("tests/data/tiny_qrels.jsonl").unwrap();
    assert_eq!(qrels.len(), 3);

    let store = MockStore::new();

    let metrics: Vec<Box<dyn RetrievalMetric>> = vec![
        Box::new(RecallAtK::new(5)),
        Box::new(PrecisionAtK::new(5)),
        Box::new(HitRateAtK::new(5)),
        Box::new(Mrr),
        Box::new(MapAtK::new(5)),
        Box::new(NdcgAtK::new(5)),
    ];

    let report = RetrievalHarness::new(&store, 5)
        .with_concurrency(2)
        .run(&qrels, &metrics)
        .await
        .unwrap();

    assert_eq!(report.metrics.len(), 6);
    for m in &report.metrics {
        assert_eq!(m.n, 3, "metric {} should score all 3 queries", m.metric);
        assert!(
            (0.0..=1.0).contains(&m.mean),
            "metric {} mean out of range: {}",
            m.metric,
            m.mean
        );
    }

    // The token-overlap mock should hit at least one relevant doc per query.
    let hit_rate = report
        .metrics
        .iter()
        .find(|m| m.metric == "hit_rate@5")
        .unwrap();
    assert!(
        hit_rate.mean > 0.5,
        "mock should score most queries; got hit_rate@5 mean = {}",
        hit_rate.mean
    );

    // Round-trip through JSON and back.
    let json = report.to_json().unwrap();
    let parsed: rig_retrieval_evals::MultiReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.metrics.len(), 6);

    // Markdown rendering is non-empty.
    let md = report.to_markdown();
    assert!(md.contains("recall@5"));
    assert!(md.contains("ndcg@5"));
}

#[tokio::test]
async fn harness_rejects_zero_k() {
    let store = MockStore::new();
    let qrels = Qrels::default();
    let metrics: Vec<Box<dyn RetrievalMetric>> = vec![Box::new(Mrr)];
    let err = RetrievalHarness::new(&store, 0)
        .run(&qrels, &metrics)
        .await
        .unwrap_err();
    match err {
        rig_retrieval_evals::Error::Config(_) => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn baseline_diff_round_trip() {
    let qrels = Qrels::load_jsonl("tests/data/tiny_qrels.jsonl").unwrap();
    let store = MockStore::new();
    let metrics: Vec<Box<dyn RetrievalMetric>> = vec![Box::new(RecallAtK::new(5))];

    let cur = RetrievalHarness::new(&store, 5)
        .run(&qrels, &metrics)
        .await
        .unwrap();
    let base = cur.clone();
    let diff = cur.diff(&base).unwrap();
    assert_eq!(diff.rows.len(), 1);
    assert!(diff.rows[0].delta.unwrap_or(1.0).abs() < 1e-9);
}
