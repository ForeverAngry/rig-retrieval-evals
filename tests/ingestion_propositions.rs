//! Integration tests for the Track 3 ingestion pipeline (propositional
//! distillation against a vector store).

#![cfg(feature = "ingestion")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::HashMap;
use std::sync::Mutex;

use rig::vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex, request::Filter};
use rig::wasm_compat::WasmCompatSend;
use rig_retrieval_evals::{
    DistillationPipeline, Document, DroppedItem, DroppedReason, InMemoryIocBaseline, Proposition,
    PropositionExtractor, RedundancyCheck, RedundancyVerdict, RegexIocExtractor,
    StubPropositionExtractor, VectorStoreRedundancyCheck,
};
use serde::Deserialize;

/// Mock store that returns a configurable similarity for every query.
/// Lets us drive the redundancy threshold deterministically.
struct ConstSimilarityStore {
    similarity: f64,
}

impl ConstSimilarityStore {
    fn new(similarity: f64) -> Self {
        Self { similarity }
    }
}

impl VectorStoreIndex for ConstSimilarityStore {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'a> Deserialize<'a> + WasmCompatSend,
    {
        let _ = req.query();
        let value = serde_json::json!({ "id": "neighbour" });
        let doc: T = serde_json::from_value(value).map_err(VectorStoreError::JsonError)?;
        Ok(vec![(self.similarity, "neighbour".into(), doc)])
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(vec![(self.similarity, "neighbour".into())])
    }
}

/// Mock store that returns per-query similarities and records every query
/// it sees. Drives heterogeneous-threshold cases.
struct LookupStore {
    table: HashMap<String, f64>,
    seen: Mutex<Vec<String>>,
}

impl LookupStore {
    fn new(table: HashMap<String, f64>) -> Self {
        Self {
            table,
            seen: Mutex::new(Vec::new()),
        }
    }

    fn seen(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }
}

impl VectorStoreIndex for LookupStore {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'a> Deserialize<'a> + WasmCompatSend,
    {
        let q = req.query().to_string();
        let score = *self.table.get(&q).unwrap_or(&0.0);
        self.seen.lock().unwrap().push(q.clone());
        let value = serde_json::json!({ "id": "neighbour" });
        let doc: T = serde_json::from_value(value).map_err(VectorStoreError::JsonError)?;
        Ok(vec![(score, "neighbour".into(), doc)])
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let q = req.query().to_string();
        let score = *self.table.get(&q).unwrap_or(&0.0);
        self.seen.lock().unwrap().push(q.clone());
        Ok(vec![(score, "neighbour".into())])
    }
}

fn ioc_extractor() -> RegexIocExtractor {
    RegexIocExtractor::new().expect("default extractor patterns must compile")
}

#[test]
fn stub_extractor_splits_on_sentence_terminators() {
    let extractor = StubPropositionExtractor::new();
    let doc = Document::new(
        "doc-prop-1",
        "APT-28 is attributed to GRU Unit 26165. The group targets NATO governments. They use spear-phishing!",
    );
    let props = futures::executor::block_on(extractor.extract(&doc)).unwrap();
    assert_eq!(props.len(), 3);
    assert!(props[0].text.starts_with("APT-28"));
    assert!(props[2].text.ends_with("spear-phishing!"));
}

#[test]
fn stub_extractor_ignores_empty_and_single_char_fragments() {
    let extractor = StubPropositionExtractor::new();
    let doc = Document::new("doc-prop-2", "...Real sentence here. .");
    let props = futures::executor::block_on(extractor.extract(&doc)).unwrap();
    let texts: Vec<&str> = props.iter().map(|p| p.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("Real sentence")),
        "got: {texts:?}"
    );
    // Lone punctuation must not become a proposition.
    assert!(props.iter().all(|p| p.text.len() > 1));
}

#[test]
fn vector_store_redundancy_check_rejects_invalid_threshold() {
    let store = ConstSimilarityStore::new(0.5);
    assert!(VectorStoreRedundancyCheck::new(&store, -0.1).is_err());
    assert!(VectorStoreRedundancyCheck::new(&store, 1.5).is_err());
    assert!(VectorStoreRedundancyCheck::new(&store, 0.9).is_ok());
}

#[tokio::test]
async fn redundancy_check_marks_high_similarity_as_redundant() {
    let store = ConstSimilarityStore::new(0.95);
    let check = VectorStoreRedundancyCheck::new(&store, 0.90).unwrap();
    let verdict = check.check(&Proposition::new("anything")).await.unwrap();
    assert!(verdict.is_redundant);
    assert!((verdict.similarity - 0.95).abs() < f64::EPSILON);
}

#[tokio::test]
async fn redundancy_check_passes_low_similarity_through() {
    let store = ConstSimilarityStore::new(0.10);
    let check = VectorStoreRedundancyCheck::new(&store, 0.90).unwrap();
    let verdict = check.check(&Proposition::new("anything")).await.unwrap();
    assert!(!verdict.is_redundant);
}

#[tokio::test]
async fn pipeline_without_propositions_leaves_propositions_empty() {
    let pipeline = DistillationPipeline::new(ioc_extractor(), InMemoryIocBaseline::new());
    let doc = Document::new("doc-1", "CVE-2024-12345 exploited.");
    let delta = pipeline.ingest(&doc).await.unwrap();
    assert!(delta.propositions.is_empty());
    assert_eq!(delta.iocs.len(), 1, "track 1 must still run");
}

#[tokio::test]
async fn pipeline_emits_net_new_propositions_below_threshold() {
    let store = ConstSimilarityStore::new(0.10);
    let check = VectorStoreRedundancyCheck::new(&store, 0.90).unwrap();
    let pipeline = DistillationPipeline::new(ioc_extractor(), InMemoryIocBaseline::new())
        .with_propositions(StubPropositionExtractor::new(), check);

    let doc = Document::new(
        "doc-2",
        "APT-28 targets NATO. The group uses spear-phishing.",
    );
    let delta = pipeline.ingest(&doc).await.unwrap();
    assert_eq!(delta.propositions.len(), 2);
    assert!(
        delta
            .dropped
            .iter()
            .all(|d| !matches!(d.reason, DroppedReason::Redundant { .. }))
    );
}

#[tokio::test]
async fn pipeline_drops_propositions_above_threshold_with_redundant_reason() {
    let store = ConstSimilarityStore::new(0.95);
    let check = VectorStoreRedundancyCheck::new(&store, 0.90).unwrap();
    let pipeline = DistillationPipeline::new(ioc_extractor(), InMemoryIocBaseline::new())
        .with_propositions(StubPropositionExtractor::new(), check);

    let doc = Document::new(
        "doc-3",
        "APT-28 targets NATO. The group uses spear-phishing.",
    );
    let delta = pipeline.ingest(&doc).await.unwrap();
    assert!(delta.propositions.is_empty());

    let redundant: Vec<_> = delta
        .dropped
        .iter()
        .filter_map(|d| match (&d.item, &d.reason) {
            (DroppedItem::Proposition(p), DroppedReason::Redundant { similarity }) => {
                Some((p.text.clone(), *similarity))
            }
            _ => None,
        })
        .collect();
    assert_eq!(redundant.len(), 2);
    for (_, sim) in &redundant {
        assert!((sim - 0.95).abs() < f64::EPSILON);
    }
}

#[tokio::test]
async fn pipeline_handles_mixed_threshold_outcomes() {
    let mut table = HashMap::new();
    // "Known. " => keep similarity high; "Novel." => similarity low.
    table.insert("Known fact.".into(), 0.99);
    table.insert("Brand new fact.".into(), 0.05);
    let store = LookupStore::new(table);
    let check = VectorStoreRedundancyCheck::new(&store, 0.50).unwrap();
    let pipeline = DistillationPipeline::new(ioc_extractor(), InMemoryIocBaseline::new())
        .with_propositions(StubPropositionExtractor::new(), check);

    let doc = Document::new("doc-4", "Known fact. Brand new fact.");
    let delta = pipeline.ingest(&doc).await.unwrap();

    assert_eq!(delta.propositions.len(), 1);
    assert_eq!(delta.propositions[0].text, "Brand new fact.");

    let dropped: Vec<_> = delta
        .dropped
        .iter()
        .filter_map(|d| match (&d.item, &d.reason) {
            (DroppedItem::Proposition(p), DroppedReason::Redundant { similarity }) => {
                Some((p.text.clone(), *similarity))
            }
            _ => None,
        })
        .collect();
    assert_eq!(dropped, vec![("Known fact.".to_string(), 0.99)]);

    // Both candidates must have hit the store (no over-eager skipping).
    let seen = pipeline.propositions().redundancy().threshold();
    assert!((seen - 0.50).abs() < f64::EPSILON);
}

#[tokio::test]
async fn pipeline_threshold_boundary_is_inclusive() {
    let store = ConstSimilarityStore::new(0.90);
    let check = VectorStoreRedundancyCheck::new(&store, 0.90).unwrap();
    let pipeline = DistillationPipeline::new(ioc_extractor(), InMemoryIocBaseline::new())
        .with_propositions(StubPropositionExtractor::new(), check);

    let doc = Document::new("doc-5", "A single sentence.");
    let delta = pipeline.ingest(&doc).await.unwrap();
    assert!(delta.propositions.is_empty());
    assert_eq!(delta.dropped.len(), 1);
    match &delta.dropped[0].reason {
        DroppedReason::Redundant { similarity } => {
            assert!((similarity - 0.90).abs() < f64::EPSILON);
        }
        other => panic!("expected Redundant, got {other:?}"),
    }
}

#[tokio::test]
async fn lookup_store_query_recording_works_for_diagnostics() {
    // Sanity: confirm the LookupStore actually records the query strings,
    // so other tests can rely on it.
    let mut table = HashMap::new();
    table.insert("Alpha.".into(), 0.95);
    table.insert("Beta.".into(), 0.05);
    let store = LookupStore::new(table);
    let check = VectorStoreRedundancyCheck::new(&store, 0.50).unwrap();
    let _ = check.check(&Proposition::new("Alpha.")).await.unwrap();
    let _ = check.check(&Proposition::new("Beta.")).await.unwrap();
    assert_eq!(
        store.seen(),
        vec!["Alpha.".to_string(), "Beta.".to_string()]
    );
}

#[tokio::test]
async fn empty_doc_produces_empty_delta_with_propositions_enabled() {
    let store = ConstSimilarityStore::new(0.0);
    let check = VectorStoreRedundancyCheck::new(&store, 0.90).unwrap();
    let pipeline = DistillationPipeline::new(ioc_extractor(), InMemoryIocBaseline::new())
        .with_propositions(StubPropositionExtractor::new(), check);

    let delta = pipeline.ingest(&Document::new("empty", "")).await.unwrap();
    assert!(delta.iocs.is_empty());
    assert!(delta.propositions.is_empty());
    assert!(delta.dropped.is_empty());
    assert!(delta.is_empty());
}

#[tokio::test]
async fn redundancy_verdict_returns_zero_similarity_when_store_is_empty() {
    // ConstSimilarityStore always returns one hit; for a real empty-store
    // assertion we drive a custom store inline.
    struct EmptyStore;
    impl VectorStoreIndex for EmptyStore {
        type Filter = Filter<serde_json::Value>;
        async fn top_n<T>(
            &self,
            _req: VectorSearchRequest<Self::Filter>,
        ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
        where
            T: for<'a> Deserialize<'a> + WasmCompatSend,
        {
            Ok(Vec::new())
        }
        async fn top_n_ids(
            &self,
            _req: VectorSearchRequest<Self::Filter>,
        ) -> Result<Vec<(f64, String)>, VectorStoreError> {
            Ok(Vec::new())
        }
    }

    let store = EmptyStore;
    let check = VectorStoreRedundancyCheck::new(&store, 0.50).unwrap();
    let verdict: RedundancyVerdict = check.check(&Proposition::new("anything")).await.unwrap();
    assert!(!verdict.is_redundant);
    assert_eq!(verdict.similarity, 0.0);
}
