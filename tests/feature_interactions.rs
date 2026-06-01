#![cfg(all(feature = "ingestion-graph", feature = "knowledge-gain"))]
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use rig_retrieval_evals::{
    CandidateDocumentGainInput, DistillationPipeline, Document, DroppedItem, DroppedReason,
    InMemoryGraphBaseline, InMemoryIocBaseline, KnowledgeGainConfig, KnowledgeGainReport,
    MetricDelta, Qrels, QueryDelta, RegexIocExtractor, ReportDiff, StubTripleExtractor,
};

#[tokio::test]
async fn ingestion_graph_delta_can_feed_knowledge_gain_candidates() {
    let pipeline = DistillationPipeline::new(
        RegexIocExtractor::new().unwrap(),
        InMemoryIocBaseline::new(),
    )
    .with_graph(
        StubTripleExtractor::new([
            ("apt28", "exploits", "doc-cve"),
            ("apt28", "exploits", "doc-known"),
        ]),
        InMemoryGraphBaseline::with_edges([("apt28", "exploits", "doc-known")]),
    );

    let delta = pipeline
        .ingest(&Document::new(
            "report-1",
            "APT28 exploited CVE-2026-0001 from 192.0.2.10.",
        ))
        .await
        .unwrap();

    assert_eq!(delta.triples.len(), 1);
    assert_eq!(delta.triples[0].object, "doc-cve");
    assert_eq!(delta.dropped.len(), 1);
    assert!(matches!(delta.dropped[0].item, DroppedItem::Triple(_)));
    assert_eq!(delta.dropped[0].reason, DroppedReason::DuplicateEdge);

    let qrels = Qrels::from_jsonl_str(
        r#"{"query_id":"q1","query":"apt28 exploit","relevant_docs":{"doc-cve":2}}"#,
    )
    .unwrap();
    let diff = ReportDiff {
        rows: vec![MetricDelta {
            metric: "recall@5".into(),
            current_mean: 1.0,
            baseline_mean: Some(0.0),
            delta: Some(1.0),
            winners: 1,
            losers: 0,
            unchanged: 0,
            query_changes: vec![QueryDelta {
                query_id: "q1".into(),
                current: 1.0,
                baseline: 0.0,
                delta: 1.0,
            }],
            current_ci: None,
            baseline_ci: None,
        }],
    };
    let config = KnowledgeGainConfig::new().with_novelty_weight(0.5);
    let candidates = delta
        .triples
        .iter()
        .map(|triple| CandidateDocumentGainInput::new(triple.object.clone()).with_novelty(0.4))
        .collect::<Vec<_>>();

    let gain = KnowledgeGainReport::from_diff(&diff, &config).with_candidate_documents(
        &qrels,
        &candidates,
        &config,
    );

    assert_eq!(gain.score, 1.0);
    assert_eq!(gain.candidate_documents.len(), 1);
    assert_eq!(gain.candidate_documents[0].doc_id, "doc-cve");
    assert!(gain.candidate_documents[0].score > 0.0);
}
