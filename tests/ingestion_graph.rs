//! Integration tests for Track 2 (knowledge-graph distillation).

#![cfg(feature = "ingestion")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use rig_retrieval_evals::{
    DistillationPipeline, Document, DroppedItem, DroppedReason, GraphBaseline,
    InMemoryGraphBaseline, InMemoryIocBaseline, RegexIocExtractor, StubTripleExtractor, Triple,
    TripleExtractor,
};

fn pipeline_for(
    triples: Vec<(&'static str, &'static str, &'static str)>,
) -> (
    DistillationPipeline<
        RegexIocExtractor,
        InMemoryIocBaseline,
        rig_retrieval_evals::NoPropositionTrack,
        rig_retrieval_evals::ActiveGraphTrack<StubTripleExtractor, InMemoryGraphBaseline>,
    >,
) {
    let pipeline = DistillationPipeline::new(
        RegexIocExtractor::new().unwrap(),
        InMemoryIocBaseline::new(),
    )
    .with_graph(
        StubTripleExtractor::new(triples),
        InMemoryGraphBaseline::new(),
    );
    (pipeline,)
}

fn doc(id: &str, text: &str) -> Document {
    Document::new(id, text)
}

#[test]
fn triple_normalises_predicate_case_and_whitespace() {
    let t = Triple::new("APT-28", "Exploits  CVE", "CVE-2024-1");
    assert_eq!(t.subject, "APT-28");
    assert_eq!(t.predicate, "exploits_cve");
    assert_eq!(t.object, "CVE-2024-1");
}

#[test]
fn triple_normalisation_collapses_whitespace_runs_and_trims() {
    let t = Triple::new("  S  ", "  Affects \t Asset  ", "  O  ");
    assert_eq!(t.subject, "S");
    assert_eq!(t.predicate, "affects_asset");
    assert_eq!(t.object, "O");
}

#[test]
fn triple_eq_is_canonicalised() {
    let a = Triple::new("A", "Affects", "B");
    let b = Triple::new("A", "affects", "B");
    let c = Triple::new("A", "AFFECTS  ", "B");
    assert_eq!(a, b);
    assert_eq!(b, c);
}

#[tokio::test]
async fn in_memory_baseline_contains_after_insert() {
    let baseline = InMemoryGraphBaseline::new();
    let t = Triple::new("apt28", "exploits", "cve-1");
    assert!(!baseline.contains(&t).await.unwrap());
    assert!(baseline.insert(t.clone()).unwrap());
    assert!(baseline.contains(&t).await.unwrap());
    // Second insert reports `false` for "already present".
    assert!(!baseline.insert(t).unwrap());
    assert_eq!(baseline.len().unwrap(), 1);
}

#[tokio::test]
async fn in_memory_baseline_from_iter_seeds_edges() {
    let baseline = InMemoryGraphBaseline::with_edges([
        ("apt28", "Exploits", "cve-1"),
        ("apt28", "uses", "rat-x"),
    ]);
    assert_eq!(baseline.len().unwrap(), 2);
    assert!(
        baseline
            .contains(&Triple::new("apt28", "exploits", "cve-1"))
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn stub_extractor_returns_normalised_triples() {
    let stub = StubTripleExtractor::new([("S", "Has Property", "O")]);
    let out = stub.extract(&doc("d", "ignored")).await.unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].predicate, "has_property");
}

#[tokio::test]
async fn pipeline_without_graph_leaves_triples_empty() {
    let pipeline = DistillationPipeline::new(
        RegexIocExtractor::new().unwrap(),
        InMemoryIocBaseline::new(),
    );
    let delta = pipeline.ingest(&doc("d", "nothing")).await.unwrap();
    assert!(delta.triples.is_empty());
}

#[tokio::test]
async fn pipeline_emits_net_new_triples_against_empty_baseline() {
    let (pipeline,) = pipeline_for(vec![
        ("apt28", "exploits", "cve-1"),
        ("apt28", "uses", "rat-x"),
    ]);
    let delta = pipeline.ingest(&doc("d", "report")).await.unwrap();
    assert_eq!(delta.triples.len(), 2);
    assert!(delta.dropped.is_empty());
}

#[tokio::test]
async fn pipeline_drops_known_triples_with_duplicate_edge_reason() {
    let baseline = InMemoryGraphBaseline::with_edges([("apt28", "exploits", "cve-1")]);
    let pipeline = DistillationPipeline::new(
        RegexIocExtractor::new().unwrap(),
        InMemoryIocBaseline::new(),
    )
    .with_graph(
        StubTripleExtractor::new([("apt28", "exploits", "cve-1"), ("apt28", "uses", "rat-x")]),
        baseline,
    );
    let delta = pipeline.ingest(&doc("d", "report")).await.unwrap();
    assert_eq!(delta.triples.len(), 1);
    assert_eq!(delta.triples[0].object, "rat-x");
    assert_eq!(delta.dropped.len(), 1);
    match &delta.dropped[0].item {
        DroppedItem::Triple(t) => assert_eq!(t.object, "cve-1"),
        other => panic!("expected DroppedItem::Triple, got {other:?}"),
    }
    assert_eq!(delta.dropped[0].reason, DroppedReason::DuplicateEdge);
}

#[tokio::test]
async fn pipeline_dedup_uses_normalised_predicate() {
    // Baseline stored as "Exploits"; candidate emitted as "EXPLOITS" — both
    // normalise to `exploits` and must collide.
    let baseline = InMemoryGraphBaseline::with_edges([("apt28", "Exploits", "cve-1")]);
    let pipeline = DistillationPipeline::new(
        RegexIocExtractor::new().unwrap(),
        InMemoryIocBaseline::new(),
    )
    .with_graph(
        StubTripleExtractor::new([("apt28", "EXPLOITS", "cve-1")]),
        baseline,
    );
    let delta = pipeline.ingest(&doc("d", "x")).await.unwrap();
    assert!(delta.triples.is_empty());
    assert_eq!(delta.dropped.len(), 1);
    assert_eq!(delta.dropped[0].reason, DroppedReason::DuplicateEdge);
}

#[tokio::test]
async fn pipeline_runs_track1_and_track2_together() {
    let pipeline = DistillationPipeline::new(
        RegexIocExtractor::new().unwrap(),
        InMemoryIocBaseline::new(),
    )
    .with_graph(
        StubTripleExtractor::new([("apt28", "exploits", "cve-2024-12345")]),
        InMemoryGraphBaseline::new(),
    );
    let delta = pipeline
        .ingest(&doc(
            "d",
            "APT-28 exploited CVE-2024-12345 from 192.0.2.10.",
        ))
        .await
        .unwrap();
    assert!(!delta.iocs.is_empty(), "track 1 should emit IoCs");
    assert_eq!(delta.triples.len(), 1, "track 2 should emit one triple");
}

#[tokio::test]
async fn empty_doc_with_track2_enabled_returns_empty_delta() {
    let pipeline = DistillationPipeline::new(
        RegexIocExtractor::new().unwrap(),
        InMemoryIocBaseline::new(),
    )
    .with_graph(
        StubTripleExtractor::new(Vec::<(&str, &str, &str)>::new()),
        InMemoryGraphBaseline::new(),
    );
    let delta = pipeline.ingest(&doc("d", "")).await.unwrap();
    assert!(delta.is_empty());
}

#[cfg(feature = "ingestion-graph")]
mod petgraph_baseline {
    use super::*;
    use rig_retrieval_evals::PetgraphBaseline;

    #[tokio::test]
    async fn petgraph_baseline_basic_insert_and_contains() {
        let baseline = PetgraphBaseline::new();
        let t = Triple::new("apt28", "exploits", "cve-1");
        assert!(!baseline.contains(&t).await.unwrap());
        assert!(baseline.insert(t.clone()).unwrap());
        assert!(baseline.contains(&t).await.unwrap());
        assert!(!baseline.insert(t).unwrap());
        assert_eq!(baseline.len().unwrap(), 1);
    }

    #[tokio::test]
    async fn petgraph_baseline_dedups_via_pipeline() {
        let baseline = PetgraphBaseline::with_edges([("apt28", "Exploits", "cve-1")]);
        let pipeline = DistillationPipeline::new(
            RegexIocExtractor::new().unwrap(),
            InMemoryIocBaseline::new(),
        )
        .with_graph(
            StubTripleExtractor::new([("apt28", "exploits", "cve-1"), ("apt28", "uses", "rat-x")]),
            baseline,
        );
        let delta = pipeline.ingest(&doc("d", "x")).await.unwrap();
        assert_eq!(delta.triples.len(), 1);
        assert_eq!(delta.dropped.len(), 1);
        assert_eq!(delta.dropped[0].reason, DroppedReason::DuplicateEdge);
    }
}
