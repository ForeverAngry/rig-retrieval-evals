//! Integration coverage for [`rig_retrieval_evals::staleness`].
//!
//! Loads the committed fixture under `tests/data/tiny_corpus_versions.jsonl`
//! and exercises both detectors against hand-rolled `RetrievedSet`s — no
//! `VectorStoreIndex` involvement, which keeps the test pure and offline.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use rig_retrieval_evals::{
    CorpusVersions, RetrievedDoc, RetrievedSet, detect_conflicts, detect_stale_hits,
};

fn ranked(query_id: &str, ids: &[&str]) -> RetrievedSet {
    RetrievedSet {
        query_id: query_id.into(),
        ranked: ids
            .iter()
            .enumerate()
            .map(|(rank, id)| RetrievedDoc {
                doc_id: (*id).into(),
                score: 1.0 / (rank as f64 + 1.0),
            })
            .collect(),
    }
}

fn load_fixture() -> CorpusVersions {
    CorpusVersions::load_jsonl("tests/data/tiny_corpus_versions.jsonl")
        .expect("load corpus versions fixture")
}

#[test]
fn fixture_resolves_implicit_and_explicit_supersession() {
    let versions = load_fixture();
    // Explicit supersession recorded on addr-2023.
    assert_eq!(versions.superseded_by("addr-2023"), Some("addr-2025"));
    // Implicit supersession via newest timestamp inside the same key.
    assert_eq!(versions.superseded_by("addr-2024"), Some("addr-2025"));
    // The newest doc in the group is fresh.
    assert_eq!(versions.superseded_by("addr-2025"), None);
    // Unversioned doc is never stale.
    assert!(!versions.is_stale("misc-doc"));
    // Unknown doc id is never stale.
    assert!(!versions.is_stale("never-seen"));
}

#[test]
fn stale_detector_flags_old_address_above_fresh_one() {
    let versions = load_fixture();
    let retrieved = ranked("q-addr", &["addr-2024", "addr-2025", "misc-doc"]);
    let report = detect_stale_hits(&retrieved, &versions, 5);
    assert_eq!(report.considered, 3);
    assert_eq!(report.stale_hits.len(), 1);
    let hit = &report.stale_hits[0];
    assert_eq!(hit.doc_id, "addr-2024");
    assert_eq!(hit.rank, 0);
    assert_eq!(hit.superseded_by, "addr-2025");
    assert!((report.stale_rate() - (1.0 / 3.0)).abs() < 1e-9);
}

#[test]
fn conflict_detector_surfaces_two_generations_in_same_window() {
    let versions = load_fixture();
    // Ranker returned both price generations in the same top-k.
    let retrieved = ranked("q-price", &["price-v1", "misc-doc", "price-v2"]);
    let report = detect_conflicts(&retrieved, &versions, 5);
    assert!(report.has_conflicts());
    assert_eq!(report.groups.len(), 1);
    assert_eq!(report.groups[0].version_key, "sku-7:price");
    assert_eq!(report.groups[0].doc_ids, vec!["price-v1", "price-v2"]);
    assert_eq!(report.conflicting_doc_count, 2);
    assert_eq!(report.considered, 3);
}

#[test]
fn conflict_detector_ignores_collisions_outside_topk() {
    let versions = load_fixture();
    let retrieved = ranked(
        "q-windowed",
        &["addr-2023", "misc-doc", "addr-2024", "addr-2025"],
    );
    // k=2 only sees addr-2023 + misc-doc, so no conflict group.
    let report = detect_conflicts(&retrieved, &versions, 2);
    assert!(!report.has_conflicts());
    assert_eq!(report.considered, 2);
    // But the stale detector still catches addr-2023 inside that window.
    let stale = detect_stale_hits(&retrieved, &versions, 2);
    assert_eq!(stale.stale_hits.len(), 1);
    assert_eq!(stale.stale_hits[0].doc_id, "addr-2023");
    assert_eq!(stale.stale_hits[0].superseded_by, "addr-2025");
}
