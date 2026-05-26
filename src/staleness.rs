//! Stale-content and conflict detection for retrieval results.
//!
//! Standard IR metrics (Recall, Precision, MRR, …) treat the corpus as a
//! flat bag of documents that either are or are not relevant for a query.
//! Real RAG corpora are versioned: the same fact (a customer's address,
//! a product price, a card's slot) is often represented by several
//! documents over time, and the *latest* one is the only one a retriever
//! should ever return. A retriever that returns the right `version_key`
//! but the wrong *generation* of it has scored a hit on the labelled
//! qrels yet silently fed the model stale context.
//!
//! This module ships two pure-Rust detectors that complement
//! [`crate::retrieval`] without changing the existing
//! [`RetrievalMetric`](crate::retrieval::RetrievalMetric) trait:
//!
//! - [`detect_stale_hits`] — flag retrieved documents whose corpus
//!   annotation says they are superseded by a newer doc id.
//! - [`detect_conflicts`] — flag groups of retrieved documents that
//!   share a `version_key`, i.e. the retriever surfaced more than one
//!   generation of the same underlying fact in the same top-k.
//!
//! Both detectors are driven by a [`CorpusVersions`] sidecar loaded from
//! a JSONL fixture so they slot into the existing
//! [`Qrels`](crate::dataset::Qrels) fixture story rather than mutating
//! the public qrels schema. Mirror the design of
//! a memvid archive can drive the eval without reshaping. Mirrors the
//! design of `rig_memvid::projection::MemoryContextPack` (declared in a
//! sibling crate that this one deliberately does not depend on).
//!
//! ## JSONL shape
//!
//! ```jsonl
//! {"doc_id":"addr-2024","version_key":"alice:address","effective_timestamp":1704067200,"superseded_by":"addr-2025"}
//! {"doc_id":"addr-2025","version_key":"alice:address","effective_timestamp":1735689600}
//! ```

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::dataset::RetrievedSet;
use crate::error::{Error, Result};

/// Side-channel version metadata for a single corpus document.
///
/// Documents not present in a [`CorpusVersions`] index are treated as
/// "unversioned": they cannot be stale and cannot contribute to a
/// conflict group.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StalenessAnnotation {
    /// Stable corpus document id matching
    /// [`crate::dataset::RetrievedDoc::doc_id`].
    pub doc_id: String,
    /// Optional supersession group key. Documents sharing a `version_key`
    /// represent the same underlying fact and only the newest one should
    /// reach the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version_key: Option<String>,
    /// Monotonic recency for ordering within a `version_key` group. Same
    /// semantics as `rig_memvid::projection::MemoryCandidate::recency`
    /// (event time preferred; frame id as a fallback proxy).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_timestamp: Option<i64>,
    /// Optional explicit "newer doc id". When present, this document is
    /// stale relative to `superseded_by` regardless of the
    /// `effective_timestamp` order — useful for fixtures that want to
    /// pin the relation without relying on timestamp math.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
}

/// JSONL-backed index of [`StalenessAnnotation`]s keyed by `doc_id`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusVersions {
    annotations: HashMap<String, StalenessAnnotation>,
}

impl CorpusVersions {
    /// Build an empty index. Useful in tests.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace an annotation. Returns `self` for chaining.
    #[must_use]
    pub fn with(mut self, annotation: StalenessAnnotation) -> Self {
        self.annotations
            .insert(annotation.doc_id.clone(), annotation);
        self
    }

    /// Load a JSONL corpus-versions file from disk. Each non-empty line
    /// must deserialize into [`StalenessAnnotation`].
    pub fn load_jsonl<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        debug!(?path, "loading corpus versions");
        let text = std::fs::read_to_string(path)?;
        Self::from_jsonl_str(&text)
    }

    /// Parse a JSONL corpus-versions payload from a string.
    pub fn from_jsonl_str(text: &str) -> Result<Self> {
        let mut annotations: HashMap<String, StalenessAnnotation> = HashMap::new();
        for (idx, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let ann: StalenessAnnotation =
                serde_json::from_str(line).map_err(|source| Error::DatasetParse {
                    line: idx + 1,
                    source,
                })?;
            annotations.insert(ann.doc_id.clone(), ann);
        }
        Ok(Self { annotations })
    }

    /// Look up an annotation by `doc_id`.
    #[must_use]
    pub fn get(&self, doc_id: &str) -> Option<&StalenessAnnotation> {
        self.annotations.get(doc_id)
    }

    /// Number of annotated documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.annotations.len()
    }

    /// True if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.annotations.is_empty()
    }

    /// Returns the `version_key` for `doc_id` if known.
    #[must_use]
    pub fn version_key_of(&self, doc_id: &str) -> Option<&str> {
        self.annotations
            .get(doc_id)
            .and_then(|a| a.version_key.as_deref())
    }

    /// Returns the explicit `superseded_by` doc id if recorded.
    #[must_use]
    pub fn superseded_by(&self, doc_id: &str) -> Option<&str> {
        let ann = self.annotations.get(doc_id)?;
        if let Some(other) = ann.superseded_by.as_deref() {
            return Some(other);
        }
        // Implicit supersession: same version_key with a strictly newer
        // effective_timestamp.
        let key = ann.version_key.as_deref()?;
        let ts = ann.effective_timestamp?;
        let mut winner: Option<(&str, i64)> = None;
        for (other_id, other) in &self.annotations {
            if other_id == doc_id {
                continue;
            }
            if other.version_key.as_deref() != Some(key) {
                continue;
            }
            let Some(other_ts) = other.effective_timestamp else {
                continue;
            };
            if other_ts <= ts {
                continue;
            }
            match winner {
                Some((_, best_ts)) if other_ts <= best_ts => {}
                _ => winner = Some((other_id.as_str(), other_ts)),
            }
        }
        winner.map(|(id, _)| id)
    }

    /// True if `doc_id` is stale: either an explicit `superseded_by` is
    /// present, or some other annotated document with the same
    /// `version_key` has a strictly newer `effective_timestamp`.
    #[must_use]
    pub fn is_stale(&self, doc_id: &str) -> bool {
        self.superseded_by(doc_id).is_some()
    }
}

/// One stale top-k hit produced by [`detect_stale_hits`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaleHit {
    /// Doc id of the stale retrieved document.
    pub doc_id: String,
    /// 0-indexed rank inside the considered top-k window.
    pub rank: usize,
    /// Doc id that supersedes `doc_id` per [`CorpusVersions`].
    pub superseded_by: String,
}

/// Per-query report of how many top-k hits were stale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StalenessReport {
    /// Query id this report was scored against.
    pub query_id: String,
    /// All stale hits found inside the top-k.
    pub stale_hits: Vec<StaleHit>,
    /// Number of top-k positions considered (`min(k, retrieved.ranked.len())`).
    pub considered: usize,
}

impl StalenessReport {
    /// Fraction of considered top-k positions that were stale,
    /// in `[0.0, 1.0]`. Returns `0.0` when no positions were considered.
    #[must_use]
    pub fn stale_rate(&self) -> f64 {
        if self.considered == 0 {
            return 0.0;
        }
        self.stale_hits.len() as f64 / self.considered as f64
    }

    /// True when at least one stale hit was found.
    #[must_use]
    pub fn has_stale_hits(&self) -> bool {
        !self.stale_hits.is_empty()
    }
}

/// One conflict group from [`detect_conflicts`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConflictGroup {
    /// Shared `version_key` the conflicting documents disagreed on.
    pub version_key: String,
    /// Doc ids inside the top-k that share `version_key`, in rank order.
    pub doc_ids: Vec<String>,
}

/// Per-query report of `version_key` collisions inside the top-k.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConflictReport {
    /// Query id this report was scored against.
    pub query_id: String,
    /// One entry per `version_key` that surfaced more than once.
    pub groups: Vec<ConflictGroup>,
    /// Total number of documents inside `groups` (i.e. how many
    /// conflicting documents the retriever returned).
    pub conflicting_doc_count: usize,
    /// Number of top-k positions considered.
    pub considered: usize,
}

impl ConflictReport {
    /// True when at least one `version_key` appeared more than once
    /// inside the considered top-k.
    #[must_use]
    pub fn has_conflicts(&self) -> bool {
        !self.groups.is_empty()
    }

    /// Fraction of considered top-k positions that were part of a
    /// conflict group, in `[0.0, 1.0]`.
    #[must_use]
    pub fn conflict_rate(&self) -> f64 {
        if self.considered == 0 {
            return 0.0;
        }
        self.conflicting_doc_count as f64 / self.considered as f64
    }
}

/// Flag every retrieved document inside the top-k that
/// [`CorpusVersions`] says is stale.
///
/// Stale = the corpus annotation records an explicit `superseded_by`,
/// or another annotated doc with the same `version_key` has a strictly
/// newer `effective_timestamp`. Documents without a `CorpusVersions`
/// entry are treated as not stale (consistent with the qrels
/// "unjudged = grade 0" convention).
#[must_use]
pub fn detect_stale_hits(
    retrieved: &RetrievedSet,
    versions: &CorpusVersions,
    k: usize,
) -> StalenessReport {
    let limit = k.min(retrieved.ranked.len());
    let window = retrieved.ranked.get(..limit).unwrap_or(&[]);
    let mut stale_hits = Vec::new();
    for (rank, doc) in window.iter().enumerate() {
        if let Some(newer) = versions.superseded_by(&doc.doc_id) {
            stale_hits.push(StaleHit {
                doc_id: doc.doc_id.clone(),
                rank,
                superseded_by: newer.to_string(),
            });
        }
    }
    StalenessReport {
        query_id: retrieved.query_id.clone(),
        stale_hits,
        considered: limit,
    }
}

/// Flag `version_key` collisions inside the top-k.
///
/// A "conflict" is two or more retrieved documents in the same window
/// that share the same `version_key` per [`CorpusVersions`]: the
/// retriever has surfaced multiple generations of the same underlying
/// fact, which forces the host to either pick one or pollute the
/// prompt with internally inconsistent context. Documents without a
/// recorded `version_key` cannot participate in a conflict group.
#[must_use]
pub fn detect_conflicts(
    retrieved: &RetrievedSet,
    versions: &CorpusVersions,
    k: usize,
) -> ConflictReport {
    let limit = k.min(retrieved.ranked.len());
    let window = retrieved.ranked.get(..limit).unwrap_or(&[]);

    // Preserve first-occurrence order of version_keys so the report is
    // stable across runs even though we use a HashMap to bucket.
    let mut order: Vec<String> = Vec::new();
    let mut buckets: HashMap<String, Vec<String>> = HashMap::new();
    for doc in window {
        let Some(key) = versions.version_key_of(&doc.doc_id) else {
            continue;
        };
        let key_owned = key.to_string();
        match buckets.get_mut(&key_owned) {
            Some(slot) => slot.push(doc.doc_id.clone()),
            None => {
                order.push(key_owned.clone());
                buckets.insert(key_owned, vec![doc.doc_id.clone()]);
            }
        }
    }

    let mut groups: Vec<ConflictGroup> = Vec::new();
    let mut conflicting_doc_count = 0usize;
    for key in order {
        let Some(doc_ids) = buckets.remove(&key) else {
            continue;
        };
        if doc_ids.len() < 2 {
            continue;
        }
        conflicting_doc_count += doc_ids.len();
        groups.push(ConflictGroup {
            version_key: key,
            doc_ids,
        });
    }

    ConflictReport {
        query_id: retrieved.query_id.clone(),
        groups,
        conflicting_doc_count,
        considered: limit,
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
    use crate::dataset::RetrievedDoc;

    fn ann(
        doc_id: &str,
        key: Option<&str>,
        ts: Option<i64>,
        sup: Option<&str>,
    ) -> StalenessAnnotation {
        StalenessAnnotation {
            doc_id: doc_id.into(),
            version_key: key.map(str::to_owned),
            effective_timestamp: ts,
            superseded_by: sup.map(str::to_owned),
        }
    }

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

    #[test]
    fn jsonl_round_trip_preserves_optional_fields() {
        let text = r#"{"doc_id":"a","version_key":"k","effective_timestamp":10}
{"doc_id":"b","version_key":"k","effective_timestamp":20}
{"doc_id":"c","superseded_by":"a"}
"#;
        let versions = CorpusVersions::from_jsonl_str(text).unwrap();
        assert_eq!(versions.len(), 3);
        assert_eq!(versions.version_key_of("a"), Some("k"));
        assert_eq!(versions.version_key_of("c"), None);
        // Explicit supersession.
        assert_eq!(versions.superseded_by("c"), Some("a"));
        // Implicit (timestamp-driven) supersession.
        assert_eq!(versions.superseded_by("a"), Some("b"));
        assert_eq!(versions.superseded_by("b"), None);
    }

    #[test]
    fn unknown_doc_is_not_stale() {
        let versions = CorpusVersions::new();
        let r = ranked("q", &["unknown-1"]);
        let report = detect_stale_hits(&r, &versions, 5);
        assert!(!report.has_stale_hits());
        assert_eq!(report.considered, 1);
        assert_eq!(report.stale_rate(), 0.0);
    }

    #[test]
    fn explicit_supersession_flags_stale_hit() {
        let versions = CorpusVersions::new()
            .with(ann("old", Some("addr"), Some(1), Some("new")))
            .with(ann("new", Some("addr"), Some(2), None));
        let r = ranked("q1", &["old", "unrelated"]);
        let report = detect_stale_hits(&r, &versions, 5);
        assert_eq!(report.stale_hits.len(), 1);
        assert_eq!(report.stale_hits[0].doc_id, "old");
        assert_eq!(report.stale_hits[0].rank, 0);
        assert_eq!(report.stale_hits[0].superseded_by, "new");
        assert!((report.stale_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn implicit_supersession_uses_latest_timestamp() {
        let versions = CorpusVersions::new()
            .with(ann("v1", Some("price"), Some(100), None))
            .with(ann("v2", Some("price"), Some(200), None))
            .with(ann("v3", Some("price"), Some(150), None));
        let r = ranked("q", &["v1", "v3"]);
        let report = detect_stale_hits(&r, &versions, 10);
        assert_eq!(report.stale_hits.len(), 2);
        // Both stale hits should point at v2 (the newest).
        assert!(report.stale_hits.iter().all(|h| h.superseded_by == "v2"));
    }

    #[test]
    fn k_window_caps_considered_positions() {
        let versions = CorpusVersions::new()
            .with(ann("old", Some("k"), Some(1), Some("new")))
            .with(ann("new", Some("k"), Some(2), None));
        let r = ranked("q", &["new", "old"]);
        // k=1 only considers the first (fresh) hit.
        let report = detect_stale_hits(&r, &versions, 1);
        assert_eq!(report.considered, 1);
        assert!(report.stale_hits.is_empty());
    }

    #[test]
    fn detect_conflicts_skips_when_only_one_per_key() {
        let versions = CorpusVersions::new()
            .with(ann("a", Some("k1"), Some(1), None))
            .with(ann("b", Some("k2"), Some(1), None));
        let r = ranked("q", &["a", "b"]);
        let report = detect_conflicts(&r, &versions, 5);
        assert!(!report.has_conflicts());
        assert_eq!(report.considered, 2);
        assert_eq!(report.conflict_rate(), 0.0);
    }

    #[test]
    fn detect_conflicts_groups_repeated_version_keys() {
        let versions = CorpusVersions::new()
            .with(ann("a1", Some("addr"), Some(1), None))
            .with(ann("a2", Some("addr"), Some(2), None))
            .with(ann("p1", Some("price"), Some(1), None))
            .with(ann("p2", Some("price"), Some(2), None))
            .with(ann("misc", None, None, None));
        let r = ranked("q42", &["a1", "p1", "a2", "p2", "misc"]);
        let report = detect_conflicts(&r, &versions, 10);
        assert_eq!(report.groups.len(), 2);
        // First-occurrence ordering: "addr" group first because a1 came
        // before p1 in the ranked list.
        assert_eq!(report.groups[0].version_key, "addr");
        assert_eq!(report.groups[0].doc_ids, vec!["a1", "a2"]);
        assert_eq!(report.groups[1].version_key, "price");
        assert_eq!(report.groups[1].doc_ids, vec!["p1", "p2"]);
        assert_eq!(report.conflicting_doc_count, 4);
        assert!((report.conflict_rate() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn version_key_collision_outside_window_is_not_a_conflict() {
        let versions = CorpusVersions::new()
            .with(ann("a1", Some("addr"), Some(1), None))
            .with(ann("a2", Some("addr"), Some(2), None));
        let r = ranked("q", &["a1", "filler", "a2"]);
        // Only the first two positions are considered.
        let report = detect_conflicts(&r, &versions, 2);
        assert!(!report.has_conflicts());
        assert_eq!(report.considered, 2);
    }
}
