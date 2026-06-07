//! Deterministic synthetic "needle in a haystack" dataset generation.
//!
//! Builds an in-memory corpus of benign-looking log lines with a handful of
//! unique indicator-of-compromise (IOC) tokens injected at known locations,
//! plus the matching BEIR-style [`Qrels`]. This mirrors the cyber / forensic
//! retrieval task — find the rare evidence buried in volume — and is fully
//! offline and reproducible from a seed, so it can drive CI benchmarks and
//! self-contained tests without shipping fixture files.
//!
//! The corpus is returned as `(doc_id, text)` pairs so the caller can build a
//! vector store, a lexical [`crate::retriever::Retriever`], or both over the
//! same documents and score them against the returned [`Qrels`].
//!
//! ```
//! use rig_retrieval_evals::synthetic::{generate, SyntheticConfig};
//!
//! let corpus = generate(&SyntheticConfig {
//!     docs: 4,
//!     lines_per_doc: 64,
//!     needles: 6,
//!     seed: 7,
//! });
//! assert_eq!(corpus.qrels.len(), 6);
//! assert_eq!(corpus.documents.len(), 4);
//! ```

use std::collections::HashMap;

use crate::dataset::{GoldQuery, Qrels};

/// Configuration for [`generate`].
#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    /// Number of documents to create (clamped to at least 1).
    pub docs: usize,
    /// Approximate number of lines per document (clamped to at least 1).
    pub lines_per_doc: usize,
    /// Number of unique IOC needles to inject (one gold query each).
    pub needles: usize,
    /// Seed controlling all pseudo-random choices (corpus is reproducible).
    pub seed: u64,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            docs: 8,
            lines_per_doc: 200,
            needles: 12,
            seed: 1,
        }
    }
}

/// A single synthetic document keyed by an opaque `doc_id` matching the ids in
/// the generated [`Qrels`].
#[derive(Debug, Clone)]
pub struct SyntheticDoc {
    /// Stable document identifier (`doc-000`, `doc-001`, …).
    pub doc_id: String,
    /// Full document text (newline-separated log lines).
    pub text: String,
}

/// An in-memory synthetic corpus plus its matching gold qrels.
#[derive(Debug, Clone)]
pub struct SyntheticCorpus {
    /// Generated documents in id order.
    pub documents: Vec<SyntheticDoc>,
    /// Gold labels: one query per injected needle, marking the single
    /// document that contains it as relevant (grade 1).
    pub qrels: Qrels,
}

/// A tiny deterministic PRNG (SplitMix64) — avoids pulling a `rand` dependency
/// and guarantees identical corpora across platforms for a given seed.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() % bound as u64) as usize
        }
    }
}

const USERS: &[&str] = &["alice", "bob", "carol", "dave", "erin", "frank"];
const ACTIONS: &[&str] = &[
    "login succeeded",
    "session opened",
    "heartbeat ok",
    "config reloaded",
    "cache flushed",
    "job completed",
];
const HOSTS: &[&str] = &["web01", "web02", "db01", "cache01", "edge01"];

/// Generate a deterministic synthetic corpus and matching gold qrels.
///
/// Each needle injects a line containing a globally-unique IOC token into one
/// document; the corresponding [`GoldQuery`] uses that token as the query and
/// marks that document as the sole relevant doc. The same [`SyntheticConfig`]
/// always yields byte-identical documents and identical qrels.
#[must_use]
pub fn generate(cfg: &SyntheticConfig) -> SyntheticCorpus {
    let docs = cfg.docs.max(1);
    let lines = cfg.lines_per_doc.max(1);
    let mut rng = SplitMix64::new(cfg.seed);

    // Decide, before writing, which (doc, line) each needle lands on so we can
    // both inject it and record the gold label.
    struct Needle {
        token: String,
        doc_idx: usize,
        line_idx: usize,
    }
    let mut needles = Vec::with_capacity(cfg.needles);
    for i in 0..cfg.needles {
        let doc_idx = rng.below(docs);
        let line_idx = rng.below(lines);
        // Tokens are globally unique and lexically distinctive so an exact
        // search has unambiguous recall.
        let token = format!("IOC-{:08x}-{i:04}", cfg.seed);
        needles.push(Needle {
            token,
            doc_idx,
            line_idx,
        });
    }

    let doc_id = |idx: usize| format!("doc-{idx:03}");

    let mut documents = Vec::with_capacity(docs);
    for d in 0..docs {
        let mut text = String::new();
        let here: Vec<&Needle> = needles.iter().filter(|n| n.doc_idx == d).collect();
        for line_idx in 0..lines {
            // Deterministic benign line.
            let ts = 1_700_000_000u64 + (d as u64 * 100_000) + line_idx as u64;
            let user = USERS.get(rng.below(USERS.len())).copied().unwrap_or("");
            let host = HOSTS.get(rng.below(HOSTS.len())).copied().unwrap_or("");
            let action = ACTIONS.get(rng.below(ACTIONS.len())).copied().unwrap_or("");
            text.push_str(&format!(
                "{ts} host={host} user={user} event=\"{action}\"\n"
            ));
            // Inject any needles anchored to this line.
            for needle in here.iter().filter(|n| n.line_idx == line_idx) {
                text.push_str(&format!(
                    "{ts} host={host} user={user} event=\"alert\" indicator={}\n",
                    needle.token
                ));
            }
        }
        documents.push(SyntheticDoc {
            doc_id: doc_id(d),
            text,
        });
    }

    let queries = needles
        .iter()
        .enumerate()
        .map(|(i, needle)| {
            let mut relevant = HashMap::new();
            relevant.insert(doc_id(needle.doc_idx), 1u8);
            GoldQuery {
                query_id: format!("needle-{i:04}"),
                query: needle.token.clone(),
                relevant_docs: relevant,
                reference_answer: None,
            }
        })
        .collect();

    SyntheticCorpus {
        documents,
        qrels: Qrels { queries },
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic_for_a_seed() {
        let cfg = SyntheticConfig {
            docs: 3,
            lines_per_doc: 50,
            needles: 5,
            seed: 7,
        };
        let a = generate(&cfg);
        let b = generate(&cfg);

        assert_eq!(a.qrels.len(), 5);
        let a_tokens: Vec<_> = a.qrels.queries.iter().map(|q| &q.query).collect();
        let b_tokens: Vec<_> = b.qrels.queries.iter().map(|q| &q.query).collect();
        assert_eq!(a_tokens, b_tokens);
        assert_eq!(a.documents[0].text, b.documents[0].text);
    }

    #[test]
    fn each_needle_token_appears_in_its_document() {
        let corpus = generate(&SyntheticConfig {
            docs: 4,
            lines_per_doc: 40,
            needles: 6,
            seed: 3,
        });
        let by_id: HashMap<&str, &str> = corpus
            .documents
            .iter()
            .map(|d| (d.doc_id.as_str(), d.text.as_str()))
            .collect();
        for query in &corpus.qrels.queries {
            let (doc_id, _grade) = query.relevant_docs.iter().next().unwrap();
            let text = by_id.get(doc_id.as_str()).unwrap();
            assert!(
                text.contains(&query.query),
                "token {} should be in {doc_id}",
                query.query
            );
        }
    }

    #[test]
    fn zero_docs_is_clamped() {
        let corpus = generate(&SyntheticConfig {
            docs: 0,
            lines_per_doc: 0,
            needles: 2,
            seed: 1,
        });
        assert_eq!(corpus.documents.len(), 1);
        assert_eq!(corpus.qrels.len(), 2);
    }
}
