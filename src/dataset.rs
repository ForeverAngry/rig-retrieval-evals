//! Labeled retrieval datasets (qrels) and accompanying corpus / answer files.
//!
//! The crate adopts a JSONL line format compatible with the BEIR ecosystem so
//! that public datasets (NQ, HotpotQA, FiQA, MS-MARCO subsets, …) can be
//! consumed directly. The canonical shape for `qrels.jsonl` is:
//!
//! ```jsonl
//! {"query_id":"q1","query":"…","relevant_docs":{"doc-7":2,"doc-9":1}}
//! {"query_id":"q2","query":"…","relevant_docs":{"doc-3":1},"reference_answer":"…"}
//! ```
//!
//! Grades in `relevant_docs` are integers 1–N where higher = more relevant.
//! Documents not listed are treated as **non-relevant** (grade 0). This matches
//! the standard TREC / BEIR qrels semantics.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::error::{Error, Result};

/// A single record in a BEIR `queries.jsonl` file.
#[derive(Deserialize)]
struct BeirQuery {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    text: String,
}

/// A single labeled query in a retrieval dataset.
///
/// Field order is stable for downstream serialization; do not reorder without
/// bumping the dataset schema version in [`Qrels`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldQuery {
    /// Stable opaque identifier for the query.
    pub query_id: String,
    /// Natural-language query text to send to the retriever.
    pub query: String,
    /// Map of `doc_id -> graded_relevance`. Documents not listed are treated
    /// as non-relevant (grade 0). Grades are typically 1–3.
    pub relevant_docs: HashMap<String, u8>,
    /// Optional reference / "gold" answer used by answer-level evaluators.
    /// Retrieval-only metrics ignore this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_answer: Option<String>,
}

impl GoldQuery {
    /// Returns `true` if `doc_id` is labeled relevant (grade ≥ 1).
    #[must_use]
    pub fn is_relevant(&self, doc_id: &str) -> bool {
        self.relevant_docs
            .get(doc_id)
            .copied()
            .is_some_and(|g| g >= 1)
    }

    /// Returns the graded relevance for `doc_id`, or 0 if unlabeled.
    #[must_use]
    pub fn grade(&self, doc_id: &str) -> u8 {
        self.relevant_docs.get(doc_id).copied().unwrap_or(0)
    }

    /// Number of distinct documents labeled relevant (grade ≥ 1).
    #[must_use]
    pub fn relevant_count(&self) -> usize {
        self.relevant_docs.values().filter(|g| **g >= 1).count()
    }
}

/// A collection of [`GoldQuery`] forming a complete retrieval dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Qrels {
    /// All labeled queries.
    pub queries: Vec<GoldQuery>,
}

impl Qrels {
    /// Load a JSONL qrels file from disk. Each line must deserialize into
    /// [`GoldQuery`]; empty lines are skipped.
    pub fn load_jsonl<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        debug!(?path, "loading qrels");
        let text = std::fs::read_to_string(path)?;
        Self::from_jsonl_str(&text)
    }

    /// Parse a JSONL qrels payload from a string. Each non-empty line is
    /// decoded into a [`GoldQuery`]. The 1-indexed line number is included in
    /// any parse error.
    pub fn from_jsonl_str(text: &str) -> Result<Self> {
        let mut queries = Vec::new();
        for (idx, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let q: GoldQuery =
                serde_json::from_str(line).map_err(|source| Error::DatasetParse {
                    line: idx + 1,
                    source,
                })?;
            queries.push(q);
        }
        Ok(Self { queries })
    }

    /// Load a downloaded BEIR / BRIGHT dataset directory into [`Qrels`].
    ///
    /// Standard IR benchmarks ship a `queries.jsonl` (`{"_id","text"}` records)
    /// and TREC-style relevance judgments at `qrels/<split>.tsv`. This reads
    /// both and produces one [`GoldQuery`] per query that has at least one
    /// positive judgment, using the BEIR **corpus ids** directly as
    /// [`GoldQuery::relevant_docs`] keys — so the retriever under test must
    /// report those same ids as its `doc_id`s.
    ///
    /// The qrels TSV is accepted in both BEIR 3-column
    /// (`query-id <tab> corpus-id <tab> score`) and TREC 4-column
    /// (`query-id <tab> iteration <tab> doc-id <tab> relevance`) layouts. A
    /// header row, blank lines, and any row whose relevance does not parse to a
    /// positive integer (grade 0) are skipped, so a leading header is handled
    /// whether or not it is present.
    ///
    /// `split` selects the file under `qrels/`, e.g. `"test"` or `"dev"`.
    pub fn from_beir<P: AsRef<Path>>(dataset_dir: P, split: &str) -> Result<Self> {
        let dir = dataset_dir.as_ref();
        debug!(?dir, %split, "loading BEIR dataset");

        // 1. query id -> text.
        let queries_path = dir.join("queries.jsonl");
        let queries_text = std::fs::read_to_string(&queries_path)?;
        let mut query_text: HashMap<String, String> = HashMap::new();
        for (idx, raw_line) in queries_text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let record: BeirQuery =
                serde_json::from_str(line).map_err(|source| Error::DatasetParse {
                    line: idx + 1,
                    source,
                })?;
            query_text.insert(record.id, record.text);
        }

        // 2. Group positive judgments by query, preserving deterministic order.
        let qrels_path = dir.join("qrels").join(format!("{split}.tsv"));
        let qrels_text = std::fs::read_to_string(&qrels_path)?;
        let mut grouped: BTreeMap<String, HashMap<String, u8>> = BTreeMap::new();
        for raw_line in qrels_text.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            let (qid, doc_id, rel) = match cols.as_slice() {
                [qid, doc_id, rel] => (*qid, *doc_id, *rel),
                [qid, _iter, doc_id, rel] => (*qid, *doc_id, *rel),
                _ => continue,
            };
            // A non-numeric / non-positive relevance (e.g. the header's
            // "score") yields grade 0 and is dropped.
            let grade: u8 = rel.trim().parse().unwrap_or(0);
            if grade == 0 {
                continue;
            }
            grouped
                .entry(qid.trim().to_string())
                .or_default()
                .insert(doc_id.trim().to_string(), grade);
        }

        // 3. Emit a GoldQuery per query that has both text and a positive judgment.
        let mut queries = Vec::with_capacity(grouped.len());
        for (qid, relevant) in grouped {
            let Some(text) = query_text.get(&qid) else {
                continue;
            };
            queries.push(GoldQuery {
                query_id: qid.clone(),
                query: text.clone(),
                relevant_docs: relevant,
                reference_answer: None,
            });
        }
        Ok(Self { queries })
    }

    /// Number of queries in the dataset.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// True if the dataset is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

/// A single retrieval observation produced by a vector store for one gold
/// query. `ranked` is sorted by descending similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedSet {
    /// The [`GoldQuery::query_id`] this retrieval corresponds to.
    pub query_id: String,
    /// Hits in ranked order (highest score first).
    pub ranked: Vec<RetrievedDoc>,
}

/// A single ranked retrieval hit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedDoc {
    /// Backend-assigned document id used to match against
    /// [`GoldQuery::relevant_docs`].
    pub doc_id: String,
    /// Similarity score reported by the backend.
    pub score: f64,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn parses_well_formed_jsonl() {
        let text = r#"{"query_id":"q1","query":"a","relevant_docs":{"d1":2,"d2":1}}
        {"query_id":"q2","query":"b","relevant_docs":{"d3":1},"reference_answer":"yes"}

        "#;
        let q = Qrels::from_jsonl_str(text).unwrap();
        assert_eq!(q.len(), 2);
        assert!(q.queries[0].is_relevant("d1"));
        assert_eq!(q.queries[0].grade("d2"), 1);
        assert_eq!(q.queries[0].grade("missing"), 0);
        assert_eq!(q.queries[1].reference_answer.as_deref(), Some("yes"));
    }

    #[test]
    fn reports_line_on_parse_error() {
        let text = "{\"query_id\":\"q1\",\"query\":\"a\",\"relevant_docs\":{}}\nnot json\n";
        let err = Qrels::from_jsonl_str(text).unwrap_err();
        match err {
            Error::DatasetParse { line, .. } => assert_eq!(line, 2),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn relevant_count_excludes_zero_grades() {
        let q = GoldQuery {
            query_id: "q".into(),
            query: "".into(),
            relevant_docs: HashMap::from([
                ("a".to_string(), 2u8),
                ("b".to_string(), 0u8),
                ("c".to_string(), 1u8),
            ]),
            reference_answer: None,
        };
        assert_eq!(q.relevant_count(), 2);
    }
}
