//! Core ingestion-pipeline data types.

use serde::{Deserialize, Serialize};

use super::graph::Triple;
use super::ioc::Ioc;
use super::proposition::Proposition;

/// A caller-parsed document fed to the [`DistillationPipeline`].
///
/// `rig-retrieval-evals` does not parse PDFs/HTML; callers extract text upstream
/// and hand it over. `sections` is optional but recommended: later tracks
/// route narrative text to the LLM-backed extractors and skip tables / code
/// blocks where appropriate.
///
/// [`DistillationPipeline`]: super::DistillationPipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Stable caller-assigned identifier. Echoed back unchanged in delta
    /// telemetry.
    pub id: String,
    /// Full document text. Track 1 (regex IoC extraction) runs over this
    /// field directly.
    pub text: String,
    /// Optional structural breakdown. Empty for callers that have not
    /// performed section detection yet.
    #[serde(default)]
    pub sections: Vec<Section>,
}

impl Document {
    /// Construct a document with no pre-parsed sections.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            sections: Vec::new(),
        }
    }

    /// Builder-style: attach pre-parsed sections.
    pub fn with_sections(mut self, sections: Vec<Section>) -> Self {
        self.sections = sections;
        self
    }
}

/// A structural slice of a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// What kind of content this section holds.
    pub kind: SectionKind,
    /// The section's text.
    pub text: String,
}

/// Section classification. Marked `#[non_exhaustive]` so future PRs can
/// add variants (e.g. `Figure`, `Code`) without breaking callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum SectionKind {
    /// Unstructured prose. Routed to the propositional / graph tracks in
    /// later PRs.
    Narrative,
    /// Tabular data (CSV, Markdown table). Skipped by narrative tracks.
    Table,
    /// Code, configuration, or shell output.
    Code,
    /// Any other section the caller wishes to surface.
    Other,
}

/// The output of one pipeline run.
///
/// The output of one pipeline run.
///
/// Marked `#[non_exhaustive]` so adding new track outputs is additive.
/// Use the provided constructors and field accessors; callers outside
/// the crate cannot use struct literals.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IngestionDelta {
    /// Net-new IoCs that the caller should commit to their IoC store.
    pub iocs: Vec<Ioc>,
    /// Net-new propositions (Track 3) whose cosine similarity against the
    /// configured store fell below the redundancy threshold.
    pub propositions: Vec<Proposition>,
    /// Net-new triples (Track 2) absent from the configured graph baseline.
    pub triples: Vec<Triple>,
    /// Items the pipeline discarded, each paired with a structured reason.
    /// Surfaced for inspectability: every drop should be assertable in a
    /// test.
    pub dropped: Vec<Dropped>,
    /// Lightweight lexical novelty score in `[0, 1]`, where `0` means no
    /// new token-set information and `1` means fully novel. Defaults to
    /// `0.0` for callers that do not run the ingestion knowledge-gain gate.
    #[serde(default)]
    pub knowledge_gain: f64,
}

impl IngestionDelta {
    /// Empty delta. Convenience for callers that need to construct one
    /// outside the crate.
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` if the document produced no net-new items in any track.
    pub fn is_empty(&self) -> bool {
        self.iocs.is_empty() && self.propositions.is_empty() && self.triples.is_empty()
    }

    /// Attach a lexical knowledge-gain score.
    #[must_use]
    pub fn with_knowledge_gain(mut self, knowledge_gain: f64) -> Self {
        self.knowledge_gain = knowledge_gain.clamp(0.0, 1.0);
        self
    }
}

/// A discarded item plus the structured reason it was dropped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dropped {
    /// What was dropped.
    pub item: DroppedItem,
    /// Why.
    pub reason: DroppedReason,
}

/// The dropped payload. `#[non_exhaustive]` so future tracks can add
/// variants without breaking callers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DroppedItem {
    /// An IoC that was already known to the baseline.
    Ioc(Ioc),
    /// A proposition that the redundancy check judged duplicative.
    Proposition(Proposition),
    /// A triple whose edge was already present in the graph baseline.
    Triple(Triple),
}

/// Structured reason a pipeline track discarded an item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DroppedReason {
    /// The IoC was already present in the baseline (Track 1 set difference).
    DuplicateIoc,
    /// The proposition's cosine similarity against the redundancy store
    /// met or exceeded the configured threshold (Track 3).
    Redundant {
        /// Cosine similarity of the most similar existing entry, in `[0, 1]`.
        similarity: f64,
    },
    /// The triple's `(subject, predicate, object)` edge was already
    /// present in the graph baseline (Track 2 set difference).
    DuplicateEdge,
}
