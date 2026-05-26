//! Chunk-stat ingestion linting.
//!
//! Operators tuning a RAG pipeline often only notice retrieval quality has
//! collapsed *after* embeddings are paid for and the index is rebuilt. The
//! chunk linter moves that gate upstream: it inspects the chunk corpus
//! before any embedding, vector store, or LLM call and flags the
//! pathological shapes that historically produce bad retrieval —
//! micro-fragments, oversized chunks, exact duplicates, and missing IDs.
//!
//! The linter is pure-data on purpose:
//!
//! - No async, no I/O, no `tokio`.
//! - No embedding model, vector store, or LLM client.
//! - Output is a deterministic [`ChunkLintReport`] suitable for fixture
//!   tests and CI gates.
//!
//! Use [`ChunkLintConfig::fatal`] to turn the report into a hard error
//! when any warning fires; the default behaviour is to surface warnings
//! and let the host decide.
//!
//! ## Example
//!
//! ```
//! use rig_retrieval_evals::ingestion::lint::{Chunk, ChunkLintConfig, lint_chunks};
//!
//! let chunks = vec![
//!     Chunk::new("a", "The quick brown fox jumps over the lazy dog."),
//!     Chunk::new("b", "The quick brown fox jumps over the lazy dog."),
//!     Chunk::unidentified("ok"),
//! ];
//! let report = lint_chunks(&chunks, &ChunkLintConfig::default());
//! assert_eq!(report.stats.count, 3);
//! assert!(report.has_warnings());
//! ```

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A caller-supplied chunk. The linter only inspects `text` length and
/// `id` presence; it does not parse content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Caller-assigned identifier. `None` is flagged by
    /// [`ChunkLintWarning::MissingIds`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The chunk's text. Length is measured in `char` count, not bytes,
    /// to keep multi-byte scripts comparable to ASCII.
    pub text: String,
}

impl Chunk {
    /// Build a chunk with an explicit identifier.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: Some(id.into()),
            text: text.into(),
        }
    }

    /// Build a chunk without an identifier. Useful for hosts that key
    /// off offsets and want the linter to flag the gap.
    pub fn unidentified(text: impl Into<String>) -> Self {
        Self {
            id: None,
            text: text.into(),
        }
    }
}

/// Linter knobs. Defaults are conservative — they fire on shapes that
/// have empirically degraded retrieval quality in practice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkLintConfig {
    /// Char count below which a chunk is considered "tiny". Default 32.
    pub tiny_chars: usize,
    /// Char count above which a chunk is considered "giant". Default
    /// 4096 — large enough that most embedders will truncate it.
    pub giant_chars: usize,
    /// Char count at or below which a chunk is considered "near-empty"
    /// (but still non-zero). Default 4. Truly empty chunks are tracked
    /// separately.
    pub near_empty_chars: usize,
    /// Fraction of total chunks that may be tiny before
    /// [`ChunkLintWarning::TooManyTinyChunks`] fires. Default 0.10.
    pub max_tiny_fraction: f64,
    /// Promote any warning to a hard error in [`lint_chunks_strict`].
    pub fatal: bool,
    /// Optional language-detection gate.
    pub language: LanguageLintConfig,
}

impl Default for ChunkLintConfig {
    fn default() -> Self {
        Self {
            tiny_chars: 32,
            giant_chars: 4096,
            near_empty_chars: 4,
            max_tiny_fraction: 0.10,
            fatal: false,
            language: LanguageLintConfig::default(),
        }
    }
}

/// Language-detection knobs for [`lint_chunks`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageLintConfig {
    /// Enable language detection. Disabled by default to avoid changing
    /// existing ingestion gates that only want shape/encoding lints.
    pub enabled: bool,
    /// Allowed ISO 639-3 language codes, lower-case (e.g. `"eng"`). Empty
    /// means detect and report unknown rates only.
    pub allowed_languages: Vec<String>,
    /// Maximum fraction of chunks whose language could not be detected.
    pub max_unknown_fraction: f64,
}

impl Default for LanguageLintConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_languages: Vec::new(),
            max_unknown_fraction: 0.20,
        }
    }
}

/// Count for one detected language code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageCount {
    /// ISO 639-3 language code returned by `whatlang`.
    pub language: String,
    /// Number of chunks detected as this language.
    pub count: u64,
}

/// Deterministic, pure-data statistics for a chunk corpus. Char counts
/// are used everywhere length appears.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkStats {
    /// Total number of chunks inspected.
    pub count: u64,
    /// Number of chunks with zero characters.
    pub empty: u64,
    /// Number of non-empty chunks at or below
    /// [`ChunkLintConfig::near_empty_chars`].
    pub near_empty: u64,
    /// Number of chunks below [`ChunkLintConfig::tiny_chars`] (excludes
    /// `empty`).
    pub tiny: u64,
    /// Number of chunks above [`ChunkLintConfig::giant_chars`].
    pub giant: u64,
    /// Number of chunks missing an `id`.
    pub missing_ids: u64,
    /// Number of chunks whose `text` collides with at least one other
    /// chunk. Counts every member of a duplicate group, not just the
    /// extras.
    pub duplicate_text: u64,
    /// Total disallowed control characters, excluding common whitespace
    /// controls (`\n`, `\r`, `\t`).
    pub control_chars: u64,
    /// Number of chunks containing a Unicode byte-order mark character.
    pub bom_chunks: u64,
    /// Minimum chunk char count (0 if `count == 0`).
    pub min_chars: u64,
    /// Maximum chunk char count (0 if `count == 0`).
    pub max_chars: u64,
    /// Mean chunk char count rounded down to the nearest integer.
    pub mean_chars: u64,
}

/// A single lint finding. Field shapes are stable; new variants are
/// additive thanks to `#[non_exhaustive]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChunkLintWarning {
    /// At least one chunk has empty `text`.
    EmptyChunks {
        /// How many empty chunks were observed.
        count: u64,
    },
    /// At least one chunk is non-empty but ≤ `near_empty_chars`.
    NearEmptyChunks {
        /// How many near-empty chunks were observed.
        count: u64,
        /// The configured threshold.
        threshold: usize,
    },
    /// The fraction of tiny chunks exceeds
    /// [`ChunkLintConfig::max_tiny_fraction`].
    TooManyTinyChunks {
        /// Tiny chunk count.
        count: u64,
        /// Total chunk count.
        total: u64,
        /// The configured `tiny_chars` threshold.
        tiny_chars: usize,
        /// The configured `max_tiny_fraction` threshold.
        max_fraction: f64,
    },
    /// At least one chunk is larger than `giant_chars` and is likely to
    /// be truncated by downstream embedders.
    GiantChunks {
        /// How many giant chunks were observed.
        count: u64,
        /// The configured threshold.
        threshold: usize,
    },
    /// Exact-text duplicates were detected.
    DuplicateChunks {
        /// Total members of duplicate groups (group sizes summed).
        count: u64,
        /// Number of distinct text values that appeared more than once.
        groups: u64,
    },
    /// At least one chunk has no `id`.
    MissingIds {
        /// How many chunks lacked an `id`.
        count: u64,
    },
    /// Encoding-level warning such as control characters or BOMs.
    Encoding {
        /// Specific encoding warning.
        warning: EncodingLintWarning,
    },
    /// Too many chunks had no detectable language.
    UnknownLanguage {
        /// Unknown-language chunk count.
        count: u64,
        /// Total chunk count.
        total: u64,
        /// Configured maximum fraction.
        max_fraction: f64,
    },
    /// Detected language codes outside the configured allow-list.
    DisallowedLanguages {
        /// Disallowed language counts.
        languages: Vec<LanguageCount>,
        /// Allowed language codes from the config.
        allowed: Vec<String>,
    },
}

/// Encoding-specific lint warning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EncodingLintWarning {
    /// One or more disallowed control characters were found.
    ControlCharacters {
        /// Total control characters across all chunks.
        count: u64,
    },
    /// One or more chunks contained a Unicode byte-order mark.
    ByteOrderMarks {
        /// Number of chunks containing a BOM character.
        count: u64,
    },
}

/// Final lint output: deterministic [`ChunkStats`] plus zero or more
/// [`ChunkLintWarning`]s, ordered by [`ChunkLintWarning`] variant.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkLintReport {
    /// Pure-data statistics for the corpus.
    pub stats: ChunkStats,
    /// All warnings raised, ordered deterministically.
    pub warnings: Vec<ChunkLintWarning>,
}

impl ChunkLintReport {
    /// `true` when at least one warning was raised.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Serialise the report to pretty JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Inspect a slice of chunks and emit a [`ChunkLintReport`].
///
/// The function never allocates an embedder, opens a network socket,
/// or mutates input — it is safe to call from CI gates and library
/// hot paths.
#[must_use]
pub fn lint_chunks(chunks: &[Chunk], config: &ChunkLintConfig) -> ChunkLintReport {
    let count = chunks.len() as u64;
    if count == 0 {
        return ChunkLintReport::default();
    }

    let mut empty = 0u64;
    let mut near_empty = 0u64;
    let mut tiny = 0u64;
    let mut giant = 0u64;
    let mut missing_ids = 0u64;
    let mut control_chars = 0u64;
    let mut bom_chunks = 0u64;
    let mut unknown_language = 0u64;
    let mut min_chars = u64::MAX;
    let mut max_chars = 0u64;
    let mut total_chars = 0u128;
    let mut text_counts: HashMap<&str, u64> = HashMap::with_capacity(chunks.len());
    let mut language_counts: BTreeMap<String, u64> = BTreeMap::new();
    let allowed_languages: Vec<String> = config
        .language
        .allowed_languages
        .iter()
        .map(|lang| lang.to_ascii_lowercase())
        .collect();

    for chunk in chunks {
        if chunk.id.is_none() {
            missing_ids = missing_ids.saturating_add(1);
        }
        let len = chunk.text.chars().count() as u64;
        if len == 0 {
            empty = empty.saturating_add(1);
        } else if len <= config.near_empty_chars as u64 {
            near_empty = near_empty.saturating_add(1);
        }
        if len > 0 && len < config.tiny_chars as u64 {
            tiny = tiny.saturating_add(1);
        }
        if len > config.giant_chars as u64 {
            giant = giant.saturating_add(1);
        }
        let bad_controls = chunk
            .text
            .chars()
            .filter(|ch| ch.is_control() && !matches!(ch, '\n' | '\r' | '\t'))
            .count() as u64;
        control_chars = control_chars.saturating_add(bad_controls);
        if chunk.text.contains('\u{feff}') {
            bom_chunks = bom_chunks.saturating_add(1);
        }
        if config.language.enabled && !chunk.text.trim().is_empty() {
            match whatlang::detect(&chunk.text) {
                Some(info) => {
                    let code = info.lang().code().to_ascii_lowercase();
                    *language_counts.entry(code).or_insert(0) += 1;
                }
                None => {
                    unknown_language = unknown_language.saturating_add(1);
                }
            }
        }
        min_chars = min_chars.min(len);
        max_chars = max_chars.max(len);
        total_chars = total_chars.saturating_add(u128::from(len));
        *text_counts.entry(chunk.text.as_str()).or_insert(0) += 1;
    }

    let duplicate_groups = text_counts.values().filter(|n| **n > 1).count() as u64;
    let duplicate_text: u64 = text_counts.values().filter(|n| **n > 1).sum();

    let mean_chars = (total_chars / u128::from(count)) as u64;
    let min_chars = if min_chars == u64::MAX { 0 } else { min_chars };

    let stats = ChunkStats {
        count,
        empty,
        near_empty,
        tiny,
        giant,
        missing_ids,
        duplicate_text,
        control_chars,
        bom_chunks,
        min_chars,
        max_chars,
        mean_chars,
    };

    let mut warnings = Vec::new();
    if empty > 0 {
        warnings.push(ChunkLintWarning::EmptyChunks { count: empty });
    }
    if near_empty > 0 {
        warnings.push(ChunkLintWarning::NearEmptyChunks {
            count: near_empty,
            threshold: config.near_empty_chars,
        });
    }
    let tiny_fraction = tiny as f64 / count as f64;
    if tiny_fraction > config.max_tiny_fraction {
        warnings.push(ChunkLintWarning::TooManyTinyChunks {
            count: tiny,
            total: count,
            tiny_chars: config.tiny_chars,
            max_fraction: config.max_tiny_fraction,
        });
    }
    if giant > 0 {
        warnings.push(ChunkLintWarning::GiantChunks {
            count: giant,
            threshold: config.giant_chars,
        });
    }
    if duplicate_groups > 0 {
        warnings.push(ChunkLintWarning::DuplicateChunks {
            count: duplicate_text,
            groups: duplicate_groups,
        });
    }
    if missing_ids > 0 {
        warnings.push(ChunkLintWarning::MissingIds { count: missing_ids });
    }
    if control_chars > 0 {
        warnings.push(ChunkLintWarning::Encoding {
            warning: EncodingLintWarning::ControlCharacters {
                count: control_chars,
            },
        });
    }
    if bom_chunks > 0 {
        warnings.push(ChunkLintWarning::Encoding {
            warning: EncodingLintWarning::ByteOrderMarks { count: bom_chunks },
        });
    }
    if config.language.enabled {
        let unknown_fraction = unknown_language as f64 / count as f64;
        if unknown_fraction > config.language.max_unknown_fraction {
            warnings.push(ChunkLintWarning::UnknownLanguage {
                count: unknown_language,
                total: count,
                max_fraction: config.language.max_unknown_fraction,
            });
        }
        if !allowed_languages.is_empty() {
            let disallowed = language_counts
                .iter()
                .filter(|(language, _count)| !allowed_languages.contains(language))
                .map(|(language, count)| LanguageCount {
                    language: language.clone(),
                    count: *count,
                })
                .collect::<Vec<_>>();
            if !disallowed.is_empty() {
                warnings.push(ChunkLintWarning::DisallowedLanguages {
                    languages: disallowed,
                    allowed: allowed_languages,
                });
            }
        }
    }

    ChunkLintReport { stats, warnings }
}

/// Strict variant of [`lint_chunks`]: returns `Err(Error::Ingestion(..))`
/// when any warning is raised and [`ChunkLintConfig::fatal`] is `true`.
/// Otherwise returns the report unchanged.
pub fn lint_chunks_strict(chunks: &[Chunk], config: &ChunkLintConfig) -> Result<ChunkLintReport> {
    let report = lint_chunks(chunks, config);
    if config.fatal && report.has_warnings() {
        return Err(Error::Ingestion(format!(
            "chunk lint failed: {} warning(s)",
            report.warnings.len()
        )));
    }
    Ok(report)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn corpus() -> Vec<Chunk> {
        vec![
            Chunk::new("a", "The quick brown fox jumps over the lazy dog."),
            Chunk::new("b", "The quick brown fox jumps over the lazy dog."), // dup
            Chunk::new("c", "tiny"),
            Chunk::new("d", ""),                                    // empty
            Chunk::new("e", "x"),                                   // near-empty
            Chunk::new("f", "z".repeat(5000)),                      // giant
            Chunk::unidentified("a healthy-sized chunk of prose."), // missing id
        ]
    }

    #[test]
    fn empty_corpus_produces_empty_report() {
        let report = lint_chunks(&[], &ChunkLintConfig::default());
        assert_eq!(report.stats, ChunkStats::default());
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn lint_flags_every_pathological_shape() {
        let report = lint_chunks(&corpus(), &ChunkLintConfig::default());
        assert_eq!(report.stats.count, 7);
        assert_eq!(report.stats.empty, 1);
        // "tiny" (4 chars) and "x" (1 char) both fall under near_empty_chars=4.
        assert_eq!(report.stats.near_empty, 2);
        assert_eq!(report.stats.giant, 1);
        assert_eq!(report.stats.missing_ids, 1);
        // duplicate_text counts every member of every duplicate group.
        assert_eq!(report.stats.duplicate_text, 2);

        let kinds: Vec<&'static str> = report
            .warnings
            .iter()
            .map(|w| match w {
                ChunkLintWarning::EmptyChunks { .. } => "empty",
                ChunkLintWarning::NearEmptyChunks { .. } => "near_empty",
                ChunkLintWarning::TooManyTinyChunks { .. } => "tiny",
                ChunkLintWarning::GiantChunks { .. } => "giant",
                ChunkLintWarning::DuplicateChunks { .. } => "dup",
                ChunkLintWarning::MissingIds { .. } => "missing",
                ChunkLintWarning::Encoding { .. } => "encoding",
                ChunkLintWarning::UnknownLanguage { .. } => "unknown_language",
                ChunkLintWarning::DisallowedLanguages { .. } => "language",
            })
            .collect();
        assert!(kinds.contains(&"empty"));
        assert!(kinds.contains(&"giant"));
        assert!(kinds.contains(&"dup"));
        assert!(kinds.contains(&"missing"));
    }

    #[test]
    fn fatal_config_promotes_warnings_to_error() {
        let config = ChunkLintConfig {
            fatal: true,
            ..ChunkLintConfig::default()
        };
        let err = lint_chunks_strict(&corpus(), &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("chunk lint failed"), "got: {msg}");
    }

    #[test]
    fn clean_corpus_passes_strict_mode() {
        let chunks = vec![
            Chunk::new("a", "healthy chunk number one. ".repeat(4)),
            Chunk::new("b", "healthy chunk number two. ".repeat(4)),
            Chunk::new("c", "healthy chunk number three. ".repeat(4)),
        ];
        let config = ChunkLintConfig {
            fatal: true,
            ..ChunkLintConfig::default()
        };
        let report = lint_chunks_strict(&chunks, &config).unwrap();
        assert!(!report.has_warnings(), "got: {:?}", report.warnings);
        assert_eq!(report.stats.count, 3);
    }

    #[test]
    fn report_json_round_trips() {
        let report = lint_chunks(&corpus(), &ChunkLintConfig::default());
        let json = report.to_json().unwrap();
        let parsed: ChunkLintReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, report);
    }

    #[test]
    fn lint_flags_encoding_artifacts() {
        let chunks = vec![
            Chunk::new("bom", "\u{feff}starts with a byte order mark"),
            Chunk::new("ctrl", "contains \u{0007} bell"),
        ];

        let report = lint_chunks(&chunks, &ChunkLintConfig::default());

        assert_eq!(report.stats.bom_chunks, 1);
        assert_eq!(report.stats.control_chars, 1);
        assert!(report.warnings.iter().any(|warning| matches!(
            warning,
            ChunkLintWarning::Encoding {
                warning: EncodingLintWarning::ByteOrderMarks { count: 1 }
            }
        )));
        assert!(report.warnings.iter().any(|warning| matches!(
            warning,
            ChunkLintWarning::Encoding {
                warning: EncodingLintWarning::ControlCharacters { count: 1 }
            }
        )));
    }

    #[test]
    fn language_lint_flags_disallowed_languages() {
        let chunks = vec![
            Chunk::new("en", "This is an English technical report about retrieval."),
            Chunk::new(
                "fr",
                "Bonjour, ceci est un rapport technique sur la recherche.",
            ),
        ];
        let config = ChunkLintConfig {
            language: LanguageLintConfig {
                enabled: true,
                allowed_languages: vec!["eng".into()],
                max_unknown_fraction: 1.0,
            },
            ..ChunkLintConfig::default()
        };

        let report = lint_chunks(&chunks, &config);

        assert!(
            report
                .warnings
                .iter()
                .any(|warning| matches!(warning, ChunkLintWarning::DisallowedLanguages { .. }))
        );
    }
}
