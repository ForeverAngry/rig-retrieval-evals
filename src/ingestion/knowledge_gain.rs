//! Lightweight ingestion novelty scoring.
//!
//! The top-level `knowledge_gain` module handles metric deltas and optional
//! embedding novelty. This ingestion-local module is intentionally cheaper:
//! normalize text to token sets and compute Jaccard novelty so ingestion
//! gates can estimate whether a candidate adds new lexical information.

use std::collections::BTreeSet;

/// Compute lexical novelty as `1 - Jaccard(candidate, baseline)`.
///
/// Returns `0.0` when both inputs are token-empty, and `1.0` when the
/// candidate has tokens but the baseline has none.
#[must_use]
pub fn jaccard_knowledge_gain(candidate: &str, baseline: &str) -> f64 {
    let candidate_tokens = normalized_tokens(candidate);
    let baseline_tokens = normalized_tokens(baseline);
    jaccard_novelty(&candidate_tokens, &baseline_tokens)
}

/// Compute lexical novelty for a set of candidate texts against a set of
/// baseline texts.
#[must_use]
pub fn corpus_jaccard_knowledge_gain<'a>(
    candidates: impl IntoIterator<Item = &'a str>,
    baseline: impl IntoIterator<Item = &'a str>,
) -> f64 {
    let candidate_tokens = candidates
        .into_iter()
        .flat_map(normalized_tokens)
        .collect::<BTreeSet<_>>();
    let baseline_tokens = baseline
        .into_iter()
        .flat_map(normalized_tokens)
        .collect::<BTreeSet<_>>();
    jaccard_novelty(&candidate_tokens, &baseline_tokens)
}

fn normalized_tokens(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

fn jaccard_novelty(candidate: &BTreeSet<String>, baseline: &BTreeSet<String>) -> f64 {
    if candidate.is_empty() && baseline.is_empty() {
        return 0.0;
    }
    if candidate.is_empty() {
        return 0.0;
    }
    if baseline.is_empty() {
        return 1.0;
    }
    let intersection = candidate.intersection(baseline).count() as f64;
    let union = candidate.union(baseline).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        1.0 - (intersection / union)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_has_no_gain() {
        assert_eq!(jaccard_knowledge_gain("alpha beta", "alpha beta"), 0.0);
    }

    #[test]
    fn disjoint_candidate_has_full_gain() {
        assert_eq!(jaccard_knowledge_gain("gamma", "alpha beta"), 1.0);
    }

    #[test]
    fn corpus_gain_merges_token_sets() {
        let gain = corpus_jaccard_knowledge_gain(["alpha gamma"], ["alpha beta"]);
        assert!(gain > 0.0);
        assert!(gain < 1.0);
    }
}
