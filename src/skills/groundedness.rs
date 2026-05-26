//! Retrieval-grounded grader.
//!
//! Bridges the skill harness back to this crate's primary surface — any
//! [`VectorStoreIndexDyn`] — so a skill evaluation can verify the agent's
//! final output is *grounded in retrievable context*, not just shaped like
//! a correct answer.
//!
//! The grader re-queries the supplied store with a configurable extract of
//! the `(task, transcript)` pair (default: `transcript.final_output`,
//! falling back to `task.prompt` when the output is empty), fetches the
//! top-`k` documents, and runs a scoring closure (default: token-recall of
//! the final output against the concatenated top-`k` payloads).
//!
//! This is a *deterministic* grounding heuristic — it does not call an
//! LLM. For a judge-driven groundedness signal use
//! [`RagasJudgeGrader`](crate::skills::RagasJudgeGrader) with the
//! `FaithfulnessMetric`.

use std::pin::Pin;
use std::sync::Arc;

use rig::vector_store::request::Filter;
use rig::vector_store::{VectorSearchRequest, VectorStoreIndexDyn};
use serde_json::Value;

use crate::skills::grader::{AsyncGrader, GraderOutcome};
use crate::skills::task::SkillTask;
use crate::skills::transcript::Transcript;

/// Closure that derives the retrieval query from a `(task, transcript)`
/// pair. Returning an empty string causes the grader to emit a
/// [`GraderOutcome::skipped`] (no signal to ground against).
pub type GroundednessQueryFn = Arc<dyn Fn(&SkillTask, &Transcript) -> String + Send + Sync>;

/// Closure that scores a candidate answer against the concatenated text of
/// the top-`k` retrieved documents. Must return a value in `[0.0, 1.0]`.
pub type GroundednessScorerFn = Arc<dyn Fn(&str, &[String]) -> f64 + Send + Sync>;

/// Closure that extracts the human-readable payload string from a raw
/// `top_n` document JSON value. Default reads the conventional `"content"`
/// string field; override for stores that use a different schema.
pub type DocumentExtractorFn = Arc<dyn Fn(&Value) -> String + Send + Sync>;

/// Default query function — `transcript.final_output`, falling back to
/// `task.prompt` if empty.
pub fn default_query_fn() -> GroundednessQueryFn {
    Arc::new(|task, transcript| {
        let trimmed = transcript.final_output.trim();
        if trimmed.is_empty() {
            task.prompt.clone()
        } else {
            transcript.final_output.clone()
        }
    })
}

/// Default document extractor — reads the `"content"` string field if
/// present, otherwise serializes the whole value.
pub fn default_document_extractor() -> DocumentExtractorFn {
    Arc::new(|doc| {
        if let Some(s) = doc.get("content").and_then(Value::as_str) {
            return s.to_string();
        }
        doc.to_string()
    })
}

/// Default scorer — token recall of `answer` against the concatenation of
/// `contexts`. Tokens are lower-cased, ASCII-alphanumeric runs split on
/// whitespace and punctuation. Returns the fraction of unique answer
/// tokens that appear in the joined context corpus, in `[0.0, 1.0]`.
pub fn default_scorer() -> GroundednessScorerFn {
    Arc::new(|answer, contexts| {
        let answer_tokens = tokenize_unique(answer);
        if answer_tokens.is_empty() {
            return 0.0;
        }
        let mut corpus = String::new();
        for c in contexts {
            corpus.push_str(c);
            corpus.push(' ');
        }
        let corpus_tokens = tokenize_unique(&corpus);
        let hits = answer_tokens
            .iter()
            .filter(|t| corpus_tokens.contains(*t))
            .count();
        hits as f64 / answer_tokens.len() as f64
    })
}

fn tokenize_unique(s: &str) -> std::collections::BTreeSet<String> {
    s.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// Async grader that verifies an agent's transcript is grounded in the
/// documents retrievable from a supplied [`VectorStoreIndexDyn`].
///
/// Score is the value returned by the scorer closure, clamped to
/// `[0.0, 1.0]`. To compose cleanly with the harness's binary pass
/// aggregation, scores are snapped to `1.0` / `0.0` against
/// `pass_threshold` (default `0.5`); the raw scorer value is preserved in
/// the outcome `notes` field for audit.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use rig::vector_store::VectorStoreIndexDyn;
/// # use rig_retrieval_evals::skills::RetrievalGroundednessGrader;
/// # fn demo(store: Arc<dyn VectorStoreIndexDyn>) {
/// let grader = RetrievalGroundednessGrader::new("grounded", store)
///     .with_k(5)
///     .with_pass_threshold(0.6);
/// # let _ = grader;
/// # }
/// ```
pub struct RetrievalGroundednessGrader {
    id: String,
    store: Arc<dyn VectorStoreIndexDyn>,
    k: u64,
    pass_threshold: f64,
    query_fn: GroundednessQueryFn,
    scorer: GroundednessScorerFn,
    extractor: DocumentExtractorFn,
}

impl RetrievalGroundednessGrader {
    /// Build a new grader bound to `store`. Defaults: `k = 5`,
    /// `pass_threshold = 0.5`, default query/scorer/extractor closures.
    pub fn new(id: impl Into<String>, store: Arc<dyn VectorStoreIndexDyn>) -> Self {
        Self {
            id: id.into(),
            store,
            k: 5,
            pass_threshold: 0.5,
            query_fn: default_query_fn(),
            scorer: default_scorer(),
            extractor: default_document_extractor(),
        }
    }

    /// Set the number of documents to fetch from the store per trial.
    pub fn with_k(mut self, k: u64) -> Self {
        self.k = k;
        self
    }

    /// Set the score above which a trial is considered grounded.
    pub fn with_pass_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold;
        self
    }

    /// Override the query-derivation closure.
    pub fn with_query_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&SkillTask, &Transcript) -> String + Send + Sync + 'static,
    {
        self.query_fn = Arc::new(f);
        self
    }

    /// Override the scoring closure. Closure must return a value in
    /// `[0.0, 1.0]`; out-of-range values are clamped.
    pub fn with_scorer<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &[String]) -> f64 + Send + Sync + 'static,
    {
        self.scorer = Arc::new(f);
        self
    }

    /// Override the document-text extractor.
    pub fn with_document_extractor<F>(mut self, f: F) -> Self
    where
        F: Fn(&Value) -> String + Send + Sync + 'static,
    {
        self.extractor = Arc::new(f);
        self
    }
}

impl AsyncGrader for RetrievalGroundednessGrader {
    fn id(&self) -> &str {
        &self.id
    }

    fn grade<'a>(
        &'a self,
        task: &'a SkillTask,
        transcript: &'a Transcript,
    ) -> Pin<Box<dyn std::future::Future<Output = GraderOutcome> + Send + 'a>> {
        let id = self.id.clone();
        let threshold = self.pass_threshold;
        let k = self.k;
        let store = self.store.clone();
        let query = (self.query_fn)(task, transcript);
        let scorer = self.scorer.clone();
        let extractor = self.extractor.clone();
        let answer = transcript.final_output.clone();

        Box::pin(async move {
            if query.trim().is_empty() {
                return GraderOutcome::skipped(id, "empty retrieval query");
            }
            let req: VectorSearchRequest<Filter<Value>> = VectorSearchRequest::builder()
                .query(query)
                .samples(k)
                .build();
            let hits = match store.top_n(req).await {
                Ok(hits) => hits,
                Err(err) => {
                    return GraderOutcome::fail(id, format!("retrieval error: {err}"));
                }
            };
            let contexts: Vec<String> = hits.iter().map(|(_, _, doc)| extractor(doc)).collect();
            let raw = scorer(&answer, &contexts).clamp(0.0, 1.0);
            let passed = raw >= threshold;
            let score = if passed { 1.0 } else { 0.0 };
            let notes = format!("grounded_score={raw:.4}; k={k}");
            GraderOutcome {
                id,
                score,
                passed,
                notes,
            }
        })
    }
}
