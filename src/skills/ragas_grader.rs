//! Bridge a [`RagasMetric`](crate::ragas::RagasMetric) into the skill harness.
//!
//! Requires both the `skills` and `ragas` features.

use std::sync::Arc;

use crate::ragas::{RagasInputs, RagasMetric, RagasScore};
use crate::skills::grader::{AsyncGrader, GraderOutcome};
use crate::skills::task::SkillTask;
use crate::skills::transcript::Transcript;

/// Closure type that projects a `(task, transcript)` pair into the
/// [`RagasInputs`] shape required by [`RagasMetric`] judges.
pub type RagasInputsFn = Arc<dyn Fn(&SkillTask, &Transcript) -> RagasInputs + Send + Sync>;

/// Wraps a [`RagasMetric`] (e.g. `FaithfulnessMetric`, `AnswerRelevanceMetric`)
/// so it can be registered as an [`AsyncGrader`] alongside deterministic
/// graders inside a [`SkillHarness`](crate::skills::SkillHarness).
///
/// The judge's continuous score in `[0.0, 1.0]` is mapped to a pass/fail by
/// comparing against `pass_threshold` (default `0.5`). A
/// [`RagasScore::not_measurable`] outcome is recorded as a
/// [`GraderOutcome::skipped`] so the harness does not penalise a sample the
/// judge legitimately abstained on.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use rig_retrieval_evals::ragas::{RagasInputs, RagasMetric, RagasScore};
/// # use rig_retrieval_evals::skills::RagasJudgeGrader;
/// # use rig_retrieval_evals::Result;
/// # use std::future::Future;
/// struct StubFaithfulness;
/// impl RagasMetric for StubFaithfulness {
///     fn name(&self) -> &'static str { "faithfulness" }
///     fn fingerprint_component(&self) -> String { "stub:faithfulness".into() }
///     fn score(&self, _: &RagasInputs) -> impl Future<Output = Result<RagasScore>> + Send {
///         async { Ok(RagasScore::measured(0.9)) }
///     }
/// }
///
/// let grader = RagasJudgeGrader::new("faithfulness", StubFaithfulness)
///     .with_pass_threshold(0.7);
/// ```
pub struct RagasJudgeGrader<M> {
    id: String,
    metric: M,
    pass_threshold: f64,
    inputs_fn: RagasInputsFn,
}

impl<M: RagasMetric> RagasJudgeGrader<M> {
    /// Build a judge grader with the default input mapping
    /// (`task.prompt → query`, `transcript.final_output → answer`, empty
    /// context, no reference) and a pass threshold of `0.5`.
    pub fn new(id: impl Into<String>, metric: M) -> Self {
        Self {
            id: id.into(),
            metric,
            pass_threshold: 0.5,
            inputs_fn: default_inputs_fn(),
        }
    }

    /// Override the score threshold above which a judge outcome is
    /// considered a pass.
    #[must_use]
    pub fn with_pass_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold;
        self
    }

    /// Override the `(task, transcript) → RagasInputs` projection. Use this
    /// when your judge needs retrieved context, a reference answer, or a
    /// custom query phrasing.
    #[must_use]
    pub fn with_inputs_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&SkillTask, &Transcript) -> RagasInputs + Send + Sync + 'static,
    {
        self.inputs_fn = Arc::new(f);
        self
    }
}

/// Default mapping: prompt → query, final_output → answer, no context, no
/// reference. Suitable for [`AnswerRelevanceMetric`](crate::ragas::AnswerRelevanceMetric)
/// and judges that can score "answer addresses prompt" without retrieval.
pub fn default_inputs_fn() -> RagasInputsFn {
    Arc::new(|task: &SkillTask, transcript: &Transcript| RagasInputs {
        query_id: task.id.clone(),
        query: task.prompt.clone(),
        answer: Some(transcript.final_output.clone()),
        context: Vec::new(),
        reference_answer: None,
    })
}

impl<M: RagasMetric + 'static> AsyncGrader for RagasJudgeGrader<M> {
    fn id(&self) -> &str {
        &self.id
    }

    fn grade<'a>(
        &'a self,
        task: &'a SkillTask,
        transcript: &'a Transcript,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = GraderOutcome> + Send + 'a>> {
        let id = self.id.clone();
        let threshold = self.pass_threshold;
        let inputs = (self.inputs_fn)(task, transcript);
        Box::pin(async move {
            match self.metric.score(&inputs).await {
                Ok(RagasScore {
                    value: Some(v),
                    rationales,
                }) => {
                    let raw = v.clamp(0.0, 1.0);
                    let passed = raw >= threshold;
                    // Snap to 0/1 so the score plays nicely with the
                    // harness's binary `pass_threshold` (default 1.0) when
                    // mixed alongside deterministic graders. The raw judge
                    // score is preserved in `notes` for audit.
                    let score = if passed { 1.0 } else { 0.0 };
                    let mut notes = format!("judge_score={raw:.4}");
                    if !rationales.is_empty() {
                        notes.push_str("; ");
                        notes.push_str(&rationales.join("; "));
                    }
                    GraderOutcome {
                        id,
                        score,
                        passed,
                        notes,
                    }
                }
                Ok(RagasScore {
                    value: None,
                    rationales,
                }) => GraderOutcome::skipped(id, rationales.join("; ")),
                Err(err) => GraderOutcome::fail(id, format!("judge error: {err}")),
            }
        })
    }
}
