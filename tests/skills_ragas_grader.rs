//! Async-grader (LLM rubric) integration test using a stub `RagasMetric`.

#![cfg(all(feature = "skills", feature = "ragas"))]
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rig_retrieval_evals::ragas::{RagasInputs, RagasMetric, RagasScore};
use rig_retrieval_evals::skills::{
    AgentRunner, AsyncGrader, ContainsGrader, RagasJudgeGrader, SkillHarness, SkillTask,
    SkillTaskSet, Transcript,
};
use rig_retrieval_evals::{Error, Result};

struct EchoRunner;
impl AgentRunner for EchoRunner {
    fn run(
        &self,
        task: &SkillTask,
        _trial: usize,
    ) -> impl Future<Output = Result<Transcript>> + Send {
        let prompt = task.prompt.clone();
        async move {
            Ok(Transcript {
                prompt: prompt.clone(),
                final_output: prompt,
                ..Default::default()
            })
        }
    }
}

/// Stub judge that returns a fixed score and counts invocations.
struct StubJudge {
    score: f64,
    calls: Arc<AtomicUsize>,
}

impl RagasMetric for StubJudge {
    fn name(&self) -> &'static str {
        "stub_judge"
    }
    fn fingerprint_component(&self) -> String {
        format!("stub:{}", self.score)
    }
    fn score(&self, _: &RagasInputs) -> impl Future<Output = Result<RagasScore>> + Send {
        let s = self.score;
        let calls = self.calls.clone();
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(RagasScore::measured(s))
        }
    }
}

#[tokio::test]
async fn ragas_judge_grader_passes_above_threshold() {
    let calls = Arc::new(AtomicUsize::new(0));
    let judge = StubJudge {
        score: 0.8,
        calls: calls.clone(),
    };
    let graders: Vec<Box<dyn AsyncGrader>> = vec![
        Box::new(ContainsGrader::present("greets", "hello")),
        Box::new(RagasJudgeGrader::new("rubric", judge).with_pass_threshold(0.7)),
    ];

    let mut suite = SkillTaskSet::new("rubric.v1");
    suite.push(
        SkillTask::new("t1", "hello world")
            .with_grader("greets")
            .with_grader("rubric"),
    );
    suite.push(SkillTask::new("t2", "hello universe").with_grader("rubric"));

    let report = SkillHarness::new(EchoRunner, graders)
        .with_trials(2)
        .with_concurrency(2)
        .run(&suite)
        .await
        .unwrap();

    assert!((report.mean_pass_rate("rubric").unwrap() - 1.0).abs() < 1e-9);
    // 2 tasks × 2 trials = 4 judge invocations.
    assert_eq!(calls.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn ragas_judge_grader_fails_below_threshold() {
    let judge = StubJudge {
        score: 0.2,
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let graders: Vec<Box<dyn AsyncGrader>> = vec![Box::new(
        RagasJudgeGrader::new("rubric", judge).with_pass_threshold(0.5),
    )];
    let mut suite = SkillTaskSet::new("rubric_fail");
    suite.push(SkillTask::new("t1", "anything").with_grader("rubric"));

    let report = SkillHarness::new(EchoRunner, graders)
        .with_trials(1)
        .run(&suite)
        .await
        .unwrap();
    assert!((report.mean_pass_rate("rubric").unwrap() - 0.0).abs() < 1e-9);
}

struct CapturingJudge {
    seen: Arc<Mutex<Vec<RagasInputs>>>,
}

impl RagasMetric for CapturingJudge {
    fn name(&self) -> &'static str {
        "capturing_judge"
    }
    fn fingerprint_component(&self) -> String {
        "capturing".into()
    }
    fn score(&self, inputs: &RagasInputs) -> impl Future<Output = Result<RagasScore>> + Send {
        let seen = self.seen.clone();
        let captured = inputs.clone();
        async move {
            seen.lock().unwrap().push(captured);
            Ok(RagasScore::measured(1.0))
        }
    }
}

#[tokio::test]
async fn ragas_judge_grader_uses_custom_inputs_fn() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let judge = CapturingJudge { seen: seen.clone() };
    let graders: Vec<Box<dyn AsyncGrader>> = vec![Box::new(
        RagasJudgeGrader::new("rubric", judge).with_inputs_fn(|task, transcript| RagasInputs {
            query_id: format!("custom:{}", task.id),
            query: format!("query: {}", task.prompt),
            answer: Some(format!("answer: {}", transcript.final_output)),
            context: vec!["retrieved context".into()],
            reference_answer: Some("gold".into()),
        }),
    )];
    let mut suite = SkillTaskSet::new("custom_inputs.v1");
    suite.push(SkillTask::new("t1", "hello").with_grader("rubric"));

    let report = SkillHarness::new(EchoRunner, graders)
        .run(&suite)
        .await
        .unwrap();

    assert!((report.mean_pass_rate("rubric").unwrap() - 1.0).abs() < 1e-9);
    let locked = seen.lock().unwrap();
    assert_eq!(locked.len(), 1);
    assert_eq!(locked[0].query_id, "custom:t1");
    assert_eq!(locked[0].query, "query: hello");
    assert_eq!(locked[0].answer.as_deref(), Some("answer: hello"));
    assert_eq!(locked[0].context, vec!["retrieved context"]);
    assert_eq!(locked[0].reference_answer.as_deref(), Some("gold"));
}

#[derive(Clone, Copy)]
enum JudgeMode {
    NotMeasurable,
    Error,
}

struct ModeJudge {
    mode: JudgeMode,
}

impl RagasMetric for ModeJudge {
    fn name(&self) -> &'static str {
        "mode_judge"
    }
    fn fingerprint_component(&self) -> String {
        "mode".into()
    }
    fn score(&self, _: &RagasInputs) -> impl Future<Output = Result<RagasScore>> + Send {
        let outcome = match self.mode {
            JudgeMode::NotMeasurable => Ok(RagasScore::not_measurable("abstained")),
            JudgeMode::Error => Err(Error::Config("judge unavailable".into())),
        };
        async move { outcome }
    }
}

#[tokio::test]
async fn ragas_judge_grader_records_not_measurable_as_skipped() {
    let graders: Vec<Box<dyn AsyncGrader>> = vec![Box::new(RagasJudgeGrader::new(
        "rubric",
        ModeJudge {
            mode: JudgeMode::NotMeasurable,
        },
    ))];
    let mut suite = SkillTaskSet::new("abstain.v1");
    suite.push(SkillTask::new("t1", "anything").with_grader("rubric"));

    let report = SkillHarness::new(EchoRunner, graders)
        .run(&suite)
        .await
        .unwrap();
    let outcome = report.trials_log[0].outcomes.get("rubric").unwrap();
    assert!(outcome.passed);
    assert_eq!(outcome.score, 1.0);
    assert_eq!(outcome.notes, "abstained");
}

#[tokio::test]
async fn ragas_judge_grader_records_judge_error_as_failure() {
    let graders: Vec<Box<dyn AsyncGrader>> = vec![Box::new(RagasJudgeGrader::new(
        "rubric",
        ModeJudge {
            mode: JudgeMode::Error,
        },
    ))];
    let mut suite = SkillTaskSet::new("error.v1");
    suite.push(SkillTask::new("t1", "anything").with_grader("rubric"));

    let report = SkillHarness::new(EchoRunner, graders)
        .run(&suite)
        .await
        .unwrap();
    let outcome = report.trials_log[0].outcomes.get("rubric").unwrap();
    assert!(!outcome.passed);
    assert_eq!(outcome.score, 0.0);
    assert!(outcome.notes.contains("judge error"));
    assert!((report.mean_pass_rate("rubric").unwrap() - 0.0).abs() < 1e-9);
}
