//! Integration test for the `skills` harness.

#![cfg(feature = "skills")]
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::future::Future;
use std::pin::Pin;

use rig_retrieval_evals::Result;
use rig_retrieval_evals::skills::{
    AgentRunner, AsyncGrader, ContainsGrader, GraderOutcome, SkillHarness, SkillTask, SkillTaskSet,
    ToolCall, ToolCallGrader, Transcript, TranscriptBudget, TriggerGrader, Usage,
};

/// Deterministic runner: echoes the prompt, optionally claims a skill,
/// and synthesises a `search` tool call when the prompt contains "search".
struct EchoRunner;

impl AgentRunner for EchoRunner {
    fn run(
        &self,
        task: &SkillTask,
        _trial: usize,
    ) -> impl Future<Output = Result<Transcript>> + Send {
        let prompt = task.prompt.clone();
        let should = task.should_trigger.clone();
        async move {
            let mut tool_calls = Vec::new();
            if prompt.contains("search") {
                tool_calls.push(ToolCall::new("search"));
            }
            Ok(Transcript {
                prompt: prompt.clone(),
                final_output: prompt,
                tool_calls,
                usage: Some(Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cost_usd: Some(0.001),
                }),
                turns: Some(1),
                skill_invoked: should,
                ..Default::default()
            })
        }
    }
}

#[tokio::test]
async fn echo_runner_passes_all_graders() {
    let graders: Vec<Box<dyn AsyncGrader>> = vec![
        Box::new(ContainsGrader::present("greets", "hello")),
        Box::new(ToolCallGrader::at_least_once("uses_search", "search")),
        Box::new(TriggerGrader::new("trigger")),
        Box::new({
            let mut b = TranscriptBudget::new("budget");
            b.max_total_tokens = Some(1000);
            b.max_turns = Some(3);
            b
        }),
    ];

    let mut suite = SkillTaskSet::new("echo.v1");
    suite.push(
        SkillTask::new("t1", "hello and search the docs")
            .with_should_trigger("docs")
            .with_grader("greets")
            .with_grader("uses_search")
            .with_grader("trigger")
            .with_grader("budget"),
    );
    suite.push(
        SkillTask::new("t2", "hello world")
            .with_should_trigger("greet")
            .with_grader("greets")
            .with_grader("trigger")
            .with_grader("budget"),
    );

    let report = SkillHarness::new(EchoRunner, graders)
        .with_trials(3)
        .with_concurrency(2)
        .run(&suite)
        .await
        .unwrap();

    assert_eq!(report.suite_id, "echo.v1");
    assert_eq!(report.n_tasks, 2);
    assert_eq!(report.trials, 3);
    assert!((report.mean_pass_rate("greets").unwrap() - 1.0).abs() < 1e-9);
    assert!((report.mean_pass_rate("trigger").unwrap() - 1.0).abs() < 1e-9);
    assert!((report.mean_pass_rate("budget").unwrap() - 1.0).abs() < 1e-9);

    // `uses_search` only applies to t1 — verify it shows up in per-grader
    // reports with a single per-task entry.
    let search_trials = report.per_grader.get("uses_search").unwrap();
    assert_eq!(search_trials.len(), 3);
    assert_eq!(search_trials[0].per_query.len(), 1);
    assert_eq!(search_trials[0].per_query[0].0, "t1");
}

#[tokio::test]
async fn missing_grader_is_a_config_error() {
    let graders: Vec<Box<dyn AsyncGrader>> =
        vec![Box::new(ContainsGrader::present("greets", "hi"))];
    let mut suite = SkillTaskSet::new("bad");
    suite.push(SkillTask::new("t1", "hi").with_grader("does_not_exist"));

    let err = SkillHarness::new(EchoRunner, graders)
        .run(&suite)
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("does_not_exist"));
}

#[tokio::test]
async fn budget_grader_fails_on_token_overrun() {
    struct BloatedRunner;
    impl AgentRunner for BloatedRunner {
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
                    usage: Some(Usage {
                        input_tokens: 10_000,
                        output_tokens: 10_000,
                        cost_usd: None,
                    }),
                    turns: Some(1),
                    ..Default::default()
                })
            }
        }
    }

    let graders: Vec<Box<dyn AsyncGrader>> = vec![Box::new({
        let mut b = TranscriptBudget::new("budget");
        b.max_total_tokens = Some(1000);
        b
    })];
    let mut suite = SkillTaskSet::new("budget.v1");
    suite.push(SkillTask::new("t1", "do stuff").with_grader("budget"));

    let report = SkillHarness::new(BloatedRunner, graders)
        .with_trials(2)
        .run(&suite)
        .await
        .unwrap();

    assert!((report.mean_pass_rate("budget").unwrap() - 0.0).abs() < 1e-9);
}

struct PartialAsyncGrader;

impl AsyncGrader for PartialAsyncGrader {
    fn id(&self) -> &str {
        "partial_async"
    }

    fn grade<'a>(
        &'a self,
        _task: &'a SkillTask,
        _transcript: &'a Transcript,
    ) -> Pin<Box<dyn Future<Output = GraderOutcome> + Send + 'a>> {
        Box::pin(async { GraderOutcome::partial("partial_async", 0.75, "partial credit") })
    }
}

#[tokio::test]
async fn custom_async_grader_is_awaited_and_thresholded() {
    let graders: Vec<Box<dyn AsyncGrader>> = vec![Box::new(PartialAsyncGrader)];
    let mut suite = SkillTaskSet::new("async.v1");
    suite.push(SkillTask::new("t1", "anything").with_grader("partial_async"));

    let report = SkillHarness::new(EchoRunner, graders)
        .with_trials(2)
        .with_pass_threshold(0.7)
        .run(&suite)
        .await
        .unwrap();

    let metric = report.per_grader.get("partial_async").unwrap();
    assert_eq!(metric.len(), 2);
    assert!((metric[0].mean - 0.75).abs() < 1e-9);
    assert!((report.mean_pass_rate("partial_async").unwrap() - 1.0).abs() < 1e-9);
}

#[tokio::test]
async fn duplicate_grader_ids_are_config_errors() {
    let graders: Vec<Box<dyn AsyncGrader>> = vec![
        Box::new(ContainsGrader::present("same", "hello")),
        Box::new(ContainsGrader::absent("same", "goodbye")),
    ];
    let mut suite = SkillTaskSet::new("duplicate.v1");
    suite.push(SkillTask::new("t1", "hello").with_grader("same"));

    let err = SkillHarness::new(EchoRunner, graders)
        .run(&suite)
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("duplicate grader id"));
}

#[tokio::test]
async fn empty_task_set_is_a_config_error() {
    let graders: Vec<Box<dyn AsyncGrader>> =
        vec![Box::new(ContainsGrader::present("greets", "hello"))];
    let suite = SkillTaskSet::new("empty.v1");

    let err = SkillHarness::new(EchoRunner, graders)
        .run(&suite)
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("skill task set is empty"));
}
