//! Integration tests for the agent harness.

#![cfg(feature = "agents")]
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::future::Future;
use std::pin::Pin;

use rig_retrieval_evals::{
    AgentEvalRunner, AgentEvalTask, AgentEvalTaskSet, AgentHarness, AgentObservation,
    AgentToolCall, Result,
};

struct SearchAgent;

impl AgentEvalRunner for SearchAgent {
    fn run<'a>(
        &'a self,
        task: &'a AgentEvalTask,
    ) -> Pin<Box<dyn Future<Output = Result<AgentObservation>> + Send + 'a>> {
        Box::pin(async move {
            let mut tool_calls = Vec::new();
            if task.prompt.contains("search") {
                tool_calls.push(AgentToolCall::new("search"));
            }
            Ok(AgentObservation {
                final_output: format!("resolved {} safely", task.prompt),
                tool_calls,
                turns: Some(2),
                ..Default::default()
            })
        })
    }
}

#[tokio::test]
async fn agent_harness_scores_output_tools_and_turn_budget() {
    let mut suite = AgentEvalTaskSet::new("agents.v1");
    suite.push(
        AgentEvalTask::new("a1", "search docs")
            .expect_output("resolved")
            .forbid_output("panic")
            .expect_tool("search")
            .with_max_turns(3),
    );

    let report = AgentHarness::new(SearchAgent)
        .with_concurrency(2)
        .run(&suite)
        .await
        .unwrap();

    assert_eq!(report.suite_id, "agents.v1");
    assert_eq!(report.n_tasks, 1);
    assert!((report.mean_score - 1.0).abs() < 1e-9);
    assert!(report.results[0].passed);
    assert_eq!(report.metric_report().metric, "agent.behavior");
}

#[tokio::test]
async fn agent_harness_reports_missing_tool() {
    let mut suite = AgentEvalTaskSet::new("agents.partial");
    suite.push(AgentEvalTask::new("a1", "no tools").expect_tool("search"));

    let report = AgentHarness::new(SearchAgent).run(&suite).await.unwrap();

    assert!(!report.results[0].passed);
    assert!(report.results[0].notes[0].contains("search"));
}

#[tokio::test]
async fn empty_agent_suite_is_a_config_error() {
    let suite = AgentEvalTaskSet::new("empty");
    let err = AgentHarness::new(SearchAgent)
        .run(&suite)
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("agent task set is empty"));
}
