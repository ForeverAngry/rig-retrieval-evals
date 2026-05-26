//! Integration tests for the memory harness.

#![cfg(feature = "memory")]
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::future::Future;
use std::pin::Pin;

use rig_retrieval_evals::{
    MemoryHarness, MemoryObservation, MemoryRunner, MemoryTask, MemoryTaskSet, Result,
};

struct EchoMemory;

impl MemoryRunner for EchoMemory {
    fn run<'a>(
        &'a self,
        task: &'a MemoryTask,
    ) -> Pin<Box<dyn Future<Output = Result<MemoryObservation>> + Send + 'a>> {
        Box::pin(async move {
            let mut retrieved = Vec::new();
            if let Some(write) = &task.write {
                retrieved.push(write.clone());
            }
            retrieved.push(format!("answer for {}", task.query));
            Ok(MemoryObservation {
                retrieved,
                reloaded: task.require_reload,
                item_count: Some(1),
                ..Default::default()
            })
        })
    }
}

#[tokio::test]
async fn memory_harness_scores_expected_and_forbidden_terms() {
    let mut suite = MemoryTaskSet::new("memory.v1");
    suite.push(
        MemoryTask::new("m1", "where is the token?")
            .with_write("alpha memory token")
            .expect_term("alpha")
            .forbid_term("omega")
            .requiring_reload(),
    );

    let report = MemoryHarness::new(EchoMemory)
        .with_concurrency(2)
        .run(&suite)
        .await
        .unwrap();

    assert_eq!(report.suite_id, "memory.v1");
    assert_eq!(report.n_tasks, 1);
    assert!((report.mean_score - 1.0).abs() < 1e-9);
    assert!(report.results[0].passed);
    assert_eq!(report.metric_report().metric, "memory.recall");
}

#[tokio::test]
async fn memory_harness_reports_partial_failures() {
    let mut suite = MemoryTaskSet::new("memory.partial");
    suite.push(MemoryTask::new("m1", "query").expect_term("missing"));

    let report = MemoryHarness::new(EchoMemory).run(&suite).await.unwrap();

    assert!(!report.results[0].passed);
    assert!((report.results[0].score - 0.0).abs() < 1e-9);
    assert!(report.results[0].notes[0].contains("missing"));
}

#[tokio::test]
async fn empty_memory_suite_is_a_config_error() {
    let suite = MemoryTaskSet::new("empty");
    let err = MemoryHarness::new(EchoMemory)
        .run(&suite)
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("memory task set is empty"));
}
