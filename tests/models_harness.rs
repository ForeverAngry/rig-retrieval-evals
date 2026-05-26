//! Integration tests for the model-behavior harness.

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::future::Future;
use std::pin::Pin;

use rig_retrieval_evals::{
    ModelBehaviorHarness, ModelBehaviorTask, ModelBehaviorTaskSet, ModelObservation, ModelRunner,
    Result,
};

struct JsonModel;

impl ModelRunner for JsonModel {
    fn run<'a>(
        &'a self,
        task: &'a ModelBehaviorTask,
    ) -> Pin<Box<dyn Future<Output = Result<ModelObservation>> + Send + 'a>> {
        Box::pin(async move {
            Ok(ModelObservation {
                output: format!(r#"{{"answer":"{} alpha"}}"#, task.prompt),
                input_tokens: Some(10),
                output_tokens: Some(8),
                ..Default::default()
            })
        })
    }
}

#[tokio::test]
async fn model_harness_scores_json_terms_and_budget() {
    let mut suite = ModelBehaviorTaskSet::new("models.v1");
    suite.push(
        ModelBehaviorTask::new("q1", "hello")
            .must_contain("alpha")
            .must_not_contain("secret")
            .requiring_json()
            .with_max_output_tokens(16),
    );

    let report = ModelBehaviorHarness::new(JsonModel)
        .with_concurrency(2)
        .run(&suite)
        .await
        .unwrap();

    assert_eq!(report.suite_id, "models.v1");
    assert_eq!(report.n_tasks, 1);
    assert!((report.mean_score - 1.0).abs() < 1e-9);
    assert!(report.results[0].passed);
    assert_eq!(report.metric_report().metric, "model.behavior");
}

#[tokio::test]
async fn model_harness_fails_missing_budget_telemetry() {
    struct NoUsage;

    impl ModelRunner for NoUsage {
        fn run<'a>(
            &'a self,
            _task: &'a ModelBehaviorTask,
        ) -> Pin<Box<dyn Future<Output = Result<ModelObservation>> + Send + 'a>> {
            Box::pin(async move {
                Ok(ModelObservation {
                    output: "plain text".to_string(),
                    ..Default::default()
                })
            })
        }
    }

    let mut suite = ModelBehaviorTaskSet::new("budget.v1");
    suite.push(ModelBehaviorTask::new("q1", "hello").with_max_output_tokens(4));

    let report = ModelBehaviorHarness::new(NoUsage)
        .run(&suite)
        .await
        .unwrap();

    assert!(!report.results[0].passed);
    assert!(report.results[0].notes[0].contains("output tokens"));
}

#[tokio::test]
async fn empty_model_suite_is_a_config_error() {
    let suite = ModelBehaviorTaskSet::new("empty");
    let err = ModelBehaviorHarness::new(JsonModel)
        .run(&suite)
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("model behavior task set is empty"));
}
