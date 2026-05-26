//! Model-behavior evaluation harnesses.
//!
//! `models` evaluates the observable behavior of a model invocation: output
//! text, JSON validity, forbidden leakage, and token budget. Provider metadata
//! remains the responsibility of companion crates such as `rig-model-catalog`.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::report::MetricReport;

/// One model-behavior task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBehaviorTask {
    /// Stable task identifier.
    pub id: String,
    /// Prompt to send to the model.
    pub prompt: String,
    /// Terms that must appear in the output.
    #[serde(default)]
    pub must_contain: Vec<String>,
    /// Terms that must not appear in the output.
    #[serde(default)]
    pub must_not_contain: Vec<String>,
    /// Require the output to parse as JSON.
    #[serde(default)]
    pub require_json: bool,
    /// Optional maximum output-token budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Free-form metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl ModelBehaviorTask {
    /// Build a task with no assertions.
    pub fn new(id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            prompt: prompt.into(),
            must_contain: Vec::new(),
            must_not_contain: Vec::new(),
            require_json: false,
            max_output_tokens: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Add a required output term.
    #[must_use]
    pub fn must_contain(mut self, term: impl Into<String>) -> Self {
        self.must_contain.push(term.into());
        self
    }

    /// Add a forbidden output term.
    #[must_use]
    pub fn must_not_contain(mut self, term: impl Into<String>) -> Self {
        self.must_not_contain.push(term.into());
        self
    }

    /// Require JSON output.
    #[must_use]
    pub fn requiring_json(mut self) -> Self {
        self.require_json = true;
        self
    }

    /// Set a maximum output-token budget.
    #[must_use]
    pub fn with_max_output_tokens(mut self, max: u64) -> Self {
        self.max_output_tokens = Some(max);
        self
    }
}

/// A named collection of model tasks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelBehaviorTaskSet {
    /// Suite identifier.
    pub id: String,
    /// Tasks in input order.
    pub tasks: Vec<ModelBehaviorTask>,
}

impl ModelBehaviorTaskSet {
    /// Build an empty suite.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tasks: Vec::new(),
        }
    }

    /// Append a task.
    pub fn push(&mut self, task: ModelBehaviorTask) {
        self.tasks.push(task);
    }

    /// Number of tasks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Whether the suite is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

/// Captured model output for one task.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelObservation {
    /// Raw user-visible model output.
    pub output: String,
    /// Optional provider-reported input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Optional provider-reported output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Free-form metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// Runs one model task.
pub trait ModelRunner: Send + Sync {
    /// Execute one prompt and capture the observable output.
    fn run<'a>(
        &'a self,
        task: &'a ModelBehaviorTask,
    ) -> Pin<Box<dyn Future<Output = Result<ModelObservation>> + Send + 'a>>;
}

/// One graded model-behavior result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBehaviorResult {
    /// Task identifier.
    pub task_id: String,
    /// Score in `[0, 1]`.
    pub score: f64,
    /// Whether every assertion passed.
    pub passed: bool,
    /// Captured output.
    pub observation: ModelObservation,
    /// Human-readable notes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// Aggregated model-behavior report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBehaviorReport {
    /// Suite identifier.
    pub suite_id: String,
    /// Number of tasks scored.
    pub n_tasks: usize,
    /// Mean score across tasks.
    pub mean_score: f64,
    /// Per-task results.
    pub results: Vec<ModelBehaviorResult>,
}

impl ModelBehaviorReport {
    /// Convert per-task scores into a generic metric report.
    #[must_use]
    pub fn metric_report(&self) -> MetricReport {
        MetricReport::from_per_query(
            "model.behavior".to_string(),
            self.results
                .iter()
                .map(|result| (result.task_id.clone(), result.score))
                .collect(),
        )
    }
}

/// Drives model-behavior tasks and grades outputs.
pub struct ModelBehaviorHarness<R: ModelRunner> {
    runner: R,
    concurrency: usize,
}

impl<R: ModelRunner> ModelBehaviorHarness<R> {
    /// Build a harness for a runner.
    pub fn new(runner: R) -> Self {
        Self {
            runner,
            concurrency: 1,
        }
    }

    /// Set maximum concurrent tasks. Values of `0` are clamped to `1`.
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Execute and grade every task.
    pub async fn run(&self, tasks: &ModelBehaviorTaskSet) -> Result<ModelBehaviorReport> {
        if tasks.is_empty() {
            return Err(Error::Config("model behavior task set is empty".into()));
        }

        let rows = stream::iter(tasks.tasks.iter().map(|task| async move {
            let observation = self.runner.run(task).await?;
            Ok::<_, Error>(grade_model_task(task, observation))
        }))
        .buffer_unordered(self.concurrency)
        .collect::<Vec<_>>()
        .await;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            results.push(row?);
        }
        results.sort_by(|a, b| a.task_id.cmp(&b.task_id));

        let mean_score = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|result| result.score).sum::<f64>() / results.len() as f64
        };

        Ok(ModelBehaviorReport {
            suite_id: tasks.id.clone(),
            n_tasks: tasks.len(),
            mean_score,
            results,
        })
    }
}

fn grade_model_task(
    task: &ModelBehaviorTask,
    observation: ModelObservation,
) -> ModelBehaviorResult {
    let output_lower = observation.output.to_lowercase();
    let mut checks = 0usize;
    let mut passed = 0usize;
    let mut notes = Vec::new();

    for term in &task.must_contain {
        checks = checks.saturating_add(1);
        if output_lower.contains(&term.to_lowercase()) {
            passed = passed.saturating_add(1);
        } else {
            notes.push(format!("missing required term `{term}`"));
        }
    }

    for term in &task.must_not_contain {
        checks = checks.saturating_add(1);
        if output_lower.contains(&term.to_lowercase()) {
            notes.push(format!("found forbidden term `{term}`"));
        } else {
            passed = passed.saturating_add(1);
        }
    }

    if task.require_json {
        checks = checks.saturating_add(1);
        if serde_json::from_str::<serde_json::Value>(&observation.output).is_ok() {
            passed = passed.saturating_add(1);
        } else {
            notes.push("output was not valid JSON".to_string());
        }
    }

    if let Some(max) = task.max_output_tokens {
        checks = checks.saturating_add(1);
        match observation.output_tokens {
            Some(tokens) if tokens <= max => {
                passed = passed.saturating_add(1);
            }
            Some(tokens) => notes.push(format!("output tokens {tokens} exceeded max {max}")),
            None => notes.push("runner did not report output tokens".to_string()),
        }
    }

    let score = if checks == 0 {
        1.0
    } else {
        passed as f64 / checks as f64
    };

    ModelBehaviorResult {
        task_id: task.id.clone(),
        score,
        passed: (score - 1.0).abs() < f64::EPSILON,
        observation,
        notes,
    }
}
