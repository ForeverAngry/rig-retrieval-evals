//! Integration test for the retrieval-grounded skill grader.

#![cfg(feature = "skills")]
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use rig::vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex, request::Filter};
use rig::wasm_compat::WasmCompatSend;
use rig_retrieval_evals::Result;
use rig_retrieval_evals::skills::{
    AgentRunner, AsyncGrader, RetrievalGroundednessGrader, SkillHarness, SkillTask, SkillTaskSet,
    Transcript,
};
use serde::Deserialize;

/// In-memory mock store. `top_n` returns documents shaped as
/// `{ "id": ..., "content": ... }` so the default extractor finds the
/// payload string.
struct MockStore {
    docs: HashMap<String, String>,
}

impl MockStore {
    fn new() -> Self {
        let mut docs = HashMap::new();
        docs.insert(
            "doc-1".to_string(),
            "the capital of france is paris".to_string(),
        );
        docs.insert(
            "doc-2".to_string(),
            "physics describes the speed of light".to_string(),
        );
        docs.insert(
            "doc-3".to_string(),
            "unrelated commentary about pasta".to_string(),
        );
        Self { docs }
    }
}

fn tokens(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

impl VectorStoreIndex for MockStore {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> std::result::Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'a> Deserialize<'a> + WasmCompatSend,
    {
        let q = tokens(req.query());
        let mut scored: Vec<(f64, String, String)> = self
            .docs
            .iter()
            .map(|(id, text)| {
                let toks = tokens(text);
                let overlap = toks.iter().filter(|t| q.contains(t)).count();
                (overlap as f64, id.clone(), text.clone())
            })
            .filter(|(s, _, _)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(req.samples() as usize);
        let mut out = Vec::with_capacity(scored.len());
        for (score, id, content) in scored {
            let v = serde_json::json!({ "id": id, "content": content });
            let doc: T = serde_json::from_value(v).map_err(VectorStoreError::JsonError)?;
            out.push((score, id, doc));
        }
        Ok(out)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> std::result::Result<Vec<(f64, String)>, VectorStoreError> {
        let q = tokens(req.query());
        let mut scored: Vec<(f64, String)> = self
            .docs
            .iter()
            .map(|(id, text)| {
                let toks = tokens(text);
                let overlap = toks.iter().filter(|t| q.contains(t)).count();
                (overlap as f64, id.clone())
            })
            .filter(|(s, _)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(req.samples() as usize);
        Ok(scored)
    }
}

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

#[tokio::test]
async fn grounded_answer_passes() {
    let store: Arc<dyn rig::vector_store::VectorStoreIndexDyn> = Arc::new(MockStore::new());
    let graders: Vec<Box<dyn AsyncGrader>> = vec![Box::new(
        RetrievalGroundednessGrader::new("grounded", store)
            .with_k(3)
            .with_pass_threshold(0.5),
    )];

    let mut suite = SkillTaskSet::new("grounded.v1");
    // EchoRunner repeats the prompt, and the prompt tokens are present in
    // doc-1, so token-recall should be 1.0.
    suite.push(SkillTask::new("t1", "the capital of france is paris").with_grader("grounded"));

    let report = SkillHarness::new(EchoRunner, graders)
        .with_trials(1)
        .run(&suite)
        .await
        .unwrap();
    assert!((report.mean_pass_rate("grounded").unwrap() - 1.0).abs() < 1e-9);
}

#[tokio::test]
async fn ungrounded_answer_fails() {
    let store: Arc<dyn rig::vector_store::VectorStoreIndexDyn> = Arc::new(MockStore::new());
    let graders: Vec<Box<dyn AsyncGrader>> = vec![Box::new(
        RetrievalGroundednessGrader::new("grounded", store)
            .with_k(3)
            .with_pass_threshold(0.5),
    )];

    let mut suite = SkillTaskSet::new("ungrounded.v1");
    // No overlap with any doc — store returns empty, token-recall = 0.
    suite.push(
        SkillTask::new("t1", "quantum chromodynamics gluon confinement").with_grader("grounded"),
    );

    let report = SkillHarness::new(EchoRunner, graders)
        .with_trials(1)
        .run(&suite)
        .await
        .unwrap();
    assert!((report.mean_pass_rate("grounded").unwrap() - 0.0).abs() < 1e-9);
}

struct EmptyOutputRunner;

impl AgentRunner for EmptyOutputRunner {
    fn run(
        &self,
        task: &SkillTask,
        _trial: usize,
    ) -> impl Future<Output = Result<Transcript>> + Send {
        let prompt = task.prompt.clone();
        async move {
            Ok(Transcript {
                prompt,
                final_output: String::new(),
                ..Default::default()
            })
        }
    }
}

#[tokio::test]
async fn empty_final_output_falls_back_to_prompt_for_retrieval_query() {
    let store: Arc<dyn rig::vector_store::VectorStoreIndexDyn> = Arc::new(MockStore::new());
    let graders: Vec<Box<dyn AsyncGrader>> = vec![Box::new(
        RetrievalGroundednessGrader::new("grounded", store)
            .with_k(3)
            .with_scorer(|_, contexts| if contexts.is_empty() { 0.0 } else { 1.0 }),
    )];

    let mut suite = SkillTaskSet::new("fallback.v1");
    suite.push(SkillTask::new("t1", "the capital of france is paris").with_grader("grounded"));

    let report = SkillHarness::new(EmptyOutputRunner, graders)
        .run(&suite)
        .await
        .unwrap();
    assert!((report.mean_pass_rate("grounded").unwrap() - 1.0).abs() < 1e-9);
}

#[tokio::test]
async fn custom_groundedness_hooks_are_used_and_scores_are_clamped() {
    let store: Arc<dyn rig::vector_store::VectorStoreIndexDyn> = Arc::new(MockStore::new());
    let graders: Vec<Box<dyn AsyncGrader>> = vec![Box::new(
        RetrievalGroundednessGrader::new("grounded", store)
            .with_k(1)
            .with_pass_threshold(1.0)
            .with_query_fn(|_, _| "capital france".to_string())
            .with_document_extractor(|doc| {
                doc.get("id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string()
            })
            .with_scorer(|_, contexts| {
                if contexts.len() == 1 && contexts.first().is_some_and(|context| context == "doc-1")
                {
                    2.0
                } else {
                    0.0
                }
            }),
    )];

    let mut suite = SkillTaskSet::new("custom_grounded.v1");
    suite.push(SkillTask::new("t1", "ignored by custom query").with_grader("grounded"));

    let report = SkillHarness::new(EchoRunner, graders)
        .run(&suite)
        .await
        .unwrap();
    let outcome = report.trials_log[0].outcomes.get("grounded").unwrap();
    assert!(outcome.passed);
    assert_eq!(outcome.score, 1.0);
    assert!(outcome.notes.contains("grounded_score=1.0000"));
}

#[tokio::test]
async fn empty_groundedness_query_is_skipped() {
    let store: Arc<dyn rig::vector_store::VectorStoreIndexDyn> = Arc::new(MockStore::new());
    let graders: Vec<Box<dyn AsyncGrader>> = vec![Box::new(
        RetrievalGroundednessGrader::new("grounded", store).with_query_fn(|_, _| String::new()),
    )];

    let mut suite = SkillTaskSet::new("skip_grounded.v1");
    suite.push(SkillTask::new("t1", "anything").with_grader("grounded"));

    let report = SkillHarness::new(EchoRunner, graders)
        .run(&suite)
        .await
        .unwrap();
    let outcome = report.trials_log[0].outcomes.get("grounded").unwrap();
    assert!(outcome.passed);
    assert_eq!(outcome.score, 1.0);
    assert_eq!(outcome.notes, "empty retrieval query");
}
