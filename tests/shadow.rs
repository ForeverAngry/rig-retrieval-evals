#![cfg(feature = "shadow")]
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::HashMap;

use rig::vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex, request::Filter};
use rig::wasm_compat::WasmCompatSend;
use rig_retrieval_evals::{EvalShadowStore, Mrr, Qrels, RecallAtK, RetrievalMetric};
use serde::Deserialize;

struct MockStore {
    docs: HashMap<String, Vec<String>>,
}

impl MockStore {
    fn from_docs(docs: &[(&str, &str)]) -> Self {
        let mut indexed = HashMap::new();
        for (id, text) in docs {
            indexed.insert((*id).to_string(), tokens(text));
        }
        Self { docs: indexed }
    }

    fn rank(&self, query: &str, k: usize) -> Vec<(f64, String)> {
        let q_tokens = tokens(query);
        let mut scored = self
            .docs
            .iter()
            .map(|(id, doc_tokens)| {
                let overlap = doc_tokens
                    .iter()
                    .filter(|token| q_tokens.contains(token))
                    .count();
                (overlap as f64, id.clone())
            })
            .filter(|(score, _)| *score > 0.0)
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        scored.truncate(k);
        scored
    }
}

fn tokens(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| {
            token
                .trim_matches(|ch: char| !ch.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

impl VectorStoreIndex for MockStore {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'a> Deserialize<'a> + WasmCompatSend,
    {
        let mut out = Vec::new();
        for (score, id) in self.rank(req.query(), req.samples() as usize) {
            let value = serde_json::json!({ "id": id });
            let doc = serde_json::from_value(value).map_err(VectorStoreError::JsonError)?;
            out.push((score, id, doc));
        }
        Ok(out)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(self.rank(req.query(), req.samples() as usize))
    }
}

#[tokio::test]
async fn shadow_report_surfaces_candidate_gain() {
    let qrels = Qrels::from_jsonl_str(
        r#"
        {"query_id":"q1","query":"rust memory retrieval","relevant_docs":{"doc-rust":1}}
        "#,
    )
    .unwrap();
    let baseline = MockStore::from_docs(&[("doc-noise", "unrelated cooking notes")]);
    let candidate = MockStore::from_docs(&[
        ("doc-noise", "unrelated cooking notes"),
        ("doc-rust", "rust memory retrieval harness"),
    ]);
    let metrics: Vec<Box<dyn RetrievalMetric>> = vec![Box::new(RecallAtK::new(1)), Box::new(Mrr)];

    let report = EvalShadowStore::new(&baseline, &candidate, 1)
        .with_concurrency(2)
        .run(&qrels, &metrics)
        .await
        .unwrap();

    let recall = report
        .diff
        .rows
        .iter()
        .find(|row| row.metric == "recall@1")
        .unwrap();
    assert_eq!(recall.baseline_mean, Some(0.0));
    assert_eq!(recall.current_mean, 1.0);
    assert_eq!(recall.delta, Some(1.0));
    assert_eq!(recall.winners, 1);
    assert_eq!(recall.losers, 0);

    let markdown = report.to_markdown();
    assert!(markdown.contains("## Baseline"));
    assert!(markdown.contains("## Delta"));
}
