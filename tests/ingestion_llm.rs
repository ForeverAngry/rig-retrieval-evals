#![cfg(feature = "ingestion")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::sync::{Arc, Mutex};

use rig::{
    OneOrMany,
    completion::{
        AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
        Usage,
    },
    extractor::ExtractorBuilder,
    message::{Text, ToolCall, ToolFunction},
    streaming::StreamingCompletionResponse,
};
use rig_retrieval_evals::{
    Document, LlmPropositionExtractor, LlmTripleExtractor, Proposition, PropositionExtractor,
    Triple, TripleExtractor,
};
use serde_json::json;

#[derive(Clone)]
struct FakeCompletionModel {
    response: FakeResponse,
    requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

#[derive(Clone)]
enum FakeResponse {
    Submit(serde_json::Value),
    TextOnly,
}

impl FakeCompletionModel {
    fn submit(arguments: serde_json::Value) -> Self {
        Self {
            response: FakeResponse::Submit(arguments),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn text_only() -> Self {
        Self {
            response: FakeResponse::TextOnly,
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn requests(&self) -> Vec<CompletionRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl CompletionModel for FakeCompletionModel {
    type Response = serde_json::Value;
    type StreamingResponse = ();
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        Self::text_only()
    }

    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = std::result::Result<CompletionResponse<Self::Response>, CompletionError>,
    > + Send {
        let response = self.response.clone();
        let requests = Arc::clone(&self.requests);
        async move {
            requests.lock().unwrap().push(request);
            let choice = match response {
                FakeResponse::Submit(arguments) => {
                    OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                        "call-1".to_string(),
                        ToolFunction::new("submit".to_string(), arguments),
                    )))
                }
                FakeResponse::TextOnly => OneOrMany::one(AssistantContent::Text(Text {
                    text: "no tool call".to_string(),
                })),
            };

            Ok(CompletionResponse {
                choice,
                usage: Usage::new(),
                raw_response: serde_json::Value::Null,
                message_id: None,
            })
        }
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> std::result::Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>
    {
        Err(CompletionError::ProviderError(
            "streaming not implemented".to_string(),
        ))
    }
}

#[tokio::test]
async fn llm_triple_adapter_normalises_structured_tool_output() {
    let model = FakeCompletionModel::submit(json!({
        "triples": [
            {
                "subject": "APT-28",
                "predicate": "  Exploits   CVE ",
                "object": "CVE-2024-12345"
            }
        ]
    }));
    let extractor = LlmTripleExtractor::new(ExtractorBuilder::new(model.clone()).build());

    let triples = extractor
        .extract(&Document::new("cti-1", "APT-28 exploited CVE-2024-12345."))
        .await
        .unwrap();

    assert_eq!(
        triples,
        vec![Triple::new("APT-28", "exploits_cve", "CVE-2024-12345")]
    );

    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].name, "submit");
    assert!(
        requests[0].tools[0]
            .parameters
            .to_string()
            .contains("triples")
    );
    assert!(requests[0].tool_choice.is_some());
}

#[tokio::test]
async fn llm_proposition_adapter_maps_structured_tool_output() {
    let model = FakeCompletionModel::submit(json!({
        "propositions": [
            "APT-28 exploited CVE-2024-12345.",
            "APT-28 used spear-phishing."
        ]
    }));
    let extractor = LlmPropositionExtractor::new(ExtractorBuilder::new(model.clone()).build());

    let propositions = extractor
        .extract(&Document::new(
            "cti-2",
            "APT-28 exploited CVE-2024-12345 and used spear-phishing.",
        ))
        .await
        .unwrap();

    assert_eq!(
        propositions,
        vec![
            Proposition::new("APT-28 exploited CVE-2024-12345."),
            Proposition::new("APT-28 used spear-phishing."),
        ]
    );

    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].name, "submit");
    assert!(
        requests[0].tools[0]
            .parameters
            .to_string()
            .contains("propositions")
    );
    assert!(requests[0].tool_choice.is_some());
}

#[tokio::test]
async fn llm_adapters_surface_missing_submit_as_ingestion_error() {
    let model = FakeCompletionModel::text_only();
    let extractor = LlmPropositionExtractor::new(ExtractorBuilder::new(model).build());

    let err = extractor
        .extract(&Document::new("cti-3", "The model refuses to call submit."))
        .await
        .unwrap_err();

    assert!(err.to_string().contains("ingestion error"));
    assert!(err.to_string().contains("No data extracted"));
}
