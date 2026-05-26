#![cfg(feature = "ingestion")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use rig::{client::CompletionClient, extractor::ExtractorBuilder, providers::ollama::Client};
use rig_retrieval_evals::{
    Document, LlmPropositionExtractor, LlmTripleExtractor, PropositionExtractor, TripleExtractor,
};

#[tokio::test]
#[ignore = "requires a running Ollama server and qwen3.5:9b or OLLAMA_MODEL"]
async fn live_ollama_extractors_work() {
    let model_name = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.5:9b".to_string());

    let client = Client::new("http://localhost:11434").unwrap();
    let model = client.completion_model(&model_name);

    let doc = Document {
        id: "doc1".to_string(),
        text: "Alice knows Bob. Bob loves Eve.".to_string(),
        sections: vec![],
    };

    let triple_extractor = ExtractorBuilder::new(model.clone())
        .preamble("Extract knowledge graph triples. Subject and object should be names. Predicate should be an action.")
        .build();
    let triple_extractor = LlmTripleExtractor::new(triple_extractor);

    let triples = triple_extractor.extract(&doc).await.unwrap();
    assert!(!triples.is_empty(), "should extract at least one triple");
    println!("Extracted Triples: {triples:#?}");

    let prop_extractor = ExtractorBuilder::new(model)
        .preamble("Extract factual propositions. Each proposition should be a standalone sentence.")
        .build();
    let prop_extractor = LlmPropositionExtractor::new(prop_extractor);

    let props = prop_extractor.extract(&doc).await.unwrap();
    assert!(!props.is_empty(), "should extract at least one proposition");
    println!("Extracted Propositions: {props:#?}");
}
