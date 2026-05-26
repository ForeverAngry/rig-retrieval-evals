use std::env;

use rig::client::CompletionClient;
use rig::extractor::ExtractorBuilder;
use rig::providers::ollama::Client;
use rig_retrieval_evals::{
    DistillationPipeline, Document, InMemoryGraphBaseline, InMemoryIocBaseline, Proposition,
    RedundancyCheck, RedundancyVerdict, RegexIocExtractor, Result,
    ingestion::llm::{LlmPropositionExtractor, LlmTripleExtractor},
};

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let model_name = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.5:9b".to_string());

    let client = Client::new("http://localhost:11434")?;
    let model = client.completion_model(&model_name);

    let doc = Document::new(
        "cti-doc",
        "APT-28 exploited CVE-2024-1234. APT-28 uses spear-phishing. It originates from Russia.",
    );

    let doc_text = doc.text.clone();
    println!("Analyzing Document:\n{doc_text}\n---\n");

    let triple_extractor = ExtractorBuilder::new(model.clone())
        .preamble("Extract threat intelligence triples.")
        .build();
    let prop_extractor = ExtractorBuilder::new(model)
        .preamble("Extract security propositions.")
        .build();

    let pipeline = DistillationPipeline::new(RegexIocExtractor::new()?, InMemoryIocBaseline::new())
        .with_graph(
            LlmTripleExtractor::new(triple_extractor),
            InMemoryGraphBaseline::new(),
        )
        .with_propositions(LlmPropositionExtractor::new(prop_extractor), AlwaysPasses);

    let delta = match pipeline.ingest(&doc).await {
        Ok(delta) => delta,
        Err(e) => {
            eprintln!("Pipeline execution failed: {e}");
            return Ok(());
        }
    };

    println!("Detected IoCs:");
    for ioc in &delta.iocs {
        println!(" - {:?} : {}", ioc.kind, ioc.value);
    }

    println!("\nDetected Triples:");
    for t in &delta.triples {
        println!(" - {} --[{}]--> {}", t.subject, t.predicate, t.object);
    }

    println!("\nDetected Propositions:");
    for p in &delta.propositions {
        println!(" - {}", p.text);
    }

    Ok(())
}

struct AlwaysPasses;
impl RedundancyCheck for AlwaysPasses {
    fn check(
        &self,
        _prop: &Proposition,
    ) -> impl std::future::Future<Output = Result<RedundancyVerdict>> + Send {
        std::future::ready(Ok(RedundancyVerdict {
            is_redundant: false,
            similarity: 0.0,
        }))
    }
}
