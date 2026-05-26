//! Minimal Track 1 (IoC) ingestion demo.
//!
//! Run with: `cargo run --example ingestion_ioc --features ingestion`

use rig_retrieval_evals::{DistillationPipeline, Document, InMemoryIocBaseline, RegexIocExtractor};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::try_init().ok();

    let extractor = RegexIocExtractor::new()?;
    let baseline = InMemoryIocBaseline::new();
    let pipeline = DistillationPipeline::new(extractor, baseline);

    let doc = Document::new(
        "whitepaper-001",
        "Threat actor APT-99 exploited CVE-2024-12345 over 192.0.2.10. \
         Beacon at https://evil.example.com. \
         Dropper md5: 098f6bcd4621d373cade4e832627b4f6.",
    );

    let delta = pipeline.ingest(&doc).await?;
    println!(
        "ingested {}: {} new IoCs, {} dropped",
        doc.id,
        delta.iocs.len(),
        delta.dropped.len()
    );
    for ioc in &delta.iocs {
        println!("  + {:?} {}", ioc.kind, ioc.value);
    }
    Ok(())
}
