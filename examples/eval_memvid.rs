//! Evaluate a tiny committed fixture corpus through `rig-memvid`.
//!
//! Run with: `cargo run --example eval_memvid --features memvid-example`

use std::collections::HashMap;

use anyhow::{Context, Result};
use rig_memvid::{
    CardSelection, MemoryCardContext, MemvidStore,
    memvid_core::{MemoryCard, MemoryCardBuilder, Polarity, PutOptions},
};
use rig_retrieval_evals::{
    CandidateDocumentGainInput, EvalShadowStore, HitRateAtK, KnowledgeGainConfig,
    KnowledgeGainReport, Mrr, NdcgAtK, Qrels, RecallAtK, RetrievalMetric,
};
use serde::Deserialize;

const CORPUS_JSONL: &str = include_str!("data/memvid_corpus.jsonl");
const QRELS_JSONL: &str = include_str!("data/memvid_qrels.jsonl");
const CARDS_JSONL: &str = include_str!("data/memvid_cards.jsonl");
const CARD_QRELS_JSONL: &str = include_str!("data/memvid_card_qrels.jsonl");

#[derive(Debug, Deserialize)]
struct CorpusDoc {
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct CardFixture {
    id: String,
    kind: CardKind,
    entity: String,
    slot: String,
    value: String,
    #[serde(default)]
    polarity: Option<CardPolarity>,
    source_doc: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CardKind {
    Fact,
    Preference,
    Profile,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CardPolarity {
    Positive,
    Negative,
    Neutral,
}

struct SeededCorpus {
    search_ids: HashMap<String, String>,
    sequence_ids: HashMap<String, u64>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let temp = tempfile::tempdir()?;
    let baseline_archive = temp.path().join("eval_memvid_baseline.mv2");
    let candidate_archive = temp.path().join("eval_memvid_candidate.mv2");
    let baseline_store = MemvidStore::builder()
        .path(&baseline_archive)
        .enable_lex()
        .open_or_create()?;
    let candidate_store = MemvidStore::builder()
        .path(&candidate_archive)
        .enable_lex()
        .open_or_create()?;

    let seeded = seed_corpus(&candidate_store)?;
    let card_ids = seed_cards(&candidate_store, &seeded.sequence_ids)?;
    let raw_qrels = remap_qrels(Qrels::from_jsonl_str(QRELS_JSONL)?, &seeded.search_ids)?;
    let card_qrels = remap_qrels(Qrels::from_jsonl_str(CARD_QRELS_JSONL)?, &card_ids)?;

    let metrics: Vec<Box<dyn RetrievalMetric>> = vec![
        Box::new(RecallAtK::new(3)),
        Box::new(HitRateAtK::new(3)),
        Box::new(Mrr),
        Box::new(NdcgAtK::new(3)),
    ];

    let raw_shadow = EvalShadowStore::new(&baseline_store, &candidate_store, 3)
        .with_concurrency(2)
        .run(&raw_qrels, &metrics)
        .await?
        .with_metadata("examples/memvid_tiny/raw_frames", "rig-memvid:lex");

    let baseline_cards = MemoryCardContext::new(baseline_store, CardSelection::EntityMentions);
    let candidate_cards = MemoryCardContext::new(candidate_store, CardSelection::EntityMentions);
    let card_shadow = EvalShadowStore::new(&baseline_cards, &candidate_cards, 3)
        .with_concurrency(2)
        .run(&card_qrels, &metrics)
        .await?
        .with_metadata(
            "examples/memvid_tiny/domain_cards",
            "rig-memvid:memory-cards",
        );
    let gain_config = KnowledgeGainConfig::new()
        .with_metric_weight("recall@3", 2.0)
        .with_metric_weight("ndcg@3", 1.0)
        .with_metric_weight("mrr", 1.0)
        .with_document_relevance_weight(1.0)
        .with_novelty_weight(0.25);
    let raw_gain = KnowledgeGainReport::from_diff(&raw_shadow.diff, &gain_config)
        .with_candidate_documents(
            &raw_qrels,
            &candidate_inputs(&seeded.search_ids, 0.15),
            &gain_config,
        );
    let card_gain = KnowledgeGainReport::from_diff(&card_shadow.diff, &gain_config)
        .with_candidate_documents(
            &card_qrels,
            &candidate_inputs(&card_ids, 0.35),
            &gain_config,
        );

    println!("## Raw Frames\n{}", raw_shadow.current.to_markdown());
    println!(
        "\n## Structured Memory Cards\n{}",
        card_shadow.current.to_markdown()
    );
    println!(
        "\n## Raw Frame Shadow Delta\n{}",
        raw_shadow.diff.to_markdown()
    );
    println!(
        "\n## Structured Memory Card Shadow Delta\n{}",
        card_shadow.diff.to_markdown()
    );
    println!("\n## Raw Frame Knowledge Gain\n{}", raw_gain.to_markdown());
    println!(
        "\n## Structured Memory Card Knowledge Gain\n{}",
        card_gain.to_markdown()
    );
    Ok(())
}

fn seed_corpus(store: &MemvidStore) -> Result<SeededCorpus> {
    let mut search_ids = HashMap::new();
    let mut sequence_ids = HashMap::new();
    for (search_id, line) in CORPUS_JSONL
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
    {
        let doc: CorpusDoc = serde_json::from_str(line)?;
        let options = PutOptions::builder().extract_triplets(false).build();
        let sequence_id = store.put_text(&doc.text, options)?;
        if search_ids
            .insert(doc.id.clone(), search_id.to_string())
            .is_some()
        {
            anyhow::bail!("duplicate corpus id: {}", doc.id);
        }
        if sequence_ids.insert(doc.id.clone(), sequence_id).is_some() {
            anyhow::bail!("duplicate corpus id: {}", doc.id);
        }
    }
    Ok(SeededCorpus {
        search_ids,
        sequence_ids,
    })
}

fn seed_cards(
    store: &MemvidStore,
    sequence_ids: &HashMap<String, u64>,
) -> Result<HashMap<String, String>> {
    let mut card_ids = HashMap::new();
    for (card_id, line) in CARDS_JSONL
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
    {
        let fixture: CardFixture = serde_json::from_str(line)?;
        let source_frame_id = sequence_ids.get(&fixture.source_doc).with_context(|| {
            format!("card references missing source_doc: {}", fixture.source_doc)
        })?;
        let build_id = u64::try_from(card_id)?.saturating_add(1);
        let card = build_card(&fixture, *source_frame_id, build_id)?;
        store.put_memory_card(card)?;
        if card_ids
            .insert(fixture.id.clone(), card_id.to_string())
            .is_some()
        {
            anyhow::bail!("duplicate card fixture id: {}", fixture.id);
        }
    }
    Ok(card_ids)
}

fn build_card(fixture: &CardFixture, source_frame_id: u64, id: u64) -> Result<MemoryCard> {
    let mut builder = match fixture.kind {
        CardKind::Fact => MemoryCardBuilder::new().fact(),
        CardKind::Preference => MemoryCardBuilder::new().preference(),
        CardKind::Profile => MemoryCardBuilder::new().profile(),
    };
    builder = builder
        .entity(&fixture.entity)
        .slot(&fixture.slot)
        .value(&fixture.value)
        .source(source_frame_id, None)
        .engine("eval_memvid_fixture", "0.1.0");
    if let Some(polarity) = fixture.polarity.as_ref() {
        builder = builder.polarity(match polarity {
            CardPolarity::Positive => Polarity::Positive,
            CardPolarity::Negative => Polarity::Negative,
            CardPolarity::Neutral => Polarity::Neutral,
        });
    }
    builder
        .build(id)
        .map_err(|err| anyhow::anyhow!("build card {}: {err}", fixture.id))
}

fn remap_qrels(mut qrels: Qrels, search_ids: &HashMap<String, String>) -> Result<Qrels> {
    for query in &mut qrels.queries {
        let mut remapped = HashMap::new();
        for (logical_id, grade) in std::mem::take(&mut query.relevant_docs) {
            let search_id = search_ids
                .get(&logical_id)
                .with_context(|| format!("qrels references missing corpus id: {logical_id}"))?;
            remapped.insert(search_id.clone(), grade);
        }
        query.relevant_docs = remapped;
    }
    Ok(qrels)
}

fn candidate_inputs(
    ids: &HashMap<String, String>,
    base_novelty: f64,
) -> Vec<CandidateDocumentGainInput> {
    ids.values()
        .map(|search_id| {
            CandidateDocumentGainInput::new(search_id.clone()).with_novelty(base_novelty)
        })
        .collect()
}
