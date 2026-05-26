//! Stress-eval `MemvidStore` against the MITRE ATT&CK Enterprise corpus.
//!
//! Downloads ATT&CK Enterprise STIX 2.1 (≈51 MB, ≈25k STIX objects, ≈2.7k
//! indexable nodes) and runs two derived qrels through `RetrievalHarness`:
//!
//! 1. **Group → Techniques used** — for each `intrusion-set` (APT group) with
//!    ≥ 5 used techniques, query = group name + aliases, relevant docs = the
//!    set of `attack-pattern` IDs reached via `uses` STIX relationships. This
//!    measures "given an adversary, surface the techniques they employ."
//! 2. **Detection-strategy → Techniques detected** — for each
//!    `x-mitre-detection-strategy`, query = strategy name, relevant = the set
//!    of `attack-pattern` IDs reached via `detects` STIX relationships. This
//!    measures "given a detection idea (rooted in process / AD / network
//!    telemetry), surface the techniques it covers."
//!
//! ## Running
//!
//! 1. Download the bundle (or set `ATTACK_STIX_PATH` to point at your copy):
//!
//!    ```bash
//!    mkdir -p /tmp/attack-stress && curl -sSL \
//!      -o /tmp/attack-stress/enterprise-attack.json \
//!      https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json
//!    ```
//!
//! 2. Run the example (release recommended — lex insertion of ≈2.7k docs is
//!    noticeably faster):
//!
//!    ```bash
//!    cargo run --release --example eval_memvid_attack --features memvid-example
//!    ```
//!
//! ## Known issue (memvid-core 2.0.139)
//!
//! On the current `rig-memvid` 0.2.0 (which pins `memvid-core` 2.0.139), the
//! embedded WAL deterministically reports a checksum mismatch on commit at
//! corpus sizes well below this bundle (reproduces at ≥ ~30 docs of ~1 KB).
//! See [docs/decisions.md](../docs/decisions.md#stress-eval-finding-memvid-core-wal-bug).
//! When the bug is hit, this example exits with code `3` and prints the
//! upstream error verbatim instead of computing the eval. Fixing the WAL is
//! an upstream `memvid-core` concern; this example will start producing
//! numbers once a fixed version is pinned.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::panic_in_result_fn
)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rig_memvid::{MemvidStore, memvid_core::PutOptions};
use rig_retrieval_evals::{
    GoldQuery, HitRateAtK, MapAtK, Mrr, NdcgAtK, PrecisionAtK, Qrels, RecallAtK, RetrievalHarness,
    RetrievalMetric,
};
use serde_json::Value;

const DEFAULT_BUNDLE_PATH: &str = "/tmp/attack-stress/enterprise-attack.json";

const INDEXABLE: &[(&str, &str)] = &[
    ("attack-pattern", "technique"),
    ("intrusion-set", "group"),
    ("malware", "malware"),
    ("tool", "tool"),
    ("course-of-action", "mitigation"),
    ("campaign", "campaign"),
    ("x-mitre-data-component", "data-component"),
    ("x-mitre-data-source", "data-source"),
    ("x-mitre-detection-strategy", "detection-strategy"),
    ("x-mitre-tactic", "tactic"),
];

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let bundle_path: PathBuf = std::env::var_os("ATTACK_STIX_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_BUNDLE_PATH));

    if !bundle_path.exists() {
        eprintln!(
            "ATT&CK STIX bundle not found at {}\n\
             Download with:\n  curl -sSL -o {} \\\n    https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json\n\
             Or set ATTACK_STIX_PATH to your local copy.",
            bundle_path.display(),
            bundle_path.display()
        );
        std::process::exit(2);
    }

    let t0 = Instant::now();
    let raw = std::fs::read(&bundle_path)
        .with_context(|| format!("read STIX bundle from {}", bundle_path.display()))?;
    let bundle: Value = serde_json::from_slice(&raw).context("parse STIX bundle JSON")?;
    let objects = bundle["objects"]
        .as_array()
        .context("STIX bundle missing 'objects' array")?;
    let parse_secs = t0.elapsed().as_secs_f64();

    // Build indexable doc set and stable stix_id -> frame_id map.
    let kind_lookup: HashMap<&str, &'static str> = INDEXABLE.iter().copied().collect();
    let mut type_counts: BTreeMap<&'static str, usize> = BTreeMap::new();

    let temp = tempfile::tempdir()?;
    let archive = temp.path().join("attack.mv2");
    let store = MemvidStore::builder()
        .path(&archive)
        .enable_lex()
        .open_or_create()?;
    let put_opts = PutOptions::builder().extract_triplets(false).build();

    let mut stix_to_frame: HashMap<String, String> = HashMap::new();
    let t1 = Instant::now();
    for obj in objects {
        let stix_type = obj["type"].as_str().unwrap_or("");
        let Some(kind) = kind_lookup.get(stix_type).copied() else {
            continue;
        };
        if obj.get("revoked").and_then(Value::as_bool).unwrap_or(false) {
            continue;
        }
        if obj
            .get("x_mitre_deprecated")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            continue;
        }
        let Some(id) = obj["id"].as_str() else {
            continue;
        };
        let name = obj["name"].as_str().unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let desc = obj["description"].as_str().unwrap_or("");
        let ext_id = obj["external_references"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|r| r["external_id"].as_str())
            .unwrap_or("");
        let aliases = obj["aliases"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .filter(|s| !s.is_empty() && *s != name)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        let platforms = obj["x_mitre_platforms"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();

        let text = format!(
            "[{kind} {ext_id}] {name}\nAliases: {aliases}\nPlatforms: {platforms}\n\n{desc}"
        );
        let frame_id = match store.put_text(&text, put_opts.clone()) {
            Ok(id) => id,
            Err(err) => {
                eprintln!(
                    "memvid put_text failed after {} docs: {err}\n\
                     This reproduces the memvid-core 2.0.139 WAL bug — see docs/decisions.md.",
                    stix_to_frame.len()
                );
                std::process::exit(3);
            }
        };
        stix_to_frame.insert(id.to_string(), frame_id.to_string());
        *type_counts.entry(kind).or_default() += 1;
    }
    if let Err(err) = store.commit() {
        eprintln!(
            "memvid commit failed after inserting {} docs: {err}\n\
             This reproduces the memvid-core 2.0.139 WAL bug — see docs/decisions.md.",
            stix_to_frame.len()
        );
        std::process::exit(3);
    }
    let insert_secs = t1.elapsed().as_secs_f64();

    // Index STIX-id → object for query-side metadata lookups.
    let stix_by_id: HashMap<&str, &Value> = objects
        .iter()
        .filter_map(|o| Some((o["id"].as_str()?, o)))
        .collect();

    // --- Qrels A: intrusion-set --uses--> attack-pattern -----------------
    let mut group_techs: HashMap<String, HashSet<String>> = HashMap::new();
    for rel in objects {
        if rel["type"] != "relationship" || rel["relationship_type"] != "uses" {
            continue;
        }
        let src = rel["source_ref"].as_str().unwrap_or("");
        let tgt = rel["target_ref"].as_str().unwrap_or("");
        if !src.starts_with("intrusion-set--") || !tgt.starts_with("attack-pattern--") {
            continue;
        }
        if !stix_to_frame.contains_key(src) || !stix_to_frame.contains_key(tgt) {
            continue;
        }
        group_techs
            .entry(src.to_string())
            .or_default()
            .insert(tgt.to_string());
    }

    let mut group_queries = Vec::new();
    for (group_id, techs) in &group_techs {
        if techs.len() < 5 {
            continue;
        }
        let obj = stix_by_id.get(group_id.as_str()).copied();
        let name = obj
            .and_then(|o| o["name"].as_str())
            .unwrap_or("(unknown group)");
        let aliases: Vec<&str> = obj
            .and_then(|o| o["aliases"].as_array())
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .filter(|s| !s.is_empty() && *s != name)
                    .collect()
            })
            .unwrap_or_default();
        let query = if aliases.is_empty() {
            name.to_string()
        } else {
            format!("{} ({})", name, aliases.join(", "))
        };
        let ext_id = obj
            .and_then(|o| o["external_references"].as_array())
            .and_then(|a| a.first())
            .and_then(|r| r["external_id"].as_str())
            .unwrap_or(group_id.as_str());
        let relevant: HashMap<String, u8> = techs
            .iter()
            .filter_map(|tid| stix_to_frame.get(tid).map(|fid| (fid.clone(), 1u8)))
            .collect();
        group_queries.push(GoldQuery {
            query_id: format!("group:{ext_id}"),
            query,
            relevant_docs: relevant,
            reference_answer: None,
        });
    }

    // --- Qrels B: x-mitre-detection-strategy --detects--> attack-pattern -
    let mut detstrat_techs: HashMap<String, HashSet<String>> = HashMap::new();
    for rel in objects {
        if rel["type"] != "relationship" || rel["relationship_type"] != "detects" {
            continue;
        }
        let src = rel["source_ref"].as_str().unwrap_or("");
        let tgt = rel["target_ref"].as_str().unwrap_or("");
        if !src.starts_with("x-mitre-detection-strategy--") || !tgt.starts_with("attack-pattern--")
        {
            continue;
        }
        if !stix_to_frame.contains_key(src) || !stix_to_frame.contains_key(tgt) {
            continue;
        }
        detstrat_techs
            .entry(src.to_string())
            .or_default()
            .insert(tgt.to_string());
    }

    let mut detstrat_queries = Vec::new();
    for (sid, techs) in &detstrat_techs {
        let obj = stix_by_id.get(sid.as_str()).copied();
        let name = obj.and_then(|o| o["name"].as_str()).unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let relevant: HashMap<String, u8> = techs
            .iter()
            .filter_map(|tid| stix_to_frame.get(tid).map(|fid| (fid.clone(), 1u8)))
            .collect();
        if relevant.is_empty() {
            continue;
        }
        detstrat_queries.push(GoldQuery {
            query_id: format!("detstrat:{sid}"),
            query: name.to_string(),
            relevant_docs: relevant,
            reference_answer: None,
        });
    }

    println!("# rig-memvid stress eval — MITRE ATT&CK Enterprise\n");
    println!("- corpus path: `{}`", bundle_path.display());
    println!("- STIX objects in bundle: {}", objects.len());
    println!("- parse time: {parse_secs:.2}s, lex-insert time: {insert_secs:.2}s");
    println!("- indexed docs: {}", stix_to_frame.len());
    let mut sorted_counts: Vec<_> = type_counts.iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(a.1));
    let breakdown = sorted_counts
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("  - by kind: {breakdown}");
    println!(
        "- group→technique queries: {} (groups with ≥5 used techniques)",
        group_queries.len()
    );
    println!(
        "- detection-strategy→technique queries: {}\n",
        detstrat_queries.len()
    );

    let metrics: Vec<Box<dyn RetrievalMetric>> = vec![
        Box::new(RecallAtK::new(5)),
        Box::new(RecallAtK::new(20)),
        Box::new(HitRateAtK::new(10)),
        Box::new(PrecisionAtK::new(10)),
        Box::new(Mrr),
        Box::new(NdcgAtK::new(10)),
        Box::new(MapAtK::new(20)),
    ];

    let harness = RetrievalHarness::new(&store, 20).with_concurrency(8);

    let t2 = Instant::now();
    let group_qrels = Qrels {
        queries: group_queries,
    };
    let group_report = harness.run(&group_qrels, &metrics).await?;
    let group_secs = t2.elapsed().as_secs_f64();

    let t3 = Instant::now();
    let det_qrels = Qrels {
        queries: detstrat_queries,
    };
    let det_report = harness.run(&det_qrels, &metrics).await?;
    let det_secs = t3.elapsed().as_secs_f64();

    println!(
        "## A. Group → Techniques used ({} queries, {group_secs:.2}s)\n{}",
        group_qrels.len(),
        group_report.to_markdown()
    );
    println!(
        "\n## B. Detection-strategy → Techniques detected ({} queries, {det_secs:.2}s)\n{}",
        det_qrels.len(),
        det_report.to_markdown()
    );

    Ok(())
}
