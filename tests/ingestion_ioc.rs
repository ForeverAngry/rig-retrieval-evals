//! Integration tests for the Track 1 ingestion pipeline (IoC filter).

#![cfg(feature = "ingestion")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use rig_retrieval_evals::{
    DistillationPipeline, Document, DroppedItem, DroppedReason, InMemoryIocBaseline, Ioc,
    IocExtractor, IocKind, RegexIocExtractor, Section, SectionKind,
};

fn extractor() -> RegexIocExtractor {
    RegexIocExtractor::new().expect("default extractor patterns must compile")
}

#[test]
fn regex_extractor_recognises_the_default_ioc_set() {
    let extractor = extractor();
    let text = "\
APT-28 exploited CVE-2024-12345 from 192.0.2.10 (and 2001:db8:1234:5678:9abc:def0:0001:0002).
Dropper sha256 a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e
ran md5 098f6bcd4621d373cade4e832627b4f6 and sha1 a94a8fef8c8d1f1cf2b5b6f7a4a4f8\
7e6f5cdfa3.
Beacon: https://evil.example.com/loader and the bare domain malware.example.org
Persistence at HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run";
    let doc = Document::new("doc-1", text);

    let iocs = extractor.extract(&doc);
    assert!(
        iocs.iter()
            .any(|i| i.kind == IocKind::Cve && i.value == "CVE-2024-12345")
    );
    assert!(
        iocs.iter()
            .any(|i| i.kind == IocKind::Ipv4 && i.value == "192.0.2.10")
    );
    assert!(iocs.iter().any(|i| i.kind == IocKind::Ipv6));
    assert!(iocs.iter().any(|i| i.kind == IocKind::Sha256));
    assert!(iocs.iter().any(|i| i.kind == IocKind::Sha1));
    assert!(iocs.iter().any(|i| i.kind == IocKind::Md5));
    assert!(
        iocs.iter()
            .any(|i| i.kind == IocKind::Url && i.value.contains("evil.example.com"))
    );
    assert!(
        iocs.iter()
            .any(|i| i.kind == IocKind::Domain && i.value == "malware.example.org")
    );
    assert!(
        iocs.iter()
            .any(|i| i.kind == IocKind::RegistryKey && i.value.starts_with("HKLM\\"))
    );
}

#[test]
fn regex_extractor_skips_domains_nested_inside_urls() {
    let extractor = extractor();
    let doc = Document::new("doc-2", "Beacon: https://evil.example.com/payload");
    let iocs = extractor.extract(&doc);
    let urls: Vec<_> = iocs.iter().filter(|i| i.kind == IocKind::Url).collect();
    let domains: Vec<_> = iocs.iter().filter(|i| i.kind == IocKind::Domain).collect();
    assert_eq!(urls.len(), 1);
    assert!(
        domains.is_empty(),
        "domain inside a URL should not be reported twice: {domains:?}"
    );
}

#[test]
fn regex_extractor_extracts_from_sections() {
    let extractor = extractor();
    let doc = Document::new("doc-3", "no iocs in body").with_sections(vec![
        Section {
            kind: SectionKind::Narrative,
            text: "See CVE-2023-99999.".into(),
        },
        Section {
            kind: SectionKind::Table,
            text: "row | 203.0.113.5 | beacon".into(),
        },
    ]);
    let iocs = extractor.extract(&doc);
    assert!(iocs.iter().any(|i| i.kind == IocKind::Cve));
    assert!(
        iocs.iter()
            .any(|i| i.kind == IocKind::Ipv4 && i.value == "203.0.113.5")
    );
}

#[tokio::test]
async fn pipeline_returns_net_new_iocs_against_empty_baseline() {
    let pipeline = DistillationPipeline::new(extractor(), InMemoryIocBaseline::new());
    let doc = Document::new("doc-4", "Patch CVE-2024-00001 on 198.51.100.7.");
    let delta = pipeline.ingest(&doc).await.unwrap();
    assert_eq!(delta.iocs.len(), 2);
    assert!(
        delta.dropped.is_empty(),
        "nothing should drop: {:?}",
        delta.dropped
    );
    assert!(!delta.is_empty());
}

#[tokio::test]
async fn pipeline_drops_known_iocs_with_duplicate_reason() {
    let baseline = InMemoryIocBaseline::new()
        .with_iocs([Ioc::new(IocKind::Cve, "CVE-2024-00001")])
        .unwrap();
    let pipeline = DistillationPipeline::new(extractor(), baseline);

    let doc = Document::new("doc-5", "Re-patch CVE-2024-00001 and add CVE-2024-00002.");
    let delta = pipeline.ingest(&doc).await.unwrap();

    let new_cves: Vec<_> = delta
        .iocs
        .iter()
        .filter(|i| i.kind == IocKind::Cve)
        .collect();
    assert_eq!(new_cves.len(), 1);
    assert_eq!(new_cves[0].value, "CVE-2024-00002");

    let dropped_cves: Vec<_> = delta
        .dropped
        .iter()
        .filter(|d| matches!(d.reason, DroppedReason::DuplicateIoc))
        .collect();
    assert_eq!(dropped_cves.len(), 1);
    match &dropped_cves[0].item {
        DroppedItem::Ioc(ioc) => assert_eq!(ioc.value, "CVE-2024-00001"),
        _ => panic!("unexpected dropped item kind"),
    }
}

#[tokio::test]
async fn pipeline_deduplicates_within_a_single_document() {
    let pipeline = DistillationPipeline::new(extractor(), InMemoryIocBaseline::new());
    let doc = Document::new(
        "doc-6",
        "CVE-2024-99999 mentioned twice: CVE-2024-99999. And again CVE-2024-99999.",
    );
    let delta = pipeline.ingest(&doc).await.unwrap();
    let cves: Vec<_> = delta
        .iocs
        .iter()
        .filter(|i| i.kind == IocKind::Cve)
        .collect();
    assert_eq!(cves.len(), 1, "intra-document duplicates must collapse");
}

#[tokio::test]
async fn pipeline_returns_empty_delta_for_text_without_iocs() {
    let pipeline = DistillationPipeline::new(extractor(), InMemoryIocBaseline::new());
    let doc = Document::new(
        "doc-7",
        "This narrative discusses ransomware in general without naming any indicators.",
    );
    let delta = pipeline.ingest(&doc).await.unwrap();
    assert!(delta.iocs.is_empty());
    assert!(delta.dropped.is_empty());
    assert!(delta.is_empty());
}

#[tokio::test]
async fn baseline_can_be_seeded_from_a_previous_delta() {
    // Run the pipeline once, then feed the resulting IoCs back into the
    // baseline. A second run on the same document must produce an empty
    // net-new set — the self-cleaning invariant the architecture promises.
    let baseline = InMemoryIocBaseline::new();
    let pipeline = DistillationPipeline::new(extractor(), baseline);

    let doc = Document::new("doc-8", "Indicators: CVE-2024-00010, 198.51.100.42.");
    let first = pipeline.ingest(&doc).await.unwrap();
    assert!(!first.iocs.is_empty());

    pipeline.baseline().extend(first.iocs.clone()).unwrap();

    let second = pipeline.ingest(&doc).await.unwrap();
    assert!(second.iocs.is_empty(), "second pass must yield no new IoCs");
    assert_eq!(second.dropped.len(), first.iocs.len());
    assert!(
        second
            .dropped
            .iter()
            .all(|d| matches!(d.reason, DroppedReason::DuplicateIoc))
    );
}
