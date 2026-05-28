# AGENTS.md

Guidance for AI coding agents working in `rig-retrieval-evals`. Mirrors
[.github/copilot-instructions.md](.github/copilot-instructions.md).

## Project

`rig-retrieval-evals` is a retrieval-quality evaluation harness for any
[`rig`](https://crates.io/crates/rig-core) `VectorStoreIndex`. Public
primitives:

- `Qrels` / `GoldQuery` — BEIR-compatible labeled datasets ([src/dataset.rs](src/dataset.rs)).
- `RetrievalMetric` trait + concrete IR metrics ([src/retrieval.rs](src/retrieval.rs)).
- `RetrievalHarness` — async driver over `VectorStoreIndexDyn` ([src/harness.rs](src/harness.rs)).
- `MetricReport` / `MultiReport` — aggregation + JSON/Markdown emission + baseline diff ([src/report.rs](src/report.rs)).

## Rules

- Rust 2024, MSRV 1.89. Library is runtime-agnostic; do not add `tokio` to
  `[dependencies]`.
- Errors: typed `thiserror` enum in [src/error.rs](src/error.rs); return
  `Result<_, Error>`. Do not introduce ad-hoc `Box<dyn Error>` or `String`
  error types.
- Never `.await` while holding a `Mutex`/`RwLock` guard. Scope-drop first.
- No `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`, `dbg!`,
  indexing/slicing, or `unreachable!` in library code — clippy
  `deny`/`forbid`. Use `?`, `ok_or(Error::...)`, `get(..)`, `match`.
  Allowed in `tests/`, `examples/`, `#[cfg(test)]` blocks (gate with
  `#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]`).
- Use `tracing` for logs; no `println!` in library code.
- Document new `pub` items with `///` rustdoc; provide a `no_run` example
  for new traits or driver types.
- Re-export new public items from [src/lib.rs](src/lib.rs).

## Feature flags

Default = `retrieval` (pure-Rust IR metrics, no LLM deps). Optional:
`embedding-novelty`, `ingestion`, `ingestion-graph`, `knowledge-gain`,
`memvid-example`, `ragas`, `shadow`, `skills`, `memory`, `models`, `agents`,
and `observe`. Future candidates include `bm25`, `cli`, and `full`. Gate
optional code with `#[cfg(feature = "...")]`.

## Validation

```sh
just check
# = cargo fmt --all -- --check
#   cargo clippy --all-targets -- -D warnings
#   cargo test --all-features
```

Integration tests live in [tests/](tests/). Examples must keep building:
`cargo build --examples`.

## Scope

Do not vendor `rig-core`. The crate must not depend on `rig-memvid`,
`rig-compose`, `rig-resources`, or `rig-mcp` — it has to evaluate **all**
of them. Update [README.md](README.md) and [CHANGELOG.md](CHANGELOG.md)
for user-visible changes.
