# rig-retrieval-evals — Copilot Instructions

See [AGENTS.md](../AGENTS.md) for the authoritative copy. Summary:

- Rust 2024, MSRV `1.89`. Library is runtime-agnostic — do not add `tokio`
  to `[dependencies]`.
- Typed `thiserror::Error` in [src/error.rs](../src/error.rs). Return
  `Result<_, Error>`; no ad-hoc `Box<dyn Error>`.
- Never `.await` while holding a `Mutex`/`RwLock` guard
  (`clippy::await_holding_lock` is `deny`).
- No `unwrap`/`expect`/`panic!`/`todo!`/`unimplemented!`/`dbg!`/indexing
  in library code (clippy `deny`/`forbid`). Use `?`, `ok_or(Error::…)`,
  `get(..)`, pattern matching.
- `unwrap`/`expect` allowed in `tests/`, `examples/`, and `#[cfg(test)]`
  blocks (gate the test module with
  `#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]`).
- Use `tracing` for logs; no `println!` in library code.
- Document new `pub` items with `///` rustdoc. Re-export from
  [src/lib.rs](../src/lib.rs).

## Validation

```sh
just check
# fmt --check + clippy --all-targets -- -D warnings + cargo test --all-features
```

Run before declaring any change done.

## Scope

The crate must not depend on `rig-memvid`, `rig-compose`, `rig-resources`,
or `rig-mcp`. It exists to evaluate retrievers built on top of them via the
`rig::vector_store::VectorStoreIndexDyn` interface.

## Planning reference

The authoritative spec is
[`rig-ecosystem/docs/evals-rag-plan.md`](https://github.com/ForeverAngry/rig-ecosystem/blob/main/docs/evals-rag-plan.md).
Out-of-scope items and reopen triggers are catalogued there.
