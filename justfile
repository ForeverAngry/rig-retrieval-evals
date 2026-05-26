# rig-retrieval-evals task runner.
#
# Install just: https://github.com/casey/just
#   brew install just
#
# Run `just` with no args to see the recipe list.

# Show all recipes.
default:
    @just --list

# Build all targets with default features.
build:
    cargo build --all-targets

# Format check + clippy + tests + msrv + doc + examples.
check: fmt clippy test msrv doc examples

# Verify code is formatted (does not mutate).
fmt:
    cargo fmt --all -- --check

# Clippy across CI and local feature gates.
clippy:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --no-default-features --all-targets -- -D warnings
    cargo clippy --no-default-features --features ragas --all-targets -- -D warnings
    cargo clippy --all-features --all-targets -- -D warnings

# Tests across CI feature combos.
test:
    cargo test --all-features --all-targets
    cargo test --no-default-features --features ragas

# MSRV gate (Rust 1.89).
msrv:
    cargo +1.89 build --all-targets

# Rustdoc with strict warnings.
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# Build every example with all features.
examples:
    cargo build --examples --all-features

# Validate the package as it would be uploaded to crates.io.
publish-dry-run:
    cargo publish --dry-run

# Preview what release-plz would bump/changelog without changing anything.
release-preview:
    release-plz update --dry-run

# Open a release PR locally (writes to a branch). Same thing CI does on push.
release-pr:
    release-plz release-pr

# Inspect the next semver bump release-plz would compute from current commits.
next-version:
    @release-plz update --dry-run 2>&1 | grep -E "(bumping|no changes|next version)" || true

# Run all checks needed for a PR / commit to main locally.
pr-ready: check publish-dry-run

# Install a git pre-push hook that runs `just pr-ready`.
install-hooks:
    #!/usr/bin/env bash
    echo '#!/usr/bin/env bash' > .git/hooks/pre-push
    echo 'set -e' >> .git/hooks/pre-push
    echo 'echo "Running just pr-ready..."' >> .git/hooks/pre-push
    echo 'just pr-ready' >> .git/hooks/pre-push
    chmod +x .git/hooks/pre-push
    echo "pre-push hook installed."
