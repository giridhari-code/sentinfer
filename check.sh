#!/usr/bin/env bash
# Builds and tests the whole workspace and writes build.log / test.log.
# Needs Rust 1.70+ (https://rustup.rs).
set -u
cd "$(dirname "$0")"
if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo not found. Install Rust first: https://rustup.rs" >&2
    exit 1
fi
rustc --version

# Some environments mount the working tree noexec; build under /tmp when needed.
TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target}"
if ! touch "$TARGET_DIR/.write_test" 2>/dev/null; then
    TARGET_DIR="/tmp/sentinel-target-$$"
fi
rm -f "$TARGET_DIR/.write_test" 2>/dev/null || true
export CARGO_TARGET_DIR="$TARGET_DIR"
export CARGO_INCREMENTAL=0

echo "== build (all crates, all targets)"
cargo build --workspace --all-targets 2>&1 | tee build.log | tail -n 40
echo "== tests (debug: avoids fat-LTO OOM on train)"
cargo test --workspace -- --test-threads=1 2>&1 | tee test.log | tail -n 80
echo
echo "Full output: build.log, test.log"
echo "Target dir: $CARGO_TARGET_DIR"
