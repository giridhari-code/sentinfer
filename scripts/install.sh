#!/usr/bin/env bash
# Install kLLM binaries from a release stage or local release build.
set -euo pipefail
PREFIX="${PREFIX:-/usr/local}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -d "$ROOT/dist" ]]; then
  STAGE="$(find "$ROOT/dist" -maxdepth 1 -type d -name 'kllm-*' | head -1)"
fi
STAGE="${STAGE:-}"

if [[ -z "$STAGE" || ! -d "$STAGE/bin" ]]; then
  echo "building release bins…"
  cargo build --release -p sentinel-serve --bin kllm
  cargo build --release -p sentinel-train --bin train --bin generate --bin tokenize
  BIN="$ROOT/target/release"
else
  BIN="$STAGE/bin"
fi

install -d "$PREFIX/bin"
install -m 755 "$BIN/kllm" "$PREFIX/bin/kllm"
[[ -f "$BIN/train" ]] && install -m 755 "$BIN/train" "$PREFIX/bin/kllm-train" || true
[[ -f "$BIN/generate" ]] && install -m 755 "$BIN/generate" "$PREFIX/bin/kllm-generate" || true
[[ -f "$BIN/tokenize" ]] && install -m 755 "$BIN/tokenize" "$PREFIX/bin/kllm-tokenize" || true

echo "installed to $PREFIX/bin"
echo "  kllm [--api-key SECRET] --checkpoint MODEL.skv --port 8080"
