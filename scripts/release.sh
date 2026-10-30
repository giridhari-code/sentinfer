#!/usr/bin/env bash
# Build release artifacts for kLLM / Sentinel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT_DIR:-$ROOT/dist}"
TARGET_TRIPLE="${TARGET_TRIPLE:-}"
VERSION="$(grep -m1 '^version' sentinel-serve/Cargo.toml | sed 's/.*"\(.*\)"/\1/')"

echo "== kLLM / Sentinel release $VERSION =="
mkdir -p "$OUT"

BUILD_ARGS=(--release -p sentinel-serve -p sentinel-train -p sentinel-model)
if [[ -n "$TARGET_TRIPLE" ]]; then
  BUILD_ARGS+=(--target "$TARGET_TRIPLE")
  BIN_ROOT="target/$TARGET_TRIPLE/release"
else
  BIN_ROOT="target/release"
fi

cargo build "${BUILD_ARGS[@]}"
cargo build --release -p sentinel-model --example bench

STAGE="$OUT/kllm-$VERSION"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/docs"

cp -f "$BIN_ROOT/kllm" "$STAGE/bin/" 2>/dev/null || cp -f "$BIN_ROOT/kllm.exe" "$STAGE/bin/"
cp -f "$BIN_ROOT/train" "$STAGE/bin/" 2>/dev/null || true
cp -f "$BIN_ROOT/generate" "$STAGE/bin/" 2>/dev/null || true
cp -f "$BIN_ROOT/tokenize" "$STAGE/bin/" 2>/dev/null || true
cp -f "$BIN_ROOT/examples/bench" "$STAGE/bin/bench" 2>/dev/null || true

cp -f README.md KLLM.md INFERENCE.md LICENSE NOTICE "$STAGE/docs/" 2>/dev/null || true
cp -f KLLM.md "$STAGE/README.md"

(
  cd "$OUT"
  tar -czf "kllm-$VERSION-linux-x86_64.tar.gz" "kllm-$VERSION"
)
echo "artifacts in $OUT"
ls -la "$OUT"
