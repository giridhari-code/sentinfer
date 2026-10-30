#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/vendor"
if [[ -f deepseek-vendor-cuda.tar.gz ]]; then
  tar xzf deepseek-vendor-cuda.tar.gz
  echo "extracted FlashMLA DeepGEMM DeepEP from tarball"
  exit 0
fi
mkdir -p FlashMLA DeepGEMM DeepEP
git clone --depth 1 https://github.com/deepseek-ai/FlashMLA.git FlashMLA
git clone --depth 1 https://github.com/deepseek-ai/DeepGEMM.git DeepGEMM
git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git DeepEP
echo "cloned upstream"
