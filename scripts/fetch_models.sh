#!/usr/bin/env bash
# Download candidate checkpoints into hf-cache via the NGC container's hf CLI.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for M in "$@"; do
  echo "=== fetching $M ==="
  docker run --rm --name hfdl-$$ \
    -v "$ROOT/hf-cache:/hf" -e HF_HOME=/hf -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    --entrypoint hf nvcr.io/nvidia/vllm:26.07-py3 \
    download "$M" --exclude "*.pth" --exclude "original/*" 2>&1 | tail -3
  echo "=== done $M ($(date +%T)) ==="
done
