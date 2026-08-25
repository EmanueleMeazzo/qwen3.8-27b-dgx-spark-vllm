#!/usr/bin/env bash
# FINAL validation of the recommended production config.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$ROOT/scripts"

bash "$R/run_experiment.sh" FINAL-recommended "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=3 MEMUTIL=0.90 MAX_NUM_SEQS=16 LADDER=1,2,4,8,16" || true

# code-heavy alternative (single-stream only, documented caveat)
bash "$R/run_experiment.sh" ALT-codeheavy-mtp6 "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=6 MEMUTIL=0.90 MAX_NUM_SEQS=16" || true
echo "FINAL VALIDATION COMPLETE"
