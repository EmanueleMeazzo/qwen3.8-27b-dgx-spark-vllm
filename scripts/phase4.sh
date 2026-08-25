#!/usr/bin/env bash
# Phase 4: stability matrix under concurrency (IMA occurs with MTP at batch>=~4).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$ROOT/scripts"

# st1: is SPEC=3 stable where 4 crashed?
bash "$R/run_experiment.sh" st1-spec3-ladder "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=3 MEMUTIL=0.90 LADDER=1,2,4,8,16" || true

# st2: SPEC=4 single-stream, throttled spec tokens at higher batch
SPECJSON='{"method":"mtp","num_speculative_tokens":4,"num_speculative_tokens_per_batch_size":[[1,2,4],[3,999,2]]}'
bash "$R/run_experiment.sh" st2-spec4-schedule "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=0 MEMUTIL=0.90 EXTRA_ARGS=--speculative-config=$SPECJSON" || true

# st3: no-spec reference ladder (expected stable)
bash "$R/run_experiment.sh" st3-spec0-ladder "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=0 MEMUTIL=0.90 LADDER=1,2,4,8,16" || true
echo "PHASE4 COMPLETE"
