#!/usr/bin/env bash
# Phase 5: concurrency stability retest (ladder fix landed after st1/st2 booted).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$ROOT/scripts"

# SPEC=3 plain under load
bash "$R/run_experiment.sh" p5a-spec3-ladder "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=3 MEMUTIL=0.90 LADDER=1,2,4,8" || true

# SPEC=4 + per-batch spec-token schedule under load
SPECJSON='{"method":"mtp","num_speculative_tokens":4,"num_speculative_tokens_per_batch_size":[[1,2,4],[3,999,2]]}'
bash "$R/run_experiment.sh" p5b-spec4-sched-ladder "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=0 MEMUTIL=0.90 LADDER=1,2,4,8 EXTRA_ARGS=--speculative-config=$SPECJSON" || true

# mitigation candidate if IMAs persist: cap max_num_seqs
bash "$R/run_experiment.sh" p5c-spec3-seqs4-ladder "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=3 MEMUTIL=0.90 MAX_NUM_SEQS=4 LADDER=1,2,4,8" || true
echo "PHASE5 COMPLETE"
