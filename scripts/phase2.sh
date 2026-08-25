#!/usr/bin/env bash
# Phase 2: spec-decode sweep + backend/cudagraph + DFlash2 on the quant winner.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$ROOT/scripts"

bash "$R/run_experiment.sh" s1-unsloth-mtp2 "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=2 MEMUTIL=0.85"
bash "$R/run_experiment.sh" s2-unsloth-mtp4 "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=4 MEMUTIL=0.85"
bash "$R/run_experiment.sh" s3-unsloth-mtp6 "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=6 MEMUTIL=0.85"
# Triton attention backend -> supports FULL_AND_PIECEWISE cudagraphs with spec decode
bash "$R/run_experiment.sh" s4-unsloth-mtp3-triton "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=3 MEMUTIL=0.85 EXTRA_ARGS=--attention-backend=TRITON_ATTN"
# DFlash2 block-diffusion drafter
bash "$R/run_experiment.sh" s5-unsloth-dflash7 "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=0 MEMUTIL=0.85 EXTRA_ARGS=--speculative-config={\"method\":\"dflash\",\"model\":\"incoai/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":7}"
echo "PHASE2 COMPLETE"
