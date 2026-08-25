#!/usr/bin/env bash
# Quantization sweep: one boot per checkpoint, full eval cycle each.
#
# Every checkpoint here keeps the in-checkpoint MTP head (verified before download --
# a quant that strips it loses ~2x decode and is not comparable), so the only variables
# are weight bytes and kernel path.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$ROOT/scripts"
SPEC="${SPEC:-3}"; MEMUTIL="${MEMUTIL:-0.90}"

run() { # run <tag> <model>
  bash "$R/run_experiment.sh" "$1" \
    "MODEL=$2 NAME=qwen38 SPEC=$SPEC MEMUTIL=$MEMUTIL MAX_NUM_SEQS=16" || true
}

for spec in "$@"; do
  run "${spec%%=*}" "${spec#*=}"
done
echo "QUANT SWEEP COMPLETE"
