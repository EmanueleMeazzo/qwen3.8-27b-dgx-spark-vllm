#!/usr/bin/env bash
# Phase 6 — the three things §4 and §5 of the README said were not measured:
#   1. how DFlash2 behaves with more than one request in flight,
#   2. whether draft acceptance is content-dependent (per-probe, not session-aggregate),
#   3. what any of this does at the context depth the recipe advertises.
#
# Two boots on the same image so the drafter is the only variable: DFlash2 (7 draft
# tokens) and the MTP=3 recommendation as the control. Every run is measured in the
# same session as its control, per the drift caveat in §8.
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$R")"

IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"
MODEL="${MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
LADDER="${LADDER:-1,2,4,8,16}"
DEPTHS="${DEPTHS:-1024,4096,16384,65536,131072}"
PORT=8000

run_one() {         # run_one <tag> <spec-tokens> <speculative-config-json>
  local tag="$1" k="$2" speccfg="$3"
  echo "=================== PHASE6 $tag ==================="
  bash "$R/stop.sh" qwen38 >/dev/null 2>&1 || true
  sleep 5
  IMAGE="$IMAGE" CMD_PREFIX= MODEL="$MODEL" NAME=qwen38 SPEC=0 MEMUTIL=0.90 \
    EXTRA_ARGS="--speculative-config=$speccfg" bash "$R/serve.sh"

  if ! timeout 1800 bash -c "until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
        docker ps --format '{{.Names}}' | grep -q vllm-qwen38 || exit 1; sleep 10; done"; then
    echo "!! $tag failed to boot"; tail -40 "$ROOT/logs/qwen38-server.log"; return 1
  fi
  echo "--- server up ---"

  # 1. probes, with acceptance attributed to each probe rather than to the session
  python3 "$R/bench.py" --port $PORT --n 3 --probes code,essay,chat --spec-tokens "$k" \
    --tag "$tag-probes" --out "$ROOT/results/$tag-probes.json"

  # 2. concurrency, on prose *and* code: if the block drafter degrades under load it
  #    shows up as falling acceptance before it shows up as falling tok/s
  python3 "$R/bench.py" --port $PORT --n 1 --probes chat --ladder "$LADDER" \
    --ladder-probes essay,code --spec-tokens "$k" \
    --tag "$tag-ladder" --out "$ROOT/results/$tag-ladder.json"

  # 3. context depth: prefill rate, decode rate, acceptance and retrieval vs depth
  python3 "$R/bench.py" --port $PORT --n 1 --probes chat --depths "$DEPTHS" --needle \
    --spec-tokens "$k" --tag "$tag-depth" --out "$ROOT/results/$tag-depth.json"

  bash "$R/stop.sh" qwen38 >/dev/null 2>&1 || true
  sleep 5
  echo "=================== DONE $tag ==================="
}

run_one P6-dflash2   7 '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
run_one P6-mtp3      3 '{"method":"mtp","num_speculative_tokens":3}'
# Third arm: the per-position data shows prose accepts almost nothing past draft
# position 3, so a shorter block is the obvious thing to try. It does not pay --
# see §4. Kept in the sweep because a negative result a reader would otherwise
# re-derive is worth publishing.
run_one P6-dflash2k4 4 '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":4}'
