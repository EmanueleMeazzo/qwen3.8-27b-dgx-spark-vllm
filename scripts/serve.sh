#!/usr/bin/env bash
# Parametric vLLM server launcher for Qwen3.8-27B on DGX Spark (GB10, aarch64).
# All knobs via env vars; every launch is logged to logs/<NAME>-server.log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/hf-cache"

# ---- knobs -----------------------------------------------------------------
IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.07-py3}"
MODEL="${MODEL:-Inferact/Qwen3.8-27B-NVFP4}"   # HF id (resolved from hf-cache) or local path under /hf
NAME="${NAME:-qwen38}"
PORT="${PORT:-8000}"
MEMUTIL="${MEMUTIL:-0.85}"                      # unified-memory fraction; 0.90+ can wedge the box
CTX="${CTX:-262144}"                            # native context; >262144 needs YARN=1
SPEC="${SPEC:-1}"                               # 0 = no spec decode; else num_speculative_tokens for MTP
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
CPUSET="${CPUSET:-5-9,15-19}"                   # GB10 big.LITTLE: X925 cores only; empty disables
YARN="${YARN:-0}"                               # 1 => rope YaRN override for CTX>262144
EXTRA_ARGS="${EXTRA_ARGS:-}"
# NGC images have no ENTRYPOINT, so the command must start with "vllm serve".
# Upstream vllm/vllm-openai sets ENTRYPOINT ["vllm","serve"], so it must not.
CMD_PREFIX="${CMD_PREFIX-vllm serve}"

CONTAINER="vllm-$NAME"
LOG="$ROOT/logs/$NAME-server.log"

# ---- compose server args ---------------------------------------------------
# shellcheck disable=SC2206
ARGS=($CMD_PREFIX)
ARGS+=(
  "$MODEL"
  --served-model-name qwen3.8-27b
  --port "$PORT"
  --max-model-len "$CTX"
  --kv-cache-dtype fp8
  --gpu-memory-utilization "$MEMUTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
)
if [ "$SPEC" != "0" ]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC}")
fi
if [ "$YARN" = "1" ]; then
  factor=$(python3 -c "print(round($CTX/262144))")
  ARGS+=(--hf-overrides "{\"text_config\": {\"rope_parameters\": {\"mrope_interleaved\": true, \"mrope_section\": [11, 11, 10], \"rope_type\": \"yarn\", \"rope_theta\": 10000000, \"partial_rotary_factor\": 0.25, \"factor\": $factor, \"original_max_position_embeddings\": 262144}}}")
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi
# shellcheck disable=SC2206
EXTRA_SPLIT=($EXTRA_ARGS)

# ---- launch ----------------------------------------------------------------
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

DOCKER_ARGS=( -d --name "$CONTAINER"
  --gpus all --ipc=host --shm-size 32g
  --ulimit memlock=-1 --ulimit stack=67108864
  -p "127.0.0.1:$PORT:$PORT"
  -v "$ROOT/hf-cache:/hf"
  -v "$ROOT/hf-cache/vllm-cache:/root/.cache/vllm"
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=0
  -e VLLM_LOGGING_LEVEL=INFO
)
[ -n "$CPUSET" ] && DOCKER_ARGS+=(--cpuset-cpus "$CPUSET")
if [ -n "${HF_TOKEN:-}" ]; then DOCKER_ARGS+=(-e HF_TOKEN="$HF_TOKEN"); fi
# extra container env, space-separated KEY=VALUE pairs (e.g. DOCKER_ENV="VLLM_USE_DEEP_GEMM=0 FOO=1")
# shellcheck disable=SC2206
DOCKER_ENV_SPLIT=(${DOCKER_ENV:-})
for kv in "${DOCKER_ENV_SPLIT[@]}"; do DOCKER_ARGS+=(-e "$kv"); done

echo ">>> $NAME: model=$MODEL spec=$SPEC memutil=$MEMUTIL seqs=$MAX_NUM_SEQS batch=$MAX_BATCHED_TOKENS ctx=$CTX cpuset=${CPUSET:-off}"
echo ">>> extra: ${EXTRA_ARGS:-none}"
CID=$(docker run "${DOCKER_ARGS[@]}" "$IMAGE" \
  "${ARGS[@]}" ${EXTRA_SPLIT[@]+"${EXTRA_SPLIT[@]}"})

# stream container logs into our log file (docker run -d only prints the id)
( sleep 2; docker logs -f "$CID" >>"$LOG" 2>&1 ) >/dev/null 2>&1 &
echo "$CID" | tee "$ROOT/logs/$NAME.container"
