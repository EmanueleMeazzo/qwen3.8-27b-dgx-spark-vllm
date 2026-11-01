#!/usr/bin/env bash
# Phase 8 — what YaRN (factor 2, 512K) costs the recommended config.
#
# Question: with YaRN at 512K, how far does single-stream decode of NVFP4 + MTP=3 fall
# from the ~28 tok/s (code) it does at short context? §5 only measured native context,
# up to 128K, and on upstream rather than on the NGC recipe.
#
# Two effects are separated, on the same image in the same session (§9 drift rule):
#   ctl   native RoPE, 262K   — short-context probes + 1K/16K/128K depths (P6 tie-in)
#   yarn  YaRN x2,     512K   — the same, plus 256K and ~496K
#   yarncfg  YaRN written into config.json instead of --hf-overrides (see below)
#
# Why yarncfg exists: vLLM builds the MTP drafter's ModelConfig straight from the
# checkpoint (config/speculative.py, hf_overrides=SpeculativeConfig.hf_config_override),
# so --hf-overrides never reaches it. The drafter keeps default RoPE and a 262,144-token
# limit, and past that its acceptance is exactly 0 (measured: 256K, 0/897, 7.35 tok/s,
# below no-spec). A local checkpoint copy (symlinked weights, YaRN in config.json) gives
# target and drafter the same RoPE.
# ctl-vs-yarn at short context = the cost of YaRN itself (it is static in vLLM, so the
# rescaled RoPE applies to every request, not only long ones). yarn across depths =
# the cost of depth.
#
# Prefill dominates: a fit to §5's 64K/128K points puts a cold 512K prompt at ~20+ min,
# and the needle check pays it a second time. Hence --no-warm (NGC has no prefix
# caching, so the warm repeat would be a third full prefill) and one bench invocation
# per deep depth, so a failure at 496K does not lose the 256K row.
#
# Usage: phase8.sh [arm ...]     (default: ctl yarn)
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$R")"

IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.07-py3}"
MODEL="${MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
SHALLOW="${SHALLOW:-1024,16384,131072}"
# 507904 = 496K: bench.py accepts a 2% overshoot on the target, and prompt + 300
# generated tokens must still fit under max-model-len 524288.
DEEP="${DEEP:-262144,507904}"
PORT=8000
TIMEOUT=3600            # socket timeout per request; must exceed the longest prefill
MTP3='{"method":"mtp","num_speculative_tokens":3}'
case "$IMAGE" in nvcr.io/*) CMDP="vllm serve" ;; *) CMDP="" ;; esac

bench() { python3 "$R/bench.py" --port $PORT --spec-tokens 3 --http-timeout $TIMEOUT "$@"; }

YARNCFG_MODEL="${YARNCFG_MODEL:-/hf/local/Qwen3.8-27B-NVFP4-yarn2}"

# run_one <tag> <yarn 0|1> <ctx> <deep depths or ""> [model]
run_one() {
  local tag="$1" yarn="$2" ctx="$3" deep="$4" model="${5:-$MODEL}"
  echo "=================== PHASE8 $tag ($(date +%F' '%T)) ==================="
  echo "    image=$IMAGE model=$model yarn=$yarn ctx=$ctx deep=${deep:-none}"
  bash "$R/stop.sh" qwen38 >/dev/null 2>&1 || true
  sleep 5
  : > "$ROOT/logs/qwen38-server.log"
  IMAGE="$IMAGE" CMD_PREFIX="$CMDP" MODEL="$model" NAME=qwen38 SPEC=0 MEMUTIL=0.90 \
    CTX="$ctx" YARN="$yarn" EXTRA_ARGS="--speculative-config=$MTP3" bash "$R/serve.sh"

  if ! timeout 1800 bash -c "until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do
        docker ps --format '{{.Names}}' | grep -q vllm-qwen38 || exit 1; sleep 10; done"; then
    echo "!! $tag FAILED TO BOOT"; tail -60 "$ROOT/logs/qwen38-server.log"
    cp "$ROOT/logs/qwen38-server.log" "$ROOT/logs/$tag-server.log" 2>/dev/null
    bash "$R/stop.sh" qwen38 >/dev/null 2>&1 || true
    return 1
  fi
  # The server must actually advertise the context asked for, or every deep row is void.
  local got
  got=$(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c \
        'import json,sys; print(json.load(sys.stdin)["data"][0].get("max_model_len"))')
  echo "--- server up: max_model_len=$got ---"
  if [ "$got" != "$ctx" ]; then
    echo "!! $tag: max_model_len $got != $ctx, aborting arm"
    cp "$ROOT/logs/qwen38-server.log" "$ROOT/logs/$tag-server.log" 2>/dev/null
    bash "$R/stop.sh" qwen38 >/dev/null 2>&1 || true
    return 1
  fi

  # 1. short context, the ~28 tok/s comparison point (code / essay / chat)
  bench --n 3 --probes code,essay,chat --tag "$tag-probes" --out "$ROOT/results/$tag-probes.json"

  # 2. depth: shallow rows in one go, each deep row on its own (saved as it finishes)
  local parts=("$ROOT/results/$tag-depth-shallow.json")
  bench --n 1 --probes chat --depths "$SHALLOW" --needle --no-warm \
    --tag "$tag-depth-shallow" --out "${parts[0]}"
  local d
  for d in ${deep//,/ }; do
    parts+=("$ROOT/results/$tag-depth-$d.json")
    bench --n 1 --probes chat --depths "$d" --needle --no-warm \
      --tag "$tag-depth-$d" --out "$ROOT/results/$tag-depth-$d.json"
  done

  # merge into the single <tag>-depth.json that summarize.py's depth_curve() reads
  python3 - "$tag" "$ROOT/results/$tag-depth.json" "${parts[@]}" <<'EOF'
import json, os, sys
tag, out, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
merged = None
for p in parts:
    if not os.path.exists(p):
        print("missing part", p); continue
    d = json.load(open(p))
    if merged is None:
        merged = d; merged["tag"] = f"{tag}-depth"; merged["parts"] = []
    else:
        merged["depths"] += d.get("depths", []); merged["needle"] = merged.get("needle", []) + d.get("needle", [])
    merged["parts"].append(os.path.basename(p))
if merged:
    json.dump(merged, open(out, "w"), indent=2); print("merged ->", out)
EOF

  cp "$ROOT/logs/qwen38-server.log" "$ROOT/logs/$tag-server.log" 2>/dev/null
  bash "$R/stop.sh" qwen38 >/dev/null 2>&1 || true
  sleep 5
  echo "=================== DONE $tag ($(date +%F' '%T)) ==================="
}

arm() {
  case "$1" in
    ctl)  run_one P8-ctl  0 262144 "" ;;
    yarn) run_one P8-yarn 1 524288 "$DEEP" ;;
    yarncfg) run_one P8-yarncfg 0 524288 "$DEEP" "$YARNCFG_MODEL" ;;
    *) echo "unknown arm: $1" >&2; return 2 ;;
  esac
}

ARMS=("$@")
[ "${#ARMS[@]}" -eq 0 ] && ARMS=(ctl yarn)
for a in "${ARMS[@]}"; do arm "$a"; done
echo "### PHASE8 COMPLETE ($(date +%F' '%T))"
