#!/usr/bin/env bash
# Phase 3: sampling study on the winning config + memutil/KV check + concurrency ladder.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$ROOT/scripts"

# One boot, three sampling regimes compared in-process
bash "$R/run_experiment.sh" p1-temp07 "MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=4 MEMUTIL=0.85" || true
# server from p1 was stopped by run_experiment; boot again manually for extra benches
MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=4 MEMUTIL=0.85 bash "$R/serve.sh"
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do sleep 10; done
python3 "$R/bench.py" --port 8000 --n 3 --probes code,essay --tag p2-greedy \
  --temperature 0 --top-p 1.0 --out "$ROOT/results/p2-greedy.json"
python3 "$R/bench.py" --port 8000 --n 3 --probes code,essay --thinking --tag p3-thinking \
  --temperature 1.0 --top-p 0.95 --out "$ROOT/results/p3-thinking.json"
bash "$R/stop.sh" qwen38 >/dev/null

# memutil 0.90 + concurrency ladder
MODEL=unsloth/Qwen3.8-27B-NVFP4 NAME=qwen38 SPEC=4 MEMUTIL=0.90 bash "$R/serve.sh"
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do sleep 10; done
python3 "$R/bench.py" --port 8000 --n 3 --probes code,essay,chat --ladder 1,2,4,8,16 \
  --tag p4-mem90-ladder --out "$ROOT/results/p4-mem90-ladder.json"
python3 "$R/quality.py" --port 8000 --out "$ROOT/results/p4-quality.json" || true
grep -aE "KV cache size|Maximum concurrency" /home/emeazzo/dev/qwen3.8/logs/qwen38-server.log | tail -2
bash "$R/stop.sh" qwen38 >/dev/null
echo "PHASE3 COMPLETE"
