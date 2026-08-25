#!/usr/bin/env bash
# One-command experiment: serve -> wait -> bench -> quality -> stop -> collect.
# Usage: run_experiment.sh TAG "ENV_KWARGS"
#   e.g. run_experiment.sh nvfp4-inferact-mtp3 "MODEL=Inferact/Qwen3.8-27B-NVFP4 NAME=t1 SPEC=3"
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="$1"; shift
ENVSTR="${1:-}"

echo "=================== EXPERIMENT $TAG ==================="
bash "$ROOT/scripts/stop.sh" qwen38 >/dev/null 2>&1 || true

# shellcheck disable=SC2086
env $ENVSTR bash "$ROOT/scripts/serve.sh"

NAME_Q=$(echo "$ENVSTR" | grep -oE 'NAME=[^ ]+' | cut -d= -f2 || true)
PORT_Q=$(echo "$ENVSTR" | grep -oE 'PORT=[^ ]+' | cut -d= -f2 || true)
LADDER_Q=$(echo "$ENVSTR" | grep -oE 'LADDER=[^ ]+' | cut -d= -f2 || true)
NAME="${NAME_Q:-qwen38}"; PORT="${PORT_Q:-8000}"; LADDER="${LADDER_Q:-${LADDER:-}}"

if ! timeout 3600 bash -c "until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; do sleep 10; done"; then
  echo "SERVER FAILED TO BECOME READY — last log lines:"
  tail -40 "$ROOT/logs/$NAME-server.log"
  exit 1
fi
echo "--- server up (boot ok) ---"

python3 "$ROOT/scripts/bench.py" --port "$PORT" --n 3 \
  --probes code,essay,chat --tag "$TAG" --out "$ROOT/results/$TAG.json"
if [ -n "${LADDER:-}" ]; then
  python3 "$ROOT/scripts/bench.py" --port "$PORT" --n 1 --probes essay --ladder "$LADDER" \
    --tag "$TAG-ladder" --out "$ROOT/results/$TAG-ladder.json"
fi
python3 "$ROOT/scripts/quality.py" --port "$PORT" --out "$ROOT/results/$TAG-quality.json" || true

bash "$ROOT/scripts/stop.sh" "$NAME" >/dev/null
sleep 5
echo "=================== DONE $TAG ==================="
