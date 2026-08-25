#!/usr/bin/env bash
# Stop a serving container started by serve.sh. Usage: ./stop.sh [name]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-qwen38}"
if [ -f "$ROOT/logs/$NAME.container" ]; then
  docker rm -f "$(cat "$ROOT/logs/$NAME.container")" >/dev/null 2>&1 || true
  rm -f "$ROOT/logs/$NAME.container"
fi
docker rm -f "vllm-$NAME" >/dev/null 2>&1 || true
echo "stopped $NAME"
