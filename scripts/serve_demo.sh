#!/usr/bin/env bash
# Serve the demo folder with cache disabled (see demo/serve.py).
#   scripts/serve_demo.sh [port]
set -euo pipefail
DEMO="$(cd "$(dirname "$0")/../demo" && pwd)"
PORT="${1:-8178}"
cd "$DEMO"
exec python3 serve.py "$PORT"
