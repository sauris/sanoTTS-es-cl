#!/usr/bin/env bash
# Serve web/ locally for browser testing. Use ?voices=local to pin the local host.
#   bash scripts/serve_web.sh [port]
set -euo pipefail
REPO=${SANO_REPO:-/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS}
PORT="${1:-8177}"
cd "$REPO/web"
exec python3 -m http.server "$PORT"
