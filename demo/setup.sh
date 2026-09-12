#!/usr/bin/env bash
# One-time setup for the demo player: copies the wasm runtime (and optionally
# voice bundles) out of a sanoTTS checkout next to this index.html.
#
#   demo/setup.sh /path/to/sanoTTS/web            # runtime only
#   demo/setup.sh /path/to/sanoTTS/web all        # runtime + every voice bundle
#   demo/setup.sh /path/to/sanoTTS/web chilean chilean-small-int8
#
# The runtime files are ~4.3 MB (the espeak-ng phonemizer data dominates), so
# they are not committed here. After this, serve this directory:
#
#   cd demo && python3 -m http.server 8178
#   open http://localhost:8178
set -euo pipefail
WEB="${1:?usage: demo/setup.sh /path/to/sanoTTS/web [all|voice-key ...]}"
shift || true

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "-- runtime --"
for f in snt_voice.js snt_voice.wasm snt_g2p.js snt_g2p.wasm snt_g2p.data; do
  cp "$WEB/$f" "$HERE/$f"
done

echo "-- voices --"
if [ "${1:-}" = "all" ]; then
  set -- "$WEB/voices"/*
fi
for key in "$@"; do
  if [ -d "$WEB/voices/$key" ]; then
    cp -r "$WEB/voices/$key" "$HERE/voices/"
  elif [ -d "$HERE/voices/$key" ]; then
    echo "   $key (already present)"
  else
    echo "   $key not found in $WEB/voices -- skipped"
  fi
done

ls -la "$HERE"
echo "DEMO_READY: cd '$HERE' && python3 -m http.server 8178"
