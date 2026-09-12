#!/usr/bin/env bash
# Rebuild the g2p wasm from the currently vendored sources (no data changes).
set -euo pipefail
WASM=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/mcu/ports/wasm
source "$HOME/work/emsdk/emsdk_env.sh" >/dev/null
cd "$WASM"
./build_g2p.sh
