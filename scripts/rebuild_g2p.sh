#!/usr/bin/env bash
# Phase 4: rebuild the snt_g2p wasm with the es-419 voice added to the data.
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
WASM="$REPO/mcu/ports/wasm"
DATA="$WASM/espeak-data-multi"

# 1. add the es-419 voice (from piper's bundled espeak-ng-data, which is what
#    the rest of the multi set was taken from)
mkdir -p "$DATA/lang/roa"
cp "$HOME/venvs/saanotts/lib/python3.11/site-packages/piper/espeak-ng-data/lang/roa/es-419" \
   "$DATA/lang/roa/es-419"
ls -la "$DATA/lang/roa/"

# 2. rebuild
source "$HOME/work/emsdk/emsdk_env.sh" >/dev/null
cd "$WASM"
./build_g2p.sh
