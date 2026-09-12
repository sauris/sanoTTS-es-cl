#!/usr/bin/env bash
# Phase 4 attempt 2: sync the Spanish espeak files (voice + dictionary) from
# piper 1.8's bundled espeak-ng-data into the wasm data set, then rebuild.
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
WASM="$REPO/mcu/ports/wasm"
DATA="$WASM/espeak-data-multi"
SRC="$HOME/venvs/saanotts/lib/python3.11/site-packages/piper/espeak-ng-data"

cp "$SRC/lang/roa/es"      "$DATA/lang/roa/es"
cp "$SRC/lang/roa/es-419"  "$DATA/lang/roa/es-419"
cp "$SRC/es_dict"          "$DATA/es_dict"

source "$HOME/work/emsdk/emsdk_env.sh" >/dev/null
cd "$WASM"
./build_g2p.sh 2>&1 | grep -E "^built" || true
echo REBUILT
