#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
export PYTHONPATH="$REPO/tools"
"$HOME/venvs/saanotts/bin/python" "$REPO/artifacts/es_cl/scripts/gen_g2p_truth.py"
export PATH="$HOME/work/emsdk/node/24.19.0_64bit/bin:$PATH"
cd "$REPO"
node mcu/ports/wasm/verify_g2p_es419_node.mjs
