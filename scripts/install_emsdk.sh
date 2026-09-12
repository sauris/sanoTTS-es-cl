#!/usr/bin/env bash
# Phase 4: install Emscripten into ~/work/emsdk (idempotent).
set -euo pipefail
if [ ! -d "$HOME/work/emsdk" ]; then
  git clone https://github.com/emscripten-core/emsdk.git "$HOME/work/emsdk"
fi
cd "$HOME/work/emsdk"
if [ ! -d "upstream" ]; then
  ./emsdk install latest
fi
./emsdk activate latest
./emsdk-env.sh 2>/dev/null || true
# verify emcc works
source ./emsdk_env.sh >/dev/null 2>&1
emcc --version | head -1
