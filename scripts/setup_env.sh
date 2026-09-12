#!/usr/bin/env bash
# One-shot environment setup for the es_CL pipeline (WSL2 Debian/Ubuntu).
# Runs apt installs (root via wsl -u root or sudo), builds both venvs, and
# compiles piper1-gpl's espeakbridge. Idempotent-ish: each step skips if done.
#
#   bash scripts/setup_env.sh
set -euo pipefail

echo "== 1. system packages (needs root) =="
if [ "$(id -u)" = "0" ]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git build-essential cmake ninja-build espeak-ng ffmpeg \
    python3-venv python3-dev || true
else
  echo "SKIP (not root): ensure git build-essential cmake ninja-build espeak-ng"
  echo "      ffmpeg python3-venv python3-dev are installed (see PIPELINE.md §1)."
fi

REPO=${SANO_REPO:-/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS}

echo "== 2. piper1-gpl clone + venv =="
if [ ! -d "$HOME/work/piper1-gpl" ]; then
  mkdir -p "$HOME/work"
  git clone --depth 1 https://github.com/OHF-Voice/piper1-gpl "$HOME/work/piper1-gpl"
fi
if [ ! -x "$HOME/venvs/piper/bin/python" ]; then
  python3 -m venv "$HOME/venvs/piper"
  "$HOME/venvs/piper/bin/pip" install -q -U pip wheel
  (cd "$HOME/work/piper1-gpl" && "$HOME/venvs/piper/bin/pip" install -e ".[train]")
fi
# torchaudio for the val_mos (UTMOS) callback; modern cmake for piper's cmake>=3.26
"$HOME/venvs/piper/bin/pip" install -q torchaudio "cmake>=3.26" scikit-build || true

echo "== 3. espeakbridge (editable installs skip the cmake build) =="
if [ ! -f "$HOME/work/piper1-gpl/src/piper/espeakbridge"*.so ] 2>/dev/null; then
  if ! ls "$HOME/work/piper1-gpl/src/piper/"espeakbridge*.so >/dev/null 2>&1; then
    export PATH="$HOME/venvs/piper/bin:$PATH"
    rm -rf /tmp/piper-build
    (cd "$HOME/work/piper1-gpl" \
      && cmake -B /tmp/piper-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      && cmake --build /tmp/piper-build -j 8 \
      && cp /tmp/piper-build/espeakbridge.so src/piper/)
  fi
fi

echo "== 4. sanoTTS research venv =="
if [ ! -x "$HOME/venvs/saanotts/bin/python" ]; then
  python3 -m venv "$HOME/venvs/saanotts"
  "$HOME/venvs/saanotts/bin/pip" install -q -U pip wheel
  "$HOME/venvs/saanotts/bin/pip" install -e "$REPO[research]"
  "$HOME/venvs/saanotts/bin/pip" install -q datasets huggingface_hub
fi

echo "== 5. base checkpoints (es_ES davefx + sharvard) =="
bash "$(dirname "$0")/download_base_ckpts.sh"

echo "== 6. verify =="
"$HOME/venvs/piper/bin/python" -c "import torch, piper; from piper.train.vits.monotonic_align import monotonic_align; print('piper venv OK, cuda:', torch.cuda.is_available())"
"$HOME/venvs/saanotts/bin/python" -c "import torch, onnx, onnxruntime, soundfile; print('saanotts venv OK, cuda:', torch.cuda.is_available())"
echo "SETUP_DONE"
