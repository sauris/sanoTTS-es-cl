#!/usr/bin/env bash
# Phase 0: create both venvs for the es_CL run.
# - ~/venvs/saanotts : sanoTTS research env (torch, onnx, piper-tts PyPI, ...)
# - ~/venvs/piper    : OHF-Voice/piper1-gpl training env (GPL, separate on purpose)
set -euo pipefail

SAANOTTS_REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS

echo "== venv: saanotts =="
python3 -m venv "$HOME/venvs/saanotts"
"$HOME/venvs/saanotts/bin/pip" install -q -U pip wheel
"$HOME/venvs/saanotts/bin/pip" install -e "$SAANOTTS_REPO[research]" 2>&1 | tail -4
"$HOME/venvs/saanotts/bin/python" -c "import torch; print('saanotts torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== venv: piper (piper1-gpl) =="
python3 -m venv "$HOME/venvs/piper"
"$HOME/venvs/piper/bin/pip" install -q -U pip wheel
cd "$HOME/work/piper1-gpl"
"$HOME/venvs/piper/bin/pip" install -e ".[train]" 2>&1 | tail -4
echo "-- build_monotonic_align.sh --"
PATH="$HOME/venvs/piper/bin:$PATH" ./build_monotonic_align.sh 2>&1 | tail -3
echo "-- setup.py build_ext --inplace --"
PATH="$HOME/venvs/piper/bin:$PATH" python3 setup.py build_ext --inplace 2>&1 | tail -3
"$HOME/venvs/piper/bin/python" -c "import torch, piper; print('piper venv torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| piper ok')"

echo "ALL_VENVS_OK"
