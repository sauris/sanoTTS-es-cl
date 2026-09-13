#!/usr/bin/env bash
# Phase 3: full distillation run for a teacher -> sanoTTS voice.
#   scripts/run_phase3_voice.sh <voice> <lang>
set -euo pipefail
VOICE=${1:?voice}
LANG=${2:?es_CL or es_CO}
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
export PYTHONPATH="$REPO/tools"
cd "$REPO"
exec "$HOME/venvs/saanotts/bin/python" tools/train_voice_from_piper.py \
  --teacher-onnx "$REPO/artifacts/voices/$LANG/$VOICE-medium/$LANG-$VOICE-medium.onnx" \
  --teacher-config "$REPO/artifacts/voices/$LANG/$VOICE-medium/$LANG-$VOICE-medium.onnx.json" \
  --text "$REPO/artifacts/data/$VOICE/distill_corpus.jsonl" \
  --out-dir "$REPO/artifacts/voices/$LANG/$VOICE-run" \
  --package-name "$VOICE" \
  --device cuda
