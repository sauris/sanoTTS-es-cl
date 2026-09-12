#!/usr/bin/env bash
# Dry-run the whole train_voice_from_piper pipeline for the es_CL voice:
# validates that every stage (including the reimplemented tools) resolves.
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
export PYTHONPATH="$REPO/tools"
cd "$REPO"
exec "$HOME/venvs/saanotts/bin/python" tools/train_voice_from_piper.py \
  --teacher-onnx "$REPO/artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx" \
  --teacher-config "$REPO/artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx.json" \
  --text "$REPO/artifacts/data/es_cl/distill_corpus.jsonl" \
  --out-dir "$REPO/artifacts/voices/es_CL/run" \
  --package-name chilean \
  --device cuda \
  --dry-run
