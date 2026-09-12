#!/usr/bin/env bash
# Phase 3: full distillation run for es_CL-huemul-medium -> sanoTTS voice.
# Resumable (stage done-files) and logged per stage.
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
export PYTHONPATH="$REPO/tools"
cd "$REPO"
mkdir -p artifacts/es_cl/logs
exec "$HOME/venvs/saanotts/bin/python" tools/train_voice_from_piper.py \
  --teacher-onnx "$REPO/artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx" \
  --teacher-config "$REPO/artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx.json" \
  --text "$REPO/artifacts/data/es_cl/distill_corpus.jsonl" \
  --out-dir "$REPO/artifacts/voices/es_CL/run" \
  --package-name chilean \
  --device cuda \
  2>&1 | tee artifacts/es_cl/logs/phase3_orchestrator.log
