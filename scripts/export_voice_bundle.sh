#!/usr/bin/env bash
# Export a sanoTTS web bundle for the es_CL voice.
#   scripts/export_voice_bundle.sh                 # full  ~1.57M voice, f16 -> web/voices/huemul
#   scripts/export_voice_bundle.sh small           # small ~340k voice,  f16 -> web/voices/huemul-small
#   scripts/export_voice_bundle.sh small int8      # small voice as per-tensor int8 (loader dequantizes)
set -euo pipefail
REPO=${SANO_REPO:-/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS}
RUN="$REPO/artifacts/voices/es_CL/run"
PY="$HOME/venvs/saanotts/bin/python"

if [ "${1:-full}" = "small" ]; then
  KEY=huemul-small
  DUR="$RUN-small/duration/duration-student.pt"
  ACU="$RUN-small/joint/latent-student.pt"
  DEC="$RUN-small/joint/decoder-student.pt"
else
  KEY=huemul
  DUR="$RUN/duration/duration-student.pt"
  ACU="$RUN/joint/latent-student.pt"
  DEC="$RUN/joint/decoder-student.pt"
fi
WEIGHTS="${2:-f16}"

PYTHONPATH="$REPO/tools" "$PY" "$REPO/tools/export_voice_bundle.py" \
  --key "$KEY" \
  --espeak-voice es-419 \
  --g2p-voice-slot 11 \
  --duration-checkpoint "$DUR" \
  --acoustic-checkpoint "$ACU" \
  --decoder-checkpoint "$DEC" \
  --out-dir "$REPO/web/voices/$KEY" \
  --weights "$WEIGHTS"
