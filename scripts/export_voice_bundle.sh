#!/usr/bin/env bash
# Export the sanoTTS web bundle(s) for the es_CL voice.
#   bash scripts/export_voice_bundle.sh            # full  ~1.5M voice -> web/voices/chilean
#   bash scripts/export_voice_bundle.sh small      # small ~450k voice -> web/voices/chilean-small
set -euo pipefail
REPO=${SANO_REPO:-/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS}
RUN="$REPO/artifacts/voices/es_CL/run"
PY="$HOME/venvs/saanotts/bin/python"

if [ "${1:-full}" = "small" ]; then
  KEY=chilean-small
  DUR="$RUN/duration-small/duration-student.pt"
  ACU="$RUN/joint-small/latent-student.pt"
  DEC="$RUN/joint-small/decoder-student.pt"
else
  KEY=chilean
  DUR="$RUN/duration/duration-student.pt"
  ACU="$RUN/joint/latent-student.pt"
  DEC="$RUN/joint/decoder-student.pt"
fi

PYTHONPATH="$REPO/tools" "$PY" "$REPO/tools/export_voice_bundle.py" \
  --key "$KEY" \
  --espeak-voice es-419 \
  --g2p-voice-slot 11 \
  --duration-checkpoint "$DUR" \
  --acoustic-checkpoint "$ACU" \
  --decoder-checkpoint "$DEC" \
  --out-dir "$REPO/web/voices/$KEY" \
  --weights f16
