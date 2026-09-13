#!/usr/bin/env bash
# Phase 7: export the web bundle for a distilled voice.
#   scripts/export_bundle_voice.sh <voice> <lang>
set -euo pipefail
VOICE=${1:?voice}
LANG=${2:?es_CL or es_CO}
REPO=${SANO_REPO:-/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS}
RUN="$REPO/artifacts/voices/$LANG/$VOICE-run"
PY="$HOME/venvs/saanotts/bin/python"

PYTHONPATH="$REPO/tools" "$PY" "$REPO/tools/export_voice_bundle.py" \
  --key "$VOICE" \
  --espeak-voice es-419 \
  --g2p-voice-slot 11 \
  --duration-checkpoint "$RUN/duration/duration-student.pt" \
  --acoustic-checkpoint "$RUN/joint/latent-student.pt" \
  --decoder-checkpoint "$RUN/joint/decoder-student.pt" \
  --out-dir "$REPO/web/voices/$VOICE" \
  --weights f16
