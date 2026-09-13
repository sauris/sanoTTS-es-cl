#!/usr/bin/env bash
# Phase 7: export the web bundle for a distilled voice.
#   scripts/export_bundle_voice.sh <voice> <lang> [small]
# "small" reads the -run-small checkpoints and exports under key <voice>-small.
set -euo pipefail
VOICE=${1:?voice}
LANG=${2:?es_CL or es_CO}
MODE=${3:-full}
REPO=${SANO_REPO:-/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS}
if [ "$MODE" = small ]; then
  RUN="$REPO/artifacts/voices/$LANG/$VOICE-run-small"
  KEY="$VOICE-small"
else
  RUN="$REPO/artifacts/voices/$LANG/$VOICE-run"
  KEY="$VOICE"
fi
PY="$HOME/venvs/saanotts/bin/python"

PYTHONPATH="$REPO/tools" "$PY" "$REPO/tools/export_voice_bundle.py" \
  --key "$KEY" \
  --espeak-voice es-419 \
  --g2p-voice-slot 11 \
  --duration-checkpoint "$RUN/duration/duration-student.pt" \
  --acoustic-checkpoint "$RUN/joint/latent-student.pt" \
  --decoder-checkpoint "$RUN/joint/decoder-student.pt" \
  --out-dir "$REPO/web/voices/$KEY" \
  --weights f16
