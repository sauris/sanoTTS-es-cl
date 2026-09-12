#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="$HOME/hf"
exec "$HOME/venvs/saanotts/bin/python" /mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/es_cl/scripts/export_speaker.py
