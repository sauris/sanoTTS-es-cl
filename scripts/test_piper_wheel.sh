#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
export PYTHONPATH="$REPO/tools"
"$HOME/venvs/saanotts/bin/python" - <<'EOF'
from piper.voice import PiperVoice
from pathlib import Path
REPO = Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS")
voice = PiperVoice.load(
    Path.home() / "work/base_ckpts/es_ES/davefx/medium/davefx-medium.onnx",
    config_path=Path.home() / "work/base_ckpts/es_ES/davefx/medium/config.json",
)
ph = voice.phonemize("¿Cuánto me demoro de aquí a Santiago?")
print("phonemes:", ph[0][:40])
ids = voice.phonemes_to_ids(ph[0])
print("ids:", ids[:12], "... total", len(ids))
EOF
