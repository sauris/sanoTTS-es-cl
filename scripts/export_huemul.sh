#!/usr/bin/env bash
# Phase 2: export es_CL-huemul-medium.onnx from the best checkpoint and render
# audition samples (held-out eval sentences + Chileanisms + seseo probes).
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
OUT="$REPO/artifacts/voices/es_CL/huemul-medium"
CKPT="$OUT/lightning_logs/version_1/checkpoints/epoch=874-val_mos=3.4628.ckpt"
VENV="$HOME/venvs/piper"

cd "$HOME/work/piper1-gpl"
"$VENV/bin/python" -m piper.train.export_onnx \
  --checkpoint "$CKPT" \
  --output-file "$OUT/es_CL-huemul-medium.onnx"
cp "$REPO/artifacts/voices/es_CL/config.json" "$OUT/es_CL-huemul-medium.onnx.json"
ls -la "$OUT" | grep -v lightning

"$VENV/bin/python" - <<'EOF'
import json
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

out_dir = Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/voices/es_CL/huemul-medium/audition")
out_dir.mkdir(parents=True, exist_ok=True)
base = out_dir.parent

voice = PiperVoice.load(base / "es_CL-huemul-medium.onnx", config_path=base / "es_CL-huemul-medium.onnx.json")

samples = [
    ("01_eval", "¿Cuánto me demoro de aquí a Santiago?"),
    ("02_eval", "La gratitud es chilena y consiste en ganar dinero para despedirse bien."),
    ("03_seseo", "Las ciencias sociales estudian el zapato y la caza."),
    ("04_yeismo", "La lluvia en la calle mojó mi caballo."),
    ("05_plain", "Voy a ir a correr con mi amiga Pepita y estamos buscando una pista de atletismo."),
    ("06_numbers", "El departamento cuesta 12.500 pesos la noche."),
    ("07_question", "¿Me puedes mandar unas fotos de ella?"),
    ("08_long", "El cerdo miniatura es diurno, sociable, amigable y juguetón, pero necesita compañía durante todo el día."),
]

for name, text in samples:
    chunks = list(
        voice.synthesize(
            text,
            syn_config=SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8),
        )
    )
    samples_array = np.concatenate([chunk.audio_int16_array for chunk in chunks])
    path = out_dir / f"{name}.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(samples_array.tobytes())
    print(f"{name}: {len(samples_array)/22050:.2f}s  {text[:50]}")
EOF
echo "AUDITION_RENDERED"
