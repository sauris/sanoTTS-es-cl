#!/usr/bin/env bash
# Phase 2b: export the best teacher checkpoint to ONNX, fix the config, and
# render audition wavs. No human gate here (overnight run): auditions land in
# artifacts/voices/<lang>/<voice>-medium/audition/ for later review.
#   scripts/export_teacher.sh <voice> <lang>
set -euo pipefail

VOICE=${1:?voice}
LANG=${2:?es_CL or es_CO}
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
OUT="$REPO/artifacts/voices/$LANG/$VOICE-medium"
VENV="$HOME/venvs/piper"

cd "$HOME/work/piper1-gpl"

# pick the checkpoint with the best UTMOS val_mos (filenames embed it)
BEST=$(ls "$OUT"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | \
  sed -n 's/.*epoch=\([0-9]*\)-val_mos=\([0-9.]*\)\.ckpt/\2 \1 &/p' | sort -gr | head -1)
if [ -z "$BEST" ]; then
  echo "NO_CHECKPOINTS" >&2
  exit 1
fi
EPOCH=$(echo "$BEST" | awk '{print $2}')
CKPT=$(echo "$BEST" | cut -d' ' -f3-)
echo "best ckpt: $CKPT (epoch $EPOCH)"
if [ "$EPOCH" -lt 500 ]; then
  echo "TRAINING_TOO_SHORT (epoch $EPOCH)" >&2
  exit 2
fi

"$VENV/bin/python" -m piper.train.export_onnx \
  --checkpoint "$CKPT" \
  --output-file "$OUT/$LANG-$VOICE-medium.onnx"
cp "$REPO/artifacts/voices/$LANG/config.json" "$OUT/$LANG-$VOICE-medium.onnx.json"

"$VENV/bin/python" - "$LANG" "$VOICE" <<'PY'
import json
import sys
from pathlib import Path

lang, voice = sys.argv[1], sys.argv[2]
out = Path(f"/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/voices/{lang}/{voice}-medium")
cfg = out / f"{lang}-{voice}-medium.onnx.json"
data = json.loads(cfg.read_text(encoding="utf-8"))
if lang == "es_CL":
    data["dataset"] = "google_chilean_spanish_slr71"
    data["dataset_attestation"] = (
        "Google Chilean Spanish Low-resource Speech (OpenSLR SLR71, https://openslr.org/71). "
        "Copyright 2018, 2019 Google, Inc. CC BY-SA 4.0. Citation: J. M. Martin-Valdivia, "
        "E. Ufimtseva, A. V. Parkhilko, A. Y. Zhila, D. Gimeno-Gomez, 'SLR71: Google Crowdsource "
        "Multilingual Speech Corpus', LREC 2020."
    )
else:
    data["dataset"] = "google_colombian_spanish_slr72"
    data["dataset_attestation"] = (
        "Google Crowdsourced Colombian Spanish Speech (OpenSLR SLR72, https://openslr.org/72). "
        "Copyright 2018, 2019 Google, Inc. CC BY-SA 4.0. Citation: A. Guevara-Rukoz, I. Demirsahin, "
        "F. He, S.-H. C. Chu, S. Sarin, K. Pipatsrisawat, A. Gutkin, A. Butryna, O. Kjartansson, "
        "'Crowdsourcing Latin American Spanish for Low-Resource Text-to-Speech', LREC 2020, pp. 6504-6513."
    )
data["language"] = {"code": lang}
data["audio"]["sample_rate"] = 22050
cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"patched {cfg}")
PY

# auditions: 2 held-out eval lines + accent probes + numbers + long sentence
"$VENV/bin/python" - "$VOICE" "$LANG" <<'PY'
import sys
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

voice, lang = sys.argv[1], sys.argv[2]
base = Path(f"/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/voices/{lang}/{voice}-medium")
out_dir = base / "audition"
out_dir.mkdir(exist_ok=True)

eval_lines = (Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS") / f"artifacts/data/{voice}/eval.txt").read_text(encoding="utf-8").splitlines()

if lang == "es_CL":
    probes = [
        ("03_seseo", "Las ciencias sociales estudian el zapato y la caza."),
        ("04_yeismo", "La lluvia en la calle moj\u00f3 mi caballo."),
        ("05_plain", "Voy a ir a correr con mi amiga Pepita y estamos buscando una pista de atletismo."),
        ("06_numbers", "El departamento cuesta 12.500 pesos la noche."),
        ("07_question", "\u00bfMe puedes mandar unas fotos de ella?"),
        ("08_long", "El cerdo miniatura es diurno, sociable, amigable y juguet\u00f3n, pero necesita compa\u00f1\u00eda durante todo el d\u00eda."),
    ]
else:
    probes = [
        ("03_seseo", "Las ciencias sociales estudian el zapato y la caza."),
        ("04_yeismo", "La lluvia en la calle moj\u00f3 mi caballo."),
        ("05_coast", "Oye, \u00bfvamos para la plaza del mercado a comer un bollo limpio con suero coste\u00f1o?"),
        ("06_numbers", "Un sombrero vueltiao fino de 23 vueltas cuesta m\u00e1s de un mill\u00f3n de pesos."),
        ("07_question", "\u00bfCu\u00e1ndo empiezan las corralejas de Sincelejo?"),
        ("08_long", "El sombrero vueltiao es el s\u00edmbolo de los sabaneros de Sucre y C\u00f3rdoba, tejido a mano con ca\u00f1a flecha en San Andr\u00e9s de Sotavento."),
    ]

samples = [("01_eval", eval_lines[0]), ("02_eval", eval_lines[1])] + probes

v = PiperVoice.load(base / f"{lang}-{voice}-medium.onnx", config_path=base / f"{lang}-{voice}-medium.onnx.json")
for name, text in samples:
    chunks = list(v.synthesize(text, syn_config=SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8)))
    arr = np.concatenate([c.audio_int16_array for c in chunks])
    with wave.open(str(out_dir / f"{name}.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(arr.tobytes())
    print(f"{name}: {len(arr)/22050:.2f}s  {text[:50]}")
PY

ls -la "$OUT" | grep -v lightning

# disk guard: keep only the exported-from (best val_mos) ckpt + last.ckpt.
# ~1 GB per val checkpoint otherwise fills the drive across voices.
find "$OUT/lightning_logs" -name "*.ckpt" ! -name "last.ckpt" \
  ! -name "$(basename "$CKPT")" -delete
echo "trimmed checkpoints (kept $(basename "$CKPT") + last.ckpt)"

echo "TEACHER_EXPORTED"
