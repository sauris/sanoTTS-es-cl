#!/usr/bin/env bash
set -euo pipefail
cp /tmp/piper-build/espeakbridge.so "$HOME/work/piper1-gpl/src/piper/"
cd "$HOME/work/piper1-gpl/src"
"$HOME/venvs/piper/bin/python" - <<'EOF'
from piper import espeakbridge
from piper.phonemize_espeak import EspeakPhonemizer
p = EspeakPhonemizer()
print("espeakbridge OK:", espeakbridge.__file__)
print(p.phonemize_espeak("hola mundo", voice_name="es-419"))
EOF
