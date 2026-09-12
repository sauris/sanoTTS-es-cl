#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/work/piper1-gpl/src"
"$HOME/venvs/piper/bin/python" - <<'EOF'
from piper.phonemize_espeak import EspeakPhonemizer
p = EspeakPhonemizer()
print("es-419:", p.phonemize("es-419", "hola mundo, ¿cómo estás?"))
EOF
