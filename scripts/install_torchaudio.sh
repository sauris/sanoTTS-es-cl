#!/usr/bin/env bash
set -euo pipefail
"$HOME/venvs/piper/bin/pip" install -q torchaudio 2>&1 | tail -1
"$HOME/venvs/piper/bin/python" - <<'EOF'
import torchaudio
import torch
print("torchaudio", torchaudio.__version__, "| torch", torch.__version__)
EOF
