#!/usr/bin/env bash
set -euo pipefail
"$HOME/venvs/saanotts/bin/pip" install -q datasets huggingface_hub 2>&1 | tail -2
"$HOME/venvs/saanotts/bin/python" - <<'EOF'
import datasets
print("datasets", datasets.__version__)
EOF
