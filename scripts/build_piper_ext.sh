#!/usr/bin/env bash
# Re-run the piper1-gpl extension builds after installing python3-dev.
set -euo pipefail
cd "$HOME/work/piper1-gpl"
echo "-- build_monotonic_align.sh --"
PATH="$HOME/venvs/piper/bin:$PATH" ./build_monotonic_align.sh 2>&1 | tail -4
echo "-- setup.py build_ext --inplace --"
PATH="$HOME/venvs/piper/bin:$PATH" python3 setup.py build_ext --inplace 2>&1 | tail -4
echo "-- verify imports --"
"$HOME/venvs/piper/bin/python" -c "import torch; import piper; from piper.train.vits.monotonic_align import monotonic_align; print('piper venv OK: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "PIPER_VENV_OK"
