#!/usr/bin/env bash
set -euo pipefail
ls -la "$HOME/work/piper1-gpl/src/piper/train/vits/monotonic_align/"
"$HOME/venvs/piper/bin/python" -c "import torch; from piper.train.vits.monotonic_align import monotonic_align; import piper; print('PIPER_VENV_OK torch', torch.__version__, 'cuda', torch.cuda.is_available())"
