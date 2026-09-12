#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/work/piper1-gpl"
"$HOME/venvs/piper/bin/python" -m piper.train.export_onnx \
  --checkpoint "$HOME/work/base_ckpts/es_ES/davefx/medium/epoch=5629-step=1605020.ckpt" \
  --output-file "$HOME/work/base_ckpts/es_ES/davefx/medium/davefx-medium.onnx"
ls -la "$HOME/work/base_ckpts/es_ES/davefx/medium/"
