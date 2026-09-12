#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/work/blast_voices/zh_CN-huayan-medium"
mkdir -p "$BASE"
for ext in onnx onnx.json; do
  f="$BASE/zh_CN-huayan-medium.$ext"
  if [ ! -s "$f" ]; then
    curl -sfL --retry 3 -o "$f" \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.$ext"
  fi
done
ls -la "$BASE"
echo ZH_READY
