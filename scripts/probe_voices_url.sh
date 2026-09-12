#!/usr/bin/env bash
set -euo pipefail
for u in \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/v1.0/en/en_US/amy/medium/en_US-amy-medium.onnx" \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -I "$u" || true)
  echo "$code $u"
done
