#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
for f in tools/verify_piper_decoder_torch_parity.py \
         tools/build_piper_vits_decoder_signature_pack.py \
         tools/run_roota_feedback_loop.py \
         tools/summarize_roota_feedback_loop.py \
         tools/export_roota_self_contained_package.py; do
  if "$HOME/venvs/saanotts/bin/python" -m py_compile "$f"; then
    echo "OK   $f"
  else
    echo "FAIL $f"
  fi
done
