#!/usr/bin/env bash
# Download 4 representative piper voices (HF rhasspy/piper-voices) for the
# multi-language g2p blast-radius check.
set -euo pipefail
BASE="$HOME/work/blast_voices"
mkdir -p "$BASE"
declare -A VOICES=(
  [en_US-amy-medium]="en/en_US/amy/medium"
  [de_DE-thorsten-medium]="de/de_DE/thorsten/medium"
  [fr_FR-upmc-medium]="fr/fr_FR/upmc/medium"
  [vi_VN-vais1000-medium]="vi/vi_VN/vais1000/medium"
  [cmn_CN-huayan-medium]="cmn/cmn_CN/huayan/medium"
)
for name in "${!VOICES[@]}"; do
  dir="${VOICES[$name]}"
  mkdir -p "$BASE/$name"
  for ext in onnx onnx.json; do
    f="$BASE/$name/$name.$ext"
    if [ ! -s "$f" ]; then
      url="https://huggingface.co/rhasspy/piper-voices/resolve/main/$dir/${name}.${ext}"
      curl -sfL --retry 3 -o "$f" "$url" || echo "WARN: failed $url"
    fi
  done
done
ls -la "$BASE"/*/ | head -30
echo BLAST_VOICES_READY
