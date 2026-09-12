#!/usr/bin/env bash
# Phase 0: download gender-matched es_ES medium base checkpoints + configs
# from HF rhasspy/piper-checkpoints into ~/work/base_ckpts (ext4 for speed).
set -euo pipefail

BASE="$HOME/work/base_ckpts"
REPO_URL="https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/es/es_ES"

mkdir -p "$BASE/es_ES/sharvard/medium" "$BASE/es_ES/davefx/medium"

for v in sharvard davefx; do
  for f in config.json train.sh MODEL_CARD dataset.jsonl.gz; do
    curl -sfL -o "$BASE/es_ES/$v/medium/$f" "$REPO_URL/$v/medium/$f" || echo "WARN: no $f for $v"
  done
done

curl -sfL --retry 3 -o "$BASE/es_ES/sharvard/medium/epoch=4899-step=215600.ckpt" \
  "$REPO_URL/sharvard/medium/epoch%3D4899-step%3D215600.ckpt"
curl -sfL --retry 3 -o "$BASE/es_ES/davefx/medium/epoch=5629-step=1605020.ckpt" \
  "$REPO_URL/davefx/medium/epoch%3D5629-step%3D1605020.ckpt"

echo "-- downloaded --"
du -sh "$BASE"/es_ES/*/medium/*.ckpt
