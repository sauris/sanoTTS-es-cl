#!/usr/bin/env bash
# Phase 2: fine-tune es_CL-huemul-medium from es_ES-davefx-medium (male base,
# speaker clm_02121). Warmstart = non-strict copy of all matching params
# (keeps Spanish phoneme embeddings; fresh optimizer/epoch counter).
set -euo pipefail

REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
CKPT="$HOME/work/base_ckpts/es_ES/davefx/medium/epoch=5629-step=1605020.ckpt"

cd "$HOME/work/piper1-gpl"
exec "$HOME/venvs/piper/bin/python" -m piper.train fit \
  --data.voice_name huemul \
  --data.csv_path "$REPO/artifacts/data/es_cl/metadata.csv" \
  --data.audio_dir "$REPO/artifacts/data/es_cl/audio" \
  --data.espeak_voice es-419 \
  --data.cache_dir "$REPO/artifacts/cache/es_cl" \
  --data.config_path "$REPO/artifacts/voices/es_CL/config.json" \
  --data.batch_size 8 \
  --data.num_workers 4 \
  --model.sample_rate 22050 \
  --model.warmstart_ckpt "$CKPT" \
  --model.learning_rate 1e-4 \
  --model.learning_rate_d 5e-5 \
  --trainer.max_epochs 1000 \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.default_root_dir "$REPO/artifacts/voices/es_CL/huemul-medium" \
  --trainer.check_val_every_n_epoch 5 \
  --trainer.log_every_n_steps 10
