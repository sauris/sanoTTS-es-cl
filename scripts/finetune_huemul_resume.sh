#!/usr/bin/env bash
# Phase 2 polish: resume the huemul fine-tune for 500 more epochs from last.ckpt.
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
CKPT="$REPO/artifacts/voices/es_CL/huemul-medium/lightning_logs/version_1/checkpoints/last.ckpt"

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
  --model.learning_rate 1e-4 \
  --model.learning_rate_d 5e-5 \
  --trainer.max_epochs 1500 \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.default_root_dir "$REPO/artifacts/voices/es_CL/huemul-medium" \
  --trainer.check_val_every_n_epoch 5 \
  --trainer.log_every_n_steps 10 \
  --ckpt_path "$CKPT"
