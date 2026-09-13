#!/usr/bin/env bash
# Phase 2: fine-tune an es_* teacher voice from an es_ES base checkpoint.
#   scripts/finetune_voice.sh copihue es_CL female
#   scripts/finetune_voice.sh vueltiao es_CO male
# Warmstart = non-strict copy of all matching params (keeps Spanish phoneme
# embeddings; fresh optimizer/epoch counter). 4070: ~90 min for 1000 epochs.
set -euo pipefail

VOICE=${1:?voice name}
LANG=${2:?es_CL or es_CO}
GENDER=${3:?male or female}

REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
if [ "$GENDER" = male ]; then
  CKPT="$HOME/work/base_ckpts/es_ES/davefx/medium/epoch=5629-step=1605020.ckpt"
else
  CKPT="$HOME/work/base_ckpts/es_ES/sharvard/medium/epoch=4899-step=215600.ckpt"
fi

# config is language-agnostic (espeak es-419 + standard 256 phoneme map);
# es_CO reuses a copy of the es_CL one.
mkdir -p "$REPO/artifacts/voices/$LANG"
[ -f "$REPO/artifacts/voices/$LANG/config.json" ] || \
  cp "$REPO/artifacts/voices/es_CL/config.json" "$REPO/artifacts/voices/$LANG/config.json"

cd "$HOME/work/piper1-gpl"
exec "$HOME/venvs/piper/bin/python" -m piper.train fit \
  --data.voice_name "$VOICE" \
  --data.csv_path "$REPO/artifacts/data/$VOICE/metadata.csv" \
  --data.audio_dir "$REPO/artifacts/data/$VOICE/audio" \
  --data.espeak_voice es-419 \
  --data.cache_dir "$REPO/artifacts/cache/$VOICE" \
  --data.config_path "$REPO/artifacts/voices/$LANG/config.json" \
  --data.batch_size 8 \
  --data.num_workers 4 \
  --model.sample_rate 22050 \
  --model.warmstart_ckpt "$CKPT" \
  --model.learning_rate 1e-4 \
  --model.learning_rate_d 5e-5 \
  --trainer.max_epochs 1000 \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.default_root_dir "$REPO/artifacts/voices/$LANG/$VOICE-medium" \
  --trainer.check_val_every_n_epoch 5 \
  --trainer.log_every_n_steps 10
