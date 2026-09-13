#!/usr/bin/env bash
# Disk guard for chande's finetune (v2): prune only files older than 10 min
# so Lightning's atomic checkpoint save is never raced.
# - val_mel ckpts: always pruned (never used for picking)
# - val_mos ckpts: keep the top-3 by the number in the filename
# Self-expires after 4 h. Runs detached; harmless if training is absent.
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
DIR="$REPO/artifacts/voices/es_CO/chande-medium/lightning_logs"
END=$(( $(date +%s) + 14400 ))
while [ "$(date +%s)" -lt "$END" ]; do
  if [ -d "$DIR" ]; then
    find "$DIR" -name "*val_mel*.ckpt" -mmin +10 -delete 2>/dev/null
    find "$DIR" -name "epoch=*val_mos*.ckpt" -mmin +10 2>/dev/null | \
      sed -n 's/.*val_mos=\([0-9.]*\)\.ckpt/\1 &/p' | sort -gr | tail -n +4 | cut -d' ' -f2- | \
      while IFS= read -r f; do [ -n "$f" ] && rm -f "$f"; done
  fi
  sleep 120
done
