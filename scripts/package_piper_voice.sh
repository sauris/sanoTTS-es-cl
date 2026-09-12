#!/usr/bin/env bash
# Assemble the rhasspy/piper-voices contribution package for es_CL-huemul-medium.
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
SRC="$REPO/artifacts/voices/es_CL/huemul-medium"
OUT="$REPO/artifacts/release/piper-voices/es/CL/huemul/medium"
mkdir -p "$OUT"

cp "$SRC/es_CL-huemul-medium.onnx" "$OUT/es_CL-huemul-medium.onnx"
cp "$SRC/es_CL-huemul-medium.onnx.json" "$OUT/es_CL-huemul-medium.onnx.json"

cat > "$OUT/MODEL_CARD" <<'EOF'
# Model card for huemul (medium)

* Language: es_CL (Spanish, Chile)
* Speakers: 1
* Quality: medium
* Samplerate: 22,050Hz

## Dataset

* URL: https://openslr.org/71 (Google Chilean Spanish Low-resource Speech, SLR71)
* License: CC BY-SA 4.0
* Attribution: Copyright 2018, 2019 Google, Inc. Citation: J. M. Martin-Valdivia,
  E. Ufimtseva, A. V. Parkhilko, A. Y. Zhila, D. Gimeno-Gomez, "SLR71: Google
  Crowdsource Multilingual Speech Corpus", LREC 2020.
* Speaker: male clm_02121, ~18.8 minutes (146 utterances).

## Training

Finetuned from es_ES-davefx-medium (rhasspy/piper-checkpoints, male) on the
clm_02121 subset for 1000 epochs (warmstart, lr 1e-4 / 5e-5, batch 8,
espeak voice es-419). Best checkpoint by UTMOS val_mos (3.46) at epoch 874.
Phonemization uses es-419 (Latin American Spanish: seseo, yeismo), which
matches the Chilean accent of the speaker.
EOF

# rhasspy/piper-voices voice dirs contain exactly: .onnx, .onnx.json, MODEL_CARD

ls -la "$OUT"
du -sh "$OUT"
echo PIPER_PACKAGE_READY
