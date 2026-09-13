#!/usr/bin/env bash
# Reduced-size (~450k param) es_CL variant: reuses the TEACHER packs and the
# signature pack from the main run (student-size-independent) and retrains
# only the students smaller. See PIPELINE.md Â§5 for the parameter arithmetic.
#
#   bash scripts/run_phase3_small.sh
set -euo pipefail
REPO=${SANO_REPO:-/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS}
RUN="$REPO/artifacts/voices/es_CL/run"
SMALL="$RUN-small"
export PYTHONPATH="$REPO/tools"
PY="$HOME/venvs/saanotts/bin/python"
TEACHER="$REPO/artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx"
CUT="$RUN/decoder-cut/es_CL-huemul-medium-decoder-from-generator-input.onnx"

step() { echo "== $* =="; }

SIG_ARGS=(
  --teacher-decoder "$CUT"
  --acoustic-checkpoint "$SMALL/acoustic/latent-student.pt"
  --signature-pack-dir "$RUN/signatures"
  --variant piperlite --channels 64,32,16,8 --activation leaky_relu
  --signature-hint-weight 0.05 --signature-temporal-weight 0.4
  --feature-hint-weight 0.05 --quiet-ceiling-weight 0.010 --quiet-ceiling-margin-db 0.5
  --adv-weight 0.075 --adv-feature-weight 0.75 --render-rows 0 --device cuda
)

mkdir -p "$SMALL/logs"

step "s3 duration-small"
if [ ! -f "$SMALL/duration/duration-student.pt" ]; then
  "$PY" "$REPO/tools/train_roota_piper_duration_student.py" \
    --pack-dir "$RUN/train-acoustic" --eval-pack-dir "$RUN/eval128" \
    --out-dir "$SMALL/duration" \
    --hidden 48 --depth 2 --kernel-size 5 \
    --duration-vocab-size 166 --duration-oov-id 0 \
    --steps 4000 --device cuda 2>&1 | tee "$SMALL/logs/s3.log"
fi

step "s4 acoustic-small"
if [ ! -f "$SMALL/acoustic/latent-student.pt" ]; then
  "$PY" "$REPO/tools/train_roota_piper_latent_student.py" \
    --pack-dir "$RUN/train-acoustic" --eval-pack-dir "$RUN/eval128" \
    --out-dir "$SMALL/acoustic" \
    --architecture token_context --hidden 48 --token-depth 2 --depth 2 --kernel-size 5 \
    --vocab-size 166 --norm-l1-weight 0.25 --delta-l1-weight 0.10 \
    --channel-stat-weight 0.05 --latent-adv-weight 0.1 --latent-adv-start-step 1500 \
    --decoder "$CUT" --steps 5000 --device cuda 2>&1 | tee "$SMALL/logs/s4.log"
fi

step "s7 decoder-small"
if [ ! -f "$SMALL/decoder/decoder-student.pt" ]; then
  "$PY" "$REPO/tools/train_roota_piper_decoder_student.py" \
    --pack-dir "$RUN/train512-decoder" --eval-pack-dir "$RUN/eval128" \
    --out-dir "$SMALL/decoder" \
    --teacher-init-checkpoint "$RUN/parity/piper-decoder-teacher.pt" \
    --teacher-init-method importance \
    --adv-start-step 1500 --steps 2800 --lr 2e-4 \
    "${SIG_ARGS[@]}" 2>&1 | tee "$SMALL/logs/s7.log"
fi

step "s8 zmix-small"
if [ ! -f "$SMALL/decoder-zmix/decoder-student.pt" ]; then
  "$PY" "$REPO/tools/train_roota_piper_decoder_student.py" \
    --pack-dir "$RUN/train512-decoder" --eval-pack-dir "$RUN/eval128" \
    --out-dir "$SMALL/decoder-zmix" \
    --init-decoder-checkpoint "$SMALL/decoder/decoder-student.pt" \
    --acoustic-latent-mix-prob 0.5 \
    --adv-start-step 1000 --steps 2500 --lr 1e-4 \
    "${SIG_ARGS[@]}" 2>&1 | tee "$SMALL/logs/s8.log"
fi

step "s9 joint-small"
if [ ! -f "$SMALL/joint/latent-student.pt" ]; then
  "$PY" "$REPO/tools/train_roota_joint_z_finetune.py" \
    --pack-dir "$RUN/train512-decoder" --teacher-decoder "$CUT" \
    --acoustic-checkpoint "$SMALL/acoustic/latent-student.pt" \
    --decoder-checkpoint "$SMALL/decoder-zmix/decoder-student.pt" \
    --out-dir "$SMALL/joint" \
    --z-anchor-weight 0.5 --steps 4000 --device cuda 2>&1 | tee "$SMALL/logs/s9.log"
fi

step "s11 package-small"
"$PY" "$REPO/tools/export_roota_self_contained_package.py" \
  --package-name huemul-small --language es_CL --voice es_CL-huemul-small \
  --acoustic-checkpoint "$SMALL/joint/latent-student.pt" \
  --duration-checkpoint "$SMALL/duration/duration-student.pt" \
  --decoder-checkpoint "$SMALL/joint/decoder-student.pt" \
  --piper-config "$REPO/artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx.json" \
  --sample-rate 22050 --duration-length-scale 1.0 \
  --out-dir "$SMALL/package"

step "bundle-small"
bash "$(dirname "$0")/export_voice_bundle.sh" small

echo "PHASE3_SMALL_DONE (total params in $SMALL/package/manifest.json)"
