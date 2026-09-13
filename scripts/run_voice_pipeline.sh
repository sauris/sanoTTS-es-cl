#!/usr/bin/env bash
# Overnight master pipeline for the three new voices (unattended run).
#   copihue   es_CL female (SLR71 clf_04310, ~16.6 min)  base es_ES-sharvard
#   vueltiao  es_CO male   (SLR72 com_06136, ~18.1 min)  base es_ES-davefx
#   chande    es_CO female (SLR72 cof_02484, ~19.8 min)  base es_ES-sharvard
# Per voice: finetune -> teacher export + auditions -> distill -> bundle.
# Then evidence for each. Resumable via artifacts/es_cl/pipeline/<voice>/*.done
# Designed to run detached:  setsid nohup bash run_voice_pipeline.sh &
set -uo pipefail

REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
SCRIPTS=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS-es-cl/scripts
LOGS=$REPO/artifacts/es_cl/logs
PIPE=$REPO/artifacts/es_cl/pipeline
DATE_STAMP=20260913
export HF_HOME="$HOME/hf"

mkdir -p "$LOGS" "$PIPE"
: > "$PIPE/status.txt"

# the sibling scripts live on drvfs with CRLF; strip and stage them
STAGE=$(mktemp -d /tmp/pipeline.XXXXXX)
for f in finetune_voice.sh export_teacher.sh run_phase3_voice.sh export_bundle_voice.sh; do
  tr -d '\015' < "$SCRIPTS/$f" > "$STAGE/$f"
  chmod +x "$STAGE/$f"
done

note() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPE/status.txt" >&2; }

run_voice() {
  local voice=$1 lang=$2 gender=$3 teacher_desc=$4
  local mdir="$PIPE/$voice"
  mkdir -p "$mdir"

  if [ -f "$mdir/bundle.done" ]; then
    note "$voice: already complete, skipping"
    return 0
  fi

  if [ ! -f "$mdir/finetune.done" ]; then
    note "$voice: finetune start"
    if bash "$STAGE/finetune_voice.sh" "$voice" "$lang" "$gender" > "$LOGS/finetune_$voice.log" 2>&1; then
      touch "$mdir/finetune.done"
      note "$voice: finetune done"
    else
      note "$voice: FINETUNE FAILED (rc=$?) -- see $LOGS/finetune_$voice.log"
      return 1
    fi
  fi

  if [ ! -f "$mdir/teacher.done" ]; then
    note "$voice: teacher export start"
    if bash "$STAGE/export_teacher.sh" "$voice" "$lang" > "$LOGS/export_teacher_$voice.log" 2>&1; then
      touch "$mdir/teacher.done"
      note "$voice: teacher exported + auditions rendered"
    else
      note "$voice: TEACHER EXPORT FAILED (rc=$?) -- see $LOGS/export_teacher_$voice.log"
      return 1
    fi
  fi

  if [ ! -f "$mdir/distill.done" ]; then
    note "$voice: distill start"
    if bash "$STAGE/run_phase3_voice.sh" "$voice" "$lang" > "$LOGS/phase3_$voice.log" 2>&1; then
      touch "$mdir/distill.done"
      note "$voice: distill done"
    else
      note "$voice: DISTILL FAILED (rc=$?) -- see $LOGS/phase3_$voice.log"
      return 1
    fi
  fi

  if [ ! -f "$mdir/bundle.done" ]; then
    note "$voice: bundle export start"
    if bash "$STAGE/export_bundle_voice.sh" "$voice" "$lang" > "$LOGS/bundle_$voice.log" 2>&1; then
      touch "$mdir/bundle.done"
      note "$voice: bundle exported -> web/voices/$voice"
    else
      note "$voice: BUNDLE FAILED (rc=$?) -- see $LOGS/bundle_$voice.log"
      return 1
    fi
  fi
  return 0
}

evidence_voice() {
  local voice=$1 lang=$2 teacher_desc=$3
  local mdir="$PIPE/$voice"
  [ -f "$mdir/evidence.done" ] && return 0

  # render the reserved 16 tatoeba sentences through the wasm runtime
  local wavs="/tmp/$voice-eval"
  rm -rf "$wavs"
  "$HOME/venvs/saanotts/bin/python" - "$REPO" <<'PY'
import json, sys
from pathlib import Path
repo = Path(sys.argv[1])
rows = json.loads((repo / "artifacts/data/es_cl/tatoeba_eval16.json").read_text(encoding="utf-8"))
Path("/tmp/tatoeba16.json").write_text(json.dumps({"sentences": rows["sentences"]}, ensure_ascii=False), encoding="utf-8")
PY
  note "$voice: evidence render (wasm)"
  if ! PATH="$HOME/work/emsdk/node/24.19.0_64bit/bin:$PATH" \
       bash -c "cd '$REPO' && node mcu/ports/wasm/dump_voice_wavs.mjs '$voice' /tmp/tatoeba16.json '$wavs'" \
       > "$LOGS/evidence_render_$voice.log" 2>&1; then
    note "$voice: EVIDENCE RENDER FAILED -- see $LOGS/evidence_render_$voice.log"
    return 1
  fi

  note "$voice: evidence scoring (whisper)"
  if "$HOME/venvs/saanotts/bin/python" "$SCRIPTS/run_evidence_voice.py" \
      --voice "$voice" --lang "$lang" --teacher-desc "$teacher_desc" \
      --wavdir "$wavs" \
      --out "$REPO/experiments/evidence/$voice-tatoeba-$DATE_STAMP.json" \
      >> "$LOGS/evidence_$voice.log" 2>&1; then
    touch "$mdir/evidence.done"
    note "$voice: evidence done"
  else
    note "$voice: EVIDENCE FAILED -- see $LOGS/evidence_$voice.log"
    return 1
  fi
}

note "pipeline master start (stage dir $STAGE)"

run_voice copihue  es_CL female \
  "es_CL-copihue-medium (es_ES-sharvard-medium <- SLR71 clf_04310)"; \
evidence_voice copihue es_CL \
  "es_CL-copihue-medium (es_ES-sharvard-medium <- SLR71 clf_04310)"

run_voice vueltiao es_CO male \
  "es_CO-vueltiao-medium (es_ES-davefx-medium <- SLR72 com_06136)"; \
evidence_voice vueltiao es_CO \
  "es_CO-vueltiao-medium (es_ES-davefx-medium <- SLR72 com_06136)"

run_voice chande   es_CO female \
  "es_CO-chande-medium (es_ES-sharvard-medium <- SLR72 cof_02484)"; \
evidence_voice chande es_CO \
  "es_CO-chande-medium (es_ES-sharvard-medium <- SLR72 cof_02484)"

note "pipeline master done"
