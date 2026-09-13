#!/usr/bin/env bash
# Small-variant pipeline for the three new voices (unattended, resumable).
# Per voice: small distill (reusing main-run packs) -> bundle -> evidence.
# Done-files under artifacts/es_cl/pipeline/<voice>-small/.
# Detached:  setsid nohup bash run_small_pipeline.sh &
set -uo pipefail

REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
SCRIPTS=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS-es-cl/scripts
LOGS=$REPO/artifacts/es_cl/logs
PIPE=$REPO/artifacts/es_cl/pipeline
DATE_STAMP=20260913
export HF_HOME="$HOME/hf"

mkdir -p "$LOGS" "$PIPE"
: > "$PIPE/status-small.txt"

STAGE=$(mktemp -d /tmp/pipeline-small.XXXXXX)
for f in run_phase3_small_voice.sh export_bundle_voice.sh; do
  tr -d '\015' < "$SCRIPTS/$f" > "$STAGE/$f"
  chmod +x "$STAGE/$f"
done

note() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPE/status-small.txt" >&2; }

run_small() {
  local voice=$1 lang=$2
  local mdir="$PIPE/$voice-small"
  mkdir -p "$mdir"

  if [ -f "$mdir/evidence.done" ]; then
    note "$voice-small: already complete, skipping"
    return 0
  fi

  if [ ! -f "$mdir/distill.done" ]; then
    note "$voice-small: distill start"
    if bash "$STAGE/run_phase3_small_voice.sh" "$voice" "$lang" > "$LOGS/phase3_small_$voice.log" 2>&1; then
      touch "$mdir/distill.done"
      note "$voice-small: distill done"
    else
      note "$voice-small: DISTILL FAILED (rc=$?) -- see $LOGS/phase3_small_$voice.log"
      return 1
    fi
  fi

  if [ ! -f "$mdir/bundle.done" ]; then
    note "$voice-small: bundle export start"
    if bash "$STAGE/export_bundle_voice.sh" "$voice" "$lang" small > "$LOGS/bundle_small_$voice.log" 2>&1; then
      touch "$mdir/bundle.done"
      note "$voice-small: bundle exported -> web/voices/$voice-small"
    else
      note "$voice-small: BUNDLE FAILED (rc=$?) -- see $LOGS/bundle_small_$voice.log"
      return 1
    fi
  fi

  local mdir_e="$PIPE/$voice-small"
  if [ ! -f "$mdir_e/evidence.done" ]; then
    local wavs="/tmp/$voice-small-eval"
    rm -rf "$wavs"
    "$HOME/venvs/saanotts/bin/python" - "$REPO" <<'PY'
import json, sys
from pathlib import Path
repo = Path(sys.argv[1])
rows = json.loads((repo / "artifacts/data/es_cl/tatoeba_eval16.json").read_text(encoding="utf-8"))
Path("/tmp/tatoeba16.json").write_text(json.dumps({"sentences": rows["sentences"]}, ensure_ascii=False), encoding="utf-8")
PY
    note "$voice-small: evidence render (wasm)"
    if ! PATH="$HOME/work/emsdk/node/24.19.0_64bit/bin:$PATH" \
         bash -c "cd '$REPO' && node mcu/ports/wasm/dump_voice_wavs.mjs '$voice-small' /tmp/tatoeba16.json '$wavs'" \
         > "$LOGS/evidence_render_${voice}_small.log" 2>&1; then
      note "$voice-small: EVIDENCE RENDER FAILED -- see $LOGS/evidence_render_${voice}_small.log"
      return 1
    fi
    note "$voice-small: evidence scoring (whisper)"
    if "$HOME/venvs/saanotts/bin/python" "$SCRIPTS/run_evidence_voice.py" \
        --voice "$voice-small" --lang "$lang" \
        --teacher-desc "$LANG-$voice-small (340k student distilled from $LANG-$voice-medium)" \
        --wavdir "$wavs" \
        --out "$REPO/experiments/evidence/$voice-small-tatoeba-$DATE_STAMP.json" \
        >> "$LOGS/evidence_${voice}_small.log" 2>&1; then
      touch "$mdir_e/evidence.done"
      note "$voice-small: evidence done"
    else
      note "$voice-small: EVIDENCE FAILED -- see $LOGS/evidence_${voice}_small.log"
      return 1
    fi
  fi
  return 0
}

note "small pipeline master start (stage dir $STAGE)"

run_small copihue  es_CL
run_small vueltiao es_CO
run_small chande   es_CO

note "small pipeline master done"
