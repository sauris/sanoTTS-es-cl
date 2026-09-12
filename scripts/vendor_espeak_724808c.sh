#!/usr/bin/env bash
# Phase 4 attempt 4: vendor espeak-ng @724808c (piper1-gpl's exact version)
# into mcu/ports/wasm/espeak so the wasm translator matches python piper 1.8.
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
WASM="$REPO/mcu/ports/wasm"
ESP="$WASM/espeak"
WORK="$HOME/work/espeak-ng-724808c"

if [ ! -d "$WORK" ]; then
  git clone https://github.com/espeak-ng/espeak-ng.git "$WORK"
  (cd "$WORK" && git checkout 724808c && git submodule update --init --recursive)
fi
(cd "$WORK" && git rev-parse --short HEAD)

LIB_FILES="common.c mnemonics.c error.c ieee80.c
  compiledata.c compiledict.c
  dictionary.c encoding.c intonation.c
  langopts.c numbers.c phoneme.c
  phonemelist.c readclause.c setlengths.c
  soundicon.c spect.c ssml.c
  synthdata.c synthesize.c tr_languages.c
  translate.c translateword.c voices.c
  wavegen.c speech.c espeak_api.c"
UCD_FILES="case.c categories.c ctype.c proplist.c scripts.c tostring.c"

# back up the old vendored tree once
if [ ! -d "$ESP/libespeak-ng.bak" ]; then
  cp -r "$ESP/libespeak-ng" "$ESP/libespeak-ng.bak"
  cp -r "$ESP/ucd-tools" "$ESP/ucd-tools.bak"
  cp -r "$ESP/include" "$ESP/include.bak" 2>/dev/null || true
fi

# wipe and re-vendor sources + headers (keep config.h)
rm -rf "$ESP/libespeak-ng" "$ESP/ucd-tools"
mkdir -p "$ESP/libespeak-ng" "$ESP/ucd-tools"
for f in $LIB_FILES; do cp "$WORK/src/libespeak-ng/$f" "$ESP/libespeak-ng/"; done
cp "$WORK/src/libespeak-ng/"*.h "$ESP/libespeak-ng/"
for f in $UCD_FILES; do cp "$WORK/src/ucd-tools/src/$f" "$ESP/ucd-tools/"; done
mkdir -p "$ESP/ucd-tools/include/ucd"
cp "$WORK/src/ucd-tools/src/include/ucd/"*.h "$ESP/ucd-tools/include/ucd/"
cp "$WORK/src/ucd-tools/src/include/ucd/ucd.h" "$ESP/ucd-tools/ucd.h" 2>/dev/null || true

# refresh public headers used by the shim
mkdir -p "$ESP/include/espeak-ng"
cp "$WORK/src/include/espeak-ng/"*.h "$ESP/include/espeak-ng/" 2>/dev/null || true
mkdir -p "$ESP/include/espeak"
cp "$WORK/src/include/espeak/"*.h "$ESP/include/espeak/" 2>/dev/null || true

ls "$ESP" | head
echo VENDORED_724808c
