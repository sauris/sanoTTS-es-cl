#!/usr/bin/env bash
# Phase 4 attempt 3: rebuild the wasm data set wholesale from piper 1.8's
# espeak-ng-data, keeping the bundle's language selection (plus es-419).
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
WASM="$REPO/mcu/ports/wasm"
DATA="$WASM/espeak-data-multi"
SRC="$HOME/venvs/saanotts/lib/python3.11/site-packages/piper/espeak-ng-data"

# languages the bundle ships (same set as before, plus es-419)
KEEP_LANGS=(
  lang/aav/vi
  lang/gmw/de
  lang/gmw/en
  lang/gmw/en-029
  lang/gmw/en-GB-scotland
  lang/gmw/en-GB-x-gbclan
  lang/gmw/en-GB-x-gbcwmd
  lang/gmw/en-GB-x-rp
  lang/gmw/en-Shaw
  lang/gmw/en-US
  lang/gmw/en-US-nyc
  lang/inc/hi
  lang/inc/ne
  lang/poz/id
  lang/roa/es
  lang/roa/es-419
  lang/roa/fr
  lang/roa/it
  lang/roa/pt
  lang/roa/pt-BR
  lang/roa/ro
  lang/sem/ar
  lang/sit/cmn
  lang/trk/tr
  lang/zle/ru
  lang/zlw/cs
)
KEEP_DICTS=(ar_dict cmn_dict cs_dict de_dict en_dict es_dict fr_dict hi_dict id_dict it_dict ne_dict pt_dict ro_dict tr_dict vi_dict)

BACKUP="$WASM/espeak-data-multi.bak-1.4.2"
if [ ! -d "$BACKUP" ]; then
  cp -r "$DATA" "$BACKUP"
fi

rm -rf "$DATA.new"
mkdir -p "$DATA.new/lang"
for f in phondata phonindex phontab intonations; do
  cp "$SRC/$f" "$DATA.new/$f"
done
for l in "${KEEP_LANGS[@]}"; do
  mkdir -p "$DATA.new/$(dirname "$l")"
  cp "$SRC/$l" "$DATA.new/$l"
done
for d in "${KEEP_DICTS[@]}"; do
  cp "$SRC/$d" "$DATA.new/$d"
done
rm -rf "$DATA.old"
mv "$DATA" "$DATA.old"
mv "$DATA.new" "$DATA"
du -sh "$DATA"

source "$HOME/work/emsdk/emsdk_env.sh" >/dev/null
cd "$WASM"
./build_g2p.sh 2>&1 | grep -E "^built" || true
echo REBUILT_FULL_1_8
