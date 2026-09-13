#!/usr/bin/env bash
# Download the SLR72 es_CO zips + line indexes into ~/work/es_co (resumable).
set -euo pipefail
DIR="$HOME/work/es_co"
mkdir -p "$DIR"
cd "$DIR"
for f in line_index_female.tsv line_index_male.tsv; do
  [ -s "$f" ] || curl -sSL -o "$f" "https://openslr.trmal.net/resources/72/$f"
done
for z in es_co_female.zip es_co_male.zip; do
  if [ ! -s "$z" ]; then
    curl -sSL -C - -o "$z.part" "https://openslr.trmal.net/resources/72/$z"
    mv "$z.part" "$z"
  fi
done
echo ALL_DOWNLOADS_DONE
ls -la "$DIR"
