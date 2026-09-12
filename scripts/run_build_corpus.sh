#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HOME/work/tatoeba"
if [ ! -s "$HOME/work/tatoeba/spa_sentences.tsv.bz2" ]; then
  curl -sfL --retry 3 -o "$HOME/work/tatoeba/spa_sentences.tsv.bz2" \
    "https://downloads.tatoeba.org/exports/per_language/spa/spa_sentences.tsv.bz2"
fi
if [ ! -s "$HOME/work/tatoeba/sentences_spa.tsv" ]; then
  bzcat "$HOME/work/tatoeba/spa_sentences.tsv.bz2" > "$HOME/work/tatoeba/sentences_spa.tsv"
fi
head -3 "$HOME/work/tatoeba/sentences_spa.tsv"
wc -l "$HOME/work/tatoeba/sentences_spa.tsv"
export HF_HOME="$HOME/hf"
exec "$HOME/venvs/saanotts/bin/python" /mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/es_cl/scripts/build_distill_corpus.py
