#!/usr/bin/env bash
set -euo pipefail
for u in \
  "https://downloads.tatoeba.org/exports/per_language/spa/spa_sentences_detailed.tsv.bz2" \
  "https://downloads.tatoeba.org/exports/per_language/spa/spa_sentences.tsv.bz2" \
  "https://downloads.tatoeba.org/exports/sentences_detailed.csv.bz2" \
  "https://downloads.tatoeba.org/sentences.tar.bz2" \
  "https://downloads.tatoeba.org/exports/sentences.csv.bz2"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -I "$u" || true)
  echo "$code $u"
done
