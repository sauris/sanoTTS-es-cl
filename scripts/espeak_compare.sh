#!/usr/bin/env bash
# Compare es vs es-419 phonemization (system espeak-ng AND piper's bundled espeak data),
# and list piper's bundled Spanish voices. Informs the Phase 4 g2p fallback.
set -euo pipefail

echo "== system espeak-ng =="
espeak-ng --version

for v in es es-419; do
  echo "--- voice: $v ---"
  for s in "ciencias" "zapato" "gracias" "lluvia" "calle" "yo" "hospital"; do
    printf '%-12s' "$s"
    espeak-ng -v "$v" -q --ipa "$s" 2>/dev/null
  done
done

echo
echo "== piper bundled espeak-ng-data =="
ls "$HOME/work/piper1-gpl/src/piper/espeak-ng-data/lang/roa/" 2>/dev/null || true
ls "$HOME/venvs/saanotts/lib/python3.11/site-packages/piper/espeak-ng-data/lang/roa/" 2>/dev/null || true
