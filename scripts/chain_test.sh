#!/usr/bin/env bash
# Dry-run the entire Phase 3 toolchain against the davefx test export:
# pack -> decoder cut -> parity (our reimplementation) -> signature pack (ours).
set -euo pipefail
REPO=/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS
export PYTHONPATH="$REPO/tools"
PY="$HOME/venvs/saanotts/bin/python"
TEACHER="$HOME/work/base_ckpts/es_ES/davefx/medium/davefx-medium.onnx"
TEACHER_JSON="$HOME/work/base_ckpts/es_ES/davefx/medium/config.json"
OUT="$REPO/artifacts/es_cl/chain-test"
mkdir -p "$OUT"

# tiny corpus: first 16 eval sentences
"$HOME/venvs/saanotts/bin/python" - <<'EOF'
import json
from pathlib import Path
src = Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/data/es_cl/eval.txt")
out = Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/es_cl/chain-test/corpus16.jsonl")
with out.open("w", encoding="utf-8") as f:
    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines()[:16]):
        if line.strip():
            f.write(json.dumps({"id": i, "text": line.strip()}, ensure_ascii=False) + "\n")
print("corpus rows:", sum(1 for _ in out.open(encoding="utf-8")))
EOF

echo "== 1. probe pack (decoder mode, 12 rows) =="
"$PY" "$REPO/tools/build_piper_vits_roota_probe_pack.py" \
  --model "$TEACHER" --config "$TEACHER_JSON" \
  --source-jsonl "$OUT/corpus16.jsonl" \
  --out-dir "$OUT/smoke12" \
  --max-rows 12 --allow-text-only-source --tensor-mode decoder \
  --noise-scale 0 --length-scale 1 --noise-w 0 2>&1 | tail -3

echo "== 2. decoder cut =="
"$PY" "$REPO/tools/extract_piper_vits_decoder_cut.py" \
  --model "$TEACHER" --pack-dir "$OUT/smoke12" --out-dir "$OUT/decoder-cut" 2>&1 | tail -5

echo "== 3. parity (reimplemented tool) =="
"$PY" "$REPO/tools/verify_piper_decoder_torch_parity.py" \
  --decoder "$OUT/decoder-cut/davefx-medium-decoder-from-generator-input.onnx" \
  --pack-dir "$OUT/smoke12" \
  --out-dir "$OUT/parity" --rows 12 --export-checkpoint 2>&1 | tail -25

echo "== 4. signature pack (reimplemented tool) =="
"$PY" "$REPO/tools/build_piper_vits_decoder_signature_pack.py" \
  --model "$OUT/decoder-cut/davefx-medium-decoder-from-generator-input.onnx" \
  --pack-dir "$OUT/smoke12" \
  --out-dir "$OUT/signatures" \
  --feed-latent-from-pack --dtype float16 2>&1 | tail -25

echo "CHAIN_TEST_DONE"
