#!/usr/bin/env python3
"""Diagnose wasm-vs-piper id mismatches at phoneme level for one sentence."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
rows = json.loads((REPO / "artifacts/es_cl/g2p_truth.json").read_text(encoding="utf-8"))
text = sys.argv[1] if len(sys.argv) > 1 else "Voy a ir a correr con mi amiga Pepita."
row = next(r for r in rows if r["text"] == text)

# reverse id map from the teacher config
cfg = json.loads((REPO / "artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx.json").read_text(encoding="utf-8"))
id_to_sym = {}
for sym, ids in cfg["phoneme_id_map"].items():
    for i in ids:
        id_to_sym[int(i)] = sym

node_script = r"""
const { resolve } = require("node:path");
(async () => {
  const web = resolve(process.argv[1], "web");
  const SaanoG2P = (await import(resolve(web, "snt_g2p.js"))).default;
  const Mod = await SaanoG2P({ locateFile: (f) => resolve(web, f) });
  const g2p = Mod.cwrap("snt_g2p_text_to_ids", "number", ["number", "number", "number"]);
  const setVoice = Mod.cwrap("snt_g2p_set_voice", "number", ["string", "number"]);
  const MAX = 512;
  const outP = Mod._malloc(MAX * 4);
  function ids(text) {
    const n = Mod.lengthBytesUTF8(text) + 1;
    const p = Mod._malloc(n);
    Mod.stringToUTF8(text, p, n);
    const k = g2p(p, outP, MAX);
    Mod._free(p);
    return Array.from(Mod.HEAP32.subarray(outP >> 2, (outP >> 2) + k));
  }
  setVoice(process.argv[3], 11);
  console.log(JSON.stringify(ids(process.argv[2])));
})();
"""
result = subprocess.run(
    [
        "/home/kuco/work/emsdk/node/24.19.0_64bit/bin/node",
        str(REPO / "artifacts/es_cl/scripts/wasm_ids.mjs"),
        str(REPO),
        text,
        sys.argv[2] if len(sys.argv) > 2 else "es-419",
    ],
    capture_output=True,
    text=True,
    cwd=str(REPO),
)
if result.returncode != 0 or not result.stdout.strip():
    print("NODE STDERR:", result.stderr[-3000:])
    print("NODE RC:", result.returncode)
    sys.exit(1)
wasm_ids = json.loads(result.stdout.strip().splitlines()[-1])
py_ids = row["ids_es419"] if (len(sys.argv) <= 2 or sys.argv[2] == "es-419") else row["ids_es"]

def sym_seq(ids):
    out = []
    for i in ids:
        if i in (0, 1, 2):
            continue  # pad/bos/eos
        out.append(id_to_sym.get(i, f"?{i}"))
    return out

w = sym_seq(wasm_ids)
p = sym_seq(py_ids)
print("text:", text)
print("wasm :", " ".join(w))
print("piper:", " ".join(p))
if len(w) == len(p):
    for a, b in zip(w, p):
        marker = "  " if a == b else "!!"
        if a != b:
            print(f"{marker} {a!r} vs {b!r}")
else:
    print(f"lengths: wasm {len(w)} vs piper {len(p)}")
