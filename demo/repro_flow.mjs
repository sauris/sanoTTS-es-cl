// Replicate demo/index.html's exact flow against the demo/ directory and
// dump the resulting PCM, so node-side can be compared with the browser.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));   // the demo/ dir
const repo = resolve(here, "..");
const web = here;
const demo = here;

const SR = 22050, MAX_IDS = 1024, OUT_CAP = 22050 * 30;
const key = "chilean-small-int8";
const text = "¡Ya po! Soy una voz chilena que vive en tu navegador, ¿cachái?";

const SaanoVoice = (await import(resolve(demo, "snt_voice.js"))).default;
const SaanoG2P = (await import(resolve(demo, "snt_g2p.js"))).default;
const V = await SaanoVoice();
const G = await SaanoG2P({ locateFile: (f) => resolve(demo, f) });

const setVoice = G.cwrap("snt_g2p_set_voice", "number", ["string", "number"]);
const g2pIds = G.cwrap("snt_g2p_text_to_ids", "number", ["number", "number", "number"]);
const synth = V.cwrap("snt_voice_synthesize", "number",
  ["number", "number", "number", "number", "number", "number", "number"]);

const meta = JSON.parse(readFileSync(resolve(demo, "voices", key, "meta.json"), "utf8"));

function widen(which) {
  const fmt = meta.weights || "f32";
  const bytes = new Uint8Array(readFileSync(resolve(demo, "voices", key, meta[which])));
  const dims = which === "front" ? meta.front_dims : meta.dec_dims;
  const head = Number(dims.meta_bytes), n = Number(dims.weight_floats);
  if (fmt === "f32") return bytes;
  if (fmt === "f16") {
    const out = new Uint8Array(head + 4 * n);
    out.set(bytes.subarray(0, head), 0);
    const src = new DataView(bytes.buffer, bytes.byteOffset + head, 2 * n);
    const dst = new DataView(out.buffer, head, 4 * n);
    for (let i = 0; i < n; i++) {
      const b = src.getUint16(i * 2, true);
      const exp = (b >> 10) & 0x1f, frac = b & 0x3ff, sign = (b & 0x8000) ? -1 : 1;
      let v;
      if (exp === 0) v = sign * frac * 5.960464477539063e-8;
      else if (exp === 31) v = frac ? NaN : sign * Infinity;
      else v = sign * Math.pow(2, exp - 25) * (1024 + frac);
      dst.setFloat32(i * 4, v, true);
    }
    return out;
  }
  const scales = which === "front" ? meta.front_scales : meta.dec_scales;
  const sizes = dims.sizes;
  const out = new Uint8Array(head + 4 * n);
  out.set(bytes.subarray(0, head), 0);
  const src = new Int8Array(bytes.buffer, bytes.byteOffset + head, n);
  const dst = new DataView(out.buffer, head, 4 * n);
  let off = 0;
  for (let t = 0; t < sizes.length; t++) {
    for (let j = 0; j < sizes[t]; j++) dst.setFloat32((off + j) * 4, scales[t] * src[off + j], true);
    off += sizes[t];
  }
  return out;
}

const front = widen("front");
const dec = widen("dec");

const rc = setVoice(meta.espeak_voice, meta.g2p_voice_slot);
console.log("set_voice rc:", rc);

const idsArr = [];
{
  const nBytes = G.lengthBytesUTF8(text) + 1;
  const textP = G._malloc(nBytes);
  G.stringToUTF8(text, textP, nBytes);
  const outP = G._malloc(MAX_IDS * 4);
  const n = g2pIds(textP, outP, MAX_IDS);
  G._free(textP);
  console.log("ids:", n);
  idsArr.push(...G.HEAP32.subarray(outP >> 2, (outP >> 2) + n));
  G._free(outP);
}

const ids = Int32Array.from(idsArr);
const idsBytes = new Uint8Array(ids.buffer, ids.byteOffset, ids.byteLength);
const idsP = V._malloc(idsBytes.length);
V.HEAPU8.set(idsBytes, idsP);
const frontP = V._malloc(front.length); V.HEAPU8.set(front, frontP);
const decP = V._malloc(dec.length); V.HEAPU8.set(dec, decP);
const outP = V._malloc(OUT_CAP * 4);
const n = synth(frontP, decP, idsP, ids.length, meta.length_scale, outP, OUT_CAP);
console.log("samples:", n);
const pcm = V.HEAPF32.slice(outP >> 2, (outP >> 2) + n);

// write wav
const buf = new ArrayBuffer(44 + n * 2);
const dv = new DataView(buf);
const wstr = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
wstr(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); wstr(8, "WAVE");
wstr(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
dv.setUint32(24, SR, true); dv.setUint32(28, SR * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
wstr(36, "data"); dv.setUint32(40, n * 2, true);
for (let i = 0; i < n; i++) {
  const v = Math.max(-1, Math.min(1, pcm[i]));
  dv.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
}
mkdirSync("/tmp/demo-repro", { recursive: true });
writeFileSync("/tmp/demo-repro/yapo-i8.wav", Buffer.from(buf));
console.log("wrote /tmp/demo-repro/yapo-i8.wav");
