// Run the page's own loadVoice/splitChunks against :8178, synth, dump wav.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));   // demo/
await import(pathToFileURL("/tmp/page_funcs.mjs").href);
const { loadVoice, splitChunks } = globalThis.__page;

const SR = 22050, MAX_IDS = 1024, OUT_CAP = 22050 * 30;
const key = process.argv[2] || "chilean-small-int8";
const text = process.argv[3] || "¡Ya po! Soy una voz chilena que vive en tu navegador, ¿cachái?";

const SaanoVoice = (await import(pathToFileURL(resolve(here, "snt_voice.js")).href)).default;
const SaanoG2P = (await import(pathToFileURL(resolve(here, "snt_g2p.js")).href)).default;
const V = await SaanoVoice();
const G = await SaanoG2P({ locateFile: (f) => resolve(here, f) });

const setVoice = G.cwrap("snt_g2p_set_voice", "number", ["string", "number"]);
const g2pIds = G.cwrap("snt_g2p_text_to_ids", "number", ["number", "number", "number"]);
const synth = V.cwrap("snt_voice_synthesize", "number",
  ["number", "number", "number", "number", "number", "number", "number"]);

const bundle = await loadVoice(key);
console.log("loaded", key, "| weights:", bundle.meta.weights, "| length_scale:", bundle.meta.length_scale);

const rc = setVoice(bundle.meta.espeak_voice, bundle.meta.g2p_voice_slot);
console.log("set_voice rc:", rc);

const chunks = splitChunks(text);
console.log("chunks:", JSON.stringify(chunks));
const pcms = [];
for (const c of chunks) {
  const nBytes = G.lengthBytesUTF8(c) + 1;
  const textP = G._malloc(nBytes);
  G.stringToUTF8(c, textP, nBytes);
  const outP = G._malloc(MAX_IDS * 4);
  const nIds = g2pIds(textP, outP, MAX_IDS);
  G._free(textP);
  const ids = Int32Array.from(G.HEAP32.subarray(outP >> 2, (outP >> 2) + nIds));
  G._free(outP);
  console.log("  chunk ids:", nIds);

  const idsBytes = new Uint8Array(ids.buffer, ids.byteOffset, ids.byteLength);
  const idsP = V._malloc(idsBytes.length);
  V.HEAPU8.set(idsBytes, idsP);
  const frontP = V._malloc(bundle.front.length); V.HEAPU8.set(bundle.front, frontP);
  const decP = V._malloc(bundle.dec.length); V.HEAPU8.set(bundle.dec, decP);
  const outCapP = V._malloc(OUT_CAP * 4);
  const n = synth(frontP, decP, idsP, ids.length, bundle.meta.length_scale, outCapP, OUT_CAP);
  [frontP, decP, idsP, outCapP].forEach((p) => V._free(p));
  if (n <= 0) throw new Error("synth failed " + n);
  pcms.push(V.HEAPF32.slice(outCapP >> 2, (outCapP >> 2) + n));
}

const total = pcms.reduce((a, p) => a + p.length, 0);
const pcm = new Float32Array(total);
let off = 0;
for (const p of pcms) { pcm.set(p, off); off += p.length; }

const buf = new ArrayBuffer(44 + total * 2);
const dv = new DataView(buf);
const wstr = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
wstr(0, "RIFF"); dv.setUint32(4, 36 + total * 2, true); wstr(8, "WAVE");
wstr(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
dv.setUint32(24, SR, true); dv.setUint32(28, SR * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
wstr(36, "data"); dv.setUint32(40, total * 2, true);
for (let k = 0; k < total; k++) {
  const v = Math.max(-1, Math.min(1, pcm[k]));
  dv.setInt16(44 + k * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
}
mkdirSync("/tmp/page-repro", { recursive: true });
writeFileSync("/tmp/page-repro/page-flow.wav", Buffer.from(buf));
console.log("wrote /tmp/page-repro/page-flow.wav");
