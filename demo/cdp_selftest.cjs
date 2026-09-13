// CDP driver: open the demo selftest in headless Edge/Chrome, poll for
// window.__SELFTEST, save the report + wav to disk.
//   node cdp_selftest.mjs <selftest-voice> <out-json> <out-wav>
const fs = require("fs");
const path = require("path");

const [selfVoice, outJson, outWav] = process.argv.slice(2);
const PORT = 9222;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getTargets() {
  const r = await fetch(`http://127.0.0.1:${PORT}/json/list`);
  return r.json();
}

async function main() {
  // find (or create) the page target
  let targets = await getTargets();
  let page = targets.find(t => t.type === "page");
  if (!page) throw new Error("no page target");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  let msgId = 0;
  const pending = new Map();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  };
  const send = (method, params = {}) => new Promise((res) => {
    const id = ++msgId;
    pending.set(id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });

  await send("Page.enable");
  const url = `http://localhost:8178/?selftest=${encodeURIComponent(selfVoice)}`;
  await send("Page.navigate", { url });

  // poll for the report (module init + wasm + synth can take a while)
  let report = null;
  for (let i = 0; i < 90; i++) {
    await sleep(1000);
    const r = await send("Runtime.evaluate", { expression: "window.__SELFTEST ? JSON.stringify(window.__SELFTEST) : null", returnByValue: true });
    if (r.result && r.result.result && r.result.result.value) {
      report = JSON.parse(r.result.result.value);
      break;
    }
  }
  ws.close();
  if (!report) throw new Error("selftest never produced a report (timeout)");
  fs.writeFileSync(outJson, JSON.stringify(report, null, 1));
  if (report.wavB64) fs.writeFileSync(outWav, Buffer.from(report.wavB64, "base64"));
  console.log(JSON.stringify({ ...report, wavB64: undefined, idsChunk0: undefined }, null, 1));
}

main().catch((e) => { console.error("CDP FAILED:", e.message); process.exit(1); });
