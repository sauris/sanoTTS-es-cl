// cdp_speak.cjs -- drive the real demo speak() path: navigate with
// ?autoplay=1, wait for the player to have audio, pull it out as base64,
// save to disk. Usage: node cdp_speak.cjs <voice> <out-wav> [text]
const fs = require("fs");

const [selfVoice, outWav, text] = process.argv.slice(2);
const PORT = 9222;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
  const page = list.find(t => t.type === "page");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let msgId = 0;
  const pending = new Map();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  };
  const send = (method, params = {}) => new Promise((res) => {
    const id = ++msgId;
    pending.set(id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });

  await send("Page.enable");
  const q = `voice=${encodeURIComponent(selfVoice)}&autoplay=1` + (text ? `&text=${encodeURIComponent(text)}` : "");
  await send("Page.navigate", { url: `http://localhost:8178/?${q}` });

  let src = null;
  for (let i = 0; i < 120; i++) {
    await sleep(1000);
    const r = await send("Runtime.evaluate", {
      expression: `(function(){ const p = document.getElementById("player"); return p && p.src && p.src.startsWith("blob:") ? p.src : null; })()`,
      returnByValue: true,
    });
    if (r.result && r.result.result && r.result.result.value) { src = r.result.result.value; break; }
  }
  if (!src) throw new Error("player never got audio");
  await sleep(2000); // let any pending status settle

  const b64 = await send("Runtime.evaluate", {
    expression: `(async () => { const b = await (await fetch(document.getElementById("player").src)).blob(); const u = new Uint8Array(await b.arrayBuffer()); let bin = ""; for (const x of u) bin += String.fromCharCode(x); return btoa(bin); })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  const bytes = Buffer.from(b64.result.result.value, "base64");
  fs.writeFileSync(outWav, bytes);
  console.log("saved", outWav, bytes.length, "bytes");
  ws.close();
}

main().catch((e) => { console.error("CDP FAILED:", e.message); process.exit(1); });
