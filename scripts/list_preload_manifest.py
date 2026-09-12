#!/usr/bin/env python3
"""List the preloaded files recorded inside web/snt_g2p.js (Emscripten metadata)."""
import re
from pathlib import Path

js = Path(r"C:\Users\kuco\Documents\dev\playgr\sanoTTS\web\snt_g2p.js").read_text(encoding="utf-8", errors="ignore")

# Emscripten embeds the preload manifest in various shapes depending on version.
patterns = [
    r'"(/?espeak[^"]*)"',
    r"'(/?espeak[^']*)'",
]
names = set()
for pat in patterns:
    for m in re.finditer(pat, js):
        names.add(m.group(1))
for name in sorted(names):
    print(name)
print(f"total: {len(names)}")
