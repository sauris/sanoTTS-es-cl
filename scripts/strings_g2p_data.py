#!/usr/bin/env python3
"""Extract printable strings near the lang/ region of snt_g2p.data."""
from pathlib import Path

data = Path(r"C:\Users\kuco\Documents\dev\playgr\sanoTTS\web\snt_g2p.data").read_bytes()
text = data.decode("latin-1")

# collect strings >= 4 chars
strings = []
start = None
for i, ch in enumerate(text):
    if 32 <= ord(ch) < 127 or ch in "/._-":
        if start is None:
            start = i
    else:
        if start is not None and i - start >= 4:
            strings.append((start, text[start:i]))
        start = None

lang_strings = [s for pos, s in strings if "lang/" in s or s.startswith("roa/") or "/es" in s or "es-" in s]
for s in lang_strings[:60]:
    print(repr(s))
print("...")
# also grep for voice-ish tokens containing 'es'
es_like = sorted({s for _, s in strings if s in {"es", "es-419", "es-es", "es-la", "es-mx", "espeak", "en-us", "en", "vi", "cmn"}})
print("exact hits:", es_like)
