#!/usr/bin/env python3
"""Inspect clause splitting of piper 1.8 EspeakPhonemizer for the failing text."""
from piper.phonemize_espeak import EspeakPhonemizer
import unicodedata

p = EspeakPhonemizer()
for text in ["¡No te preocupes, sé feliz!", "Dicen que María estuvo enferma, pero ahora tiene buen aspecto."]:
    for voice in ("es-419", "es"):
        raw = p.phonemize(voice, text)
        print(f"{voice}: {text!r}")
        print(f"  sentences: {len(raw)}")
        # also dump the clause-level internals
        from piper import espeakbridge
        espeakbridge.set_voice(voice)
        clauses = espeakbridge.get_phonemes(text)
        for i, (ph, term, eos) in enumerate(clauses):
            print(f"  clause {i}: ph={ph!r} term={term!r} eos={eos}")
