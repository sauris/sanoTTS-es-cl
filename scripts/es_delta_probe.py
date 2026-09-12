#!/usr/bin/env python3
"""Phase 4 recon: how do es vs es-419 phoneme ID sequences differ on our corpus?

Phonemizes a sample of the distillation corpus with both espeak voices via
piper-tts and reports the per-codepoint deltas, so a JS-side remap fallback can
be judged: if a small id->id remap makes es output identical to es-419 output
on virtually every row, the browser can keep the `es` voice and remap.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from piper.phonemize_espeak import EspeakPhonemizer

REPO = Path(__file__).resolve().parents[3]
CORPUS = REPO / "artifacts/data/es_cl/distill_corpus.jsonl"
SAMPLE = 2000

def main() -> None:
    phon = EspeakPhonemizer()
    from piper.phoneme_ids import phonemes_to_ids

    rows = []
    with CORPUS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line)["text"])
    rows = rows[:SAMPLE]

    deltas: Counter[str] = Counter()
    rows_differing = 0
    mismatch_examples: list[str] = []
    for text in rows:
        try:
            ph_es = phon.phonemize("es", text)
            ph_419 = phon.phonemize("es-419", text)
        except Exception:
            continue
        ids_es = [tuple(phonemes_to_ids(p)) for p in ph_es if p]
        ids_419 = [tuple(phonemes_to_ids(p)) for p in ph_419 if p]
        if ids_es == ids_419:
            continue
        rows_differing += 1
        flat_es = [i for chunk in ids_es for i in chunk]
        flat_419 = [i for chunk in ids_419 for i in chunk]
        if len(flat_es) != len(flat_419):
            deltas["<length-mismatch>"] += 1
            if len(mismatch_examples) < 8:
                mismatch_examples.append(text)
            continue
        for a, b in zip(flat_es, flat_419):
            if a != b:
                deltas[f"{a}->{b}"] += 1
        if len(mismatch_examples) < 8:
            mismatch_examples.append(text)

    total = len(rows)
    print(f"sampled rows: {total}")
    print(f"rows differing: {rows_differing} ({rows_differing/total*100:.2f}%)")
    print("\nid deltas:")
    for delta, count in deltas.most_common(30):
        print(f"  {delta:12s} {count}")
    print("\nexamples:")
    for text in mismatch_examples:
        print(f"  - {text[:90]}")


if __name__ == "__main__":
    main()
