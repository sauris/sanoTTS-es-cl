#!/usr/bin/env python3
"""Phase 1, steps 2-5: export speaker clm_02121 to the Piper fine-tune layout.

- artifacts/data/es_cl/audio/*.wav   22050 Hz mono 16-bit
- artifacts/data/es_cl/metadata.csv  utt_id|text  (piper1-gpl single-speaker format)
- artifacts/data/es_cl/eval.txt      16 held-out utterances (never trained on)
- artifacts/data/es_cl/ATTRIBUTION.txt  SLR71 / CC BY-SA 4.0 record

Cleanup is deliberately minimal: strip whitespace, drop empty/undecodable
audio and utterances over 300 chars, and add the missing opening ¿ before
question segments (espeak uses it for prosody).
"""
from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "artifacts/data/es_cl"
AUDIO_DIR = OUT_DIR / "audio"

SPEAKER_ID = 2121
TARGET_SR = 22050
MAX_CHARS = 300
NUM_EVAL = 16
SEED = 20260912

ATTRIBUTION = """Google Chilean Spanish Low-resource Speech (OpenSLR SLR71)
https://openslr.org/71

Citation (LREC 2020):
J. M. Martin-Valdivia, E. Ufimtseva, A. V. Parkhilko, A. Y. Zhila,
D. Gimeno-Gomez, "SLR71: Google Crowdsource Multilingual Speech Corpus",
Language Resources and Evaluation Conference (LREC 2020), Marseille, 2020.

Copyright 2018, 2019 Google, Inc.
License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
https://creativecommons.org/licenses/by-sa/4.0/

This file records attribution for every artifact in this directory and for
all derived models (es_CL-huemul-medium Piper fine-tune, distilled sanoTTS
voice). Derived artifacts must carry this attribution and remain compatible
with CC BY-SA 4.0 (sanoTTS training/G2P side is GPLv3; both permit
attribution + share-alike redistribution).

Selected subset: male speaker clm_02121 (146 utterances, ~18.8 min at 48 kHz).
"""

QUESTION_FIX = re.compile(r"([^.!?\n]*\?)")


def fix_opening_question(text: str) -> str:
    """Add the missing ¿ at the start of each question segment."""

    def _add(m: re.Match) -> str:
        seg = m.group(1)
        return "¿" + seg

    if "¿" in text:
        return text
    return QUESTION_FIX.sub(_add, text)


def clean_text(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if len(text) > MAX_CHARS:
        return None
    text = fix_opening_question(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("ylacombe/google-chilean-spanish", "male", split="train")
    paths = ds.cast_column("audio", Audio(decode=False))["audio"]
    rows = []
    for i in range(len(ds)):
        row = ds[i]
        if int(row["speaker_id"]) != SPEAKER_ID:
            continue
        text = clean_text(row["text"])
        if text is None:
            print(f"skip (text): {row['text'][:60]!r}")
            continue
        audio = row["audio"]
        sr = audio["sampling_rate"]
        wav = np.asarray(audio["array"], dtype=np.float32)
        if wav.size == 0 or not np.isfinite(wav).all():
            print("skip (audio)")
            continue
        if sr != TARGET_SR:
            # 48000 -> 22050 = 147/320 after gcd
            from math import gcd

            g = gcd(TARGET_SR, sr)
            wav = resample_poly(wav, TARGET_SR // g, sr // g).astype(np.float32)
        name = Path(paths[i]["path"]).stem
        sf.write(AUDIO_DIR / f"{name}.wav", np.clip(wav, -1.0, 1.0), TARGET_SR, subtype="PCM_16")
        rows.append((name, text))

    rng = random.Random(SEED)
    rng.shuffle(rows)
    eval_rows = rows[:NUM_EVAL]
    train_rows = sorted(rows[NUM_EVAL:])

    with (OUT_DIR / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", lineterminator="\n")
        for name, text in train_rows:
            writer.writerow([name, text])

    (OUT_DIR / "eval.txt").write_text(
        "\n".join(text for _, text in sorted(eval_rows)) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")

    durs = [(AUDIO_DIR / f"{n}.wav") for n, _ in train_rows]
    total = sum(sf.info(p).duration for p in durs)
    print(f"train utts: {len(train_rows)}  ({total/60:.1f} min @ {TARGET_SR} Hz)")
    print(f"eval utts:  {len(eval_rows)}")
    print(f"audio dir:  {AUDIO_DIR}")


if __name__ == "__main__":
    main()
