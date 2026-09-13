#!/usr/bin/env python3
"""Export an SLR71 (google-chilean-spanish) speaker to the Piper fine-tune layout.

Generalization of export_speaker.py (huemul, clm_02121):
  python export_slr71_speaker.py --voice copihue --config female --speaker 4310

- artifacts/data/<voice>/audio/*.wav   22050 Hz mono 16-bit
- artifacts/data/<voice>/metadata.csv  utt_id|text  (piper1-gpl single-speaker format)
- artifacts/data/<voice>/eval.txt      16 held-out utterances (never trained on)
- artifacts/data/<voice>/ATTRIBUTION.txt  SLR71 / CC BY-SA 4.0 record
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly

REPO = Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS")

TARGET_SR = 22050
MAX_CHARS = 300
NUM_EVAL = 16
SEED = 20260912

ATTRIBUTION_TMPL = """Google Chilean Spanish Low-resource Speech (OpenSLR SLR71)
https://openslr.org/71

Citation (LREC 2020):
J. M. Martin-Valdivia, E. Ufimtseva, A. V. Parkhilko, A. Y. Zhila,
D. Gimeno-Gomez, "SLR71: Google Crowdsource Multilingual Speech Corpus",
Language Resources and Evaluation Conference (LREC 2020), Marseille, 2020.

Copyright 2018, 2019 Google, Inc.
License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
https://creativecommons.org/licenses/by-sa/4.0/

This file records attribution for every artifact in this directory and for
all derived models (es_CL-{voice}-medium Piper fine-tune, distilled sanoTTS
voice). Derived artifacts must carry this attribution and remain compatible
with CC BY-SA 4.0 (sanoTTS training/G2P side is GPLv3; both permit
attribution + share-alike redistribution).

Selected subset: {gender} speaker {gender_prefix}_{speaker_id:05d} ({stats}).
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True)
    parser.add_argument("--config", required=True, choices=("female", "male"))
    parser.add_argument("--speaker", required=True, type=int)
    parser.add_argument("--minutes", required=True, type=float)
    parser.add_argument("--utts", required=True, type=int)
    args = parser.parse_args()

    gender_prefix = "clf" if args.config == "female" else "clm"
    out_dir = REPO / f"artifacts/data/{args.voice}"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ylacombe/google-chilean-spanish", args.config, split="train")
    paths = ds.cast_column("audio", Audio(decode=False))["audio"]
    rows = []
    for i in range(len(ds)):
        row = ds[i]
        if int(row["speaker_id"]) != args.speaker:
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
        sf.write(audio_dir / f"{name}.wav", np.clip(wav, -1.0, 1.0), TARGET_SR, subtype="PCM_16")
        rows.append((name, text))

    rng = random.Random(SEED)
    rng.shuffle(rows)
    eval_rows = rows[:NUM_EVAL]
    train_rows = sorted(rows[NUM_EVAL:])

    with (out_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", lineterminator="\n")
        for name, text in train_rows:
            writer.writerow([name, text])

    (out_dir / "eval.txt").write_text(
        "\n".join(text for _, text in sorted(eval_rows)) + "\n", encoding="utf-8"
    )
    attribution = ATTRIBUTION_TMPL.format(
        voice=args.voice,
        gender=args.config,
        speaker_id=args.speaker,
        gender_prefix=gender_prefix,
        stats=f"{len(rows)} utterances kept, ~{args.minutes} min at 48 kHz",
    )
    (out_dir / "ATTRIBUTION.txt").write_text(attribution, encoding="utf-8")

    total = sum(sf.info(audio_dir / f"{n}.wav").duration for n, _ in train_rows)
    print(f"train utts: {len(train_rows)}  ({total/60:.1f} min @ {TARGET_SR} Hz)")
    print(f"eval utts:  {len(eval_rows)}")
    print(f"audio dir:  {audio_dir}")


if __name__ == "__main__":
    main()
