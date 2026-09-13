#!/usr/bin/env python3
"""Export an SLR72 (es_CO) speaker to the Piper fine-tune layout.

  python export_es_co_speaker.py --voice chande --gender female --speaker 03397 \
      --minutes 17.2 --utts 120

Audio comes straight from the openslr zips (48 kHz wavs, resampled to
22050 Hz mono 16-bit). Layout and cleanup rules match export_speaker.py:
- artifacts/data/<voice>/audio/*.wav
- artifacts/data/<voice>/metadata.csv   utt_id|text
- artifacts/data/<voice>/eval.txt       16 held-out utterances
- artifacts/data/<voice>/ATTRIBUTION.txt  SLR72 / CC BY-SA 4.0 record
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS")

TARGET_SR = 22050
MAX_CHARS = 300
NUM_EVAL = 16
SEED = 20260912

ATTRIBUTION_TMPL = """Google Crowdsourced Colombian Spanish Speech (OpenSLR SLR72)
https://openslr.org/72

Citation (LREC 2020):
A. Guevara-Rukoz, I. Demirsahin, F. He, S.-H. C. Chu, S. Sarin, K. Pipatsrisawat,
A. Gutkin, A. Butryna, O. Kjartansson, "Crowdsourcing Latin American Spanish
for Low-Resource Text-to-Speech", Proceedings of The 12th Language Resources
and Evaluation Conference (LREC 2020), Marseille, 2020, pp. 6504-6513.

Copyright 2018, 2019 Google, Inc.
License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
https://creativecommons.org/licenses/by-sa/4.0/

This file records attribution for every artifact in this directory and for
all derived models (es_CO-{voice}-medium Piper fine-tune, distilled sanoTTS
voice). Derived artifacts must carry this attribution and remain compatible
with CC BY-SA 4.0 (sanoTTS training/G2P side is GPLv3; both permit
attribution + share-alike redistribution).

Selected subset: {gender} speaker {gender_prefix}_{speaker} ({stats}).
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
    parser.add_argument("--gender", required=True, choices=("female", "male"))
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--minutes", required=True, type=float)
    parser.add_argument("--utts", required=True, type=int)
    parser.add_argument(
        "--dir",
        default=None,
        help="directory holding zips + line indexes (default ~/work/es_co)",
    )
    args = parser.parse_args()

    gender_prefix = "cof" if args.gender == "female" else "com"
    data_dir = Path(args.dir) if args.dir else Path.home() / "work/es_co"
    zip_path = data_dir / f"es_co_{args.gender}.zip"
    out_dir = REPO / f"artifacts/data/{args.voice}"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # transcripts: line_index_<gender>.tsv  "<file stem>\t<text>"
    index_path = data_dir / f"line_index_{args.gender}.tsv"
    texts: dict[str, str] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            texts[parts[0].strip()] = parts[1]

    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".wav"):
                continue
            stem = Path(info.filename).stem
            parts = stem.split("_")
            if len(parts) < 3 or parts[1] != args.speaker:
                continue
            raw = zf.read(info)
            try:
                wav, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=True)
            except Exception as exc:
                print(f"skip (decode): {stem}: {exc}")
                continue
            wav = wav.mean(axis=1)
            text = clean_text(texts.get(stem, ""))
            if text is None:
                print(f"skip (text): {texts.get(stem, '')[:60]!r}")
                continue
            if wav.size == 0 or not np.isfinite(wav).all():
                print("skip (audio)")
                continue
            if sr != TARGET_SR:
                from math import gcd

                g = gcd(TARGET_SR, sr)
                wav = resample_poly(wav, TARGET_SR // g, sr // g).astype(np.float32)
            sf.write(audio_dir / f"{stem}.wav", np.clip(wav, -1.0, 1.0), TARGET_SR, subtype="PCM_16")
            rows.append((stem, text))

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
        gender=args.gender,
        gender_prefix=gender_prefix,
        speaker=args.speaker,
        stats=f"{len(rows)} utterances kept, ~{args.minutes} min at 48 kHz",
    )
    (out_dir / "ATTRIBUTION.txt").write_text(attribution, encoding="utf-8")

    total = sum(sf.info(audio_dir / f"{n}.wav").duration for n, _ in train_rows)
    print(f"train utts: {len(train_rows)}  ({total/60:.1f} min @ {TARGET_SR} Hz)")
    print(f"eval utts:  {len(eval_rows)}")
    print(f"audio dir:  {audio_dir}")


if __name__ == "__main__":
    main()
