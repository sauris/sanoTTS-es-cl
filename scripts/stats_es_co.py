#!/usr/bin/env python3
"""Rank SLR72 (es_CO) speakers by total audio duration, per gender.

Reads the two openslr zips in ~/work/es_co without extracting: parses each
wav header (fmt + data chunk sizes) inside the zip members to get durations.

Gotcha carried over from SLR71: the numeric speaker_id is NOT unique across
genders (the gender lives in the filename prefix cof_/com_), so stats are
keyed by (gender, speaker_id).
"""
from __future__ import annotations

import argparse
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

ZIPS = {
    "female": Path.home() / "work/es_co/es_co_female.zip",
    "male": Path.home() / "work/es_co/es_co_male.zip",
}


def wav_duration(header: bytes) -> float | None:
    """Duration of a wav from its first bytes (handles LIST chunks before data)."""
    if len(header) < 44 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        return None
    pos = 12
    byte_rate = None
    while pos + 8 <= len(header):
        chunk_id = header[pos : pos + 4]
        (chunk_size,) = struct.unpack("<I", header[pos + 4 : pos + 8])
        if chunk_id == b"fmt ":
            # fmt body: audio_format u16, channels u16, sample_rate u32,
            # byte_rate u32 at offset 8 of the body
            byte_rate = struct.unpack("<I", header[pos + 8 + 8 : pos + 8 + 12])[0]
        elif chunk_id == b"data":
            if byte_rate:
                return chunk_size / byte_rate
            return None
        pos += 8 + chunk_size + (chunk_size % 2)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default=None,
        help="directory holding es_co_<gender>.zip (default ~/work/es_co)",
    )
    args = parser.parse_args()
    base = Path(args.dir) if args.dir else Path.home() / "work/es_co"
    for gender, default_path in ZIPS.items():
        zip_path = base / default_path.name
        if not zip_path.exists():
            print(f"missing: {zip_path}")
            continue
        stats: dict[str, list[float]] = defaultdict(list)
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if not info.filename.endswith(".wav"):
                    continue
                stem = Path(info.filename).stem
                # cof_<speaker>_<uid>.wav
                parts = stem.split("_")
                if len(parts) < 3 or not parts[1].isdigit():
                    continue
                speaker = parts[1]
                with zf.open(info) as handle:
                    header = handle.read(256)
                dur = wav_duration(header)
                if dur is None:
                    print(f"  unparsable: {info.filename}")
                    continue
                stats[speaker].append(dur)
        ranked = sorted(
            stats.items(), key=lambda kv: sum(kv[1]), reverse=True
        )
        print(f"\n== {gender} ({len(ranked)} speakers) ==")
        for speaker, durs in ranked[:8]:
            print(
                f"  {speaker}: {sum(durs)/60:6.1f} min  {len(durs):4d} utts"
            )


if __name__ == "__main__":
    main()
