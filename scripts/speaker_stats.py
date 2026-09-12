#!/usr/bin/env python3
"""Phase 1, step 1: per-speaker duration stats for ylacombe/google-chilean-spanish.

Ranks speakers by total clean duration across both gender configs so the
fine-tune speaker can be picked by data volume (plan Phase 1.2).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Audio, load_dataset

OUT = Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/es_cl/data/speaker_stats.json")

def main() -> None:
    stats: dict[tuple[str, str], dict] = defaultdict(lambda: {"seconds": 0.0, "utts": 0, "lengths": [], "sample_path": None})
    for gender in ("female", "male"):
        ds = load_dataset("ylacombe/google-chilean-spanish", gender, split="train")
        paths = ds.cast_column("audio", Audio(decode=False))["audio"]
        print(f"{gender}: {len(ds)} rows, columns={ds.column_names}", flush=True)
        for i in range(len(ds)):
            row = ds[i]
            # Numeric speaker_ids are NOT unique across the female/male configs
            # (the gender lives in the original filename, e.g. clf_09334_...).
            spk = (gender, str(row["speaker_id"]))
            audio = row["audio"]
            dur = len(audio["array"]) / audio["sampling_rate"]
            rec = stats[spk]
            rec["seconds"] += dur
            rec["utts"] += 1
            rec["lengths"].append(dur)
            if rec["sample_path"] is None:
                rec["sample_path"] = paths[i].get("path")

    table = []
    for (gender, spk), rec in stats.items():
        table.append(
            {
                "speaker_id": spk,
                "gender": gender,
                "minutes": round(rec["seconds"] / 60, 1),
                "utts": rec["utts"],
                "median_utt_s": round(statistics.median(rec["lengths"]), 2),
                "max_utt_s": round(max(rec["lengths"]), 2),
                "sample_path": rec["sample_path"],
            }
        )
    table.sort(key=lambda r: r["minutes"], reverse=True)

    print(f"\n{'speaker':<9}{'gender':<9}{'min':>7}{'utts':>7}{'med_s':>8}{'max_s':>8}  file")
    for r in table:
        print(
            f"{r['speaker_id']:<9}{r['gender']:<9}{r['minutes']:>7}{r['utts']:>7}"
            f"{r['median_utt_s']:>8}{r['max_utt_s']:>8}  {r['sample_path']}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {OUT}", file=sys.stderr)

if __name__ == "__main__":
    main()
