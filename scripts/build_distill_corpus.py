#!/usr/bin/env python3
"""Phase 3 prep: build the es_CL distillation text corpus (~8k rows).

Sources:
- SLR71 transcripts (both genders), cleaned with the same rules as the
  fine-tune export, deduplicated.
- Tatoeba spa sentences (CC BY 2.0 FR, attributed) to top up to 8k rows.

Excluded from training text:
- the 16 fine-tune holdout utterances (artifacts/data/es_cl/eval.txt)
- the 16 es_ES evidence sentences (experiments/evidence/ood-tatoeba-20260908.json)
- a reserved fixed-seed sample of 16 Tatoeba sentences for the Phase 6 es_CL
  evidence (written to artifacts/data/es_cl/tatoeba_eval16.json)

The corpus is deterministically shuffled so the orchestrator's front-held-out
smoke12/eval128 splits are representative of both sources.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "artifacts/data/es_cl"
TATOEBA_TSV = Path.home() / "work/tatoeba/sentences_spa.tsv"
OUT = DATA / "distill_corpus.jsonl"
RESERVED = DATA / "tatoeba_eval16.json"
TARGET_ROWS = 8000
SEED = 20260912


def normalize(text: str) -> str | None:
    text = unicodedata.normalize("NFC", str(text)).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text) < 5 or len(text) > 220:
        return None
    if re.fullmatch(r"[\W\d]+", text):
        return None
    return text


def main() -> None:
    excluded: set[str] = set()
    for line in (DATA / "eval.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            excluded.add(normalize(line))  # type: ignore[arg-type]
    evidence = REPO / "experiments/evidence/ood-tatoeba-20260908.json"
    ev = json.loads(evidence.read_text(encoding="utf-8"))
    for result in ev.get("results", []):
        if str(result.get("language")) == "es":
            for row in result.get("per_row", []):
                value = normalize(str(row.get("text") or ""))
                if value:
                    excluded.add(value)

    rows: list[tuple[str, str]] = []

    slr_texts: set[str] = set()
    for gender in ("female", "male"):
        ds = load_dataset("ylacombe/google-chilean-spanish", gender, split="train")
        for row in ds:
            text = normalize(row["text"])
            if text is None or text in excluded or text in slr_texts:
                continue
            slr_texts.add(text)
            rows.append(("slr71", text))
    print(f"SLR71 unique usable: {len(slr_texts)}")

    tatoeba: list[str] = []
    with TATOEBA_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for parts in reader:
            if len(parts) < 2:
                continue
            # per-language export: id \t text (3 columns if lang kept)
            text = normalize(parts[-1])
            if text is None or text in excluded or text in slr_texts:
                continue
            tatoeba.append(text)
    print(f"Tatoeba spa usable: {len(tatoeba)}")

    rng = random.Random(SEED)
    rng.shuffle(tatoeba)
    reserved = sorted(set(tatoeba[:16]))
    tatoeba = [t for t in tatoeba if t not in set(reserved)]
    (RESERVED).write_text(
        json.dumps(
            {
                "seed": SEED,
                "source": "https://downloads.tatoeba.org/exports/sentences.csv (lang=spa)",
                "note": "reserved for the es_CL OOD evidence; never used as training text",
                "sentences": reserved,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    need = TARGET_ROWS - len(rows)
    if need > 0:
        for text in tatoeba[:need]:
            rows.append(("tatoeba", text))
    print(f"total corpus rows: {len(rows)}")

    order = list(range(len(rows)))
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(order)
    with OUT.open("w", encoding="utf-8") as handle:
        for index in order:
            source, text = rows[index]
            handle.write(json.dumps({"id": f"{index:05d}", "text": text, "source": source}, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(order)} rows)")


if __name__ == "__main__":
    main()
