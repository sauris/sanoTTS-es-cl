#!/usr/bin/env python3
"""Phase 3 prep, generalized: build the distillation text corpus (~8k rows)
for any of the new voices (copihue / chande / vueltiao).

Sources:
- SLR71 transcripts (both genders), HF ylacombe/google-chilean-spanish
- SLR72 transcripts (both genders), openslr line_index TSVs
- Tatoeba spa sentences (CC BY 2.0 FR, attributed) to top up to 8k rows

Excluded from training text:
- the 16 fine-tune holdouts of EVERY new voice (eval.txt per voice)
- the 16 es_ES evidence sentences (experiments/evidence/ood-tatoeba-20260908.json)
- the 16 reserved Tatoeba sentences (artifacts/data/es_cl/tatoeba_eval16.json)
  -- loaded, never regenerated, so all voices share the same OOD eval set

  python build_distill_corpus2.py --voice copihue
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset

REPO = Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS")
VOICES = ("copihue", "chande", "vueltiao")
TATOEBA_TSV = Path.home() / "work/tatoeba/sentences_spa.tsv"
ES_CO_TSVS = {
    "slr72-female": Path.home() / "work/es_co/line_index_female.tsv",
    "slr72-male": Path.home() / "work/es_co/line_index_male.tsv",
}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True, choices=VOICES)
    args = parser.parse_args()

    out = REPO / f"artifacts/data/{args.voice}/distill_corpus.jsonl"
    reserved_path = REPO / "artifacts/data/es_cl/tatoeba_eval16.json"
    reserved = set(json.loads(reserved_path.read_text(encoding="utf-8"))["sentences"])

    excluded: set[str] = set(reserved)
    for voice in VOICES:
        eval_txt = REPO / f"artifacts/data/{voice}/eval.txt"
        if eval_txt.exists():
            for line in eval_txt.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = normalize(line)
                    if value:
                        excluded.add(value)
    evidence = REPO / "experiments/evidence/ood-tatoeba-20260908.json"
    ev = json.loads(evidence.read_text(encoding="utf-8"))
    for result in ev.get("results", []):
        if str(result.get("language")) == "es":
            for row in result.get("per_row", []):
                value = normalize(str(row.get("text") or ""))
                if value:
                    excluded.add(value)

    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

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

    co_texts: set[str] = set()
    for source, tsv in ES_CO_TSVS.items():
        for line in tsv.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            text = normalize(parts[1])
            if text is None or text in excluded or text in slr_texts or text in co_texts:
                continue
            co_texts.add(text)
            rows.append((source, text))
    print(f"SLR72 unique usable: {len(co_texts)}")

    tatoeba: list[str] = []
    with TATOEBA_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for parts in reader:
            if len(parts) < 2:
                continue
            text = normalize(parts[-1])
            if text is None or text in excluded or text in slr_texts or text in co_texts:
                continue
            tatoeba.append(text)
    print(f"Tatoeba spa usable: {len(tatoeba)}")

    need = TARGET_ROWS - len(rows)
    if need > 0:
        rng = random.Random(SEED)
        rng.shuffle(tatoeba)
        for text in tatoeba[:need]:
            rows.append(("tatoeba", text))
    else:
        # more SLR text than needed: deterministic shuffle + trim
        rng = random.Random(SEED)
        rng.shuffle(rows)
        rows = rows[:TARGET_ROWS]
    print(f"total corpus rows: {len(rows)}")

    order = list(range(len(rows)))
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(order)
    with out.open("w", encoding="utf-8") as handle:
        for index in order:
            source, text = rows[index]
            handle.write(
                json.dumps({"id": f"{index:05d}", "text": text, "source": source}, ensure_ascii=False)
                + "\n"
            )
    print(f"wrote {out} ({len(order)} rows)")


if __name__ == "__main__":
    main()
