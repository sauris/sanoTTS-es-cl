#!/usr/bin/env python3
"""Blast-radius ground truth: piper 1.8 ids for 5 languages (en, de, fr, vi, zh)."""
import json
from pathlib import Path

from piper import PiperVoice

BASE = Path.home() / "work/blast_voices"
OUT = Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/es_cl/g2p_truth_blast.json")

CASES = [
    ("en", "en_US-amy-medium", "Hello world, this is a quick test."),
    ("de", "de_DE-thorsten-medium", "Guten Tag Welt, wie geht es dir?"),
    ("fr", "fr_FR-upmc-medium", "Bonjour le monde, comment allez-vous?"),
    ("vi", "vi_VN-vais1000-medium", "Xin chào bạn, thời tiết hôm nay thế nào?"),
    ("zh", "zh_CN-huayan-medium", "你好世界，今天天气怎么样？"),
]


def main() -> None:
    rows = []
    for lang, name, text in CASES:
        voice = PiperVoice.load(BASE / name / f"{name}.onnx", config_path=BASE / name / f"{name}.onnx.json")
        sentences = voice.phonemize(text)
        ids = [i for s in sentences if s for i in voice.phonemes_to_ids(s)]
        rows.append({"lang": lang, "text": text, "ids": ids})
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
