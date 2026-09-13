#!/usr/bin/env python3
"""Phase 8, parametrized: Whisper-small transcription + CER/WER evidence for
any of the new voices. Same protocol as run_evidence.py (es_CL huemul).

  python run_evidence_voice.py --voice copihue --lang es_CL \
      --teacher-desc "es_CL-copihue-medium (es_ES-sharvard-medium <- SLR71 clf_04310)" \
      --wavdir /tmp/copihue-eval \
      --out <repo>/experiments/evidence/copihue-tatoeba-20260913.json
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import jiwer
import whisper

REPO = Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS")
SENTENCES = json.loads((REPO / "artifacts/data/es_cl/tatoeba_eval16.json").read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ü", "u")
    text = re.sub(r"[¿?¡!.,;:\"'()\-—…]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--teacher-desc", required=True)
    parser.add_argument("--wavdir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--date", default="2026-09-13")
    args = parser.parse_args()

    wav_dir = Path(args.wavdir)
    model = whisper.load_model("small")
    rows = []
    for i, text in enumerate(SENTENCES["sentences"]):
        wav = wav_dir / f"{i:03d}.wav"
        result = model.transcribe(str(wav), language="es", temperature=0.0, fp16=False, beam_size=None)
        hyp = result["text"].strip()
        ref_n, hyp_n = normalize(text), normalize(hyp)
        cer = jiwer.cer(ref_n, hyp_n)
        wer = jiwer.wer(ref_n, hyp_n)
        rows.append({"index": i, "text": text, "transcript": hyp, "cer": round(cer, 4), "wer": round(wer, 4)})
        print(f"{i:02d} cer={cer:.3f} wer={wer:.3f}  {text[:55]}")
        print(f"    heard: {hyp[:90]}")

    cer_mean = sum(r["cer"] for r in rows) / len(rows)
    wer_mean = sum(r["wer"] for r in rows) / len(rows)
    evidence = {
        "date": args.date,
        "voice": args.voice,
        "teacher": args.teacher_desc,
        "protocol": (
            "16 held-out Tatoeba spa sentences (fixed seed 20260912, reserved at corpus "
            "build time, never in training text), rendered by the browser wasm runtime "
            "(snt_voice + snt_g2p es-419, weights as shipped), transcribed by Whisper small "
            "(language=es, greedy, fp32), CER/WER against the source with "
            "case/punctuation-insensitive normalization. Same protocol family as "
            "ood-tatoeba-20260908.json."
        ),
        "results": [
            {
                "system": args.voice,
                "language": args.lang,
                "model": "whisper-small",
                "cer": round(cer_mean, 4),
                "wer": round(wer_mean, 4),
                "n": len(rows),
                "per_row": rows,
            }
        ],
    }
    Path(args.out).write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCER={cer_mean:.4f}  WER={wer_mean:.4f}  -> {args.out}")


if __name__ == "__main__":
    main()
