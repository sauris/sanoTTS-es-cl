#!/usr/bin/env python3
"""Phase 4 gate prep: ground-truth phoneme ids for the es-419/es wasm parity gate.

Writes JSON: [{text, ids_es419, ids_es}] for a spread of corpus sentences plus
seseo/yeismo probes, using python piper (1.8) with the huemul teacher config
(es-419) and the davefx base config (es).
"""
from __future__ import annotations

import json
from pathlib import Path

from piper import PiperVoice

REPO = Path(__file__).resolve().parents[3]
TEACHER = REPO / "artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx"
TEACHER_JSON = REPO / "artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx.json"
DAVEFX = Path.home() / "work/base_ckpts/es_ES/davefx/medium/davefx-medium.onnx"
DAVEFX_JSON = Path.home() / "work/base_ckpts/es_ES/davefx/medium/config.json"
OUT = REPO / "artifacts/es_cl/g2p_truth.json"

PROBES = [
    "Hola mundo.",
    "Las ciencias sociales estudian el zapato y la caza.",
    "La lluvia en la calle mojó mi caballo.",
    "¿Cuánto me demoro de aquí a Santiago?",
    "El ceviche es delicioso, pero el pastel de choclo es mejor.",
    "Voy a ir a correr con mi amiga Pepita.",
    "Necesito dárselo.",
    "Tengo pensado quedarme una semana allí.",
    "No confundas cometas con asteroides.",
    "Dicen que María estuvo enferma la semana pasada.",
    "¡No te preocupes, sé feliz!",
    "Las instituciones castellanoleonesas son reticentes.",
    "Ella habla inglés como si fuera su lengua materna.",
    "El departamento cuesta 12.500 pesos la noche.",
    "Todo el mundo conoce el monte Fuji.",
    "Tienes que masticar la comida antes de tragarla.",
    "Acabo de recibir vuestro mensaje.",
    "¿Quién es el actor estadounidense más famoso?",
    "Quiero instalar paneles solares.",
    "La pronunciación del francés es difícil.",
]


def main() -> None:
    v419 = PiperVoice.load(TEACHER, config_path=TEACHER_JSON)
    ves = PiperVoice.load(DAVEFX, config_path=DAVEFX_JSON)
    rows = []
    for text in PROBES:
        def ids_for(voice: PiperVoice) -> list[int]:
            sentences = voice.phonemize(text)
            chunks = [voice.phonemes_to_ids(s) for s in sentences if s]
            return [i for chunk in chunks for i in chunk]

        rows.append({"text": text, "ids_es419": ids_for(v419), "ids_es": ids_for(ves)})
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    same = sum(1 for r in rows if r["ids_es419"] == r["ids_es"])
    print(f"wrote {OUT}: {len(rows)} rows ({same} identical between es and es-419)")


if __name__ == "__main__":
    main()
