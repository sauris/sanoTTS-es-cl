#!/usr/bin/env python3
"""Re-render the 8 audition wavs for an es_CO teacher from its existing ONNX
(after the 08_long attribution fix: Chinú, Tuchín, San Andrés de Sotavento).
  python rerender_audition_esco.py <voice> <lang>
"""
import sys
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

voice, lang = sys.argv[1], sys.argv[2]
base = Path(f"/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/voices/{lang}/{voice}-medium")
out_dir = base / "audition"
out_dir.mkdir(exist_ok=True)

eval_lines = (Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS") / f"artifacts/data/{voice}/eval.txt").read_text(encoding="utf-8").splitlines()

probes = [
    ("03_seseo", "Las ciencias sociales estudian el zapato y la caza."),
    ("04_yeismo", "La lluvia en la calle mojó mi caballo."),
    ("05_coast", "Oye, ¿vamos para la plaza del mercado a comer un bollo limpio con suero costeño?"),
    ("06_numbers", "Un sombrero vueltiao fino de 23 vueltas cuesta más de un millón de pesos."),
    ("07_question", "¿Cuándo empiezan las corralejas de Sincelejo?"),
    ("08_long", "El sombrero vueltiao es el símbolo de los sabaneros de Sucre y Córdoba, tejido a mano con caña flecha en Chinú, Tuchín y San Andrés de Sotavento."),
]

samples = [("01_eval", eval_lines[0]), ("02_eval", eval_lines[1])] + probes

v = PiperVoice.load(base / f"{lang}-{voice}-medium.onnx", config_path=base / f"{lang}-{voice}-medium.onnx.json")
for name, text in samples:
    chunks = list(v.synthesize(text, syn_config=SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8)))
    arr = np.concatenate([c.audio_int16_array for c in chunks])
    with wave.open(str(out_dir / f"{name}.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(arr.tobytes())
    print(f"{name}: {len(arr)/22050:.2f}s")
print("AUDITION_RERENDERED")
