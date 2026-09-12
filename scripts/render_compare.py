#!/usr/bin/env python3
"""Render the same sentences through teacher and student(s) for A/B listening.

Uses the dashboard engine: one synthesize() call writes
{teacher, oracle-decoder, student} wavs; we copy them under stable names into
artifacts/es_cl/compare/.

  python render_compare.py [--small]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "tools"

SENTENCES = [
    ("01", "¿Cuánto me demoro de aquí a Santiago?"),
    ("02", "Las ciencias sociales estudian el zapato y la caza."),
    ("03", "La lluvia en la calle mojó mi caballo."),
    ("04", "Voy a ir a correr con mi amiga Pepita y estamos buscando una pista de atletismo."),
    ("05", "El departamento cuesta 12.500 pesos la noche."),
    ("06", "El cerdo miniatura es diurno, sociable, amigable y juguetón."),
    ("07", "Tengo pensado quedarme una semana allí, si el tiempo acompaña."),
    ("08", "¡No te preocupes, sé feliz!"),
]


def load_dashboard(acoustic: Path, decoder_onnx: Path, student_ckpt: Path, out_dir: Path):
    import importlib.util

    def load_mod(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    dash = load_mod("dash_engine", TOOLS / "serve_roota_arbitrary_tts_dashboard.py")
    ns = argparse.Namespace(
        host="127.0.0.1", port=0,
        acoustic_checkpoint=acoustic,
        decoder=decoder_onnx, decoder_backend="student", decoder_student_checkpoint=student_ckpt,
        audio_enhancer_checkpoint=None, postprocess_gain=1.0, postprocess_filter="none",
        duration_checkpoint=None, duration_source="student", duration_length_scale=1.0,
        piper_model=REPO / "artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx",
        piper_config=REPO / "artifacts/voices/es_CL/huemul-medium/es_CL-huemul-medium.onnx.json",
        out_dir=out_dir, device="cpu", noise_scale=0.0, length_scale=1.0, noise_w=0.0,
        sentence_silence=0.12, sibilant_inject_beta=0.0, sibilant_calib=None,
        text_chunking="none", dashboard_title="compare", dashboard_subtitle="",
        default_text="",
    )
    return dash.DashboardState(ns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()

    run = REPO / "artifacts/voices/es_CL/run"
    cut = run / "decoder-cut/es_CL-huemul-medium-decoder-from-generator-input.onnx"
    student = load_dashboard(
        run / "joint/latent-student.pt", cut, run / "joint/decoder-student.pt",
        REPO / "artifacts/es_cl/compare/_engine_full",
    )
    small = None
    if args.small:
        srun = REPO / "artifacts/voices/es_CL/run-small"
        small = load_dashboard(
            srun / "joint/latent-student.pt", cut, srun / "joint/decoder-student.pt",
            REPO / "artifacts/es_cl/compare/_engine_small",
        )

    out_dir = REPO / "artifacts/es_cl/compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    for nn, text in SENTENCES[: args.rows]:
        r = student.synthesize(text, duration_source="student")
        audio_dir = Path(r["out_dir"]) / "audio" if False else student.args.out_dir / "audio"
        rid = r["id"]
        shutil.copyfile(audio_dir / f"{rid}-teacher.wav", out_dir / f"{nn}_teacher.wav")
        shutil.copyfile(audio_dir / f"{rid}-student.wav", out_dir / f"{nn}_student_full.wav")
        if small is not None:
            s = small.synthesize(text, duration_source="student")
            shutil.copyfile(
                small.args.out_dir / "audio" / f"{s['id']}-student.wav",
                out_dir / f"{nn}_student_small.wav",
            )
        print(f"{nn}: {text[:60]}")

    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
