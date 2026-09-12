#!/usr/bin/env python3
"""Five-lane Root A feedback gate: render + score the eval pack's held-out rows.

Reconstructed replacement for the tool referenced by
docs/roota-language-porting-recipe.md (Stage 7) and tools/train_voice_from_piper.py
(stage s10_gate). It drives the dashboard tool's synthesis engine
(tools/serve_roota_arbitrary_tts_dashboard.py DashboardState) on each eval row and
scores every lane with SCOREQ:

    teacher_piper               full Piper teacher ONNX (text -> waveform)
    teacher_latent_piper_dec    teacher latent -> cut Piper decoder ONNX
    teacher_latent_student_dec  teacher latent -> compact student decoder
    student_full                student duration + student acoustic -> student decoder
    student_oracle_duration     oracle duration + student acoustic -> student decoder

Outputs <out-dir>/scoreq-summary.json (consumed by
tools/summarize_roota_feedback_loop.py) plus per-lane render directories.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent


def load_local_module(name: str, path: Path) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-pack", type=Path, required=True)
    parser.add_argument("--acoustic-checkpoint", type=Path, required=True)
    parser.add_argument("--duration-checkpoint", type=Path, default=None)
    parser.add_argument("--piper-model", type=Path, required=True)
    parser.add_argument("--piper-config", type=Path, required=True)
    parser.add_argument(
        "--decoder-onnx",
        type=Path,
        default=None,
        help="cut Piper decoder ONNX. Default: resolved from --decoder-report or the pack dir.",
    )
    parser.add_argument(
        "--decoder-student-checkpoint",
        type=Path,
        default=None,
        help="compact decoder checkpoint. Default: <decoder-report dir>/decoder-student.pt.",
    )
    parser.add_argument("--decoder-report", type=Path, default=None)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--param-budget", type=int, default=1_500_000)
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument("--force-score", action="store_true")
    parser.add_argument("--duration-length-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def resolve_decoder_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    decoder_onnx = args.decoder_onnx
    student_ckpt = args.decoder_student_checkpoint
    if decoder_onnx is None:
        if args.decoder_report and args.decoder_report.is_file():
            report = json.loads(args.decoder_report.read_text(encoding="utf-8"))
            for key in ("teacher_decoder", "decoder", "teacher_decoder_onnx"):
                value = report.get(key)
                if value and Path(value).is_file():
                    decoder_onnx = Path(value)
                    break
    if student_ckpt is None and args.decoder_report is not None:
        candidate = args.decoder_report.parent / "decoder-student.pt"
        if candidate.is_file():
            student_ckpt = candidate
    if decoder_onnx is None or not decoder_onnx.is_file():
        raise SystemExit(f"could not resolve the cut decoder ONNX (tried {decoder_onnx})")
    if student_ckpt is None or not student_ckpt.is_file():
        raise SystemExit(f"could not resolve the student decoder checkpoint (tried {student_ckpt})")
    return decoder_onnx, student_ckpt


def engine_args(
    *,
    dashboard_mod: Any,
    acoustic: Path,
    duration: Path | None,
    piper_model: Path,
    piper_config: Path,
    decoder_onnx: Path,
    student_ckpt: Path,
    backend: str,
    out_dir: Path,
    duration_source: str,
    duration_length_scale: float,
    device: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        host="127.0.0.1",
        port=0,
        acoustic_checkpoint=acoustic,
        decoder=decoder_onnx,
        decoder_backend=backend,
        decoder_student_checkpoint=student_ckpt,
        audio_enhancer_checkpoint=None,
        postprocess_gain=1.0,
        postprocess_filter="none",
        duration_checkpoint=duration,
        duration_source=duration_source,
        duration_length_scale=duration_length_scale,
        piper_model=piper_model,
        piper_config=piper_config,
        out_dir=out_dir,
        device=device,
        noise_scale=0.0,
        length_scale=1.0,
        noise_w=0.0,
        sentence_silence=0.12,
        sibilant_inject_beta=0.0,
        sibilant_calib=None,
        text_chunking="none",
        dashboard_title="feedback-loop",
        dashboard_subtitle="",
        default_text="",
    )


def score_dir(scoreq: Any, lane_dir: Path, suffix: str) -> dict[str, Any]:
    values: list[float] = []
    wavs = sorted(lane_dir.glob(f"*{suffix}"))
    for wav in wavs:
        try:
            values.append(float(scoreq.predict(str(wav))))
        except Exception:  # noqa: BLE001 - per-file scoring is best effort
            pass
    return {
        "n": len(values),
        "wavs": len(wavs),
        "scoreq_mean": statistics.mean(values) if values else None,
        "scoreq_std": statistics.stdev(values) if len(values) > 1 else None,
        "dir": str(lane_dir),
    }


def main() -> None:
    args = parse_args()
    summary_path = args.out_dir / "scoreq-summary.json"
    if summary_path.is_file() and not (args.force_render or args.force_score):
        print(f"already scored: {summary_path}")
        return

    dashboard = load_local_module(
        "feedback_loop_dashboard_engine", TOOLS / "serve_roota_arbitrary_tts_dashboard.py"
    )
    decoder_onnx, student_ckpt = resolve_decoder_paths(args)
    if args.duration_checkpoint is None:
        raise SystemExit("--duration-checkpoint is required for the student_full lane")

    rows = json.loads((args.eval_pack / "rows.json").read_text(encoding="utf-8"))
    texts = [str(row.get("text") or "").strip() for row in rows if str(row.get("text") or "").strip()]
    texts = texts[: max(1, args.rows)]
    if not texts:
        raise SystemExit(f"no texts in {args.eval_pack}/rows.json")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lane_teacher = args.out_dir / "lane-teacher-piper-onnx"
    lane_student = args.out_dir / "lane-student-decoder"
    common = dict(
        acoustic=args.acoustic_checkpoint,
        duration=args.duration_checkpoint,
        piper_model=args.piper_model,
        piper_config=args.piper_config,
        decoder_onnx=decoder_onnx,
        student_ckpt=student_ckpt,
        duration_length_scale=args.duration_length_scale,
        device=args.device,
    )
    onnx_engine = dashboard.DashboardState(
        engine_args(
            dashboard_mod=dashboard,
            backend="onnx",
            out_dir=lane_teacher,
            duration_source="oracle",
            **common,
        )
    )
    student_engine = dashboard.DashboardState(
        engine_args(
            dashboard_mod=dashboard,
            backend="student",
            out_dir=lane_student,
            duration_source="student",
            **common,
        )
    )

    render_log: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        try:
            # teacher + teacher-latent -> Piper decoder
            a = onnx_engine.synthesize(text, duration_source="oracle")
            # teacher-latent -> student decoder AND oracle-duration student lane
            c_e = student_engine.synthesize(text, duration_source="oracle")
            # full student (learned durations)
            d = student_engine.synthesize(text, duration_source="student")
            render_log.append(
                {
                    "index": index,
                    "text": text,
                    "teacher": a["metadata"]["id"],
                    "student_oracle": c_e["metadata"]["id"],
                    "student_full": d["metadata"]["id"],
                    "teacher_rms": a["metadata"].get("teacher_rms"),
                    "student_rms": d["metadata"].get("student_rms"),
                }
            )
        except Exception as exc:  # noqa: BLE001 - log and continue with other rows
            render_log.append({"index": index, "text": text, "error": str(exc)})

    from scoreq_loader import load_scoreq_class

    scoreq = load_scoreq_class()(data_domain="synthetic", mode="nr", use_onnx=True)

    card = student_engine.model_card
    parameters = {
        "acoustic": int(card.get("acoustic_parameters") or 0),
        "duration": int(card.get("duration_parameters") or 0),
        "decoder": int(card.get("decoder_parameters") or 0),
        "budget": int(args.param_budget),
    }
    parameters["total"] = parameters["acoustic"] + parameters["duration"] + parameters["decoder"]
    parameters["within_budget"] = parameters["total"] <= parameters["budget"]

    lanes = {
        "teacher_piper": score_dir(scoreq, lane_teacher / "audio", "-teacher.wav"),
        "teacher_latent_piper_dec": score_dir(
            scoreq, lane_teacher / "audio", "-oracle-decoder.wav"
        ),
        "teacher_latent_student_dec": score_dir(
            scoreq, lane_student / "audio", "-oracle-decoder.wav"
        ),
        "student_full": score_dir(scoreq, lane_student / "audio", "-student.wav"),
    }
    # student_oracle_duration lane: rendered under duration_source=oracle; its
    # wavs share the lane dir with full-student renders, so select by render id
    # from the metadata we logged at render time.
    oracle_ids = {row.get("student_oracle") for row in render_log if row.get("student_oracle")}
    oracle_wavs = [
        (lane_student / "audio") / f"{rid}-student.wav"
        for rid in sorted(oracle_ids)
        if (lane_student / "audio" / f"{rid}-student.wav").is_file()
    ]
    values = []
    for wav in oracle_wavs:
        try:
            values.append(float(scoreq.predict(str(wav))))
        except Exception:  # noqa: BLE001
            pass
    lanes["student_oracle_duration"] = {
        "n": len(values),
        "wavs": len(oracle_wavs),
        "scoreq_mean": statistics.mean(values) if values else None,
        "scoreq_std": statistics.stdev(values) if len(values) > 1 else None,
        "dir": str(lane_student / "audio"),
    }

    teacher_mean = lanes["teacher_piper"]["scoreq_mean"]
    deltas = {
        name: (
            None
            if teacher_mean is None or lane["scoreq_mean"] is None
            else round(lane["scoreq_mean"] - teacher_mean, 4)
        )
        for name, lane in lanes.items()
        if name != "teacher_piper"
    }

    summary = {
        "candidate_label": args.candidate_label,
        "rows_requested": len(texts),
        "rows_rendered": sum(1 for r in render_log if not r.get("error")),
        "lanes": lanes,
        "deltas_vs_teacher": deltas,
        "parameters": parameters,
        "decoder_report": str(args.decoder_report) if args.decoder_report else None,
        "decoder_onnx": str(decoder_onnx),
        "decoder_student_checkpoint": str(student_ckpt),
        "eval_pack": str(args.eval_pack),
        "render_log": render_log,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "render_log"}, indent=2))


if __name__ == "__main__":
    main()
