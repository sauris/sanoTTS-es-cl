#!/usr/bin/env python3
"""Build a decoder activation-signature pack from a Piper decoder-cut ONNX.

Reconstructed replacement for the tool referenced by
docs/roota-language-porting-recipe.md (Stage 5) and tools/train_voice_from_piper.py
(stage s6_signatures).

For every chunk tensor in a decoder-mode probe pack, the cut ONNX is run with
the latent taken from the pack (--feed-latent-from-pack) and the requested
--node value names exposed as extra graph outputs. Each captured activation is
pooled per latent frame with EXACTLY the trainer's math
(tools/train_roota_piper_decoder_student.py pooled_activation_signature):

    mean   = adaptive_avg_pool1d(act, latent_frames)
    logrms = log1p(sqrt(adaptive_avg_pool1d(act**2, latent_frames) + 1e-12))

and stored as {key}_mean / {key}_logrms arrays [channels, latent_frames] plus
a latent_frames scalar, in one NPZ per chunk, indexed by
decoder-signature-manifest.jsonl ({source_tensor_npz, signature_npz}) as
consumed by load_signature_index()/load_signature_tensors() in the decoder
trainer. A summary.json marks the directory done for resumable orchestrators.

NODE_LABELS maps a Piper decoder-cut ONNX value name to a semantic signature
key. The names below are the current piper1-gpl torch.onnx.export (legacy
exporter) names, verified against a 2026 piper1-gpl medium export.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

# value name (graph) -> semantic signature key
NODE_LABELS: dict[str, str] = {
    "/dec/conv_pre/Conv_output_0": "conv_pre",
    "/dec/ups.0/ConvTranspose_output_0": "up0_raw",
    "/dec/Div_output_0": "stage0_mix",
    "/dec/ups.1/ConvTranspose_output_0": "up1_raw",
    "/dec/Div_1_output_0": "stage1_mix",
    "/dec/ups.2/ConvTranspose_output_0": "up2_raw",
    "/dec/Div_2_output_0": "stage2_mix",
    "/dec/conv_post/Conv_output_0": "pre_tanh",
    "/dec/Tanh_output_0": "audio",
    # decoder-cut keeps the final output under the requested name:
    "output": "audio",
    # the decoder-cut latent input keeps the teacher graph's value name
    # (the tensor feeding /dec/conv_pre/Conv; piper 1.8-era exports)
    "/Mul_7_output_0": "latent",
    # legacy (2026-06 Nepali-era) export names kept for compatibility
    "x.191": "conv_pre",
    "x.195": "up0_raw",
}

DTYPES = {"float16": np.float16, "float32": np.float32}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="decoder-cut ONNX")
    parser.add_argument("--pack-dir", type=Path, required=True, help="decoder-mode probe pack")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feed-latent-from-pack", action="store_true", default=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument(
        "--node",
        action="append",
        default=None,
        help="ONNX value name to capture; repeatable. Defaults to every NODE_LABELS name present in the graph.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all chunks")
    return parser.parse_args()


def adaptive_avg_pool1d(x: np.ndarray, frames: int) -> np.ndarray:
    """NumPy equivalent of torch F.adaptive_avg_pool1d for [C, T]."""
    channels, time = x.shape
    if time == frames:
        return x.copy()
    out = np.empty((channels, frames), dtype=np.float64)
    for i in range(frames):
        start = (i * time) // frames
        end = -((-(i + 1) * time) // frames)
        end = max(end, start + 1)
        out[:, i] = x[:, start:end].mean(axis=1)
    return out


def pooled_signature(act: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirror pooled_activation_signature exactly (see module docstring)."""
    mean = adaptive_avg_pool1d(act, frames)
    mean_sq = adaptive_avg_pool1d(np.square(act.astype(np.float64)), frames)
    logrms = np.log1p(np.sqrt(mean_sq + 1e-12))
    return mean, logrms


def graph_value_names(model: onnx.ModelProto) -> set[str]:
    names = {i.name for i in model.graph.input}
    names.update(o.name for o in model.graph.output)
    for node in model.graph.node:
        names.update(out for out in node.output if out)
    return names


def build_session(model_path: Path, nodes: list[str]) -> tuple[ort.InferenceSession, str]:
    model = onnx.load(str(model_path))
    present = graph_value_names(model)
    existing_outputs = {o.name for o in model.graph.output}
    missing = [n for n in nodes if n not in present]
    if missing:
        raise SystemExit(f"{model_path.name}: value names not in graph: {missing}")
    for name in nodes:
        if name not in existing_outputs:
            model.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    inputs = [i.name for i in session.get_inputs()]
    if len(inputs) != 1:
        raise SystemExit(f"expected one latent input, got {inputs}")
    return session, inputs[0]


def pack_chunks(pack_dir: Path, limit: int) -> list[dict[str, Any]]:
    rows = json.loads((pack_dir / "rows.json").read_text(encoding="utf-8"))
    chunks: list[dict[str, Any]] = []
    for row in rows:
        for chunk in row.get("chunks", []):
            chunks.append(chunk)
    if limit > 0:
        chunks = chunks[:limit]
    if not chunks:
        raise SystemExit(f"no chunks under {pack_dir}")
    return chunks


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"decoder ONNX not found: {args.model}")
    if not (args.pack_dir / "rows.json").is_file():
        raise SystemExit(f"pack rows.json not found under {args.pack_dir}")

    model_present = graph_value_names(onnx.load(str(args.model)))
    if args.node:
        nodes = list(dict.fromkeys(args.node))
    else:
        nodes = [name for name in NODE_LABELS if name in model_present]
    if not nodes:
        raise SystemExit("no signature nodes resolved from the graph")
    # label -> value name, exactly one array per label ("audio" prefers the
    # graph output over the Tanh node output when both are present)
    chosen: dict[str, str] = {}
    for name in nodes:
        label = NODE_LABELS.get(name)
        if label is None:
            raise SystemExit(f"{name}: not in NODE_LABELS; add it there")
        if label not in chosen or name == "output":
            chosen[label] = name
    by_name = {name: label for label, name in chosen.items()}

    session, latent_input = build_session(args.model, list(chosen.values()))
    chunks = pack_chunks(args.pack_dir, args.max_rows)

    sig_dir = args.out_dir / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)
    store_dtype = DTYPES[args.dtype]
    manifest_path = args.out_dir / "decoder-signature-manifest.jsonl"
    manifest_rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, float]] = {}
    for chunk in chunks:
        tensor_path = Path(str(chunk.get("tensor_npz") or ""))
        if not tensor_path.is_file():
            tensor_path = args.pack_dir / tensor_path.name
        with np.load(tensor_path) as tensors:
            latent = np.asarray(tensors["generator_input"], dtype=np.float32)
        frames = int(latent.shape[-1])
        outputs = session.run(list(chosen.values()), {latent_input: latent})
        saved: dict[str, np.ndarray] = {
            "latent_frames": np.asarray([frames], dtype=np.int64)
        }
        for name, array in zip(chosen.values(), outputs, strict=True):
            # [batch, channels, time] (audio may carry an extra singleton dim)
            # -> [channels, time]
            act = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[-1])
            if act.shape[1] < frames:
                raise SystemExit(
                    f"{tensor_path.name}: activation {name} time {act.shape[1]} < latent frames {frames}"
                )
            mean, logrms = pooled_signature(act, frames)
            label = by_name[name]
            saved[f"{label}_mean"] = mean.astype(store_dtype)
            saved[f"{label}_logrms"] = logrms.astype(store_dtype)
            s = stats.setdefault(label, {"max_abs_mean": 0.0, "max_abs_logrms": 0.0})
            s["max_abs_mean"] = max(s["max_abs_mean"], float(np.abs(mean).max()))
            s["max_abs_logrms"] = max(s["max_abs_logrms"], float(np.abs(logrms).max()))
        sig_path = sig_dir / f"{tensor_path.stem}.sig.npz"
        np.savez_compressed(sig_path, **saved)
        manifest_rows.append(
            {
                "source_tensor_npz": str(tensor_path.resolve()),
                "signature_npz": str(sig_path.resolve()),
                "latent_frames": frames,
            }
        )
    written = len(manifest_rows)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in manifest_rows:
            manifest.write(json.dumps(row) + "\n")

    summary = {
        "model": str(args.model),
        "pack_dir": str(args.pack_dir),
        "rows": written,
        "dtype": args.dtype,
        "nodes": list(chosen.values()),
        "labels": sorted(chosen),
        "activation_stats": stats,
        "manifest": str(manifest_path),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "activation_stats"}, indent=2))


if __name__ == "__main__":
    main()
