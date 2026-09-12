#!/usr/bin/env python3
"""Verify PyTorch parity of a Piper decoder-cut ONNX and export the teacher.

Reconstructed replacement for the tool referenced by
docs/roota-language-porting-recipe.md (Stage 4) and tools/train_voice_from_piper.py
(stage s5_parity). It:

1. reads the decoder-cut ONNX (generator_input -> waveform) produced by
   tools/extract_piper_vits_decoder_cut.py;
2. rebuilds the Piper medium VITS decoder (Generator with resblock "2",
   upsample rates 8/8/4, initial channel 256) as a standalone PyTorch module
   with the clean parameter names the decoder trainer expects
   (conv_pre, ups.N, stages.N.branches.B.convs.J, conv_post);
3. copies weights from the ONNX initializers by walking the graph's
   Conv/ConvTranspose nodes in execution order and validating every shape;
4. proves parity on pack latents (same thresholds as the decoder-cut
   validation: mean abs <= 5e-4, cosine >= 0.9999);
5. with --export-checkpoint, writes piper-decoder-teacher.pt containing
   {"state_dict", "config", "parity"} for
   tools/train_roota_piper_decoder_student.py --teacher-init-checkpoint.

The exported state dict is plain (weight norms are removed in the ONNX export
already), so every tensor loads directly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn


# Piper medium (and high) generator profile: resblock "2", kernels (3,5,7),
# upsample rates (8,8,4), kernels (16,16,8), initial 256. Per-branch resblock
# dilations are read from the graph: this export uses (1,2),(2,6),(3,12).
TEACHER_CHANNELS = (256, 128, 64, 32)
RESBLOCK_KERNELS = (3, 5, 7)
DEFAULT_BRANCH_DILATIONS = ((1, 2), (2, 6), (3, 12))
UPSAMPLE_RATES = (8, 8, 4)
UPSAMPLE_KERNELS = (16, 16, 8)
LRELU_SLOPE = 0.1


class ResBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=d,
                padding=((kernel_size - 1) * d) // 2,
            )
            for d in dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x


class Stage(nn.Module):
    def __init__(self, channels: int, branch_dilations: tuple[tuple[int, ...], ...]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            ResBranch(channels, kernel, dilations)
            for kernel, dilations in zip(RESBLOCK_KERNELS, branch_dilations, strict=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = None
        for branch in self.branches:
            out = branch(x)
            total = out if total is None else total + out
        return total / len(self.branches)


class PiperDecoderTeacher(nn.Module):
    """Standalone Piper generator decoder with trainer-facing parameter names."""

    def __init__(
        self,
        latent_channels: int = 192,
        branch_dilations: tuple[tuple[int, ...], ...] = DEFAULT_BRANCH_DILATIONS,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = TEACHER_CHANNELS
        if len(branch_dilations) != len(RESBLOCK_KERNELS):
            raise RuntimeError(
                f"expected {len(RESBLOCK_KERNELS)} branch dilation tuples, got {branch_dilations}"
            )
        self.branch_dilations = tuple(tuple(d) for d in branch_dilations)
        self.conv_pre = nn.Conv1d(latent_channels, c0, 7, 1, padding=3)
        self.ups = nn.ModuleList(
            nn.ConvTranspose1d(
                cin,
                cout,
                k,
                stride=s,
                padding=(k - s) // 2,
            )
            for cin, cout, k, s in zip(
                (c0, c1, c2), (c1, c2, c3), UPSAMPLE_KERNELS, UPSAMPLE_RATES, strict=True
            )
        )
        self.stages = nn.ModuleList(Stage(c, self.branch_dilations) for c in (c1, c2, c3))
        self.conv_post = nn.Conv1d(c3, 1, 7, 1, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for up, stage in zip(self.ups, self.stages, strict=True):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            x = stage(x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, required=True, help="decoder-cut ONNX")
    parser.add_argument("--pack-dir", type=Path, required=True, help="eval pack directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32, help="max chunks to verify")
    parser.add_argument("--export-checkpoint", action="store_true")
    parser.add_argument("--max-row-mean-abs", type=float, default=5e-4)
    parser.add_argument("--min-row-cosine", type=float, default=0.9999)
    return parser.parse_args()


def initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}


def graph_branch_dilations(model: onnx.ModelProto) -> tuple[tuple[int, ...], ...]:
    """Read per-branch resblock conv dilations off the graph's Conv attributes."""
    by_resblock: dict[int, dict[int, set[int]]] = {}
    for node in model.graph.node:
        if node.op_type != "Conv" or not node.name or "/dec/resblocks." not in node.name:
            continue
        parts = node.name.split("/")
        resblock_part = next((p for p in parts if p.startswith("resblocks.")), None)
        conv_part = next((p for p in parts if p.startswith("convs.")), None)
        if resblock_part is None or conv_part is None:
            continue
        resblock_index = int(resblock_part.split(".")[1])
        conv_index = int(conv_part.split(".")[1])
        attrs = {a.name: a for a in node.attribute}
        if "dilations" not in attrs:
            continue
        dilations = list(attrs["dilations"].ints)
        if len(dilations) != 1:
            raise RuntimeError(f"{node.name}: unexpected dilation rank {dilations}")
        by_resblock.setdefault(resblock_index, {}).setdefault(conv_index, set()).add(dilations[0])
    if not by_resblock:
        raise RuntimeError("no /dec/resblocks.* conv dilations found in graph")
    branch_count = 3
    if len(by_resblocks := {k: v for k, v in by_resblock.items()}) % branch_count != 0:
        raise RuntimeError(f"unexpected resblock count {len(by_resblocks)}")
    branches: list[tuple[int, ...]] = []
    for branch_index in range(branch_count):
        sample = by_resblock.get(branch_index) or by_resblock.get(branch_index + branch_count)
        if not sample:
            raise RuntimeError(f"missing resblock entries for branch {branch_index}")
        conv_indices = sorted(sample)
        pairs: list[int] = []
        for conv_index in conv_indices:
            values = sample[conv_index]
            if len(values) != 1:
                raise RuntimeError(f"branch {branch_index} conv {conv_index}: inconsistent dilations {values}")
            pairs.append(values.pop())
        branches.append(tuple(pairs))
    return tuple(branches)


def build_teacher_module(model: onnx.ModelProto) -> PiperDecoderTeacher:
    return PiperDecoderTeacher(branch_dilations=graph_branch_dilations(model))


def load_weights(model: onnx.ModelProto, module: PiperDecoderTeacher) -> dict[str, Any]:
    """Copy weights from `dec.*` initializers to the clean-named module.

    Piper exports keep plain parameter names (dec.conv_pre.weight,
    dec.ups.N.weight, dec.resblocks.{3*stage+branch}.convs.{0,1}.weight,
    dec.conv_post.weight), so the mapping is by name with strict shape checks.
    """
    inits = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}
    state: dict[str, torch.Tensor] = {}
    mapping: list[dict[str, Any]] = []

    def take(export_key: str, target_key: str, expected: tuple[int, ...]) -> None:
        if export_key not in inits:
            raise RuntimeError(f"decoder ONNX is missing initializer {export_key}")
        value = inits[export_key]
        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"{export_key}: shape {tuple(value.shape)} != expected {expected}"
            )
        state[target_key] = torch.from_numpy(np.ascontiguousarray(value)).float()
        mapping.append({"export": export_key, "target": target_key, "shape": list(expected)})

    c0, c1, c2, c3 = TEACHER_CHANNELS
    take("dec.conv_pre.weight", "conv_pre.weight", (c0, 192, 7))
    take("dec.conv_pre.bias", "conv_pre.bias", (c0,))
    for i, (cin, cout, k) in enumerate(((c0, c1, 16), (c1, c2, 16), (c2, c3, 8))):
        take(f"dec.ups.{i}.weight", f"ups.{i}.weight", (cin, cout, k))
        take(f"dec.ups.{i}.bias", f"ups.{i}.bias", (cout,))
    for stage_index, ch in enumerate((c1, c2, c3)):
        for branch_index, kernel in enumerate(RESBLOCK_KERNELS):
            resblock_index = 3 * stage_index + branch_index
            conv_count = len(module.branch_dilations[branch_index])
            for conv_index in range(conv_count):
                prefix = f"stages.{stage_index}.branches.{branch_index}.convs.{conv_index}"
                export_prefix = f"dec.resblocks.{resblock_index}.convs.{conv_index}"
                take(f"{export_prefix}.weight", f"{prefix}.weight", (ch, ch, kernel))
                take(f"{export_prefix}.bias", f"{prefix}.bias", (ch,))
    take("dec.conv_post.weight", "conv_post.weight", (1, c3, 7))

    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mapping mismatch: missing={missing}, unexpected={unexpected}")
    return {"mapped_tensors": len(mapping), "mapping": mapping}


def pack_chunk_latents(pack_dir: Path, limit: int) -> list[tuple[str, np.ndarray, int]]:
    rows_path = pack_dir / "rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    chunks: list[tuple[str, np.ndarray, int]] = []
    for row in rows:
        for chunk in row.get("chunks", []):
            tensor_path = Path(str(chunk.get("tensor_npz") or ""))
            if not tensor_path.is_file():
                tensor_path = pack_dir / tensor_path.name
            with np.load(tensor_path) as tensors:
                latent = np.asarray(tensors["generator_input"], dtype=np.float32)
            chunks.append((str(chunk.get("tensor_npz")), latent, int(chunk.get("audio_samples", -1))))
            if len(chunks) >= limit:
                return chunks
    return chunks


def main() -> None:
    args = parse_args()
    if not args.decoder.is_file():
        raise SystemExit(f"decoder ONNX not found: {args.decoder}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(args.decoder))
    module = build_teacher_module(model)
    load_info = load_weights(model, module)
    module.eval()

    session = ort.InferenceSession(str(args.decoder), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    if len(input_names) != 1:
        raise SystemExit(f"expected exactly one decoder input, got {input_names}")
    latent_name = input_names[0]

    chunks = pack_chunk_latents(args.pack_dir, args.rows)
    if not chunks:
        raise SystemExit(f"no chunks found under {args.pack_dir}")

    comparisons: list[dict[str, Any]] = []
    errors: list[str] = []
    with torch.no_grad():
        for source, latent, audio_samples in chunks:
            ort_audio = np.asarray(
                session.run(None, {latent_name: latent})[0], dtype=np.float32
            ).reshape(-1)
            torch_audio = module(torch.from_numpy(latent)).numpy().reshape(-1)
            if ort_audio.size != torch_audio.size:
                errors.append(f"{source}: sample mismatch {ort_audio.size} != {torch_audio.size}")
                continue
            diff = ort_audio - torch_audio
            mean_abs = float(np.mean(np.abs(diff)))
            max_abs = float(np.max(np.abs(diff)))
            denom = float(np.linalg.norm(ort_audio) * np.linalg.norm(torch_audio))
            cos = float(np.dot(ort_audio, torch_audio) / denom) if denom > 0 else 1.0
            comparisons.append(
                {
                    "source": source,
                    "samples": int(ort_audio.size),
                    "mean_abs": mean_abs,
                    "max_abs": max_abs,
                    "cosine": cos,
                }
            )
            if mean_abs > args.max_row_mean_abs:
                errors.append(f"{source}: mean abs {mean_abs:.8f} > {args.max_row_mean_abs}")
            if cos < args.min_row_cosine:
                errors.append(f"{source}: cosine {cos:.8f} < {args.min_row_cosine}")

    parity = {
        "passed": not errors,
        "decoder": str(args.decoder),
        "pack_dir": str(args.pack_dir),
        "chunks": len(comparisons),
        "max_mean_abs": float(max((c["mean_abs"] for c in comparisons), default=0.0)),
        "max_max_abs": float(max((c["max_abs"] for c in comparisons), default=0.0)),
        "min_cosine": float(min((c["cosine"] for c in comparisons), default=1.0)),
        "teacher_channels": list(TEACHER_CHANNELS),
        "weight_mapping": load_info,
        "errors": errors,
        "comparisons": comparisons,
    }
    report_path = args.out_dir / "piper-decoder-torch-parity.json"
    report_path.write_text(json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.export_checkpoint:
        checkpoint = {
            "state_dict": {k: v.clone() for k, v in module.state_dict().items()},
            "config": {
                "kind": "piper-decoder-teacher",
                "teacher_channels": list(TEACHER_CHANNELS),
                "resblock_kernels": list(RESBLOCK_KERNELS),
                "resblock_dilations": [list(d) for d in module.branch_dilations],
                "upsample_rates": list(UPSAMPLE_RATES),
                "upsample_kernel_sizes": list(UPSAMPLE_KERNELS),
                "latent_channels": 192,
                "source_decoder": str(args.decoder),
            },
            "parity": {k: v for k, v in parity.items() if k not in ("comparisons", "weight_mapping")},
        }
        ckpt_path = args.out_dir / "piper-decoder-teacher.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"exported {ckpt_path}")

    print(
        json.dumps(
            {k: v for k, v in parity.items() if k not in ("comparisons", "weight_mapping")},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not parity["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
