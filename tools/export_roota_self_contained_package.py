#!/usr/bin/env python3
"""Export a Root A voice as raw fp16 runtime assets (self-contained package).

Reconstructed replacement for the tool referenced by
docs/roota-language-porting-recipe.md (Stage 9) and tools/train_voice_from_piper.py
(stage s11_export). Loads the three student checkpoints, casts every tensor to
float16, concatenates them into one deterministic little-endian blob and writes:

    weights.fp16.bin            all component tensors, in manifest order
    manifest.json               tensor offsets/shapes + parameter counts
    piper-phoneme-config.json   phoneme_id_map + espeak voice contract
    runtime-kernels.md          which kernels the package assumes
    README.md                   what this package is and is not

The manifest is the source of truth for parameter counts
(total_parameters) and byte sizes (weights_size_bytes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-name", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--acoustic-checkpoint", type=Path, required=True)
    parser.add_argument("--duration-checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--piper-config", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--duration-length-scale", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_component_state(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = None
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            state = value
            break
    if state is None:
        raise SystemExit(f"{checkpoint_path}: no model_state_dict/state_dict found")
    config = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    return state, config


RUNTIME_KERNELS = """# Runtime kernels assumed by this package

- Duration student: embedding + 1D conv stack (kernel 5), per-token frame counts
- Acoustic student: token-context 1D convs predicting the 192-dim Piper latent
- Decoder student: piperlite HiFi-GAN-style decoder (ConvTranspose upsample +
  residual banks + tanh head)
- Frontend: Piper/espeak-ng phoneme IDs are the input contract; the text
  frontend itself is NOT included in this package
"""


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    components: dict[str, dict[str, Any]] = {}
    blob = bytearray()
    total_parameters = 0
    for role, path in (
        ("duration", args.duration_checkpoint),
        ("acoustic", args.acoustic_checkpoint),
        ("decoder", args.decoder_checkpoint),
    ):
        state, config = load_component_state(path)
        entries = []
        role_params = 0
        for name in sorted(state):
            tensor = state[name].detach().cpu().to(torch.float16).contiguous()
            numel = int(tensor.numel())
            entries.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": "float16",
                    "offset": len(blob),
                    "numel": numel,
                }
            )
            blob.extend(tensor.numpy().tobytes())
            role_params += numel
        components[role] = {
            "checkpoint": str(path),
            "config": config,
            "parameters": role_params,
            "tensors": entries,
        }
        total_parameters += role_params

    weights_path = args.out_dir / "weights.fp16.bin"
    weights_path.write_bytes(bytes(blob))

    piper_config = json.loads(args.piper_config.read_text(encoding="utf-8"))
    phoneme_config = {
        "num_symbols": int(piper_config.get("num_symbols") or 256),
        "phoneme_type": piper_config.get("phoneme_type", "espeak"),
        "espeak_voice": (piper_config.get("espeak") or {}).get("voice"),
        "phoneme_id_map": piper_config.get("phoneme_id_map", {}),
    }
    (args.out_dir / "piper-phoneme-config.json").write_text(
        json.dumps(phoneme_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "package_name": args.package_name,
        "language": args.language,
        "voice": args.voice,
        "sample_rate": int(args.sample_rate),
        "duration_length_scale": float(args.duration_length_scale),
        "weights": "float16, little-endian, concatenated in component order",
        "weights_file": weights_path.name,
        "weights_size_bytes": len(blob),
        "total_parameters": total_parameters,
        "components": {
            role: {"parameters": info["parameters"], "checkpoint": info["checkpoint"]}
            for role, info in components.items()
        },
        "tensor_index": components,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "runtime-kernels.md").write_text(RUNTIME_KERNELS, encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        f"# {args.package_name}\n\n"
        f"Tiny Piper-distilled neural runtime package ({total_parameters:,} parameters, "
        f"{len(blob):,} bytes of fp16 weights) for the {args.voice} voice "
        f"({args.language}). It starts from Piper-compatible phoneme IDs and produces "
        f"{args.sample_rate} Hz waveform audio. It is not a fully standalone "
        "arbitrary-text TTS runtime unless the espeak-ng frontend is included.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "package": str(args.out_dir),
                "total_parameters": total_parameters,
                "weights_size_bytes": len(blob),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
