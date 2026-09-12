#!/usr/bin/env python3
"""Export a sanoTTS browser voice bundle (front_f16.bin + dec_f16.bin + meta.json).

Reconstructed replacement for the tool referenced by web/index.html and
mcu/ports/wasm/build_voices.sh. The binary layout is the snt_voice wasm
contract (mcu/src/snt_front_f32.c and mcu/src/snt_piperlite.c):

front blob = meta.bin header + weights
    header: 18 int32 LE words
        magic 0x534E4652 ('SNFR'), version 1,
        d_vocab, d_hidden, d_depth, d_kernel, d_max_tokens, d_max_duration,
        a_vocab, a_hidden, a_token_depth, a_depth, a_kernel, a_out,
        adapter_mode(0), adapter_kernel(0), adapter_rank(0), n_tensors
    then n_tensors x (offset,size) int32 LE pairs (offsets in FLOATS)
    then the weights, float32 in slot order (or float16 with meta.weights=f16)

    slot order (DurationStudent + ContextualLatentStudent):
      0 embedding.weight [d_vocab,d_hidden]
      1 input_proj.weight [d_hidden,d_hidden+3,1]   2 input_proj.bias
      then per duration block: scale, conv1.weight, conv1.bias,
                               conv2.weight, conv2.bias
      output.weight [1,d_hidden,1], output.bias
      a: embedding.weight, token_input_proj.weight [h,h+2,1], .bias,
      per token block x5, frame_input_proj.weight [h,h+3,1], .bias,
      per frame block x5, output.weight [a_out,h,1], output.bias

dec blob = meta.bin header + weights
    header: 12 int32 LE words
        magic 0x534E504C ('SNPL'), version 1,
        in_ch, c0, c1, c2, c3, pf_channels(0), pf_layers(0), pf_kernel(9),
        pf_scale bits(0.25f), n_tensors
    then n_tensors x (offset,size) pairs, then weights in slot order:

      0 pre.weight [c0,in_ch,7]   1 pre.bias
      per stage s in 0..2: up.weight [in_c,out_c,k16/16/8], up.bias,
        then 3 branches (kernels 3,5,7) x (conv1.weight, conv1.bias,
        conv2.weight, conv2.bias)
      post.weight [1,c3,7], post.bias

meta.json mirrors the shipped spanish bundle (see web/voices/spanish/meta.json)
including the fp16 opt-in ("weights":"f16") and the widened-sha256 checks the
browser loader enforces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

FRONT_MAGIC = 0x534E4652
DEC_MAGIC = 0x534E504C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True, help="voice key, e.g. chilean")
    parser.add_argument("--espeak-voice", required=True, help="e.g. es-419")
    parser.add_argument("--g2p-voice-slot", type=int, required=True)
    parser.add_argument("--duration-checkpoint", type=Path, required=True)
    parser.add_argument("--acoustic-checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--weights", choices=("f16", "f32"), default="f16")
    return parser.parse_args()


def load_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            config = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
            return value, config
    raise SystemExit(f"{path}: no model state dict found")


def block_slots(state: dict[str, torch.Tensor], prefix: str) -> list[np.ndarray]:
    """scale, conv1.weight, conv1.bias, conv2.weight, conv2.bias."""
    return [
        state[f"{prefix}.scale"].reshape(-1),
        state[f"{prefix}.net.0.weight"].reshape(-1),
        state[f"{prefix}.net.0.bias"].reshape(-1),
        state[f"{prefix}.net.2.weight"].reshape(-1),
        state[f"{prefix}.net.2.bias"].reshape(-1),
    ]


def build_front(
    duration_state: dict[str, torch.Tensor],
    duration_config: dict[str, Any],
    acoustic_state: dict[str, torch.Tensor],
    acoustic_config: dict[str, Any],
) -> tuple[bytes, dict[str, int], int]:
    d_hidden = int(duration_config["hidden"])
    d_depth = int(duration_config["depth"])
    d_kernel = int(duration_config["kernel_size"])
    d_max_tokens = int(duration_config.get("max_tokens") or 512)
    d_max_duration = int(duration_config.get("max_duration") or 80)
    d_vocab = int(duration_config.get("vocab_size") or duration_state["embedding.weight"].shape[0])

    a_hidden = int(acoustic_config["hidden"])
    a_token_depth = int(acoustic_config["token_depth"])
    a_depth = int(acoustic_config["depth"])
    a_kernel = int(acoustic_config["kernel_size"])
    a_out = int(acoustic_config.get("out_channels") or acoustic_state["output.weight"].shape[0])
    a_vocab = int(acoustic_config.get("vocab_size") or acoustic_state["embedding.weight"].shape[0])

    tensors: list[np.ndarray] = [
        duration_state["embedding.weight"].reshape(-1),
        duration_state["input_proj.weight"].reshape(-1),
        duration_state["input_proj.bias"].reshape(-1),
    ]
    for i in range(d_depth):
        tensors.extend(block_slots(duration_state, f"blocks.{i}"))
    tensors.extend(
        [
            duration_state["output.weight"].reshape(-1),
            duration_state["output.bias"].reshape(-1),
        ]
    )
    tensors.extend(
        [
            acoustic_state["embedding.weight"].reshape(-1),
            acoustic_state["token_input_proj.weight"].reshape(-1),
            acoustic_state["token_input_proj.bias"].reshape(-1),
        ]
    )
    for i in range(a_token_depth):
        tensors.extend(block_slots(acoustic_state, f"token_blocks.{i}"))
    tensors.extend(
        [
            acoustic_state["frame_input_proj.weight"].reshape(-1),
            acoustic_state["frame_input_proj.bias"].reshape(-1),
        ]
    )
    for i in range(a_depth):
        tensors.extend(block_slots(acoustic_state, f"frame_blocks.{i}"))
    tensors.extend(
        [
            acoustic_state["output.weight"].reshape(-1),
            acoustic_state["output.bias"].reshape(-1),
        ]
    )

    dims = {
        "d_vocab": d_vocab,
        "d_hidden": d_hidden,
        "d_depth": d_depth,
        "d_kernel": d_kernel,
        "d_max_tokens": d_max_tokens,
        "d_max_duration": d_max_duration,
        "a_vocab": a_vocab,
        "a_hidden": a_hidden,
        "a_token_depth": a_token_depth,
        "a_depth": a_depth,
        "a_kernel": a_kernel,
        "a_out": a_out,
        "adapter_mode": 0,
        "adapter_kernel": 0,
        "adapter_rank": 0,
    }
    header_words = [FRONT_MAGIC, 1] + [dims[k] for k in (
        "d_vocab", "d_hidden", "d_depth", "d_kernel", "d_max_tokens", "d_max_duration",
        "a_vocab", "a_hidden", "a_token_depth", "a_depth", "a_kernel", "a_out",
        "adapter_mode", "adapter_kernel", "adapter_rank",
    )] + [len(tensors)]
    return serialize(header_words, tensors, meta_offset=72)


def build_dec(dec_state: dict[str, torch.Tensor], dec_config: dict[str, Any]) -> tuple[bytes, dict[str, int], int]:
    in_ch = int(dec_config["in_channels"])
    channels = [int(c) for c in dec_config["channels"]]
    if len(channels) != 4:
        raise SystemExit(f"piperlite bundle expects 4 channel widths, got {channels}")
    c0, c1, c2, c3 = channels
    if str(dec_config.get("variant")) != "piperlite":
        raise SystemExit(f"only piperlite decoders export, got {dec_config.get('variant')!r}")

    up_specs = ((c0, c1, 16), (c1, c2, 16), (c2, c3, 8))
    tensors: list[np.ndarray] = [
        dec_state["pre.weight"].reshape(-1),
        dec_state["pre.bias"].reshape(-1),
    ]
    for stage, (cin, cout, _k) in enumerate(up_specs):
        tensors.append(dec_state[f"up{stage}.weight"].reshape(-1))
        tensors.append(dec_state[f"up{stage}.bias"].reshape(-1))
        for branch in range(3):
            bank = dec_state
            tensors.append(bank[f"res{stage}.blocks.{branch}.conv1.weight"].reshape(-1))
            tensors.append(bank[f"res{stage}.blocks.{branch}.conv1.bias"].reshape(-1))
            tensors.append(bank[f"res{stage}.blocks.{branch}.conv2.weight"].reshape(-1))
            tensors.append(bank[f"res{stage}.blocks.{branch}.conv2.bias"].reshape(-1))
    tensors.extend(
        [
            dec_state["post.weight"].reshape(-1),
            dec_state["post.bias"].reshape(-1),
        ]
    )

    dims = {
        "in_ch": in_ch,
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "pf_channels": 0,
        "pf_layers": 0,
        "pf_kernel": 9,
        "pf_scale": 0.25,
    }
    header_words = (
        [DEC_MAGIC, 1, dims["in_ch"], c0, c1, c2, c3,
         dims["pf_channels"], dims["pf_layers"], dims["pf_kernel"]] 
        + [struct.unpack("<i", struct.pack("<f", dims["pf_scale"]))[0]]
        + [len(tensors)]
    )
    return serialize(header_words, tensors, meta_offset=48)


def serialize(
    header_words: list[int], tensors: list[np.ndarray], *, meta_offset: int
) -> tuple[bytes, dict[str, int], int]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for tensor in tensors:
        size = int(tensor.numel if hasattr(tensor, "numel") else tensor.size)
        size = int(np.asarray(tensor).size)
        offsets.append((cursor, size))
        cursor += size
    table = b"".join(struct.pack("<ii", off, size) for off, size in offsets)
    header = struct.pack(f"<{len(header_words)}i", *header_words)
    meta_bytes = meta_offset + len(table)
    weights = np.concatenate([np.asarray(t, dtype=np.float32).reshape(-1) for t in tensors])
    return header + table + weights.tobytes(), {"meta_bytes": meta_bytes}, int(weights.size)


def widen_and_hash(blob: bytes, meta_bytes: int, weight_floats: int, dtype: str) -> tuple[bytes, str]:
    """Cast the weight payload and return (blob, sha256 of the WIDENED bytes).

    The browser loader verifies meta.json's front_widened_sha256 against the
    fp32-widened blob, so the hash must be computed the same way.
    """
    header = blob[:meta_bytes]
    weights = np.frombuffer(blob, dtype=np.float32, count=weight_floats, offset=meta_bytes)
    if dtype == "f16":
        stored = weights.astype(np.float16)
    else:
        stored = weights
    stored_bytes = stored.tobytes()
    # widened = header + f32 bytes (exactly what widenF16() produces in the browser)
    widened = header + weights.astype(np.float32).tobytes()
    digest = hashlib.sha256(widened).hexdigest()
    return header + stored_bytes, digest


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    duration_state, duration_config = load_state(args.duration_checkpoint)
    acoustic_state, acoustic_config = load_state(args.acoustic_checkpoint)
    dec_state, dec_config = load_state(args.decoder_checkpoint)

    front_f32, front_meta, front_floats = build_front(duration_state, duration_config, acoustic_state, acoustic_config)
    dec_f32, dec_meta, dec_floats = build_dec(dec_state, dec_config)

    front_blob, front_sha = widen_and_hash(front_f32, front_meta["meta_bytes"], front_floats, args.weights)
    dec_blob, dec_sha = widen_and_hash(dec_f32, dec_meta["meta_bytes"], dec_floats, args.weights)

    front_path = args.out_dir / ("front_f16.bin" if args.weights == "f16" else "front_f32.bin")
    dec_path = args.out_dir / ("dec_f16.bin" if args.weights == "f16" else "dec_f32.bin")
    front_path.write_bytes(front_blob)
    dec_path.write_bytes(dec_blob)

    front_dims = {**front_meta, "weight_floats": front_floats}
    dec_dims = {**dec_meta, "weight_floats": dec_floats}
    meta = {
        "key": args.key,
        "espeak_voice": args.espeak_voice,
        "g2p_voice_slot": args.g2p_voice_slot,
        "length_scale": args.length_scale,
        "sample_rate": args.sample_rate,
        "front_bytes": len(front_blob),
        "dec_bytes": len(dec_blob),
        "front_dims": front_dims,
        "dec_dims": dec_dims,
        "checkpoints": {
            "duration": str(args.duration_checkpoint),
            "acoustic": str(args.acoustic_checkpoint),
            "decoder": str(args.decoder_checkpoint),
        },
        "front": front_path.name,
        "front_widened_sha256": front_sha,
        "dec": dec_path.name,
        "dec_widened_sha256": dec_sha,
        "weights": args.weights,
        "weights_note": (
            "float16 storage, widened to float32 by the loader before it reaches the runtime"
            if args.weights == "f16"
            else "float32 storage"
        ),
    }
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "front_bytes": len(front_blob),
        "dec_bytes": len(dec_blob),
        "front_sha256": front_sha[:16] + "...",
        "dec_sha256": dec_sha[:16] + "...",
        "meta": str(meta_path),
    }, indent=2))


if __name__ == "__main__":
    main()
