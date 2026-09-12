#!/usr/bin/env python3
"""Validate the signature pack with the decoder trainer's own loader."""
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/tools")
import numpy as np

from train_roota_piper_decoder_student import load_signature_index, load_signature_tensors

sig_dir = Path("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/es_cl/chain-test/signatures")
index = load_signature_index(sig_dir)
print("index entries:", len(index))
keys = ["stage0_mix", "stage1_mix", "stage2_mix", "pre_tanh", "audio"]
first_source = next(iter(index))
with np.load(first_source) as z:
    frames = int(np.asarray(z["generator_input"]).shape[-1])
tensors = load_signature_tensors(
    index[first_source], keys, frames, phase_bins=0, exact_feature_keys=[]
)
for name, value in tensors.items():
    print(f"{name:22s} {value.shape} {value.dtype}")
print("SIGNATURE_LOADER_OK")
