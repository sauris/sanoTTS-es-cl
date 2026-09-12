#!/usr/bin/env python3
"""Debug: compare my PiperDecoderTeacher stage-by-stage against the cut ONNX."""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

sys.path.insert(0, "/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/tools")
from verify_piper_decoder_torch_parity import PiperDecoderTeacher, load_weights

import torch.nn.functional as F

cut = Path(sys.argv[1])
pack_npz = Path(sys.argv[2])

model = onnx.load(str(cut))
module = PiperDecoderTeacher()
load_weights(model, module)
module.eval()

# expose intermediates
probe_nodes = [
    "/dec/conv_pre/Conv_output_0",
    "/dec/ups.0/ConvTranspose_output_0",
    "/dec/Div_output_0",
    "/dec/ups.1/ConvTranspose_output_0",
    "/dec/Div_1_output_0",
    "/dec/ups.2/ConvTranspose_output_0",
    "/dec/Div_2_output_0",
    "/dec/conv_post/Conv_output_0",
]
existing = {o.name for o in model.graph.output}
for n in probe_nodes:
    if n not in existing:
        model.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
latent_name = sess.get_inputs()[0].name

with np.load(pack_npz) as z:
    latent = np.asarray(z["generator_input"], dtype=np.float32)

outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {latent_name: latent})))
x = torch.from_numpy(latent)

with torch.no_grad():
    a = module.conv_pre(x)
    def cmp(label, mine, ort_name):
        ref = outs[ort_name]
        mine_np = mine.numpy()
        if mine_np.shape != ref.shape:
            print(f"{label:12s} SHAPE MISMATCH mine={mine_np.shape} ort={ref.shape}")
            return None
        d = float(np.abs(mine_np - ref).max())
        c = float(np.dot(mine_np.ravel(), ref.ravel()) / (np.linalg.norm(mine_np) * np.linalg.norm(ref) + 1e-12))
        print(f"{label:12s} max_abs={d:.6f} cos={c:.6f} shape={ref.shape}")
        return d

    if cmp("conv_pre", a, "/dec/conv_pre/Conv_output_0") is not None and False:
        pass
    # continue chain regardless
    r = F.leaky_relu(a, 0.1)
    u0 = module.ups[0](r)
    if cmp("ups.0", u0, "/dec/ups.0/ConvTranspose_output_0") is None:
        sys.exit()
    s0 = module.stages[0](u0)
    if cmp("stage0", s0, "/dec/Div_output_0") is None:
        sys.exit()
    r1 = F.leaky_relu(s0, 0.1)
    u1 = module.ups[1](r1)
    if cmp("ups.1", u1, "/dec/ups.1/ConvTranspose_output_0") is None:
        sys.exit()
    s1 = module.stages[1](u1)
    if cmp("stage1", s1, "/dec/Div_1_output_0") is None:
        sys.exit()
    r2 = F.leaky_relu(s1, 0.1)
    u2 = module.ups[2](r2)
    if cmp("ups.2", u2, "/dec/ups.2/ConvTranspose_output_0") is None:
        sys.exit()
    s2 = module.stages[2](u2)
    if cmp("stage2", s2, "/dec/Div_2_output_0") is None:
        sys.exit()
    rp = F.leaky_relu(s2)
    p = module.conv_post(rp)
    if cmp("conv_post", p, "/dec/conv_post/Conv_output_0") is None:
        sys.exit()
    t = torch.tanh(p)
    cmp("tanh(audio)", t, "output")
