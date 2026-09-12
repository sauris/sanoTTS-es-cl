#!/usr/bin/env python3
import onnx

m = onnx.load("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/voices/es_CL/run/decoder-cut/es_CL-huemul-medium-decoder-from-generator-input.onnx")
print("INPUTS:", [i.name for i in m.graph.input])
print("OUTPUTS:", [o.name for o in m.graph.output])
for node in m.graph.node:
    if node.name in ("/dec/conv_pre/Conv", "dec.conv_pre/Conv") or node.op_type == "Conv":
        print("first conv:", node.name, "inputs:", list(node.input))
        break
