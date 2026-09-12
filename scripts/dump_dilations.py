#!/usr/bin/env python3
import onnx

m = onnx.load("/mnt/c/Users/kuco/Documents/dev/playgr/sanoTTS/artifacts/es_cl/chain-test/decoder-cut/davefx-medium-decoder-from-generator-input.onnx")
for n in m.graph.node:
    if n.op_type == "Conv" and n.name and "resblocks" in n.name:
        d = [a for a in n.attribute if a.name == "dilations"]
        print(n.name, list(d[0].ints) if d else "none")
