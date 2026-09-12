#!/usr/bin/env python3
"""List /dec/ scoped node value names + the Conv feeding dec (latent candidate)."""
import sys
import onnx

model = onnx.load(sys.argv[1])
for node in model.graph.node:
    label = node.name or node.output[0]
    if "/dec/" in label or (node.name and node.name.startswith("dec")) or "dec." in label:
        print(f"{node.op_type:16s} name={label[:60]:60s} out={node.output[0][:60]}")
