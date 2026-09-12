#!/usr/bin/env python3
"""Dump stage-0 subgraph: conv attributes, Add inputs, Div constant."""
import sys
from pathlib import Path

import onnx
from onnx import numpy_helper

model = onnx.load(sys.argv[1])
inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
by_input = {}
for node in model.graph.node:
    for out in node.output:
        by_input[out] = node

interesting = [
    "/dec/ups.0/ConvTranspose_output_0",
    "/dec/resblocks.0/LeakyRelu_output_0",
    "/dec/resblocks.0/convs.0/Conv_output_0",
    "/dec/resblocks.0/Add_output_0",
    "/dec/resblocks.0/LeakyRelu_1_output_0",
    "/dec/resblocks.0/convs.1/Conv_output_0",
    "/dec/resblocks.0/Add_1_output_0",
    "/dec/resblocks.1/convs.0/Conv_output_0",
    "/dec/Add_output_0",
    "/dec/Constant_output_0",
    "/dec/Div_output_0",
]
for node in model.graph.node:
    label = node.name or node.output[0]
    if label in ("/dec/resblocks.0/Conv",) or (node.name and node.name.startswith("/dec/resblocks.0")) or node.name in ("/dec/Add", "/dec/Constant", "/dec/Div"):
        attrs = {a.name: (list(a.ints) if a.ints else a.i if a.type == 2 else a.f) for a in node.attribute}
        print(f"{node.op_type:14s} {node.name}")
        print(f"   inputs: {list(node.input)}")
        print(f"   outputs: {list(node.output)}")
        if attrs:
            print(f"   attrs: {attrs}")
# what feeds the Adds?
def producers(name):
    n = by_input.get(name)
    return n.name if n else ("INPUT" if any(i.name == name for i in model.graph.input) else "CONST?")
for target in ("/dec/resblocks.0/Add_output_0", "/dec/resblocks.0/Add_1_output_0", "/dec/Add_output_0", "/dec/Div_output_0"):
    n = by_input.get(target)
    if n:
        print(f"\n{target} producers:")
        for inp in n.input:
            print(f"    {inp}  <- {producers(inp)}")
