#!/usr/bin/env python3
"""Inspect a Piper teacher ONNX: decoder-region node names, initializer names,
and candidate generator_input tensor."""
import sys
from pathlib import Path

import onnx
from onnx import numpy_helper

model = onnx.load(sys.argv[1])
graph = model.graph

print("== inputs ==")
for i in graph.input:
    print("  ", i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
print("== outputs ==")
for o in graph.output:
    print("  ", o.name)

names = {}
for node in graph.node:
    for out in node.output:
        if out:
            names[out] = node

print("== decoder-side value names (dec/ups/generator-ish) ==")
count = 0
for node in graph.node:
    label = node.name or node.output[0]
    if any(tag in label for tag in ("dec", "ups", "Div", "Tanh", "conv_post", "conv_pre")):
        if count < 80:
            print(f"  {node.op_type:16s} {label[:70]:70s} -> {node.output[0][:50]}")
        count += 1
print(f"(total matching: {count})")

print("== initializer name patterns (first 40) ==")
inits = [i.name for i in graph.initializer]
for n in inits[:40]:
    print("  ", n)
print(f"(total initializers: {len(inits)})")

init_tensors = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
conv_like = [n for n, a in init_tensors.items() if a.ndim == 3 and a.shape[0] in (256, 128, 64, 32, 1) and a.shape[2] in (3, 5, 7, 8, 16)]
print("== conv-ish initializers (in|out|k == teacher shapes) ==")
for n in conv_like[:80]:
    print("  ", n, init_tensors[n].shape)
