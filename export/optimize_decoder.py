"""Graph surgery on the merged decoder: stop re-processing the cross-attention cache on every step.

Runs after export_decoder.py (and quantize.py), on each decoder file, in place:

    python optimize_decoder.py ../models/decoder_merged.onnx ../models/decoder_merged.int8.onnx

The cross-attention cache is computed once by the prefill branch and never changes. The exported
cached-step branch still did two things with it on every step (found with bench/ort_profile.py):

1. Transpose each layer's cross key, K -> Kᵀ, for Q·Kᵀ (6 Transposes of batch×12×587×64 per step).
   Now the prefill branch writes the key already transposed (batch×12×64×enc_seq) and the cached
   branch uses it as is.
2. Copy the cache to its outputs with Identity, only because both If branches must have the same
   outputs (12 copies per step). Now the cached branch outputs an empty tensor instead. The
   pipeline never fetches those outputs on cached steps.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

NUM_LAYERS, NUM_HEADS, HEAD_DIM = 6, 12, 64


def branches(model):
    (if_node,) = [n for n in model.graph.node if n.op_type == "If"]
    graphs = {a.name: a.g for a in if_node.attribute}
    return graphs["else_branch"], graphs["then_branch"]  # prefill (use_cache False), cached step


def set_dims(value_info, dims):
    """dims: str = symbolic, int = fixed, None = unknown."""
    shape = value_info.type.tensor_type.shape
    shape.ClearField("dim")
    for d in dims:
        dim = shape.dim.add()
        if isinstance(d, str):
            dim.dim_param = d
        elif d is not None:
            dim.dim_value = d


def rename_input(graph, old, new):
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new


def transpose_key_in_prefill(prefill, layer):
    out = prefill.output[4 * layer + 3]  # logits, then per layer: self k, self v, cross k, cross v
    assert out.name.endswith(f"present.{layer}.encoder.key"), out.name
    inner = out.name + "_untransposed"
    for node in prefill.node:
        node.output[:] = [inner if o == out.name else o for o in node.output]
    rename_input(prefill, out.name, inner)  # the prefill's own attention keeps the original layout
    prefill.node.append(helper.make_node("Transpose", [inner], [out.name], perm=[0, 1, 3, 2]))
    set_dims(out, ["batch", NUM_HEADS, HEAD_DIM, "enc_seq"])


def drop_key_transpose_in_step(step, layer):
    past = f"past_key_values.{layer}.encoder.key"
    (t,) = [n for n in step.node if n.op_type == "Transpose" and n.input[0] == past]
    assert list(helper.get_attribute_value(t.attribute[0])) == [0, 1, 3, 2]
    step.node.remove(t)
    rename_input(step, t.output[0], past)


def empty_cross_outputs_in_step(step):
    empty = numpy_helper.from_array(np.zeros((0, NUM_HEADS, 0, HEAD_DIM), np.float32))
    for out in step.output:
        if ".encoder." not in out.name:
            continue
        (copy,) = [n for n in step.node if n.op_type == "Identity" and n.output[0] == out.name]
        step.node.remove(copy)
        step.node.append(helper.make_node("Constant", [], [out.name], value=empty))
        set_dims(out, [None] * 4)


def optimize(path: Path):
    model = onnx.load(path)
    prefill, step = branches(model)
    for layer in range(NUM_LAYERS):
        transpose_key_in_prefill(prefill, layer)
        drop_key_transpose_in_step(step, layer)
        key = next(i for i in model.graph.input if i.name == f"past_key_values.{layer}.encoder.key")
        set_dims(key, ["batch", NUM_HEADS, HEAD_DIM, "enc_seq"])
    empty_cross_outputs_in_step(step)
    for out in model.graph.output:  # the two branches now give different shapes here
        if ".encoder." in out.name:
            set_dims(out, [None] * 4)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"optimized {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    for path in parser.parse_args().paths:
        optimize(path)


if __name__ == "__main__":
    main()
