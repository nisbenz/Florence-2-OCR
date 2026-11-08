"""Export the Florence-2-base decoder to one ONNX graph with an If node.

Offline only (PyTorch). Writes models/decoder_merged.onnx.

The decoder is traced twice, then both traces are merged:
  prefill (else branch): no cache yet, computes the cross-attention KV from the encoder output
  decode  (then branch): reuses the cache, adds self-attention KV for the new tokens
Weights that both traces share are stored once in the outer graph.

Inputs:
  input_ids                         (B, T)          int64
  encoder_hidden_states             (B, S, 768)     float32
  encoder_attention_mask            (B, S)          float32
  past_key_values.{i}.decoder.key   (B, 12, P, 64)  self-attention cache, i = 0..5 (same for .value)
  past_key_values.{i}.encoder.key   (B, 12, S, 64)  cross-attention cache (same for .value)
  use_cache_branch                  (1,)            bool: False on the first step, True after
On the first step the past inputs are ignored but must still be fed (use P = 0).

Outputs: logits (B, T, vocab) and present.{i}.{decoder,encoder}.{key,value},
which are fed back as the past inputs on the next step.
"""

import argparse
import hashlib
import tempfile
from pathlib import Path

import onnx
import torch
from onnx import helper

from export_encoders import OPSET, load_model

NUM_LAYERS = 6
KV_NAMES = [
    f"{i}.{part}.{kv}" for i in range(NUM_LAYERS) for part in ("decoder", "encoder") for kv in ("key", "value")
]
PAST_NAMES = [f"past_key_values.{n}" for n in KV_NAMES]
PRESENT_NAMES = [f"present.{n}" for n in KV_NAMES]


def flatten(past_key_values):
    # The model's cache is a tuple per layer: (self_k, self_v, cross_k, cross_v).
    return [t for layer in past_key_values for t in layer]


class DecoderPrefill(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm = model.language_model

    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask):
        out = self.lm(
            attention_mask=encoder_attention_mask,
            encoder_outputs=(encoder_hidden_states,),
            decoder_input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        return (out.logits, *flatten(out.past_key_values))


class DecoderWithPast(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm = model.language_model

    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask, *past):
        past_key_values = tuple(tuple(past[i : i + 4]) for i in range(0, len(past), 4))
        out = self.lm(
            attention_mask=encoder_attention_mask,
            encoder_outputs=(encoder_hidden_states,),
            decoder_input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return (out.logits, *flatten(out.past_key_values))


def kv_axes(name):
    # Self-attention cache grows with generated tokens; cross-attention cache has the encoder length.
    return {0: "batch", 2: "past_seq" if ".decoder." in name else "enc_seq"}


def export_branches(model, workdir):
    batch, enc_seq = 2, 587
    encoder_hidden_states = torch.randn(batch, enc_seq, 768)
    encoder_attention_mask = torch.ones(batch, enc_seq)
    shared_axes = {
        "input_ids": {0: "batch", 1: "dec_seq"},
        "encoder_hidden_states": {0: "batch", 1: "enc_seq"},
        "encoder_attention_mask": {0: "batch", 1: "enc_seq"},
        "logits": {0: "batch", 1: "dec_seq"},
    }
    present_axes = {n: {0: "batch", 2: "total_seq" if ".decoder." in n else "enc_seq"} for n in PRESENT_NAMES}

    # Trace prefill with 2 tokens so the causal-mask path is part of the graph.
    prefill_ids = torch.tensor([[2, 0]] * batch)
    with torch.no_grad():
        past = flatten(
            model.language_model(
                attention_mask=encoder_attention_mask,
                encoder_outputs=(encoder_hidden_states,),
                decoder_input_ids=prefill_ids,
                use_cache=True,
                return_dict=True,
            ).past_key_values
        )

        torch.onnx.export(
            DecoderPrefill(model),
            (prefill_ids, encoder_hidden_states, encoder_attention_mask),
            workdir / "prefill.onnx",
            input_names=["input_ids", "encoder_hidden_states", "encoder_attention_mask"],
            output_names=["logits", *PRESENT_NAMES],
            dynamic_axes={**shared_axes, **present_axes},
            opset_version=OPSET,
            dynamo=False,
        )

        torch.onnx.export(
            DecoderWithPast(model),
            (torch.tensor([[100]] * batch), encoder_hidden_states, encoder_attention_mask, *past),
            workdir / "decode.onnx",
            input_names=["input_ids", "encoder_hidden_states", "encoder_attention_mask", *PAST_NAMES],
            output_names=["logits", *PRESENT_NAMES],
            dynamic_axes={**shared_axes, **present_axes, **{n: kv_axes(n) for n in PAST_NAMES}},
            opset_version=OPSET,
            dynamo=False,
        )


def tensor_key(init):
    data = init.raw_data or onnx.numpy_helper.to_array(init).tobytes()
    return init.data_type, tuple(init.dims), hashlib.sha256(data).hexdigest()


def as_branch(graph, prefix, shared, shared_inits):
    """Turn a traced graph into an If branch: weights move to the outer graph, deduplicated by content."""
    rename = {}
    for init in graph.initializer:
        key = tensor_key(init)
        if key not in shared:
            shared[key] = f"w{len(shared)}"
            new_init = onnx.TensorProto()
            new_init.CopyFrom(init)
            new_init.name = shared[key]
            shared_inits.append(new_init)
        rename[init.name] = shared[key]

    # Prefix every value the branch computes, so the two branches and the outer graph never share a name.
    for node in graph.node:
        for i, name in enumerate(node.output):
            if name:
                rename[name] = prefix + name
                node.output[i] = prefix + name
        node.name = prefix + node.name
    for node in graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = rename.get(name, name)

    # A graph output can be a graph input passed straight through: it needs a node to define it.
    outputs = []
    for out in graph.output:
        source = rename.get(out.name, out.name)
        if not source.startswith(prefix):
            graph.node.append(helper.make_node("Identity", [source], [prefix + out.name]))
            source = prefix + out.name
        value = onnx.ValueInfoProto()
        value.CopyFrom(out)
        value.name = source
        outputs.append(value)

    branch = helper.make_graph(list(graph.node), prefix.rstrip("/"), [], outputs)
    return branch


def merge(workdir, out_path):
    prefill = onnx.load(workdir / "prefill.onnx")
    decode = onnx.load(workdir / "decode.onnx")

    shared, shared_inits = {}, []
    else_branch = as_branch(prefill.graph, "prefill/", shared, shared_inits)
    then_branch = as_branch(decode.graph, "decode/", shared, shared_inits)

    inputs = list(decode.graph.input)
    names = {i.name for i in inputs}
    inputs += [i for i in prefill.graph.input if i.name not in names]
    inputs.append(helper.make_tensor_value_info("use_cache_branch", onnx.TensorProto.BOOL, [1]))

    output_names = ["logits", *PRESENT_NAMES]
    if_node = helper.make_node(
        "If", ["use_cache_branch"], output_names, then_branch=then_branch, else_branch=else_branch
    )
    # Sequence dims differ between the branches, so the outer outputs only fix the rank.
    outputs = [
        helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [None] * (3 if n == "logits" else 4))
        for n in output_names
    ]

    graph = helper.make_graph([if_node], "florence2_decoder", inputs, outputs, shared_inits)
    merged = helper.make_model(graph, opset_imports=decode.opset_import, ir_version=decode.ir_version)
    onnx.checker.check_model(merged)
    onnx.save(merged, out_path)

    sizes = {p.name: p.stat().st_size >> 20 for p in (workdir / "prefill.onnx", workdir / "decode.onnx", out_path)}
    print(f"Sizes in MB: {sizes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, _ = load_model()
    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        export_branches(model, Path(tmp))
        merge(Path(tmp), args.out / "decoder_merged.onnx")

    print(f"Wrote {args.out}/decoder_merged.onnx")


if __name__ == "__main__":
    main()
