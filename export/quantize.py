"""int8 versions of the exported graphs. Runs after export_encoders.py and export_decoder.py.

Writes <name>.int8.onnx next to each fp32 graph; the server picks per graph which one to load (INT8).

- Encoders: dynamic quantization of MatMul (int8 weights offline, activations quantized per call).
- Decoder: quantize_dynamic finds nothing to do there. Its weights are top-level initializers used
  from inside the If branches, and it only looks for weights in the branch's own scope. The
  weight-only MatMulNBits quantizer follows outer-scope weights, so the decoder uses that (8 bits,
  blocks of 32, accuracy_level 4 = int8 compute).

    python quantize.py --models ../models
"""

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

DECODER_BITS, DECODER_BLOCK = 8, 32


def quantize_encoder(src: Path, dst: Path):
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])


def quantize_decoder(src: Path, dst: Path):
    config = DefaultWeightOnlyQuantConfig(
        block_size=DECODER_BLOCK, is_symmetric=True, accuracy_level=4, bits=DECODER_BITS
    )
    quantizer = MatMulNBitsQuantizer(
        onnx.load(src), bits=DECODER_BITS, block_size=DECODER_BLOCK, is_symmetric=True, accuracy_level=4,
        algo_config=config,
    )
    quantizer.process()
    onnx.save(quantizer.model.model, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()
    for name, quantize in [
        ("vision_encoder", quantize_encoder),
        ("text_encoder", quantize_encoder),
        ("decoder_merged", quantize_decoder),
    ]:
        src, dst = args.models / f"{name}.onnx", args.models / f"{name}.int8.onnx"
        quantize(src, dst)
        print(f"{dst.name}: {src.stat().st_size / 1e6:.0f} MB -> {dst.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
