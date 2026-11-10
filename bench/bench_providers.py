"""Latency benchmark: OpenVINO EP vs plain CPU EP in onnxruntime.

Needs the onnxruntime-openvino wheel (it replaces plain onnxruntime and still
ships the CPU EP), so run it from its own venv:

    uv venv bench/.venv --python 3.12
    uv pip install -p bench/.venv onnxruntime-openvino numpy
    bench/.venv/bin/python bench/bench_providers.py
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parent.parent / "models"

PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "openvino": [("OpenVINOExecutionProvider", {"device_type": "CPU"})],
}

# Values for the symbolic dims. Decoder is benchmarked as one cached step,
# which is the per-token cost inside the generation loop.
ENC_SEQ = 577 + 10  # image tokens + prompt tokens


def dims(past_seq):
    return {"batch": 1, "dec_seq": 1, "past_seq": past_seq, "enc_seq": ENC_SEQ}


DTYPES = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(bool)": np.bool_}


def make_inputs(sess, past_seq, rng):
    feeds = {}
    d = dims(past_seq)
    for i in sess.get_inputs():
        shape = [d[s] if isinstance(s, str) else s for s in i.shape]
        dtype = DTYPES[i.type]
        if i.name == "use_cache_branch":
            feeds[i.name] = np.array([True])
        elif i.name == "input_ids":
            feeds[i.name] = rng.integers(0, 50000, size=shape, dtype=np.int64)
        elif "attention_mask" in i.name:
            feeds[i.name] = np.ones(shape, dtype=dtype)
        else:
            feeds[i.name] = rng.standard_normal(shape).astype(dtype)
    return feeds


def bench(model, provider, feeds, warmup, runs):
    opts = ort.SessionOptions()
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(MODELS / model), opts, providers=PROVIDERS[provider])
    load_s = time.perf_counter() - t0
    active = sess.get_providers()[0]
    for _ in range(warmup):
        sess.run(None, feeds)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        out = sess.run(None, feeds)
        times.append((time.perf_counter() - t) * 1000)
    return active, load_s, times, out[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["vision_encoder.onnx", "text_encoder.onnx", "decoder_merged.onnx"])
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--past-seq", type=int, default=32, help="KV cache length for the decoder step")
    args = p.parse_args()

    avail = ort.get_available_providers()
    print(f"onnxruntime {ort.__version__}, providers: {avail}")
    if "OpenVINOExecutionProvider" not in avail:
        print("WARNING: OpenVINO EP not available; install onnxruntime-openvino")

    print(f"\n{'model':<22}{'provider':<11}{'load s':>8}{'mean ms':>10}{'p50 ms':>9}"
          f"{'p90 ms':>9}{'min ms':>9}{'max |diff|':>12}")
    for model in args.models:
        rng = np.random.default_rng(0)
        probe = ort.InferenceSession(str(MODELS / model), providers=["CPUExecutionProvider"])
        feeds = make_inputs(probe, args.past_seq, rng)
        del probe
        ref = None
        for name in PROVIDERS:
            if name == "openvino" and "OpenVINOExecutionProvider" not in avail:
                continue
            active, load_s, t, out = bench(model, name, feeds, args.warmup, args.runs)
            if active == "CPUExecutionProvider" and name != "cpu":
                print(f"  ({name} fell back to CPU EP)")
            diff = "" if ref is None else f"{np.abs(out - ref).max():.2e}"
            ref = out if ref is None else ref
            q = statistics.quantiles(t, n=10)
            print(f"{model:<22}{name:<11}{load_s:>8.2f}{statistics.mean(t):>10.1f}"
                  f"{statistics.median(t):>9.1f}{q[8]:>9.1f}{min(t):>9.1f}{diff:>12}")


if __name__ == "__main__":
    main()
