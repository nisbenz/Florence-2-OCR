"""Where one request's time goes: preprocessing, each ONNX graph, and the Python beam-search glue.

Runs the pipeline directly (no HTTP), so use the server venv:

    python bench/profile_stages.py --image samples/example.jpg --threads 4
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import onnxruntime as ort
from PIL import Image

SERVER = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(SERVER))
import pipeline as pl  # noqa: E402


def timed(totals, counts, name, fn):
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        totals[name] += time.perf_counter() - t
        counts[name] += 1
        return out

    return wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=SERVER.parent / "models")
    parser.add_argument("--threads", type=int, default=0, help="intra-op threads, 0 = ORT default")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--int8", nargs="*", default=[], help="graphs to load as <name>.int8.onnx")
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")  # as in the server
    p = pl.OcrPipeline(args.models, options, set(args.int8))
    image = Image.open(args.image)
    p([image])  # warm-up

    totals, counts = defaultdict(float), defaultdict(int)
    pl.preprocess = timed(totals, counts, "preprocess", pl.preprocess)
    for name in ("vision", "text_encoder", "decoder"):
        session = getattr(p, name)
        session.run = timed(totals, counts, name, session.run)

    start = time.perf_counter()
    p([image] * args.batch)
    total = time.perf_counter() - start
    glue = total - sum(totals.values())
    print(f"threads={args.threads or 'default'} batch={args.batch} total {total:.2f} s")
    for name, s in [*totals.items(), ("beam-search glue (numpy)", glue)]:
        calls = f"{counts[name]:>5} calls" if name in counts else ""
        print(f"  {name:<26}{s:>7.2f} s {100 * s / total:>5.1f}% {calls}")


if __name__ == "__main__":
    main()
