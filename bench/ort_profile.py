"""Per-operator time inside each ONNX graph, from ONNX Runtime's own profiler (enable_profiling).

Runs one request through the pipeline with profiling on, then sums the "Node" events per op type.
The raw JSON traces open in https://ui.perfetto.dev. Use the server venv:

    python bench/ort_profile.py --image samples/example.jpg --threads 4 --int8 vision_encoder text_encoder decoder_merged
"""

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import onnxruntime as ort
from PIL import Image

SERVER = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(SERVER))
from pipeline import OcrPipeline  # noqa: E402


def summarize(trace: Path, top: int):
    by_op, count = defaultdict(float), defaultdict(int)
    for event in json.loads(trace.read_text()):
        if event.get("cat") == "Node" and event["name"].endswith("_kernel_time"):
            op = event["args"]["op_name"]
            by_op[op] += event["dur"] / 1000  # µs -> ms
            count[op] += 1
    total = sum(by_op.values())
    for op, ms in sorted(by_op.items(), key=lambda x: -x[1])[:top]:
        print(f"    {op:<24}{ms:>9.1f} ms {100 * ms / total:>5.1f}% {count[op]:>7} runs")
    print(f"    {'total kernel time':<24}{total:>9.1f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=SERVER.parent / "models")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--int8", nargs="*", default=[])
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--keep", type=Path, help="directory to keep the JSON traces in")
    args = parser.parse_args()

    out = args.keep or Path(tempfile.mkdtemp())
    out.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.enable_profiling = True
    options.profile_file_prefix = str(out / "ort")
    p = OcrPipeline(args.models, options, set(args.int8))
    p([Image.open(args.image)])
    for name in ("vision", "text_encoder", "decoder"):
        trace = Path(getattr(p, name).end_profiling())
        print(f"{name}  ({trace.name})")
        summarize(trace, args.top)


if __name__ == "__main__":
    main()
