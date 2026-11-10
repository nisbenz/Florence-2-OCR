"""Accuracy of the pipeline itself (no HTTP, no Docker), to screen a model variant before building an image.

    python bench/accuracy_direct.py --samples samples --references references --int8 vision_encoder
"""

import argparse
import sys
from pathlib import Path

import onnxruntime as ort
from PIL import Image

SERVER = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(SERVER))
from accuracy import score  # noqa: E402
from pipeline import OcrPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=SERVER.parent / "models")
    parser.add_argument("--int8", nargs="*", default=[])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    p = OcrPipeline(args.models, options, set(args.int8))
    scores = []
    for path in sorted(args.samples.iterdir()):
        text = p([Image.open(path)])[0]
        ref = (args.references / f"{path.stem}.txt").read_text(encoding="utf-8")
        scores.append(score(text, ref))
        print(f"{100 * scores[-1]:6.1f}%  {path.name}  {text[:60]!r}")
    print(f"accuracy: {100 * sum(scores) / len(scores):.2f}% over {len(scores)} images")


if __name__ == "__main__":
    main()
