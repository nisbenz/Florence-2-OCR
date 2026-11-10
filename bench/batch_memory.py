"""Peak RAM and time per batch size, to choose MAX_BATCH under the 4 GB limit.

Each batch size runs in a fresh process, so the peak RSS belongs to that size alone. Runs the pipeline
directly (no HTTP), so use the server venv.

    python bench/batch_memory.py --image samples/example.jpg --sizes 1 2 4 8
"""

import argparse
import resource
import subprocess
import sys
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent / "server"


def measure(image_path: Path, size: int, models: Path):
    sys.path.insert(0, str(SERVER))
    from PIL import Image

    from pipeline import OcrPipeline

    pipeline = OcrPipeline(models)
    image = Image.open(image_path)
    loaded_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    pipeline([image])  # warm-up at batch 1
    start = time.perf_counter()
    pipeline([image] * size)
    seconds = time.perf_counter() - start
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KB
    print(f"{size:>5}{loaded_mb:>10.0f}{peak_mb:>10.0f}{seconds:>9.2f}{size / seconds:>9.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--models", type=Path, default=SERVER.parent / "models")
    parser.add_argument("--one", type=int, help=argparse.SUPPRESS)  # internal: measure one size
    args = parser.parse_args()

    if args.one:
        measure(args.image, args.one, args.models)
        return
    print(f"{'batch':>5}{'load MB':>10}{'peak MB':>10}{'batch s':>9}{'img/s':>9}")
    for size in args.sizes:
        cmd = [sys.executable, __file__, "--image", str(args.image), "--models", str(args.models), "--one", str(size)]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
