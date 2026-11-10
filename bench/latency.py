"""Single-request latency, as the challenge scores it (section 6, step 1): one request at a time.

Warm-up requests are sent first and not counted.

    python bench/latency.py --url http://localhost:8000 --samples samples --rounds 3
"""

import argparse
from pathlib import Path

import httpx

from common import load_samples, percentile, post_ocr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3, help="passes over the sample set")
    args = parser.parse_args()

    samples = load_samples(args.samples)
    times, failed = [], 0
    with httpx.Client() as client:
        for i in range(args.warmup):
            post_ocr(client, args.url, *samples[i % len(samples)])
        for _ in range(args.rounds):
            for name, data in samples:
                status, _, seconds = post_ocr(client, args.url, name, data)
                if status == 200:
                    times.append(seconds)
                else:
                    failed += 1

    if not times:
        raise SystemExit("Every request failed")
    print(f"requests: {len(times)} ok, {failed} failed")
    for p in (50, 95, 99):
        print(f"p{p}: {percentile(times, p) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
