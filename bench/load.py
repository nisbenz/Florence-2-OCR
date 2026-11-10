"""Throughput under concurrency, as the challenge scores it (section 6, step 2).

Concurrency rises step by step. In each step, `concurrency` clients send requests back to back for
`--duration` seconds. A request succeeds if it returns 200 with a "text" field within 30 s, and a step
passes at >= 99% success. The result is the highest requests/second of a passing step.
The official run uses 2-minute steps (--duration 120).

    python bench/load.py --url http://localhost:8000 --samples samples --steps 1 2 4 8 16
"""

import argparse
import threading
import time
from pathlib import Path

import httpx

from common import load_samples, percentile, post_ocr

PASS_RATE = 0.99


def run_step(url, samples, concurrency, duration):
    results = []  # (ok, seconds), appended from every client thread
    lock = threading.Lock()
    end = time.perf_counter() + duration

    def client_loop(offset):
        i = offset
        with httpx.Client() as client:
            while time.perf_counter() < end:
                status, text, seconds = post_ocr(client, url, *samples[i % len(samples)])
                with lock:
                    results.append((status == 200 and text is not None, seconds))
                i += 1

    threads = [threading.Thread(target=client_loop, args=(c,)) for c in range(concurrency)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    ok = [s for good, s in results if good]
    return {
        "sent": len(results),
        "success": len(ok) / len(results) if results else 0.0,
        "rps": len(ok) / elapsed,
        "p50": percentile(ok, 50) if ok else float("nan"),
        "p95": percentile(ok, 95) if ok else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--duration", type=float, default=30, help="seconds per step (official: 120)")
    args = parser.parse_args()

    samples = load_samples(args.samples)
    best = None
    print(f"{'conc':>5}{'sent':>7}{'success':>9}{'rps':>8}{'p50 s':>8}{'p95 s':>8}")
    for concurrency in args.steps:
        r = run_step(args.url, samples, concurrency, args.duration)
        passed = r["success"] >= PASS_RATE
        print(f"{concurrency:>5}{r['sent']:>7}{r['success']:>9.1%}{r['rps']:>8.2f}"
              f"{r['p50']:>8.2f}{r['p95']:>8.2f}  {'pass' if passed else 'FAIL'}")
        if passed and (best is None or r["rps"] > best):
            best = r["rps"]

    print(f"\nmax sustained rps (>= 99% success): {best:.2f}" if best else "\nno step passed")


if __name__ == "__main__":
    main()
