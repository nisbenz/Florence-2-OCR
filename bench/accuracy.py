"""Accuracy against references/, as the challenge scores it (section 4.5).

Per image: max(0, 1 - CER), CER = edit_distance(text, reference) / len(reference). Averaged over images.
references/<image stem>.txt holds the original model's text for samples/<image>.

    python bench/accuracy.py --url http://localhost:8000 --samples samples --references references
"""

import argparse
from pathlib import Path

import httpx

from common import load_samples, post_ocr


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance over characters."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(text: str, reference: str) -> float:
    if not reference:
        return 1.0 if not text else 0.0
    return max(0.0, 1 - edit_distance(text, reference) / len(reference))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--references", type=Path, default=Path("references"))
    args = parser.parse_args()

    scores = []
    with httpx.Client() as client:
        for name, data in load_samples(args.samples):
            reference = (args.references / f"{Path(name).stem}.txt").read_text(encoding="utf-8")
            status, text, _ = post_ocr(client, args.url, name, data)
            s = score(text, reference) if status == 200 else 0.0
            scores.append(s)
            print(f"{s:6.1%}  {name}" + ("" if status == 200 else f"  (HTTP {status})"))

    print(f"\naccuracy: {sum(scores) / len(scores):.2%} over {len(scores)} images")


if __name__ == "__main__":
    main()
