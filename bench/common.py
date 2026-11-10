"""Shared helpers for the bench scripts: sample loading, one /ocr call, percentiles."""

import math
import time
from pathlib import Path

import httpx

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TIMEOUT_S = 30  # a request slower than this counts as failed, as in the official load test


def load_samples(samples_dir: Path) -> list[tuple[str, bytes]]:
    paths = sorted(p for p in samples_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"No .jpg/.png images in {samples_dir}")
    return [(p.name, p.read_bytes()) for p in paths]


def post_ocr(client: httpx.Client, url: str, name: str, data: bytes) -> tuple[int, str | None, float]:
    """Returns (status, text or None, seconds). Status 0 means the request didn't complete."""
    t = time.perf_counter()
    try:
        r = client.post(f"{url}/ocr", files={"image": (name, data)}, timeout=TIMEOUT_S)
        text = r.json().get("text") if r.status_code == 200 else None
        return r.status_code, text, time.perf_counter() - t
    except (httpx.HTTPError, ValueError):
        return 0, None, time.perf_counter() - t


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]
