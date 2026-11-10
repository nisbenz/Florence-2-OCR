"""FastAPI backend. No PyTorch: ONNX Runtime for the model, transformers only for CLIPImageProcessor.

Requests go into a bounded queue (full -> fast 503). One inference worker thread takes a batch when
MAX_BATCH requests are waiting or MAX_WAIT_MS has passed since the first one, whichever comes first.
It never waits for the queue to fill. Requests that can no longer finish within 30 s are dropped.

Settings come from environment variables. MAX_BATCH=2 because bench/batch_memory.py measured a
3.8 GB peak at batch 4, too close to the 4 GB limit. MAX_WAIT_MS and QUEUE_SIZE are still placeholders.
"""

import asyncio
import io
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import onnxruntime as ort
from fastapi import FastAPI, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from pipeline import OcrPipeline

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "2"))
MAX_WAIT_MS = float(os.environ.get("MAX_WAIT_MS", "20"))
QUEUE_SIZE = int(os.environ.get("QUEUE_SIZE", "32"))


def cpu_quota() -> int:
    """CPUs allowed by docker --cpus (cgroup v2 cpu.max), else the visible CPU count."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, round(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


# ONNX Runtime threads per graph. ORT's own default is one per physical core of the host, which
# ignores docker --cpus and oversubscribes the quota (p50 9.1 s vs 3.5 s at --cpus=4), so default
# to the quota.
ORT_THREADS = int(os.environ.get("ORT_THREADS", "0")) or cpu_quota()
# Idle ORT threads spin-wait for work by default (session.intra_op.allow_spinning). Under a --cpus
# quota the spinning burns CPU time the working threads need: off, p50 2.3 s -> 1.2 s at --cpus=4.
ORT_SPIN = os.environ.get("ORT_SPIN", "0")
# Graphs to load as <name>.int8.onnx (export/quantize.py), comma-separated. Empty = all fp32.
INT8 = {g for g in os.environ.get("INT8", "vision_encoder,text_encoder,decoder_merged").split(",") if g}
DEADLINE_S = 30.0  # the load test counts anything slower as failed
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

log = logging.getLogger("ocr")


@dataclass
class Job:
    image: Image.Image
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    arrived: float = field(default_factory=time.monotonic)

    def finish(self, result):
        # The worker is a plain thread: hand the result back to the event loop that owns the future.
        self.loop.call_soon_threadsafe(_set_result, self.future, result)


def _set_result(future, result):
    if not future.done():  # the client may have gone away already
        future.set_result(result)


class Worker:
    def __init__(self, pipeline: OcrPipeline):
        self.pipeline = pipeline
        self.jobs: queue.Queue[Job] = queue.Queue(maxsize=QUEUE_SIZE)
        self.last_batch_s = 0.0  # how long the previous batch took, to predict whether a job can still finish
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, name="inference", daemon=True)

    def take_batch(self) -> list[Job]:
        try:
            batch = [self.jobs.get(timeout=0.5)]
        except queue.Empty:
            return []
        until = time.monotonic() + MAX_WAIT_MS / 1000
        while len(batch) < MAX_BATCH:
            remaining = until - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self.jobs.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def run(self):
        while not self.stopping.is_set():
            batch = self.take_batch()
            now = time.monotonic()
            live = []
            for job in batch:
                if now - job.arrived + self.last_batch_s > DEADLINE_S:
                    job.finish((503, {"error": "server overloaded, request dropped"}))
                else:
                    live.append(job)
            if not live:
                continue

            start = time.monotonic()
            try:
                texts = self.pipeline([job.image for job in live])
                results = [(200, {"text": t}) for t in texts]
            except Exception:
                log.exception("inference failed")
                results = [(500, {"error": "inference failed"})] * len(live)
            self.last_batch_s = time.monotonic() - start
            for job, result in zip(live, results):
                job.finish(result)


def decode_upload(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()  # decode now, so a truncated or corrupt file fails here and not in the worker
    return image


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.worker = None

    def load():
        options = ort.SessionOptions()
        options.intra_op_num_threads = ORT_THREADS
        options.add_session_config_entry("session.intra_op.allow_spinning", ORT_SPIN)
        worker = Worker(OcrPipeline(MODELS_DIR, options, INT8))
        worker.thread.start()
        app.state.worker = worker
        log.warning("model loaded: %d ORT threads, spin %s, int8 graphs: %s", ORT_THREADS, ORT_SPIN, sorted(INT8) or "none")

    # Load in the background so /health can answer 503 while loading instead of refusing connections.
    loading = asyncio.create_task(asyncio.to_thread(load))
    yield
    await loading
    if app.state.worker:
        app.state.worker.stopping.set()


app = FastAPI(title="Florence-2 OCR", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def bad_request(request, exc):
    return JSONResponse({"error": "send the image as multipart/form-data field 'image'"}, status_code=400)


@app.get("/health")
async def health():
    if app.state.worker is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(image: UploadFile):
    worker = app.state.worker
    if worker is None:
        return JSONResponse({"error": "model is still loading"}, status_code=503)

    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "image larger than 10 MB"}, status_code=413)
    try:
        decoded = await asyncio.to_thread(decode_upload, data)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return JSONResponse({"error": "not a valid JPEG or PNG image"}, status_code=400)

    loop = asyncio.get_running_loop()
    job = Job(decoded, loop.create_future(), loop)
    try:
        worker.jobs.put_nowait(job)
    except queue.Full:
        return JSONResponse({"error": "server busy, try again"}, status_code=503)

    status, body = await job.future
    return JSONResponse(body, status_code=status)
