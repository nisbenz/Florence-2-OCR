"""FastAPI backend skeleton. No PyTorch, no transformers: ONNX Runtime only (not wired in yet)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the ONNX graphs, tokenizer and prompt embeddings here, then mark the model ready.
    app.state.ready = False
    yield
    # Shutdown: stop the inference worker here.


app = FastAPI(title="Florence-2 OCR", lifespan=lifespan)
