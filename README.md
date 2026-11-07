# Florence-2 OCR

A CPU-only OCR backend built on `microsoft/Florence-2-base`. Image in, text out.

The goal is to serve the `<OCR>` task with the lowest latency and the highest
requests per second possible, within 4 GB of RAM, a few CPU threads and no GPU,
while keeping accuracy at 70% or above against the original model's output.

> Status: early draft. Only the project structure exists so far.

## Constraints

- **No PyTorch and no `transformers` in the server.** The final Docker image
  must not contain `torch`.
- PyTorch is used **offline only**, in `export/`, to load the original weights
  and convert them to ONNX.
- No pre-converted or pre-quantized Florence-2. Everything starts from the
  original `microsoft/Florence-2-base` weights.
- No response caching. Every request runs inference.

## API

- `GET /health`: returns `{"status": "ok"}` only once the model is loaded.
- `POST /ocr`: `multipart/form-data` with one `image` field (JPEG or PNG),
  returns `{"text": "..."}`.

The server listens on port `8000`.

## Layout

```
export/   offline scripts: original weights -> ONNX graphs (PyTorch allowed here only)
server/   the backend: preprocessing, tokenizer, ONNX inference, HTTP API (no PyTorch)
bench/    accuracy, latency and load-test scripts
```

## Planned approach

1. Export three ONNX graphs: vision encoder, text encoder, and a decoder with a KV cache.
2. Reimplement image preprocessing and use the Hugging Face `tokenizers` library
   in the server.
3. Serve it with FastAPI, a bounded request queue and static batching.
4. Measure accuracy, p50/p95 latency and requests per second before optimizing anything.
