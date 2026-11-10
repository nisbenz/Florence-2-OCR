# Design decisions

Each decision, why it was made, and the evidence. Numbers come from `journal.md`.
"Open" means the decision is still to be made or still needs evidence.

## Architecture

| Decision | Why | Evidence |
|---|---|---|
| Layout: `export/`, `server/`, `bench/`, `journal.md`, `Dockerfile` | Required by the challenge (§8) | — |
| PyTorch only offline, in `export/` and the Docker build stage | Rule 4.4: no torch in the server | Final image has no `torch` (checked) |
| Three ONNX graphs: vision encoder, text encoder (image tokens + `<OCR>` prompt), decoder | Each stage has a different shape and cost, so each can be quantized and tuned separately | Matches PyTorch: vision 1.6e-4, text encoder 2.9e-3 max abs diff |
| `<OCR>` prompt embeddings precomputed (`ocr_prompt_embeds.npy`) | The prompt never changes, so its embedding lookup is done once at export | Exact match |
| Decoder as **one** graph with an `If` node (prefill / cached step) | Two graphs store the decoder weights twice under a 4 GB limit | 520 MB vs 1013 MB |
| Model pinned to revision `5ca5edf`, `transformers==4.49.0` in export | The remote modeling code must not change under the export; 5.x breaks it | — |
| Multi-stage Dockerfile that exports at build time | GitHub rejects files > 100 MB; the rules prefer building from the original weights | `/health` ok in ~3 s |

## Generation

| Decision | Why | Evidence |
|---|---|---|
| Beam search, 3 beams, early stopping, no repeated 3-grams, forced BOS/EOS | The reference is the original model with its default settings; matching them matches the reference text | Text identical to PyTorch, single and batched |
| Max new tokens 1024 | Same as Microsoft's example; the decoder can't go past 1024 positions; never cut text | Cost: worst-case latency (open: lower it?) |
| Beam search reimplemented in NumPy, step for step like transformers 4.49 `BeamSearchScorer` | No transformers generation in the server | Identical text |
| Finished images leave the batch | They would otherwise keep costing decoder work | — |
| Cross-attention cache kept per image, moved only when an image leaves the batch, not fetched on cached steps | It depends only on the image: the K beams hold identical copies, so per-beam gathers were pure copying | Glue 0.65 s → 0.07 s per request |
| Decoder graph surgery (`export/optimize_decoder.py`): cross keys stored as Kᵀ by prefill; cached step has no `Transpose` of them and outputs empty tensors instead of `Identity` copies | The profiler showed 2/3 of decoder time was re-transposing and copying a cache that never changes | Decoder 0.88 → 0.39 s; Docker p50 1.2 → 0.94 s, RPS 0.81 → 1.06; text identical to PyTorch |

## Preprocessing and tokenizer

| Decision | Why | Evidence |
|---|---|---|
| `CLIPImageProcessor` from transformers (no torch needed) instead of own NumPy code | The challenge author said own preprocessing is a waste of time (verbal; **get it in writing**) | Max diff 0.0 vs Florence2Processor, incl. L/RGBA |
| HF `tokenizers` with the original `tokenizer.json` | Same tokenizer as the model, no torch | Identical text |

## Server

| Decision | Why | Evidence |
|---|---|---|
| FastAPI + one inference worker thread | ORT already uses all allowed cores inside one run; more workers would split the same CPUs | Batch 2 takes 2× batch 1 on 4 threads |
| Bounded queue (32), fast `503` when full | Back-pressure instead of unbounded latency | Load test: no crash at concurrency 8 |
| Batch at `MAX_BATCH` or `MAX_WAIT_MS` (20 ms), never wait for a full queue | Bounded wait for the first request | — |
| Drop requests that can't finish within 30 s | The load test counts them as failed anyway; don't spend CPU on them | — |
| Decode the upload in the request handler; 10 MB → `413`, bad image → `400` | Bad input never reaches the worker or crashes it | All bad-input cases return 4xx |
| `MAX_BATCH=2` | Chosen by Claude when the decision was skipped; batch 4 was 3.8 GB in fp32 | **Open:** batch 1 gives the same RPS, better latency and 1.36 GB vs 2.12 GB |

## ONNX Runtime settings

| Decision | Why | Evidence |
|---|---|---|
| `intra_op_num_threads` = the cgroup CPU quota (`/sys/fs/cgroup/cpu.max`) | ORT's default is one thread per host core and ignores `--cpus`, so threads get throttled | p50 9.1 s → 3.5 s |
| `session.intra_op.allow_spinning=0` | Spinning threads burn the `--cpus` quota the working threads need | p50 2.3 s → 1.2 s, RPS 0.56 → 0.81 |
| Encoders: dynamic int8 quantization of MatMul | Docs recommend dynamic for transformers; AVX-VNNI on the dev CPU | Vision 1.09 → 0.57 s, 100%\* |
| Decoder: 8-bit weight-only `MatMulNBits` (block 32, accuracy_level 4) | `quantize_dynamic` can't reach weights the `If` branches use from the outer scope | Decoder 1.27 → 0.89 s, 546 → 270 MB, 100%\* |
| 8-bit, not 4-bit, decoder | 4-bit isn't faster, and carries more accuracy risk | 0.85 s vs 0.89 s |

\* 6 synthetic images only. All quantization results must be re-checked on the real samples.

## Tried and rejected

| Option | Result |
|---|---|
| Separate (lower) decoder thread count | 3 threads ties 4, fewer is slower once spinning is off |
| Static int8 QDQ for the vision encoder | 860 ms vs 577 ms dynamic |
| ORT transformer optimizer fusions on the vision encoder | No attention fusion; 662 ms vs 580 ms int8 |
| 2 server processes × 2 threads | RPS 1.13 vs 1.06, p50 1.34 s vs 0.94 s |
| Continuous batching | Not built: batch 2 takes 2× batch 1 on 4 CPUs, so bigger batches add no throughput |
| 4-bit decoder | Not faster than 8-bit |

## Open

- `MAX_BATCH` 1 vs 2, and `MAX_WAIT_MS`.
- Max new tokens below 1024 as a latency safety cap.
- Greedy (`NUM_BEAMS=1`): p50 0.83 s, RPS 1.20 vs 0.94 s / 1.06; 100% on synthetic only. Needs the real samples.
- Official CPU count (TBA) and evaluation CPU: int8 gains depend on VNNI; re-measure there.
- Written waiver for the built-in preprocessing.
