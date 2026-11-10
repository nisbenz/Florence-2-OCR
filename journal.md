# Development journal

## Approach

Florence-2-base, `<OCR>` task, served on CPU with no PyTorch in the server.

1. **Export offline** (`export/`, the only place torch is allowed): three ONNX graphs.
   - `vision_encoder.onnx`: DaViT, `pixel_values` (N×3×768×768) → 577 image tokens.
   - `text_encoder.onnx`: image tokens + the fixed `<OCR>` prompt embeddings → encoder states (587 tokens).
   - `decoder_merged.onnx`: one graph with an `If` node. Branch 1 is prefill, branch 2 is a cached
     step. Keeping one graph stores the decoder weights once: 520 MB instead of 1013 MB as two graphs.
2. **Server** (`server/`): FastAPI, `CLIPImageProcessor` for preprocessing, HF `tokenizers`, ONNX Runtime,
   beam search (3 beams) reimplemented in NumPy to match `generate()` step for step.
3. **Back-pressure:** bounded queue (full → fast `503`), one inference worker that takes a batch at
   `MAX_BATCH` requests or `MAX_WAIT_MS`, requests that can't finish within 30 s dropped.
4. **Measure** (`bench/`) after every change and log it below.

## Experiment log

| Date | Change | Accuracy | p50 latency | Max RPS | Kept? |
|---|---|---|---|---|---|
| 2026-11-10 | Baseline: first working server in Docker (`--memory=4g --cpus=4`, MAX_BATCH 2, fp32, ORT default threads) | 100%\* | 9.1 s | 0.17 | ✅ |
| 2026-11-10 | `ORT_THREADS=4` (intra-op threads = `--cpus`) | 100%\* | 3.5 s | 0.36 | ✅ |
| 2026-11-10 | int8: dynamic MatMul quantization of both encoders + 8-bit weight-only `MatMulNBits` decoder; threads default to the cgroup quota | 100%\* | 2.5 s | 0.50 | ✅ |

\* 6 synthetic images with references from our own PyTorch run (starter repo not released yet).
`--cpus=4` is a stand-in until the official CPU count is announced. Rows come from
`bench/docker_bench.sh`. Peak container memory: 3.41 GB, 3.40 GB, then 2.19 GB with int8.

## Problems and how they were solved

- **`transformers` 5.x breaks Florence-2's remote modeling code.** Pinned to 4.49.0, and the model to
  revision `5ca5edf` so the remote code can't change under the export.
- **torchvision didn't match the CPU-only torch build.** Installed both from the CPU wheel index.
- **Two decoder graphs (prefill + cached step) doubled the weights.** Merged them into one `If`-node
  graph (−493 MB on disk and in RAM).
- **Matching PyTorch's output exactly.** The model's own config uses beam search (3 beams, early
  stopping, no repeated 3-grams, forced BOS/EOS), so I kept those defaults to match the reference
  text, and the NumPy beam search follows transformers 4.49's `BeamSearchScorer` step for step.
  Text now matches PyTorch exactly, alone and batched.

## Measurements and notes

### Quantization, per graph (2026-11-10, `bench/profile_stages.py` + `bench/accuracy_direct.py`, 4 threads, host)

| Variant | Size | Stage time | Accuracy\* |
|---|---|---|---|
| Vision encoder fp32 → int8 (dynamic, MatMul) | 366 → 113 MB | 1.09 → 0.57 s | 100% |
| Text encoder fp32 → int8 (dynamic, MatMul) | 173 → 46 MB | 0.18 → 0.07 s | 100% |
| Decoder fp32 → 8-bit `MatMulNBits` (block 32) | 546 → 270 MB | 1.27 → 0.89 s (55 steps) | 100% |
| Decoder fp32 → 4-bit `MatMulNBits` (block 32) | 546 → 222 MB | 1.27 → 0.85 s | 100% |

- `quantize_dynamic` left the decoder untouched (546 MB → 546 MB). In the merged graph the weights are
  top-level initializers used from inside the `If` branches, and the quantizer only looks for weights
  inside the branch. ORT's weight-only `MatMulNBits` quantizer follows outer-scope weights: 110 MatMuls
  converted, 48 left (attention Q·K and ·V, activation × activation, nothing to quantize).
- 4-bit isn't faster than 8-bit, so I kept 8-bit (less accuracy risk).
- After quantization the NumPy beam-search glue is ~29% of a request (0.65 s of 2.27 s). Next target.
- \* Only 6 synthetic images. Quantization must be re-checked on the real samples when the starter
  repo is out.

### Thread oversubscription under `--cpus` (2026-11-10)

In Docker the baseline was 3× slower than on the host (p50 9.1 s vs 2.9 s). ONNX Runtime's default
`intra_op_num_threads=0` starts one thread per physical core of the **host** (10 on the i5-13400F),
but `--cpus=4` is only a CPU-time quota: 10 threads share 4 CPUs' worth of time and get throttled,
and the threads that spin-wait burn the quota for nothing. Setting the thread count to the quota
fixed it: p50 9.1 s → 3.5 s, max RPS 0.17 → 0.36.

### Peak RAM per batch size (2026-11-10, one synthetic image, 16 host threads, no Docker)

| Batch | Peak RAM | Batch time | Images/s |
|---|---|---|---|
| 1 | 2.4 GB | 3.3 s | 0.31 |
| 2 | 2.7 GB | 5.9 s | 0.34 |
| 4 | 3.8 GB | 10.4 s | 0.39 |
| 8 | 6.8 GB | 19.8 s | 0.40 |

Batching barely raises throughput: the CPU is already busy at batch 1. Batch 4 peaks at 3.8 GB,
too close to the 4 GB limit, so `MAX_BATCH` defaults to 2.

### Where one request's time goes (2026-11-10, `bench/profile_stages.py`, batch 1, 55 decoder steps)

| Stage | 2 threads | 4 threads | 8 threads |
|---|---|---|---|
| Vision encoder | 1.89 s | 1.19 s | 0.91 s |
| Text encoder | 0.25 s | 0.30 s | 0.13 s |
| Decoder (55 calls) | 1.31 s | 1.27 s | 1.72 s |
| NumPy beam-search glue | 0.42 s | 0.44 s | 0.55 s |
| **Total** | 3.88 s | 3.21 s | 3.33 s |

- The vision encoder scales with threads; the decoder doesn't, and gets *slower* at 8 threads.
  Each decoder step is tiny (3 rows × 1 token), so thread synchronization costs more than it saves.
  → Different thread counts per graph is worth testing.
- About 13–17% is Python/NumPy glue between decoder steps (log-softmax over the 51289-token vocab,
  n-gram bans, gathering the KV cache and encoder states for the surviving beams).

## What I learned

- Data-flow order is not build order: export and a PyTorch reference come first, because everything
  later is checked against their tensors and text.
- Beam search with 3 beams means a batch of N images is 3N decoder rows, which is what drives RAM.
