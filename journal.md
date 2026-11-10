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
| 2026-11-10 | Beam search: stop gathering the cross-attention cache per beam and fetching it every step | 100%\* | 2.3 s | 0.56 | ✅ |
| 2026-11-10 | `session.intra_op.allow_spinning=0` (no spin-waiting under the CPU quota) | 100%\* | 1.2 s | 0.81 | ✅ |
| 2026-11-10 | `MAX_BATCH=1` instead of 2 (measured, not adopted yet: open decision) | 100%\* | 1.2 s | 0.83 | ⏸ |
| 2026-11-10 | Decoder graph surgery: cross keys stored transposed, no per-step `Transpose` / `Identity` copies | 100%\* | 0.94 s | 1.06 | ✅ |
| 2026-11-10 | Separate decoder threads: 1 / 2 / 3 (encoders stay at 4) | 100%\* | 1.08 / 0.98 / 0.95 s | 0.91 / 1.02 / 1.06 | ❌ |
| 2026-11-10 | Greedy decoding (`NUM_BEAMS=1`) instead of 3 beams (open decision: needs real samples) | 100%\* | 0.83 s | 1.20 | ⏸ |
| 2026-11-10 | 2 server processes × 2 threads (`WORKERS=2`, `MAX_BATCH=1`) | 100%\* | 1.34 s | 1.13 | ❌ |

\* 6 synthetic images with references from our own PyTorch run (starter repo not released yet).
`--cpus=4` is a stand-in until the official CPU count is announced. Rows come from
`bench/docker_bench.sh`. Peak container memory: 3.41 GB, 3.40 GB, 2.19 GB (int8), 2.13 GB, 2.12 GB,
1.36 GB (batch 1). Design decisions and their reasons: `DESIGN.md`.

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

### The vision encoder (2026-11-10)

It is now about half of a request (0.58 s of ~1.1 s on 4 threads). ORT profile, one image, int8:

| Part | Time |
|---|---|
| Stage 0 (192×192 feature map) | 109 ms |
| Stage 1 | 54 ms |
| Stage 2 (deepest, most blocks) | 273 ms |
| Stage 3 | 23 ms |
| Conv patch embeddings | 37 ms |
| Other (fused / unnamed nodes) | 95 ms |

Window attention 255 ms and channel attention 204 ms. No single hotspot like the decoder's: the
largest nodes are three 11 ms Convs, then `DynamicQuantizeMatMul` (≈43% overall) spread over 97
layers. It is real math, not redundant work.

| Tried | Time | Mean abs diff vs fp32 | Verdict |
|---|---|---|---|
| Dynamic int8 (current) | 576 ms | 0.147 | kept |
| Same, `ORT_ENABLE_EXTENDED` (no NCHWc layout conversions) | 580 ms | 0.146 | ❌ no gain |
| Weight-only `MatMulNBits` 8-bit | 1031 ms | 0.022 | ❌ slower (kernel built for small-M decoding) |
| Weight-only `MatMulNBits` 4-bit | 1142 ms | 0.190 | ❌ |
| Static QDQ int8 (earlier) | 860 ms | — | ❌ |
| Transformer optimizer fusions (earlier) | 662 ms | — | ❌ |

Note: dynamic int8 is the noisiest of these (mean diff 0.147 vs 0.022 for 8-bit `MatMulNBits`).
Fine on the synthetic images, but it is the first thing to suspect if accuracy drops on real samples.

**Lower input resolution**, checked in PyTorch before exporting anything (vision time in PyTorch fp32):

| Size | Image tokens | Vision time | Text vs 768 reference (3 synthetic images) |
|---|---|---|---|
| 768 | 577 | 1.36 s | identical |
| 640 | 401 | 0.97 s | 1 of 3 changed (`42.50` → `42:50`) |
| 576 | 325 | 0.73 s | 3 of 3 changed: lowercased, parts dropped |
| 512 | 257 | 0.61 s | 3 of 3 changed: whole lines missing |

The model was trained at 768 and its reading degrades fast below it, even on large clean synthetic
text. A missing line costs far more CER than the time saved. Rejected.

Conclusion: with ONNX Runtime on CPU, the vision encoder is close to its floor here. What's left is
thread count (it scales with threads, unlike the decoder) on the official CPU count.

### Sustained load, 10 minutes (2026-11-10, Docker `--cpus=4`, concurrency 4)

628 requests, 100% success, 1.04 RPS, 0 errors in the server log. Container memory per minute:
1.98, 1.98, 1.98, 1.98, 2.00, 1.98, 1.98, 2.00, 1.98, 1.98, 1.98 GB: flat, no leak.

### Experiments that didn't help (2026-11-10)

**Separate decoder thread count.** Before spinning was turned off the decoder got slower with more
threads, so fewer decoder threads looked promising. With spinning off it doesn't: 3 threads ties 4,
1–2 are slower. One thread count for everything.

**Static int8 (QDQ) for the vision encoder**, calibrated on the samples, to skip quantizing
activations on every call (`DynamicQuantizeMatMul` was 43% of the vision encoder):
860 ms vs 577 ms for dynamic, same error vs fp32. ORT doesn't turn this QDQ pattern into faster
kernels here. Conv layers are only 44 ms of the vision encoder, so quantizing them wasn't worth trying.

**ORT transformer optimizer** (`onnxruntime.transformers.optimizer`, model_type bert/vit) on the vision
encoder: fused 24 BiasGelu and 51 SkipLayerNormalization, but no Attention (DaViT's window attention
doesn't match the pattern). Slower: 1138 vs 1088 ms fp32, 662 vs 580 ms int8. ORT's own load-time
optimizations (`ORT_ENABLE_ALL`, the default) already do what matters.

**Several server processes** (2 × 2 threads): RPS 1.06 → 1.13, but single-request p50 0.94 → 1.34 s
(each request gets half the threads), and concurrency 2 gained nothing (0.76 RPS, likely connections
landing unevenly on the two processes). Not worth the latency on 4 CPUs.

**Continuous batching: not built.** It only pays if a bigger batch raises throughput, and on 4 CPUs
batch 2 takes exactly twice as long as batch 1. Revisit if the official CPU count is much larger.

**Greedy decoding** is faster (p50 0.94 → 0.83 s, RPS 1.06 → 1.20) and scored 100% here, but the
synthetic images are easy; greedy can drift from the beam-search reference on real text. Decide with
the real samples.

### Profiling the decoder with ONNX Runtime's profiler (2026-11-10, `bench/ort_profile.py`)

`enable_profiling` on each session, one request, kernel time summed per op type (int8, 4 threads):

| Decoder op | Before | After surgery |
|---|---|---|
| `Identity` (648 runs) | 323 ms | gone |
| `Transpose` | 226 ms (2652 runs) | 47 ms (2322 runs) |
| `MatMulNBits` (the real work) | 205 ms | 197 ms |
| Total kernel time inside the branches | ~955 ms | ~467 ms |

(`If` is the container and counts everything inside it, so it's left out of the total.)

Two thirds of the decoder's time wasn't math. In the cached-step branch:

- **12 `Identity` per step** copied the whole cross-attention cache (~65 MB) to the outputs, only
  because both `If` branches must have the same outputs. Not fetching those outputs (previous fix)
  didn't stop ORT from computing them.
- **6 `Transpose` per step** turned each layer's cross key K (batch×12×587×64) into Kᵀ for Q·Kᵀ.
  The keys never change after prefill, so this redid the same work 55 times per request.

`export/optimize_decoder.py` edits the graph after export: the prefill branch writes the cross keys
already transposed, the cached branch uses them directly (Transpose deleted) and outputs empty tensors
instead of copies (Identity → empty Constant). Decoder 0.88 s → 0.39 s per request, request
1.61 s → 1.11 s (host, 4 threads, spinning off, same files otherwise). Text still identical to PyTorch
with the fp32 graphs, single and batched.

Lesson: the profiler found what reading the export code didn't. The exporter produces a correct
graph, not a graph that knows which inputs are constant across calls.

### The beam-search glue was the cross-attention cache (2026-11-10)

After int8, 0.65 s of a 2.27 s request was Python/NumPy between decoder steps. Timing each piece at
real shapes (3 beam rows, 587 encoder tokens):

| Per step | Time |
|---|---|
| Gather the cross-attention cache for the surviving beams (12 tensors of 3×12×587×64) | 12.2 ms |
| Gather the self-attention cache | 0.15 ms |
| Gather encoder states + mask | 0.19 ms |
| log-softmax over 51289 tokens | 0.17 ms |
| top-2k with argpartition | 0.19 ms |
| no-repeat 3-gram bans | 0.01 ms |

12 ms × 55 steps ≈ the whole 0.65 s. And it was useless work: the cross-attention cache depends only on
the image, so the K beams of an image hold identical copies and never change after prefill. Beam
reordering stays inside one image, so the cache only has to move when an image finishes and leaves
the batch. The decoder also returned the unchanged cache on every step (`present.*.encoder.*`,
~65 MB copied out of ORT); now a cached step asks only for logits and the self-attention cache.

Glue 0.65 s → 0.07 s, request 2.27 s → 1.66 s (host, 4 threads). Text still identical to PyTorch with
the fp32 graphs, single and batched (`export/check_pipeline.py`).

### Spin-waiting under a CPU quota (2026-11-10)

ORT's worker threads spin (busy-wait) between parallel sections so they react fast. On a full machine
that costs nothing, but under `--cpus=4` every spinning microsecond is taken from the same 4-CPU
budget the working threads need. Decoder steps are tiny and frequent, so threads spin a lot.
`session.intra_op.allow_spinning=0`: p50 2.3 s → 1.2 s, max RPS 0.56 → 0.81.

### Batch size after these fixes (2026-11-10, Docker `--cpus=4`)

| MAX_BATCH | Max RPS | p50 at concurrency 4 | Peak RAM |
|---|---|---|---|
| 1 | 0.83 | 4.9 s | 1.36 GB |
| 2 | 0.81 | 5.4 s | 2.12 GB |

On 4 threads one image already keeps every core busy, so a batch of 2 takes twice as long as one
image (3.34 s vs 1.66 s on the host): same throughput, worse latency, more RAM.

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
