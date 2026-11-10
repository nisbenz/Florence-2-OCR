"""Image bytes -> OCR text on ONNX Runtime. No PyTorch.

Generation is beam search with Florence-2's own defaults (config.json, text_config):
3 beams, early stopping, no repeated 3-grams, forced BOS (0) first, forced EOS (2) at the
last step, max_new_tokens 1024 as in Microsoft's example. The scoring follows transformers
4.49 (BeamSearchScorer) step for step, so the text matches PyTorch's generate().
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

from preprocess import load_processor, preprocess

DECODER_START, BOS, EOS = 2, 0, 2
NUM_BEAMS = 3
MAX_NEW_TOKENS = 1024
NO_REPEAT_NGRAM = 3
NUM_LAYERS, NUM_HEADS, HEAD_DIM = 6, 12, 64
# Self-attention cache: grows by one token per step, follows the beams.
SELF_KV = [f"{i}.decoder.{kv}" for i in range(NUM_LAYERS) for kv in ("key", "value")]
# Cross-attention cache: computed once from the encoder output, identical for every beam of an image.
CROSS_KV = [f"{i}.encoder.{kv}" for i in range(NUM_LAYERS) for kv in ("key", "value")]


def log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def banned_ngram_tokens(seq):
    """Tokens that would repeat a 3-gram already in seq (transformers' NoRepeatNGramLogitsProcessor)."""
    if len(seq) + 1 < NO_REPEAT_NGRAM:
        return []
    # Every earlier position where the last two tokens appeared: the token after them is banned.
    match = (seq[:-2] == seq[-2]) & (seq[1:-1] == seq[-1])
    return seq[2:][match]


class Hypotheses:
    """Finished beams of one image, keeping the best num_beams (transformers' BeamHypotheses)."""

    def __init__(self, num_beams: int):
        self.num_beams = num_beams
        self.beams = []  # (score, token ids)
        self.worst_score = 1e9

    def add(self, tokens, sum_logprobs, generated_len):
        score = sum_logprobs / generated_len  # length_penalty = 1.0
        if len(self.beams) < self.num_beams or score > self.worst_score:
            self.beams.append((score, tokens))
            if len(self.beams) > self.num_beams:
                worst = min(range(len(self.beams)), key=lambda i: self.beams[i][0])
                del self.beams[worst]
                self.worst_score = min(s for s, _ in self.beams)
            else:
                self.worst_score = min(score, self.worst_score)

    def best(self):
        return max(self.beams, key=lambda b: b[0])[1]


class OcrPipeline:
    def __init__(
        self,
        models_dir: Path,
        options: ort.SessionOptions | None = None,
        int8: set[str] = frozenset(),
        decoder_options: ort.SessionOptions | None = None,
        num_beams: int = NUM_BEAMS,
    ):
        """int8: graph names ("vision_encoder", "text_encoder", "decoder_merged") to load as <name>.int8.onnx.
        decoder_options: separate session options for the decoder (e.g. another thread count).
        num_beams: 3 matches the reference; 1 is greedy decoding."""

        def session(name, opts=options):
            path = models_dir / (f"{name}.int8.onnx" if name in int8 else f"{name}.onnx")
            return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

        self.num_beams = num_beams
        self.vision = session("vision_encoder")
        self.text_encoder = session("text_encoder")
        self.decoder = session("decoder_merged", decoder_options or options)
        self.processor = load_processor(models_dir)
        self.tokenizer = Tokenizer.from_file(str(models_dir / "tokenizer.json"))
        self.prompt_embeds = np.load(models_dir / "ocr_prompt_embeds.npy")
        self.self_outputs = ["logits"] + [f"present.{n}" for n in SELF_KV]
        self.cross_outputs = [f"present.{n}" for n in CROSS_KV]

    def __call__(self, images: list[Image.Image]) -> list[str]:
        pixel_values = preprocess(self.processor, images)
        (image_features,) = self.vision.run(None, {"pixel_values": pixel_values})
        prompt = np.repeat(self.prompt_embeds, len(images), axis=0)
        hidden, mask = self.text_encoder.run(None, {"image_features": image_features, "prompt_embeds": prompt})
        return [self.decode(ids) for ids in self.generate(hidden, mask.astype(np.float32))]

    def decode(self, ids):
        # Same as the reference: batch_decode(skip_special_tokens=False), then post_process_generation('<OCR>').
        text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=False)
        return text.replace("<s>", "").replace("</s>", "")

    def run_decoder(self, input_ids, hidden, mask, self_past=None, cross=None):
        """Prefill when self_past is None: returns (logits, self cache, cross cache).
        A cached step returns (logits, self cache) and doesn't fetch the cross cache, which the decoder
        only passes through unchanged (fetching it would copy ~65 MB out of ONNX Runtime per step)."""
        prefill = self_past is None
        if prefill:
            # The prefill branch ignores the past inputs, but they must still be fed (length 0).
            n = len(input_ids)
            self_past = [np.zeros((n, NUM_HEADS, 0, HEAD_DIM), np.float32)] * len(SELF_KV)
            # Cross keys are stored transposed (batch, heads, head_dim, enc_seq): export/optimize_decoder.py.
            cross = [np.zeros((n, NUM_HEADS, HEAD_DIM, 0) if kv.endswith("key") else (n, NUM_HEADS, 0, HEAD_DIM),
                              np.float32) for kv in CROSS_KV]
        feeds = {
            "input_ids": input_ids,
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": mask,
            "use_cache_branch": np.array([not prefill]),
            **{f"past_key_values.{n}": p for n, p in zip(SELF_KV, self_past)},
            **{f"past_key_values.{n}": c for n, c in zip(CROSS_KV, cross)},
        }
        outputs = self.self_outputs + (self.cross_outputs if prefill else [])
        logits, *present = self.decoder.run(outputs, feeds)
        if prefill:
            return logits[:, -1], present[: len(SELF_KV)], present[len(SELF_KV) :]
        return logits[:, -1], present

    def generate(self, hidden, mask):
        """Beam search for a batch of images. Returns the generated token ids per image, start token excluded."""
        n_images, k = len(hidden), self.num_beams
        max_length = 1 + MAX_NEW_TOKENS

        # Prefill once per image: the K beams start identical, so copy the result K times afterwards.
        logits, past, cross = self.run_decoder(np.full((n_images, 1), DECODER_START, np.int64), hidden, mask)
        rows = np.repeat(np.arange(n_images), k)
        logits, past, hidden, mask = logits[rows], [p[rows] for p in past], hidden[rows], mask[rows]
        cross = [c[rows] for c in cross]

        seqs = np.full((n_images * k, 1), DECODER_START, np.int64)
        # Only the first beam is live at the start, or all K would pick the same tokens.
        beam_scores = np.tile(np.array([0.0] + [-1e9] * (k - 1), np.float32), n_images)
        active = list(range(n_images))  # image index of each group of K rows still generating
        hyps = [Hypotheses(k) for _ in range(n_images)]
        cur_len = 1

        while True:
            # Same processors in the same order as transformers: no-repeat n-gram, forced BOS, forced EOS.
            scores = log_softmax(logits.astype(np.float32))
            for r, seq in enumerate(seqs):
                scores[r, banned_ngram_tokens(seq)] = -np.inf
            if cur_len == 1:
                scores[:] = -np.inf
                scores[:, BOS] = 0
            elif cur_len == max_length - 1:
                scores[:] = -np.inf
                scores[:, EOS] = 0
            scores = (scores + beam_scores[:, None]).reshape(len(active), -1)

            vocab = logits.shape[-1]
            top = np.argpartition(-scores, 2 * k, axis=1)[:, : 2 * k]
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1, kind="stable")
            top, top_scores = np.take_along_axis(top, order, axis=1), np.take_along_axis(top_scores, order, axis=1)

            next_rows, next_tokens, next_scores, still_active = [], [], [], []
            for g, image in enumerate(active):
                chosen = []
                for rank, (flat, score) in enumerate(zip(top[g], top_scores[g])):
                    row, token = g * k + flat // vocab, flat % vocab
                    if token == EOS:
                        # An EOS outside the top K candidates is ignored, as in transformers.
                        if rank < k:
                            hyps[image].add(seqs[row][1:], float(score), cur_len - 1)
                    else:
                        chosen.append((row, token, score))
                    if len(chosen) == k:
                        break
                # early_stopping=True: an image is done once it has K finished beams.
                # Finished images leave the batch, so they no longer cost decoder work.
                if len(hyps[image].beams) < k:
                    still_active.append(image)
                    for row, token, score in chosen:
                        next_rows.append(row)
                        next_tokens.append(token)
                        next_scores.append(score)

            if not still_active:
                active = still_active
                break
            if len(still_active) < len(active):
                # An image finished: drop its rows from the per-image tensors. This is the only time
                # they move; beam reordering stays within an image, where all K rows are identical.
                kept = [g for g, image in enumerate(active) if image in still_active]
                image_rows = (np.array(kept)[:, None] * k + np.arange(k)).ravel()
                hidden, mask = hidden[image_rows], mask[image_rows]
                cross = [c[image_rows] for c in cross]
            active = still_active
            next_rows = np.array(next_rows)
            seqs = np.concatenate([seqs[next_rows], np.array(next_tokens)[:, None]], axis=1)
            beam_scores = np.array(next_scores, np.float32)
            cur_len += 1
            if cur_len >= max_length:
                break

            past = [p[next_rows] for p in past]
            logits, past = self.run_decoder(seqs[:, -1:], hidden, mask, past, cross)

        # Images that hit the limit: their open beams count as finished, as in transformers' finalize().
        for g, image in enumerate(active):
            for row in range(g * k, (g + 1) * k):
                hyps[image].add(seqs[row][1:], float(beam_scores[row]), cur_len - 1)

        return [h.best() for h in hyps]
