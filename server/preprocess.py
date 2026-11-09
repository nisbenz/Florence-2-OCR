"""Image preprocessing: bytes -> pixel_values (B, 3, 768, 768) float32.

Uses the model's own CLIPImageProcessor from transformers (no torch needed), configured by
models/preprocessor_config.json, which export/export_encoders.py writes.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor


def load_processor(models_dir: Path) -> CLIPImageProcessor:
    # Florence-2's config leaves do_convert_rgb off, so RGBA, palette or grayscale input would fail.
    return CLIPImageProcessor.from_pretrained(models_dir, do_convert_rgb=True)


def decode_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def preprocess(processor: CLIPImageProcessor, images: list[Image.Image]) -> np.ndarray:
    return processor(images=images, return_tensors="np")["pixel_values"]
