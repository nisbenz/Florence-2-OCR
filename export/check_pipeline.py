"""Check that server/pipeline.py (ONNX, no torch) gives the same text as PyTorch's generate().

Runs in the export venv, after export_encoders.py and export_decoder.py. The reference uses
Microsoft's example settings: num_beams=3, do_sample=False, max_new_tokens=1024.
Compares one image at a time and all images in one batch.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from export_encoders import load_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from pipeline import OcrPipeline  # noqa: E402

LINES = [
    "INVOICE 2026-11-09 TOTAL 42.50",
    "Florence-2 runs on ONNX Runtime without PyTorch.",
    "Order #8812 shipped to 14 Rue de la Paix, Paris. Qty: 3 x 19.99 EUR. Thank you!",
]


def synthetic_images():
    images = []
    for n in range(1, len(LINES) + 1):
        im = Image.new("RGB", (1000, 160 + 70 * n), "white")
        draw = ImageDraw.Draw(im)
        for i, line in enumerate(LINES[:n]):
            draw.text((30, 40 + 70 * i), line, fill="black", font=ImageFont.load_default(size=36))
        images.append(im)
    return images


def reference(model, processor, images):
    inputs = processor(text=["<OCR>"] * len(images), images=[im.convert("RGB") for im in images], return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    texts = processor.batch_decode(ids, skip_special_tokens=False)
    # Batched outputs are padded to the longest one; the pad tokens aren't part of the text.
    return [processor.post_process_generation(t, task="<OCR>", image_size=im.size)["<OCR>"].replace("<pad>", "")
            for t, im in zip(texts, images)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", type=Path, help="extra test images (default: synthetic ones only)")
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()

    images = synthetic_images() + [Image.open(p) for p in args.images]
    model, processor = load_model()
    pipeline = OcrPipeline(args.models)

    expected = reference(model, processor, images)
    single = [pipeline([im])[0] for im in images]
    batched = pipeline(images)
    expected_single = [reference(model, processor, [im])[0] for im in images]

    ok = True
    for i, (e, es, s, b) in enumerate(zip(expected, expected_single, single, batched)):
        match = e == es == s == b
        ok &= match
        print(f"[{'ok' if match else 'MISMATCH'}] image {i}: {e!r}")
        if not match:
            print(f"    torch batched {e!r}\n    torch single  {es!r}\n    onnx single   {s!r}\n    onnx batched  {b!r}")
    print("all match" if ok else "some outputs differ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
