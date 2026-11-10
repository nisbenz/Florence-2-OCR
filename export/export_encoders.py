"""Export the Florence-2-base encoders to ONNX.

Offline only: this is the one place PyTorch is allowed. It writes to models/:
  vision_encoder.onnx  pixel_values (B, 3, 768, 768) -> image_features (B, 577, 768)
  text_encoder.onnx    image_features + prompt_embeds -> encoder_hidden_states, encoder_attention_mask
  ocr_prompt_embeds.npy  embeddings of the fixed <OCR> prompt, computed once here
  preprocessor_config.json  image preprocessing settings, read by the server
  tokenizer.json         the original tokenizer, for the `tokenizers` library in the server

The decoder export is not written yet.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

MODEL_ID = "microsoft/Florence-2-base"
# Pinned so the remote modeling code and weights can't change under us.
REVISION = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
OPSET = 17


def get_imports_without_flash_attn(filename):
    # The remote modeling code lists flash_attn as an import even though it is
    # optional; without this, loading fails on a CPU-only machine.
    return [imp for imp in get_imports(filename) if imp != "flash_attn"]


class VisionEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model._encode_image(pixel_values)


class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.encoder = model.language_model.get_encoder()

    def forward(self, image_features, prompt_embeds):
        # Same merge as Florence2ForConditionalGeneration.generate: image tokens first, then the prompt.
        inputs_embeds, attention_mask = self.model._merge_input_ids_with_image_features(
            image_features, prompt_embeds
        )
        hidden_states = self.encoder(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        ).last_hidden_state
        return hidden_states, attention_mask


def load_model():
    with patch("transformers.dynamic_module_utils.get_imports", get_imports_without_flash_attn):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, processor = load_model()

    # A blank image is enough for tracing: only the shapes matter here.
    inputs = processor(text="<OCR>", images=Image.new("RGB", (768, 768)), return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        prompt_embeds = model.get_input_embeddings()(inputs["input_ids"])
        image_features = model._encode_image(pixel_values)

        np.save(args.out / "ocr_prompt_embeds.npy", prompt_embeds.numpy())
        # The server builds its CLIPImageProcessor from this file, so it never touches the Hub.
        processor.image_processor.save_pretrained(args.out)
        processor.tokenizer.backend_tokenizer.save(str(args.out / "tokenizer.json"))

        torch.onnx.export(
            VisionEncoder(model),
            (pixel_values,),
            args.out / "vision_encoder.onnx",
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={"pixel_values": {0: "batch"}, "image_features": {0: "batch"}},
            opset_version=OPSET,
            dynamo=False,
        )

        torch.onnx.export(
            TextEncoder(model),
            (image_features, prompt_embeds),
            args.out / "text_encoder.onnx",
            input_names=["image_features", "prompt_embeds"],
            output_names=["encoder_hidden_states", "encoder_attention_mask"],
            dynamic_axes={
                "image_features": {0: "batch"},
                "prompt_embeds": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
                "encoder_attention_mask": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

    print(f"Wrote encoders and prompt embeddings to {args.out}/")


if __name__ == "__main__":
    main()
