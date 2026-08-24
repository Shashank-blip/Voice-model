"""Export the fine-tuned model to int8 ONNX so the runtime needs onnxruntime
only -- no torch in the deployed package."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from training.train import BASE_MODEL, MAX_LENGTH, IntentModel

ROOT = Path(__file__).parent.parent


def main() -> None:
    models_dir = ROOT / "models"
    manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))

    model = IntentModel(len(manifest["labels"]))
    model.load_state_dict(torch.load(models_dir / "intent.pt", map_location="cpu"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(models_dir / "tokenizer")

    dummy = tokenizer("hello", return_tensors="pt", padding="max_length",
                      truncation=True, max_length=MAX_LENGTH)
    fp32 = models_dir / "intent.fp32.onnx"
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), fp32,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                      "logits": {0: "batch"}},
        opset_version=17,
        # torch >= 2.5 defaults to the dynamo-based exporter, which needs the
        # optional `onnxscript` package. Pin the legacy TorchScript-based
        # exporter so export works with only the declared training deps.
        dynamo=False)

    quantize_dynamic(fp32, models_dir / "intent.onnx", weight_type=QuantType.QInt8)
    fp32.unlink()
    size_mb = (models_dir / "intent.onnx").stat().st_size / 1e6
    print(f"exported models/intent.onnx ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
