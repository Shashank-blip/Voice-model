"""Export the fine-tuned model to int8 ONNX so the runtime needs onnxruntime
only -- no torch in the deployed package."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python training/export_onnx.py` (script's own directory on
# sys.path[0], not the repo root) as well as `python -m training.export_onnx`
# (repo root already on sys.path). Must run before the local-package
# imports below.
_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORTS))

import json

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
