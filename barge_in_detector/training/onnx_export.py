"""Export an E5/BERT classifier for CPU inference; quantization is validated separately."""

import argparse
import json
import shutil
from pathlib import Path


def export_model(checkpoint, destination):
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.transformers.optimizer import optimize_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint, destination = Path(checkpoint), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, attn_implementation="eager"
        )
        .cpu()
        .eval()
    )
    if model.config.label2id != {"backchannel": 0, "barge_in": 1}:
        raise ValueError("Unexpected output label orientation")

    class Logits(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask):
            return self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=torch.zeros_like(input_ids),
            ).logits

    example = tokenizer(["yes i understand", "please wait"], padding=True, return_tensors="pt")
    raw = destination / "model-raw.onnx"
    with torch.no_grad():
        torch.onnx.export(
            Logits(model),
            (example["input_ids"], example["attention_mask"]),
            str(raw),
            dynamo=False,
            opset_version=17,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
        )
    onnx.checker.check_model(str(raw))
    optimized = optimize_model(
        str(raw),
        model_type="bert",
        num_heads=model.config.num_attention_heads,
        hidden_size=model.config.hidden_size,
        opt_level=1,
        use_gpu=False,
    )
    fp32, int8 = destination / "model-fp32.onnx", destination / "model-int8.onnx"
    optimized.save_model_to_file(str(fp32))
    quantize_dynamic(
        str(fp32),
        str(int8),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm", "Attention"],
        # ORT's fused BERT operators do not carry intermediate value_info.
        # This graph is exported entirely in FP32, so that is the safe fallback.
        extra_options={"MatMulConstBOnly": True, "DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    shutil.copyfile(checkpoint / "tokenizer.json", destination / "tokenizer.json")
    metadata = {
        "schema_version": 1,
        "labels": ["backchannel", "barge_in"],
        "max_length": 64,
        "pad_id": tokenizer.pad_token_id,
        "pad_token": tokenizer.pad_token,
        "model_file": "model-int8.onnx",
        "calibrated": False,
        "threshold": 0.5,
        "quantization": "Dynamic signed int8, per-channel; MatMul/Gemm/Attention. Embeddings remain fp32.",
        "fused_operator_counts": optimized.get_fused_operator_statistics(),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_model(args.checkpoint, args.out), indent=2))


if __name__ == "__main__":
    main()
