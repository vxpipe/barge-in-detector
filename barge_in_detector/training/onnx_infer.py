"""CPU-only ONNX inference with the shared 64-token, two-label contract."""

import json
from pathlib import Path

import numpy as np


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2 or not np.isfinite(logits).all():
        raise ValueError("Expected finite [batch, 2] logits")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp[:, 1] / exp.sum(axis=1)).tolist()


class OnnxClassifier:
    def __init__(self, directory, *, threads=4, model_file=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if (
            metadata["schema_version"] != 1
            or metadata["max_length"] != 64
            or metadata["labels"] != ["backchannel", "barge_in"]
            or threads < 1
        ):
            raise ValueError("Unsupported classifier contract")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=64, direction="right")
        self.tokenizer.enable_padding(pad_id=metadata["pad_id"], pad_token=metadata["pad_token"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(directory / (model_file or metadata["model_file"])),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, texts):
        if not texts:
            return []
        encoded = self.tokenizer.encode_batch(texts)
        inputs = {
            "input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64),
        }
        return probabilities(self.session.run(["logits"], inputs)[0])
