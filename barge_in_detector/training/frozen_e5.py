"""Frozen E5 embeddings and a portable binary linear head (CPU inference)."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

PREFIX = "query: "
MAX_LENGTH = 64
LABELS = ["backchannel", "barge_in"]


@dataclass
class LinearHead:
    weights: list[float]
    intercept: float

    def __post_init__(self):
        if not self.weights or not all(math.isfinite(x) for x in [*self.weights, self.intercept]):
            raise ValueError("Head weights must be nonempty and finite")

    def predict(self, embeddings):
        probabilities = []
        for row in embeddings:
            if len(row) != len(self.weights):
                raise ValueError("Embedding dimension does not match head")
            if not all(math.isfinite(x) for x in row):
                raise ValueError("Embeddings must be finite")
            logit = self.intercept + math.fsum(
                w * float(x) for w, x in zip(self.weights, row, strict=True)
            )
            probabilities.append(
                1 / (1 + math.exp(-logit))
                if logit >= 0
                else math.exp(logit) / (1 + math.exp(logit))
            )
        return probabilities

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "labels": LABELS,
                    "weights": self.weights,
                    "intercept": self.intercept,
                    "prefix": PREFIX,
                    "max_length": MAX_LENGTH,
                    "pooling": "masked_mean_l2",
                    "calibrated": False,
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("labels") != LABELS:
            raise ValueError("Unsupported head labels")
        expected = {
            "schema_version": 1,
            "prefix": PREFIX,
            "max_length": MAX_LENGTH,
            "pooling": "masked_mean_l2",
        }
        if any(data.get(k) != v for k, v in expected.items()):
            raise ValueError("Unsupported embedding contract")
        return cls(data["weights"], data["intercept"])


def masked_mean_normalize(hidden, mask):
    import torch.nn.functional as functional

    hidden = hidden.float().masked_fill(~mask.bool().unsqueeze(-1), 0.0)
    pooled = hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    return functional.normalize(pooled, p=2, dim=1)


def encode(model, tokenizer, texts, *, batch_size=128):
    import numpy as np
    import torch

    if batch_size < 1 or not texts:
        raise ValueError("Positive batch size and nonempty texts required")
    model.eval().requires_grad_(False)
    device = next(model.parameters()).device
    embeddings = []
    with torch.inference_mode():
        for offset in range(0, len(texts), batch_size):
            batch = tokenizer(
                [PREFIX + t for t in texts[offset : offset + batch_size]],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            hidden = model(**batch).last_hidden_state
            embeddings.append(masked_mean_normalize(hidden, batch["attention_mask"]).cpu().numpy())
    return np.concatenate(embeddings)


class FrozenE5:
    """Load the complete downloaded artifact locally; serving always uses CPU."""

    def __init__(self, path):
        from transformers import AutoModel, AutoTokenizer

        path = Path(path)
        self.head = LinearHead.load(path / "head.json")
        self.tokenizer = AutoTokenizer.from_pretrained(path / "encoder", local_files_only=True)
        self.model = (
            AutoModel.from_pretrained(
                path / "encoder", local_files_only=True, add_pooling_layer=False
            )
            .cpu()
            .eval()
        )
        self.model.requires_grad_(False)

    def predict(self, texts, *, batch_size=128):
        return self.head.predict(encode(self.model, self.tokenizer, texts, batch_size=batch_size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Already-cleaned overlapping caller utterance")
    parser.add_argument("--model", type=Path, default=Path("models/frozen-e5-run1"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    probability = FrozenE5(args.model).predict([args.text])[0]
    print(json.dumps({"p_barge_in": probability, "calibrated": False}))


if __name__ == "__main__":
    main()
