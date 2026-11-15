import json
import math

import pytest

from barge_in_detector.training.frozen_e5 import LinearHead, masked_mean_normalize


def test_linear_head_roundtrip_and_positive_class(tmp_path):
    head = LinearHead([2.0, -1.0], 0.5)
    path = tmp_path / "head.json"
    head.save(path)
    loaded = LinearHead.load(path)
    assert loaded.predict([[1.0, 2.0], [0.0, 0.0]]) == pytest.approx([1 / (1 + math.exp(-0.5))] * 2)
    assert loaded.predict([[1000.0, 0.0], [-1000.0, 0.0]]) == [1.0, 0.0]


def test_head_rejects_invalid_features():
    head = LinearHead([1.0, 2.0], 0.0)
    with pytest.raises(ValueError, match="dimension"):
        head.predict([[1.0]])
    with pytest.raises(ValueError, match="finite"):
        head.predict([[float("nan"), 0.0]])


def test_head_rejects_wrong_label_order(tmp_path):
    path = tmp_path / "head.json"
    LinearHead([1.0], 0.0).save(path)
    data = json.loads(path.read_text())
    data["labels"].reverse()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="labels"):
        LinearHead.load(path)


def test_pooling_ignores_padding_and_normalizes():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [999.0, 999.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = masked_mean_normalize(hidden, mask)
    assert pooled.tolist()[0] == pytest.approx([0.6, 0.8])
    assert torch.linalg.vector_norm(pooled).item() == pytest.approx(1.0)


def test_encoding_prefix_budget_and_freeze():
    torch = pytest.importorskip("torch")
    from barge_in_detector.training.frozen_e5 import encode

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert texts == ["query: wait", "query: जी बिल्कुल"]
            assert kwargs == {
                "padding": True,
                "truncation": True,
                "max_length": 64,
                "return_tensors": "pt",
            }
            return {
                "input_ids": torch.zeros(2, 2, dtype=torch.long),
                "attention_mask": torch.ones(2, 2, dtype=torch.long),
            }

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, **kwargs):
            from types import SimpleNamespace

            assert not self.training
            assert not torch.is_grad_enabled()
            assert not self.weight.requires_grad
            return SimpleNamespace(last_hidden_state=torch.ones(2, 2, 3))

    result = encode(Model(), Tokenizer(), ["wait", "जी बिल्कुल"])
    assert result.shape == (2, 3)
