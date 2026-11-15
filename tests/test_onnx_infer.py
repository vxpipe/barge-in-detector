import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from barge_in_detector.training.onnx_infer import OnnxClassifier, probabilities


def test_probability_is_stable_and_rejects_invalid_outputs():
    result = probabilities(np.array([[10000, 10000], [-10000, 10000]], dtype=np.float32))
    assert result == [0.5, 1.0]
    with pytest.raises(ValueError):
        probabilities(np.array([[float("nan"), 1]]))
    with pytest.raises(ValueError):
        probabilities(np.zeros((2, 3)))


def test_serving_uses_cpu_masks_and_the_64_token_budget(tmp_path, monkeypatch):
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "yes": 2, "wait": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "labels": ["backchannel", "barge_in"],
                "max_length": 64,
                "pad_id": 1,
                "pad_token": "[PAD]",
                "model_file": "model-int8.onnx",
            }
        )
    )
    calls = []

    class Session:
        def __init__(self, path, **kwargs):
            assert kwargs["providers"] == ["CPUExecutionProvider"]
            assert kwargs["sess_options"].intra_op_num_threads == 2

        def run(self, outputs, inputs):
            calls.append(inputs)
            return [np.array([[0, 1], [1, 0]], dtype=np.float32)]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            InferenceSession=Session,
            SessionOptions=SimpleNamespace,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99),
        ),
    )
    model = OnnxClassifier(tmp_path, threads=2)
    assert len(model.predict(["yes " * 90, "wait"])) == 2
    assert calls[0]["input_ids"].shape == (2, 64)
    assert calls[0]["input_ids"].dtype == np.int64
    assert calls[0]["attention_mask"][0].sum() == 64
    assert calls[0]["attention_mask"][1].sum() == 1
    assert model.predict([]) == []
