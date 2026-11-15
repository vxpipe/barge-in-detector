import numpy as np
import pytest


def test_export_matches_torch_with_padding_and_truncation(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from barge_in_detector.training.onnx_export import export_model
    from barge_in_detector.training.onnx_infer import OnnxClassifier

    torch.manual_seed(17)
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "yes": 2, "wait": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    )
    config = transformers.BertConfig(
        vocab_size=4,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        label2id={"backchannel": 0, "barge_in": 1},
        id2label={0: "backchannel", 1: "barge_in"},
    )
    model = transformers.BertForSequenceClassification(config).eval()
    checkpoint, output = tmp_path / "checkpoint", tmp_path / "onnx"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    export_model(checkpoint, output)
    texts = ["yes " * 90, "wait", "yes wait"]
    inputs = tokenizer(texts, truncation=True, max_length=64, padding=True, return_tensors="pt")
    with torch.no_grad():
        expected = model(**inputs).logits.softmax(dim=-1)[:, 1].numpy()
    fp32 = OnnxClassifier(output, threads=1, model_file="model-fp32.onnx")
    np.testing.assert_allclose(fp32.predict(texts), expected, atol=1e-5)
    np.testing.assert_allclose(fp32.predict(["wait"]), expected[1:2], atol=1e-5)
    int8 = OnnxClassifier(output, threads=1)
    np.testing.assert_allclose(int8.predict(texts), expected, atol=0.02)
