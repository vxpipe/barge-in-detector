import pytest

from barge_in_detector.training.lexical import LexicalModel, export_model


def test_export_reload_probability_and_token_budget(tmp_path):
    np = pytest.importorskip("numpy")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("sklearn")
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = ["yes indeed", "okay go on", "wait please", "stop now"]
    chars = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False, sublinear_tf=True)
    words = TfidfVectorizer(
        ngram_range=(1, 2), lowercase=False, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"
    )
    features = hstack([chars.fit_transform(texts), words.fit_transform(texts) * 0.25]).tocsr()
    classifier = LogisticRegression().fit(
        features, ["backchannel", "backchannel", "barge_in", "barge_in"]
    )
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "wait": 3}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer_path = tmp_path / "source-tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    destination = tmp_path / "model"
    export_model(chars, words, classifier, 0.25, tokenizer_path, destination)
    loaded = LexicalModel(destination)
    assert loaded.predict(texts) == pytest.approx(
        classifier.predict_proba(features)[:, 1], abs=1e-12
    )
    assert loaded.truncate("wait " * 100).split() == ["wait"] * 62
    assert loaded.truncate("ज़रा रुकिए") == "ज़रा रुकिए"
    assert all(np.isfinite(loaded.predict(["wait " * 100])))
