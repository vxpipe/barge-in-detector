"""Portable character/word linear classifier with the shared 64-token budget."""

import argparse
import json
import shutil
from pathlib import Path


def export_model(chars, words, classifier, word_scale, tokenizer_path, destination):
    if list(classifier.classes_) != ["backchannel", "barge_in"]:
        raise ValueError("Unexpected classifier label order")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    data = {
        "schema_version": 1,
        "labels": ["backchannel", "barge_in"],
        "max_length": 64,
        "calibrated": False,
        "word_scale": word_scale,
        "weights": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "characters": {"vocabulary": chars.vocabulary_, "idf": chars.idf_.tolist()},
        "words": {"vocabulary": words.vocabulary_, "idf": words.idf_.tolist()},
    }
    (destination / "classifier.json").write_text(json.dumps(data, ensure_ascii=False) + "\n")
    shutil.copyfile(tokenizer_path, destination / "tokenizer.json")


class LexicalModel:
    def __init__(self, directory):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from tokenizers import Tokenizer

        directory = Path(directory)
        data = json.loads((directory / "classifier.json").read_text())
        if (
            data["schema_version"] != 1
            or data["labels"] != ["backchannel", "barge_in"]
            or data["max_length"] != 64
        ):
            raise ValueError("Unsupported classifier contract")
        self.chars = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            lowercase=False,
            sublinear_tf=True,
            vocabulary=data["characters"]["vocabulary"],
        )
        self.words = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=False,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
            vocabulary=data["words"]["vocabulary"],
        )
        self.chars.idf_ = np.asarray(data["characters"]["idf"])
        self.words.idf_ = np.asarray(data["words"]["idf"])
        self.weights = np.asarray(data["weights"])
        self.intercept, self.word_scale = data["intercept"], data["word_scale"]
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=64, direction="right")

    def truncate(self, text):
        encoding = self.tokenizer.encode(text)
        end = max((end for start, end in encoding.offsets), default=0)
        return text[:end]

    def predict(self, texts):
        from scipy.sparse import hstack
        from scipy.special import expit

        texts = [self.truncate(text) for text in texts]
        features = hstack(
            [self.chars.transform(texts), self.words.transform(texts) * self.word_scale]
        ).tocsr()
        return expit(features @ self.weights + self.intercept).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Already-cleaned overlapping caller utterance")
    parser.add_argument("--model", type=Path, default=Path("models/lexical-run1/portable"))
    args = parser.parse_args()
    probability = LexicalModel(args.model).predict([args.text])[0]
    print(json.dumps({"p_barge_in": probability, "calibrated": False}))


if __name__ == "__main__":
    main()
