"""Compare character and word phrase features using the fixed multilingual dataset.

uv run --group baseline python -m experiments.compare_lexical_features
Diagnostic candidate: full cleaned strings, not a serving export.
"""

import csv
import json
import time
import warnings
from pathlib import Path

import joblib
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from barge_in_detector.splitting import SPLITS, read_csv
from experiments.audit_reviewed_errors import reviewed_predictions, score, sha256


def main():
    started = time.perf_counter()
    warnings.simplefilter("error", ConvergenceWarning)
    out = Path("models/lexical-run1")
    if out.exists():
        raise ValueError("Preserve existing experiment")
    splits = {s: read_csv(Path(f"data/splits/{s}.csv")) for s in SPLITS}
    original = json.loads(Path("reports/e5-small-run1/source-manifest.json").read_text())
    for name in SPLITS:
        assert sha256(Path(f"data/splits/{name}.csv")) == original[f"data/splits/{name}.csv"]
    chars = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False, sublinear_tf=True)
    words = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    # Words are a complementary feature, not a tokenizer assumption for every script.
    train_texts = [r["text_clean"] for r in splits["train"]]
    val_texts = [r["text_clean"] for r in splits["validation"]]
    train_char, train_word = chars.fit_transform(train_texts), words.fit_transform(train_texts)
    val_char, val_word = chars.transform(val_texts), words.transform(val_texts)

    def predicted(rows, classifier, features):
        return [
            {**r, "p_barge_in": float(p)}
            for r, p in zip(rows, classifier.predict_proba(features)[:, 1], strict=True)
        ]

    trials, best = [], None
    with threadpool_limits(limits=1):
        for scale in (0.25, 1.0, 4.0):
            for c in (1.0, 10.0, 100.0):
                model = LogisticRegression(
                    C=c, class_weight="balanced", max_iter=1000, random_state=20261115
                )
                model.fit(
                    hstack([train_char, train_word * scale]).tocsr(),
                    [r["label"] for r in splits["train"]],
                )
                assert list(model.classes_) == ["backchannel", "barge_in"]
                metrics = score(
                    predicted(
                        splits["validation"], model, hstack([val_char, val_word * scale]).tocsr()
                    )
                )
                trials.append({"word_scale": scale, "C": c, "validation": metrics})
                if best is None or metrics["macro_f1"] > best[0]:
                    best = (metrics["macro_f1"], scale, c, model)
        _, scale, c, model = best
        challenge = reviewed_predictions(
            Path("data/evaluation/english-challenge-v1.csv"),
            Path("data/evaluation/reviews/english-challenge-v1.json"),
            Path("reports/e5-small-run1/challenge-predictions.csv"),
        )
        predictions = {}
        for name, rows in {
            "validation": splits["validation"],
            "test": splits["test"],
            "challenge": challenge,
        }.items():
            texts = [r["text_clean"] for r in rows]
            features = hstack([chars.transform(texts), words.transform(texts) * scale]).tocsr()
            predictions[name] = predicted(rows, model, features)
    out.mkdir(parents=True)
    joblib.dump(
        {"chars": chars, "words": words, "word_scale": scale, "classifier": model},
        out / "local-experiment.joblib",
    )
    seen = {r["text_clean"] for rows in splits.values() for r in rows}
    report = {
        "protocol": "Character 2-5 grams plus word 1-2 grams; train-only TF-IDF and balanced logistic regression; validation-only selection from fixed grid; threshold 0.5. Development evidence, no serving export.",
        "selected": {"word_scale": scale, "C": c},
        "trials": trials,
        "validation": score(predictions["validation"]),
        "test": score(predictions["test"]),
        "challenge": {
            "all": score(predictions["challenge"]),
            "unseen": score([r for r in predictions["challenge"] if r["text_clean"] not in seen]),
            "by_category": {
                category: score([r for r in predictions["challenge"] if r["category"] == category])
                for category in sorted({r["category"] for r in challenge})
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
        "input_sha256": {
            str(p): sha256(p)
            for p in [
                Path(__file__).relative_to(Path.cwd()),
                *[Path(f"data/splits/{s}.csv") for s in SPLITS],
            ]
        },
    }
    dest = Path("reports/lexical-run1")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, rows in predictions.items():
        with (dest / f"predictions-{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["language", "label", "text_clean", "p_barge_in"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
