"""Fixed validation-selected comparison of three sparse multilingual classifiers.

Run only after the authorized GPU experiment is downloaded and destroyed.
All features are learned from training text; heldouts are loaded after selection.
These are development comparisons, not evidence of realistic-prior calibration.
"""

import csv
import json
import random
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from tokenizers import Tokenizer

from barge_in_detector.splitting import read_csv
from experiments.audit_reviewed_errors import reviewed_predictions, score, sha256
from experiments.compare_phrase_supplement import summaries


def main():
    started = time.perf_counter()
    warnings.simplefilter("error", ConvergenceWarning)
    model_dir, report_dir = Path("models/lightweight-v2"), Path("reports/lightweight-v2")
    if model_dir.exists() or report_dir.exists():
        raise ValueError("Preserve previous run; select a new run name")
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    split_dir = Path("data/experiments/original-plus-phrase-v1/splits")
    train, validation = (read_csv(split_dir / f"{s}.csv") for s in ("train", "validation"))
    tokenizer_path = Path("huggingface/model/candidates/e5-phrase-run1/checkpoint/tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=64, direction="right")

    def truncate(text):
        return text[: max((end for _, end in tokenizer.encode(text).offsets), default=0)]

    chars = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False, sublinear_tf=True)
    words = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    texts = [truncate(r["text_clean"]) for r in train]
    train_chars, train_words = chars.fit_transform(texts), words.fit_transform(texts)
    val_texts = [truncate(r["text_clean"]) for r in validation]
    val_chars, val_words = chars.transform(val_texts), words.transform(val_texts)
    labels = [r["label"] for r in train]
    trials, selected = [], {}

    def predictions(rows, classifier, matrix):
        assert list(classifier.classes_) == ["backchannel", "barge_in"]
        return [
            {**r, "p_barge_in": float(p)}
            for r, p in zip(rows, classifier.predict_proba(matrix)[:, 1], strict=True)
        ]

    with threadpool_limits(limits=1):
        for family in ("logistic", "linear_svm", "complement_nb"):
            for scale in (0.0, 0.25, 1.0):
                matrix = hstack([train_chars, train_words * scale]).tocsr()
                val_matrix = hstack([val_chars, val_words * scale]).tocsr()
                for strength in (0.1, 1.0, 10.0):
                    fit_start = time.perf_counter()
                    if family == "logistic":
                        classifier = LogisticRegression(
                            C=strength,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=20261115,
                        )
                    elif family == "linear_svm":
                        classifier = CalibratedClassifierCV(
                            LinearSVC(
                                C=strength,
                                class_weight="balanced",
                                max_iter=5000,
                                random_state=20261115,
                            ),
                            method="sigmoid",
                            ensemble=False,
                            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=20261115),
                        )
                    else:
                        classifier = ComplementNB(alpha=strength)
                    classifier.fit(matrix, labels)
                    metrics = score(predictions(validation, classifier, val_matrix))
                    trial = {
                        "family": family,
                        "word_scale": scale,
                        "C" if family != "complement_nb" else "alpha": strength,
                        "validation": metrics,
                        "fit_seconds": time.perf_counter() - fit_start,
                    }
                    trials.append(trial)
                    # Ties retain the first candidate in the prespecified order.
                    if (
                        family not in selected
                        or metrics["macro_f1"] > selected[family][0]["validation"]["macro_f1"]
                    ):
                        selected[family] = (trial, classifier)
            print(f"Selected {family} using validation only", flush=True)

        selection = {name: trial for name, (trial, _) in selected.items()}
        overall = max(selection, key=lambda name: selection[name]["validation"]["macro_f1"])
        (report_dir / "selection.json").write_text(
            json.dumps(
                {
                    "rule": "Highest combined-validation macro F1; listed order breaks ties. Saved before heldout loading.",
                    "overall": overall,
                    "families": selection,
                    "trials": trials,
                },
                indent=2,
            )
            + "\n"
        )

        challenge = reviewed_predictions(
            Path("data/evaluation/english-challenge-v1.csv"),
            Path("data/evaluation/reviews/english-challenge-v1.json"),
            Path("reports/e5-small-run1/challenge-predictions.csv"),
        )
        sets = {
            "combined_validation": validation,
            "combined_test": read_csv(split_dir / "test.csv"),
            "original_test": read_csv(Path("data/splits/test.csv")),
            "supplement_test": read_csv(
                Path("data/supplements/phrase-coverage-v1/splits/test.csv")
            ),
            "challenge": challenge,
        }
        results, benchmarks = {}, {}
        for family, (trial, classifier) in selected.items():
            scale = trial["word_scale"]

            def predict(texts, classifier=classifier, scale=scale):
                texts = [truncate(text) for text in texts]
                matrix = hstack([chars.transform(texts), words.transform(texts) * scale]).tocsr()
                return classifier.predict_proba(matrix)[:, 1]

            results[family] = {}
            for name, rows in sets.items():
                probs = predict([r["text_clean"] for r in rows])
                predicted = [
                    {**r, "p_barge_in": float(p)} for r, p in zip(rows, probs, strict=True)
                ]
                results[family][name] = summaries(predicted)
                with (report_dir / f"predictions-{family}-{name}.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["language", "label", "text_clean", "p_barge_in"],
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows(predicted)
            joblib.dump(
                {"chars": chars, "words": words, "word_scale": scale, "classifier": classifier},
                model_dir / f"{family}.joblib",
            )

            texts = [r["text_clean"] for r in sets["combined_test"]]
            random.Random(20261115).shuffle(texts)
            stress = (
                "please stop for a moment because i need to correct the information you are reading "
                * 12
            )
            assert len(tokenizer.encode(stress).ids) == 64
            benchmarks[family] = {}
            for scenario, inputs in (("multilingual_test", texts), ("full_64_tokens", [stress])):
                durations = []
                for i in range(1100):
                    tick = time.perf_counter_ns()
                    probability = predict([inputs[i % len(inputs)]])[0]
                    elapsed = (time.perf_counter_ns() - tick) / 1e6
                    assert 0 <= probability <= 1
                    if i >= 100:
                        durations.append(elapsed)
                benchmarks[family][scenario] = dict(
                    zip(
                        ("p50_ms", "p95_ms", "p99_ms"),
                        np.percentile(durations, [50, 95, 99]).tolist(),
                        strict=True,
                    )
                )

    report = {
        "protocol": "27 fixed sparse classifiers; train-only TF-IDF; shared 64-token truncation; validation-only selection per family; threshold 0.5. SVM sigmoid uses three-fold out-of-fold training scores, not evaluation labels.",
        "selection": selection,
        "overall_validation_winner": overall,
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(sets["combined_test"]),
        },
        "results": results,
        "model_path": benchmarks,
        "whole_path": None,
        "whole_path_reason": "Vxpipe boundary is not implemented",
        "benchmark": "Owner-designated local CPU, batch one, one thread, 100 warmups/1000 measured; tokenization through probability, load excluded.",
        "elapsed_seconds": time.perf_counter() - started,
        "input_sha256": {
            str(p): sha256(p)
            for p in [
                Path(__file__).relative_to(Path.cwd()),
                tokenizer_path,
                *[split_dir / f"{s}.csv" for s in ("train", "validation", "test")],
            ]
        },
        "limitations": "Synthetic heldouts and reused English development challenge; no realistic-prior calibration or fresh human evaluation. No test-based model selection.",
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "overall_validation_winner": overall,
                "results": {
                    family: {name: metrics["all"]["accuracy"] for name, metrics in rows.items()}
                    for family, rows in results.items()
                },
                "model_path": benchmarks,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
