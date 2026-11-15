"""Compare a learned character baseline with the frozen encoder on identical splits.

uv run --group baseline python -m experiments.compare_character_baseline

Vocabulary/IDF/weights use train only; C uses validation macro F1 only.
The reviewed challenge is scored once after selection. No model is exported.
"""

import argparse
import csv
import json
import platform
import subprocess
import warnings
from pathlib import Path

import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from barge_in_detector.splitting import SPLITS, read_csv
from experiments.audit_reviewed_errors import reviewed_predictions, score, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/character-baseline-v1"))
    args = parser.parse_args()
    splits = {s: read_csv(Path(f"data/splits/{s}.csv")) for s in SPLITS}
    encoder_manifest_path = Path("reports/e5-small-run1/source-manifest.json")
    manifest = json.loads(encoder_manifest_path.read_text())
    for name in SPLITS:
        path = Path(f"data/splits/{name}.csv")
        if sha256(path) != manifest[path.as_posix()]:
            raise ValueError("Splits differ from the frozen encoder's training data")

    # Fixed before observing this experiment's challenge predictions.
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), lowercase=False, sublinear_tf=True
    )
    train = vectorizer.fit_transform([r["text_clean"] for r in splits["train"]])
    validation = vectorizer.transform([r["text_clean"] for r in splits["validation"]])
    labels = [r["label"] for r in splits["train"]]

    def with_probabilities(model, rows, features):
        index = list(model.classes_).index("barge_in")
        return [
            {**r, "p_barge_in": float(p)}
            for r, p in zip(rows, model.predict_proba(features)[:, index], strict=True)
        ]

    trials = []
    best = None
    warnings.simplefilter("error", ConvergenceWarning)
    with threadpool_limits(limits=1):
        for c in (0.1, 1.0, 10.0):
            model = LogisticRegression(
                C=c, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=20261115
            )
            model.fit(train, labels)
            metrics = score(with_probabilities(model, splits["validation"], validation))
            trials.append({"C": c, "validation": metrics, "iterations": model.n_iter_.tolist()})
            if best is None or metrics["macro_f1"] > best[0]:
                best = (metrics["macro_f1"], c, model)
        _, chosen_c, model = best
        source = Path("data/evaluation/english-challenge-v1.csv")
        reviews = Path("data/evaluation/reviews/english-challenge-v1.json")
        encoder_predictions = Path("reports/e5-small-run1/challenge-predictions.csv")
        challenge = reviewed_predictions(source, reviews, encoder_predictions)
        predicted = {
            s: with_probabilities(
                model, rows, vectorizer.transform([r["text_clean"] for r in rows])
            )
            for s, rows in {
                "validation": splits["validation"],
                "test": splits["test"],
                "challenge": challenge,
            }.items()
        }
    all_texts = {r["text_clean"] for rows in splits.values() for r in rows}

    def challenge_metrics(rows):
        return {
            "all": score(rows),
            "unseen": score([r for r in rows if r["text_clean"] not in all_texts]),
            "by_category": {
                c: score([r for r in rows if r["category"] == c])
                for c in sorted({r["category"] for r in rows})
            },
        }

    tracked_inputs = [Path(f"data/splits/{s}.csv") for s in SPLITS] + [
        source,
        reviews,
        encoder_predictions,
        encoder_manifest_path,
        Path(__file__),
        Path("experiments/audit_reviewed_errors.py"),
        Path("barge_in_detector/training/metrics.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    report = {
        "protocol": "Train-only character TF-IDF and logistic regression; C selected by validation macro F1; fixed threshold 0.5. Diagnostic, uncalibrated; no serving/export change.",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "input_sha256": {
            str(p.relative_to(Path.cwd()) if p.is_absolute() else p): sha256(p)
            for p in tracked_inputs
        },
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "threads": 1,
        },
        "configuration": {
            "analyzer": "char",
            "ngram_range": [2, 5],
            "sublinear_tf": True,
            "class_weight": "balanced",
            "C_grid": [0.1, 1.0, 10.0],
            "chosen_C": chosen_c,
            "seed": 20261115,
            "threshold": 0.5,
        },
        "feature_count": len(vectorizer.vocabulary_),
        "trials": trials,
        "baseline": {
            "validation": score(predicted["validation"]),
            "test": score(predicted["test"]),
            "challenge": challenge_metrics(predicted["challenge"]),
        },
        "encoder": {
            **{
                s: json.loads(Path(f"reports/e5-small-run1/eval-{s}.json").read_text())
                for s in ("validation", "test")
            },
            "challenge": challenge_metrics(challenge),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ["case_id", "category", "language", "text_clean", "label", "p_barge_in"]
    with (args.out / "challenge-predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predicted["challenge"])
    print(json.dumps({"chosen_C": chosen_c, "baseline": report["baseline"]}, indent=2))


if __name__ == "__main__":
    main()
