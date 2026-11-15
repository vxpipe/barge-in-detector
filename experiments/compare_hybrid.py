"""Test character + frozen E5 features with train-only fitting and validation selection.

Run locally after downloading frozen-e5-run1:
uv run --no-sync python -m experiments.compare_hybrid
"""

import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from barge_in_detector.splitting import SPLITS, read_csv
from barge_in_detector.training.frozen_e5 import FrozenE5, encode
from experiments.audit_reviewed_errors import reviewed_predictions, score, sha256
from experiments.compare_frozen_e5 import attach_probabilities


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    warnings.simplefilter("error", ConvergenceWarning)
    out = Path("models/hybrid-run1")
    if out.exists():
        raise ValueError("Output already exists")
    splits = {s: read_csv(Path(f"data/splits/{s}.csv")) for s in SPLITS}
    manifest = json.loads(Path("reports/e5-small-run1/source-manifest.json").read_text())
    for name in SPLITS:
        assert sha256(Path(f"data/splits/{name}.csv")) == manifest[f"data/splits/{name}.csv"]
    frozen = FrozenE5(Path("models/frozen-e5-run1"))
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), lowercase=False, sublinear_tf=True
    )
    chars = {"train": vectorizer.fit_transform([r["text_clean"] for r in splits["train"]])}
    chars["validation"] = vectorizer.transform([r["text_clean"] for r in splits["validation"]])
    embeddings = {
        s: encode(frozen.model, frozen.tokenizer, [r["text_clean"] for r in splits[s]])
        for s in ("train", "validation")
    }
    labels = [r["label"] for r in splits["train"]]
    trials, best = [], None
    # Fixed grid before observing this candidate's test/challenge predictions.
    with threadpool_limits(limits=1):
        for scale in (0.25, 1.0, 4.0):
            train = hstack([chars["train"], csr_matrix(embeddings["train"] * scale)]).tocsr()
            validation = hstack(
                [chars["validation"], csr_matrix(embeddings["validation"] * scale)]
            ).tocsr()
            for c in (1.0, 10.0, 100.0):
                classifier = LogisticRegression(
                    C=c, class_weight="balanced", max_iter=1000, random_state=20261115
                )
                classifier.fit(train, labels)
                metrics = score(
                    attach_probabilities(
                        splits["validation"], classifier.predict_proba(validation)[:, 1]
                    )
                )
                trials.append({"embedding_scale": scale, "C": c, "validation": metrics})
                if best is None or metrics["macro_f1"] > best[0]:
                    best = (metrics["macro_f1"], scale, c, classifier)
        _, scale, c, classifier = best
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
            if name not in embeddings:
                embeddings[name] = encode(
                    frozen.model, frozen.tokenizer, [r["text_clean"] for r in rows]
                )
                chars[name] = vectorizer.transform([r["text_clean"] for r in rows])
            features = hstack([chars[name], csr_matrix(embeddings[name] * scale)]).tocsr()
            predictions[name] = attach_probabilities(rows, classifier.predict_proba(features)[:, 1])
    out.mkdir(parents=True)
    np.savez_compressed(out / "features.npz", **embeddings)
    (out / "head.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "classes": classifier.classes_.tolist(),
                "weights": classifier.coef_[0].tolist(),
                "intercept": float(classifier.intercept_[0]),
                "vocabulary": vectorizer.vocabulary_,
                "idf": vectorizer.idf_.tolist(),
                "embedding_scale": scale,
                "encoder": "../frozen-e5-run1/encoder",
                "C": c,
            }
        )
        + "\n"
    )
    report = {
        "protocol": "Character TF-IDF plus frozen pretrained E5 embeddings, balanced logistic regression; train-only features/weights; validation-only grid selection; threshold 0.5. Candidate assessed on development challenge after selection.",
        "selected": {"embedding_scale": scale, "C": c},
        "trials": trials,
        "validation": score(predictions["validation"]),
        "test": score(predictions["test"]),
        "challenge": {
            "all": score(predictions["challenge"]),
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
                Path("models/frozen-e5-run1/encoder/model.safetensors"),
                *[Path(f"data/splits/{s}.csv") for s in SPLITS],
            ]
        },
    }
    report_dir = Path("reports/hybrid-run1")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, rows in predictions.items():
        with (report_dir / f"predictions-{name}.csv").open("w", newline="") as f:
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
