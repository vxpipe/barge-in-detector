"""One fixed-architecture original-versus-additive data ablation, on local CPU.

uv run --group baseline python -m experiments.compare_phrase_supplement
Original model, data, thresholds and evaluation files are preserved. Hyperparameters
are frozen to the previously selected lexical model; no test-based tuning.
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
from barge_in_detector.training.lexical import LexicalModel
from experiments.audit_reviewed_errors import reviewed_predictions, score, sha256


def summaries(rows):
    result = {
        "all": score(rows),
        "by_language": {
            language: score([r for r in rows if r["language"] == language])
            for language in sorted({r["language"] for r in rows})
        },
    }
    if rows and "category" in rows[0]:
        result["by_category"] = {
            category: score([r for r in rows if r["category"] == category])
            for category in sorted({r["category"] for r in rows})
        }
    return result


def main():
    started = time.perf_counter()
    warnings.simplefilter("error", ConvergenceWarning)
    model_dir, report_dir = Path("models/lexical-phrase-v1"), Path("reports/lexical-phrase-v1")
    if model_dir.exists() or report_dir.exists():
        raise ValueError("Preserve existing experiment; use a new run name for another ablation")
    original = {s: read_csv(Path("data/splits") / f"{s}.csv") for s in SPLITS}
    supplement = {
        s: read_csv(Path("data/supplements/phrase-coverage-v1/splits") / f"{s}.csv") for s in SPLITS
    }
    train = original["train"] + supplement["train"]
    selected = json.loads(Path("reports/lexical-run1/report.json").read_text())["selected"]
    baseline = LexicalModel("models/lexical-run1/portable")
    # Both models receive the same 64-token serving inputs for evaluation.
    train_texts = [baseline.truncate(r["text_clean"]) for r in train]
    chars = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False, sublinear_tf=True)
    words = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    scale = selected["word_scale"]
    feature_started = time.perf_counter()
    matrix = hstack(
        [chars.fit_transform(train_texts), words.fit_transform(train_texts) * scale]
    ).tocsr()
    feature_seconds = time.perf_counter() - feature_started
    classifier = LogisticRegression(
        C=selected["C"], class_weight="balanced", max_iter=1000, random_state=20261115
    )
    challenge = reviewed_predictions(
        Path("data/evaluation/english-challenge-v1.csv"),
        Path("data/evaluation/reviews/english-challenge-v1.json"),
        Path("reports/e5-small-run1/challenge-predictions.csv"),
    )
    sets = {
        "validation": original["validation"],
        "test": original["test"],
        "supplement_validation": supplement["validation"],
        "supplement_test": supplement["test"],
        "challenge": challenge,
    }
    train_seen = {r["text_clean"] for r in train}
    assert not train_seen.intersection(
        r["text_clean"] for key, rows in sets.items() if key != "challenge" for r in rows
    )
    # Some challenge strings existed in the original dataset before this work;
    # new supplement training was explicitly screened against all challenge rows.
    assert not {r["text_clean"] for r in supplement["train"]}.intersection(
        r["text_clean"] for r in challenge
    )
    predictions, results = {}, {}
    with threadpool_limits(limits=1):
        fit_started = time.perf_counter()
        classifier.fit(matrix, [r["label"] for r in train])
        fit_seconds = time.perf_counter() - fit_started
        assert list(classifier.classes_) == ["backchannel", "barge_in"]
        for name, rows in sets.items():
            texts = [baseline.truncate(r["text_clean"]) for r in rows]
            features = hstack([chars.transform(texts), words.transform(texts) * scale]).tocsr()
            new_probs, old_probs = classifier.predict_proba(features)[:, 1], baseline.predict(texts)
            new = [{**r, "p_barge_in": float(p)} for r, p in zip(rows, new_probs, strict=True)]
            old = [{**r, "p_barge_in": float(p)} for r, p in zip(rows, old_probs, strict=True)]
            predictions[name] = new
            results[name] = {
                "original_model": summaries(old),
                "additive_model": summaries(new),
                "fixed": sum(
                    (a["p_barge_in"] >= 0.5) != (a["label"] == "barge_in")
                    and (b["p_barge_in"] >= 0.5) == (b["label"] == "barge_in")
                    for a, b in zip(old, new, strict=True)
                ),
                "regressed": sum(
                    (a["p_barge_in"] >= 0.5) == (a["label"] == "barge_in")
                    and (b["p_barge_in"] >= 0.5) != (b["label"] == "barge_in")
                    for a, b in zip(old, new, strict=True)
                ),
            }
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    joblib.dump(
        {"chars": chars, "words": words, "word_scale": scale, "classifier": classifier},
        model_dir / "local-experiment.joblib",
    )
    for name, rows in predictions.items():
        with (report_dir / f"predictions-{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["language", "label", "text_clean", "p_barge_in"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "protocol": "Fixed architecture/C/word scale/0.5 threshold; additive training using original plus supplement train. 64-token input budget. All 14 languages. No tuning on test or reviewed challenge.",
        "selected_before_this_run": selected,
        "original_train_count": len(original["train"]),
        "supplement_train_count": len(supplement["train"]),
        "training_language_counts": {
            language: sum(r["language"] == language for r in train)
            for language in sorted({r["language"] for r in train})
        },
        "training_inputs_truncated": sum(
            a != b["text_clean"] for a, b in zip(train_texts, train, strict=True)
        ),
        "results": results,
        "feature_build_seconds": feature_seconds,
        "classifier_fit_seconds": fit_seconds,
        "training_seconds": feature_seconds + fit_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "input_sha256": {
            str(p): sha256(p)
            for p in [
                Path(__file__).relative_to(Path.cwd()),
                Path("data/supplements/phrase-coverage-v1/utterances.csv"),
                *[Path("data/splits") / f"{s}.csv" for s in SPLITS],
                *[Path("data/supplements/phrase-coverage-v1/splits") / f"{s}.csv" for s in SPLITS],
            ]
        },
        "limitations": "Synthetic test labels are unverified; challenge reused for development. Fresh owner review, calibration and integration acceptance still required.",
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "counts": {"old_train": len(original["train"]), "new_train": len(train)},
                "seconds": report["elapsed_seconds"],
                "results": {
                    k: {
                        "original": v["original_model"]["all"],
                        "additive": v["additive_model"]["all"],
                        "fixed": v["fixed"],
                        "regressed": v["regressed"],
                    }
                    for k, v in results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
