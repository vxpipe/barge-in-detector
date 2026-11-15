"""Frozen pretrained E5 + linear classifier on the existing multilingual splits.

uv run --group train --group baseline python -m experiments.compare_frozen_e5
"""

import argparse
import csv
import json
import platform
import shutil
import time
import warnings
from pathlib import Path

import sklearn
import torch
import transformers
from huggingface_hub import snapshot_download
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoTokenizer

from barge_in_detector.splitting import SPLITS, read_csv
from barge_in_detector.training.frozen_e5 import MAX_LENGTH, PREFIX, LinearHead, encode
from experiments.audit_reviewed_errors import reviewed_predictions, score, sha256

BASE_MODEL = "intfloat/multilingual-e5-small"
BASE_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def attach_probabilities(rows, probabilities):
    return [{**r, "p_barge_in": float(p)} for r, p in zip(rows, probabilities, strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models/frozen-e5-run1"))
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output exists; preserve experiments by using a new directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This run requires the authorized GPU")
    started = time.perf_counter()
    torch.manual_seed(20261115)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    warnings.simplefilter("error", ConvergenceWarning)
    source = json.loads(args.source_manifest.read_text())
    for path, digest in source["files_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed since snapshot: {path}")
    manifest = json.loads(Path("reports/e5-small-run1/source-manifest.json").read_text())
    for split in SPLITS:
        path = Path(f"data/splits/{split}.csv")
        if sha256(path) != manifest[str(path)]:
            raise ValueError("Dataset differs from the original encoder experiment")
    splits = {s: read_csv(Path(f"data/splits/{s}.csv")) for s in SPLITS}
    checkpoint = snapshot_download(
        BASE_MODEL,
        revision=BASE_REVISION,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "README.md",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(
        checkpoint, local_files_only=True, add_pooling_layer=False
    ).to(args.device)
    model.eval().requires_grad_(False)
    loaded = time.perf_counter()
    features = {
        s: encode(model, tokenizer, [r["text_clean"] for r in splits[s]])
        for s in ("train", "validation")
    }
    encoded = time.perf_counter()
    labels = [r["label"] for r in splits["train"]]
    trials, best = [], None
    with threadpool_limits(limits=1):
        for c in (0.1, 1.0, 10.0):
            fit_started = time.perf_counter()
            classifier = LogisticRegression(
                C=c, class_weight="balanced", max_iter=1000, random_state=20261115
            )
            classifier.fit(features["train"], labels)
            fit_seconds = time.perf_counter() - fit_started
            assert list(classifier.classes_) == ["backchannel", "barge_in"]
            validation_rows = attach_probabilities(
                splits["validation"], classifier.predict_proba(features["validation"])[:, 1]
            )
            metrics = score(validation_rows)
            trials.append({"C": c, "fit_seconds": fit_seconds, "validation": metrics})
            if best is None or metrics["macro_f1"] > best[0]:
                best = (metrics["macro_f1"], c, classifier)
        _, chosen_c, classifier = best
        selected = time.perf_counter()
        # Load challenge labels only after model selection. Never refit on them.
        challenge = reviewed_predictions(
            Path("data/evaluation/english-challenge-v1.csv"),
            Path("data/evaluation/reviews/english-challenge-v1.json"),
            Path("reports/e5-small-run1/challenge-predictions.csv"),
        )
        for name, rows in (("test", splits["test"]), ("challenge", challenge)):
            features[name] = encode(model, tokenizer, [r["text_clean"] for r in rows])
        predictions = {
            name: attach_probabilities(rows, classifier.predict_proba(features[name])[:, 1])
            for name, rows in {
                "validation": splits["validation"],
                "test": splits["test"],
                "challenge": challenge,
            }.items()
        }
    head = LinearHead(classifier.coef_[0].tolist(), float(classifier.intercept_[0]))
    args.out.mkdir(parents=True)
    head.save(args.out / "head.json")
    reloaded = LinearHead.load(args.out / "head.json")
    parity = max(
        abs(a - b["p_barge_in"])
        for a, b in zip(
            reloaded.predict(features["challenge"]), predictions["challenge"], strict=True
        )
    )
    assert parity < 1e-6
    model.save_pretrained(args.out / "encoder")
    tokenizer.save_pretrained(args.out / "encoder")
    shutil.copy2(args.source_manifest, args.out / "source-manifest.json")
    checkpoint_hash = sha256(Path(checkpoint) / "model.safetensors")
    seen = {r["text_clean"] for rows in splits.values() for r in rows}
    challenge_rows = predictions["challenge"]
    report = {
        "protocol": "Frozen original pretrained encoder; query prefix; masked mean + L2 normalization; FP32 embeddings. Train-only balanced logistic regression; validation-only C selection; threshold 0.5; no calibration or challenge tuning.",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "base_model_sha256": checkpoint_hash,
        "source_commit": source["source_commit"],
        "source_snapshot_sha256": sha256(args.source_manifest),
        "configuration": {
            "prefix": PREFIX,
            "max_length": MAX_LENGTH,
            "budget_includes_prefix_and_special_tokens": True,
            "batch_size": 128,
            "C_grid": [0.1, 1.0, 10.0],
            "chosen_C": chosen_c,
            "class_weight": "balanced",
            "threshold": 0.5,
            "encoder_trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "feature_count": features["train"].shape[1],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "timing_seconds": {
            "download_and_load": loaded - started,
            "encode_train_validation": encoded - loaded,
            "head_selection": selected - encoded,
            "total_through_export": time.perf_counter() - started,
        },
        "trials": trials,
        "head_reload_max_probability_difference": parity,
        "validation": score(predictions["validation"]),
        "test": score(predictions["test"]),
        "challenge": {
            "all": score(challenge_rows),
            "unseen": score([r for r in challenge_rows if r["text_clean"] not in seen]),
            "by_category": {
                c: score([r for r in challenge_rows if r["category"] == c])
                for c in sorted({r["category"] for r in challenge_rows})
            },
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, rows in predictions.items():
        with (args.out / f"predictions-{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["language", "label", "text_clean", "p_barge_in"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
    hashes = {
        str(p.relative_to(args.out)): sha256(p) for p in sorted(args.out.rglob("*")) if p.is_file()
    }
    (args.out / "artifacts.sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            {
                "chosen_C": chosen_c,
                "test": report["test"],
                "challenge": report["challenge"],
                "timing_seconds": report["timing_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
