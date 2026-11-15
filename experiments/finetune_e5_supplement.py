"""Fixed three-seed E5 fine-tuning; choose on validation before held-out analysis.

Run on the single authorized RTX 5090 inside gpubox:
uv run --group train --group baseline --group export python -m experiments.finetune_e5_supplement
"""

import csv
import gc
import json
import platform
import shutil
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from barge_in_detector.splitting import SPLITS, read_csv
from barge_in_detector.training.evaluate import predict
from barge_in_detector.training.train import main as train
from experiments.audit_reviewed_errors import reviewed_predictions, score, sha256

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SEEDS = (20261115, 20261116, 20261117)
ROOT = Path("models/e5-phrase-run1")
SPLIT_DIR = Path("data/experiments/original-plus-phrase-v1/splits")


def evaluation_sets():
    sets = {f"combined_{s}": read_csv(SPLIT_DIR / f"{s}.csv") for s in ("validation", "test")}
    for name, root in (
        ("original", Path("data/splits")),
        ("supplement", Path("data/supplements/phrase-coverage-v1/splits")),
    ):
        for s in ("validation", "test"):
            sets[f"{name}_{s}"] = read_csv(root / f"{s}.csv")
    sets["challenge"] = reviewed_predictions(
        Path("data/evaluation/english-challenge-v1.csv"),
        Path("data/evaluation/reviews/english-challenge-v1.json"),
        Path("reports/e5-small-run1/challenge-predictions.csv"),
    )
    return sets


def summarize(rows):
    result = {"all": score(rows)}
    if rows and "category" in rows[0]:
        result["by_category"] = {
            c: score([r for r in rows if r["category"] == c])
            for c in sorted({r["category"] for r in rows})
        }
        groups = {r["group"] for r in rows if r.get("group")}
        result["prefix_groups"] = {
            g: [
                {k: r[k] for k in ("text_clean", "label", "p_barge_in")}
                for r in rows
                if r.get("group") == g
            ]
            for g in sorted(groups)
        }
    return result


def main():
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This experiment requires the authorized RTX 5090")
    if ROOT.exists():
        raise ValueError("Preserve earlier runs; choose a new run name")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    ROOT.mkdir(parents=True)
    reports = ROOT / "reports"
    reports.mkdir()
    splits = {s: read_csv(SPLIT_DIR / f"{s}.csv") for s in SPLITS}
    assert [len(splits[s]) for s in SPLITS] == [4918, 915, 908]
    assert all(len({r["language"] for r in rows}) == 14 for rows in splits.values())
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not {r["text_clean"] for r in splits[a]} & {r["text_clean"] for r in splits[b]}
    started = time.perf_counter()
    base = Path("models/base-e5-pinned")
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=base,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ],
    )
    histories = {}
    durations = {}
    for seed in SEEDS:
        dest = ROOT / f"seed-{seed}"
        seed_started = time.perf_counter()
        train(
            [
                "--model",
                str(base),
                "--out",
                str(dest),
                "--splits",
                str(SPLIT_DIR),
                "--epochs",
                "5",
                "--batch-size",
                "32",
                "--lr",
                "3e-5",
                "--warmup",
                "0.1",
                "--max-length",
                "64",
                "--seed",
                str(seed),
            ]
        )
        torch.cuda.synchronize()
        durations[seed] = time.perf_counter() - seed_started
        histories[seed] = json.loads((dest / "metrics.json").read_text())
        shutil.copyfile(dest / "metrics.json", reports / f"training-seed-{seed}.json")
        gc.collect()
        torch.cuda.empty_cache()
    selected = max(SEEDS, key=lambda s: histories[s]["best"]["validation"]["macro_f1"])
    selection = {
        "seed": selected,
        "epoch": histories[selected]["best"]["epoch"],
        "criterion": "Highest combined-validation macro F1, seed order breaks ties; fixed before test/challenge loading",
        "validation": histories[selected]["best"]["validation"],
    }
    (reports / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    shutil.copytree(ROOT / f"seed-{selected}", ROOT / "checkpoint")
    # Held-out analysis begins only after the checkpoint has been selected.
    sets = evaluation_sets()
    results = {}
    for seed in SEEDS:
        checkpoint = ROOT / f"seed-{seed}"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = (
            AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
            .cuda()
            .eval()
        )
        result = {}
        for name, rows in sets.items():
            probs = predict(model, tokenizer, rows, batch_size=128, max_length=64)
            predicted = [{**r, "p_barge_in": p} for r, p in zip(rows, probs, strict=True)]
            result[name] = summarize(predicted)
            with (reports / f"predictions-seed-{seed}-{name}.csv").open("w", newline="") as f:
                fields = [*rows[0]]
                if "p_barge_in" not in fields:
                    fields.append("p_barge_in")
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(predicted)
            if seed == selected and name == "challenge":
                single = predict(model, tokenizer, rows, batch_size=1, max_length=64)
                result["batch_single_parity"] = {
                    "max_probability_difference": max(
                        abs(a - b) for a, b in zip(probs, single, strict=True)
                    ),
                    "label_disagreements": sum(
                        (a >= 0.5) != (b >= 0.5) for a, b in zip(probs, single, strict=True)
                    ),
                }
        results[seed] = result
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "base_model": MODEL,
        "base_revision": REVISION,
        "seeds": list(SEEDS),
        "selection": selection,
        "training_seconds_by_seed": durations,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "split_counts": {s: len(rows) for s, rows in splits.items()},
        "train_language_counts": {
            lang: sum(r["language"] == lang for r in splits["train"])
            for lang in sorted({r["language"] for r in splits["train"]})
        },
        "input_sha256": {
            str(SPLIT_DIR / f"{s}.csv"): sha256(SPLIT_DIR / f"{s}.csv") for s in SPLITS
        },
        "base_weights_sha256": sha256(base / "model.safetensors"),
        "code_sha256": sha256(Path(__file__)),
        "results": results,
        "limitations": "Synthetic heldouts; reviewed English is reused development data with historical overlap. Uncalibrated threshold 0.5. No production promotion.",
    }
    (reports / "gpu-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selection": selection,
                "training_seconds": durations,
                "reviewed_by_seed": {s: r["challenge"]["all"] for s, r in results.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
