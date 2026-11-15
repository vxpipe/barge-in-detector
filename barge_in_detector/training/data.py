"""Read versioned dataset splits without owning their generation or assignment."""

from pathlib import Path

from barge_in_detector.splitting import read_csv as read_csv

DEFAULT_DATASET = "HashNuke/barge-in-detector-data"


def resolve_splits(splits, *, repo_id=DEFAULT_DATASET, revision=None):
    if splits is not None:
        directory = Path(splits)
        resolved_revision = None
    else:
        from huggingface_hub import HfApi, snapshot_download

        resolved_revision = HfApi().dataset_info(repo_id, revision=revision).sha
        root = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=resolved_revision,
                allow_patterns=[
                    "data/splits/*.csv",
                    "data/evaluation/*.csv",
                    "data/evaluation/reviews/*.json",
                    "metadata/languages.yaml",
                ],
            )
        )
        directory = root / "data/splits"
    for split in ("train", "validation", "test"):
        if not (directory / f"{split}.csv").is_file():
            raise ValueError(f"Dataset split missing: {directory / split}.csv")
    return directory, resolved_revision
