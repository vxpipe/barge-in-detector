import sys
from types import SimpleNamespace

import pytest

from barge_in_detector.training.data import read_csv, resolve_splits


def make_splits(path):
    path.mkdir(parents=True)
    for name in ["train", "validation", "test"]:
        (path / f"{name}.csv").write_text("text_clean,label,language\nstop,barge_in,en\n")


def test_local_splits_do_not_require_hub(tmp_path):
    make_splits(tmp_path / "splits")
    directory, revision = resolve_splits(tmp_path / "splits")
    assert revision is None
    assert read_csv(directory / "test.csv")[0]["label"] == "barge_in"
    with pytest.raises(ValueError, match="missing"):
        resolve_splits(tmp_path / "absent")


def test_hub_download_uses_resolved_immutable_revision(tmp_path, monkeypatch):
    make_splits(tmp_path / "data/splits")
    seen = {}

    def download(**kwargs):
        seen.update(kwargs)
        return str(tmp_path)

    fake = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(
            dataset_info=lambda *a, **k: SimpleNamespace(sha="fixed-commit")
        ),
        snapshot_download=download,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    directory, revision = resolve_splits(None, repo_id="owner/data", revision="v1")
    assert revision == "fixed-commit"
    assert seen["revision"] == "fixed-commit"
    assert seen["repo_type"] == "dataset"
    assert directory == tmp_path / "data/splits"
