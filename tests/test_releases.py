import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from barge_in_detector.releases import allowed_path, build_release, publish_release


def test_additive_model_candidate_keeps_existing_root_release(tmp_path):
    root, artifacts, manifest = fixture_release(tmp_path)
    data = json.loads(manifest.read_text())
    prefix = "candidates/e5-phrase-run1/checkpoint"
    names = list(data["artifacts_sha256"])
    data["artifacts_sha256"] = {}
    for name in names:
        destination = artifacts / prefix / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((artifacts / name).read_bytes())
        data["artifacts_sha256"][f"{prefix}/{name}"] = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()
    for suffix in (
        "onnx/model-fp32.onnx",
        "onnx/model-int8.onnx",
        "onnx/metadata.json",
        "onnx/tokenizer.json",
        "README.md",
        "provenance.json",
        "artifacts.sha256.json",
    ):
        name = f"candidates/e5-phrase-run1/{suffix}"
        p = artifacts / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("example")
        data["artifacts_sha256"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
    data["model_subdirectory"] = prefix
    manifest.write_text(json.dumps(data))
    release = build_release("model", root=root, manifest_path=manifest, artifacts=artifacts)
    assert "model.safetensors" not in release.files
    assert f"{prefix}/model.safetensors" in release.files
    assert "candidates/e5-phrase-run1/onnx/model-int8.onnx" in release.files


@pytest.mark.parametrize(
    "name",
    [
        "data/supplements/phrase-coverage-v1/utterances.csv",
        "data/supplements/phrase-coverage-v1/splits/train.csv",
        "data/supplements/phrase-coverage-v1/README.md",
        "data/experiments/original-plus-phrase-v1/splits/test.csv",
        "data/experiments/original-plus-phrase-v1/README.md",
    ],
)
def test_additive_dataset_paths_are_allowed(name):
    assert allowed_path("dataset", name)


@pytest.mark.parametrize(
    "kind,name",
    [
        ("model", "candidates/run/checkpoint/optimizer.pt"),
        ("model", "candidates/run/ssh.json"),
        ("model", "candidates/run/checkpoint/training.py"),
        ("model", "candidates/run/../model.safetensors"),
        ("dataset", "data/supplements/run/generate.py"),
        ("dataset", "data/supplements/run/private.json"),
        ("dataset", "data/experiments/run/splits/../../secret.csv"),
    ],
)
def test_candidate_paths_do_not_relax_release_boundary(kind, name):
    assert not allowed_path(kind, name)


def fixture_release(tmp_path, kind="model"):
    root = tmp_path / "source"
    root.mkdir()
    cards = root / "releases/cards" / kind
    cards.mkdir(parents=True)
    (cards / "README.md").write_text("---\nlicense: mit\n---\nExperimental release\n")
    (root / "LICENSE").write_text("MIT")
    artifacts = root / "models/run"
    artifacts.mkdir(parents=True)
    names = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    if kind == "dataset":
        names = ["data/splits/train.csv", "data/splits/validation.csv", "data/splits/test.csv"]
        artifacts = root
    hashes = {}
    for name in names:
        path = artifacts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "source_repo": "https://github.com/vxpipe/barge-in-detector",
        "code_revision": "a" * 40,
    }
    manifest["artifacts_sha256" if kind == "model" else "files_sha256"] = hashes
    if kind == "model":
        manifest.update(
            dataset_repo="HashNuke/barge-in-detector-data",
            dataset_revision="b" * 40,
            model_id="HashNuke/barge-in-detector",
        )
    path = root / "release.json"
    path.write_text(json.dumps(manifest))
    return root, artifacts, path


@pytest.mark.parametrize("kind", ["model", "dataset"])
def test_manifest_selects_only_release_files(tmp_path, kind):
    root, artifacts, manifest = fixture_release(tmp_path, kind)
    (artifacts / "optimizer.pt").write_text("private checkpoint")
    result = build_release(kind, root=root, manifest_path=manifest, artifacts=artifacts)
    assert "optimizer.pt" not in result.files
    assert "README.md" in result.files
    assert "LICENSE" in result.files
    assert ("release.json" if kind == "model" else "dataset-release.json") in result.files


def test_modified_artifact_rejected(tmp_path):
    root, artifacts, manifest = fixture_release(tmp_path)
    (artifacts / "config.json").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        build_release("model", root=root, manifest_path=manifest, artifacts=artifacts)


@pytest.mark.parametrize("name", ["../secret.json", "/secret.json", "optimizer.pt", "training.py"])
def test_disallowed_paths_rejected(tmp_path, name):
    root, artifacts, manifest = fixture_release(tmp_path)
    data = json.loads(manifest.read_text())
    data["artifacts_sha256"][name] = "a" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="path"):
        build_release("model", root=root, manifest_path=manifest, artifacts=artifacts)


def test_symlink_rejected(tmp_path):
    root, artifacts, manifest = fixture_release(tmp_path)
    target = artifacts / "config.json"
    target.unlink()
    target.symlink_to(root / "LICENSE")
    with pytest.raises(ValueError, match="symlink"):
        build_release("model", root=root, manifest_path=manifest, artifacts=artifacts)


def test_requires_immutable_dataset_revision(tmp_path):
    root, artifacts, manifest = fixture_release(tmp_path)
    data = json.loads(manifest.read_text())
    data["dataset_revision"] = "main"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="dataset_revision"):
        build_release("model", root=root, manifest_path=manifest, artifacts=artifacts)


def test_atomic_upload_pins_parent_and_does_not_delete_remote_files(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(CommitOperationAdd=lambda **kw: SimpleNamespace(**kw), HfApi=lambda: None),
    )
    root, artifacts, manifest = fixture_release(tmp_path)
    release = build_release("model", root=root, manifest_path=manifest, artifacts=artifacts)
    calls = []
    api = SimpleNamespace(
        repo_info=lambda **kw: SimpleNamespace(sha="c" * 40),
        create_commit=lambda **kw: calls.append(kw) or SimpleNamespace(oid="d" * 40),
    )
    assert publish_release(release, api=api) == "d" * 40
    assert calls[0]["parent_commit"] == "c" * 40
    assert calls[0]["repo_type"] == "model"
    assert {op.path_in_repo for op in calls[0]["operations"]} == set(release.files)


def test_dataset_language_manifest_uses_factor_catalogue(tmp_path):
    root, artifacts, manifest = fixture_release(tmp_path, "dataset")
    language = root / "generation/factors/languages.yaml"
    language.parent.mkdir(parents=True)
    language.write_text("languages: []\n")
    data = json.loads(manifest.read_text())
    data["files_sha256"]["metadata/languages.yaml"] = hashlib.sha256(
        language.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(data))
    release = build_release("dataset", root=root, manifest_path=manifest)
    assert release.files["metadata/languages.yaml"] == language


def test_dry_run_is_offline_and_refresh_is_explicit(tmp_path, monkeypatch, capsys):
    from barge_in_detector import releases

    root, artifacts, manifest = fixture_release(tmp_path)
    monkeypatch.setattr(releases, "ROOT", root)
    monkeypatch.setattr(releases, "publish_release", lambda *_: pytest.fail("unexpected upload"))
    argv = ["model", "--manifest", str(manifest), "--artifacts", str(artifacts)]
    before = manifest.read_bytes()
    assert releases.main(argv) == 0
    assert "Offline dry run" in capsys.readouterr().out
    assert manifest.read_bytes() == before
    (artifacts / "config.json").write_text("updated export")
    assert releases.main([*argv, "--refresh-hashes"]) == 0
    assert manifest.read_bytes() != before
    assert releases.main(argv) == 0


def test_publish_checks_dataset_bytes_against_source_commit(tmp_path, monkeypatch):
    from barge_in_detector import releases

    root, artifacts, manifest = fixture_release(tmp_path, "dataset")
    release = build_release("dataset", root=root, manifest_path=manifest)
    commands = []

    def git(args, **kwargs):
        commands.append(args)
        if args[1] == "ls-remote":
            return SimpleNamespace(stdout="c" * 40 + "\trefs/heads/main\n")
        return SimpleNamespace(stdout=b"different source bytes")

    monkeypatch.setattr(releases.subprocess, "run", git)
    with pytest.raises(ValueError, match="differ from code_revision"):
        releases.verify_source(release, root)
    assert commands[1] == ["git", "merge-base", "--is-ancestor", "a" * 40, "c" * 40]
