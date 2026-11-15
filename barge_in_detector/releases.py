"""Manifest-selected HF releases; no model/training dependencies required."""

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/vxpipe/barge-in-detector"
REPOS = {"model": "HashNuke/barge-in-detector", "dataset": "HashNuke/barge-in-detector-data"}
MANIFEST_NAMES = {"model": "release.json", "dataset": "dataset-release.json"}
HASH_KEYS = {"model": "artifacts_sha256", "dataset": "files_sha256"}


def allowed_path(kind, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name:
        return False
    if kind == "model":
        candidate = re.fullmatch(r"candidates/[a-z0-9][a-z0-9-]*/(.+)", name)
        if candidate:
            name = candidate[1]
            if name in {
                "README.md",
                "LICENSE",
                "NOTICE",
                "provenance.json",
                "artifacts.sha256.json",
            }:
                return True
            name = re.sub(r"^(?:checkpoint|seed-\d{8})/", "", name)
        if re.fullmatch(r"releases/[a-z0-9][a-z0-9-]*\.json", name):
            return True
        return bool(
            re.fullmatch(
                r"(?:config\.json|tokenizer\.json|tokenizer_config\.json|special_tokens_map\.json|"
                r"added_tokens\.json|vocab\.(?:txt|json)|merges\.txt|sentencepiece\.bpe\.model|"
                r"spiece\.model|model(?:-\d{5}-of-\d{5})?\.safetensors|model\.safetensors\.index\.json|"
                r"(?:onnx/)?model(?:_quantized|_int8|-fp32|-int8)?\.onnx(?:_data)?|"
                r"onnx/(?:metadata|tokenizer)\.json)",
                name,
            )
        )
    return bool(
        re.fullmatch(
            r"(?:data/(?:export|splits|evaluation)/[a-zA-Z0-9_-]+\.csv|"
            r"data/evaluation/reviews/[a-zA-Z0-9_-]+\.json|metadata/languages\.yaml|"
            r"data/supplements/[a-z0-9][a-z0-9-]*/(?:utterances\.csv|README\.md|"
            r"splits/(?:train|validation|test)\.csv)|"
            r"data/experiments/[a-z0-9][a-z0-9-]*/(?:README\.md|"
            r"splits/(?:train|validation|test)\.csv))",
            name,
        )
    )


def checked_file(base, relative):
    path = base / relative
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlink is not a release input: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing release file: {relative}")
    return path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class Release:
    kind: str
    repo_id: str
    metadata: dict
    files: dict[str, Path]


def build_release(kind, *, root, manifest_path, artifacts=None, verify_hashes=True):
    metadata = json.loads(checked_file(manifest_path.parent, manifest_path.name).read_text())
    if metadata.get("source_repo") != SOURCE:
        raise ValueError(f"source_repo must be {SOURCE}")
    pins = ["code_revision"] + (["dataset_revision"] if kind == "model" else [])
    for key in pins:
        if not re.fullmatch(r"[0-9a-f]{40}", metadata.get(key, "")):
            raise ValueError(f"{key} must be an immutable 40-character commit SHA")
    if kind == "model" and (
        metadata.get("model_id") != REPOS["model"]
        or metadata.get("dataset_repo") != REPOS["dataset"]
    ):
        raise ValueError("model_id and dataset_repo must name the project HF repositories")
    hashes = metadata.get(HASH_KEYS[kind])
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Manifest must select files with a nonempty hash mapping")
    model_subdirectory = metadata.get("model_subdirectory", "")
    if model_subdirectory and not re.fullmatch(
        r"candidates/[a-z0-9][a-z0-9-]*/checkpoint", model_subdirectory
    ):
        raise ValueError("Invalid model_subdirectory")
    prefix = f"{model_subdirectory}/" if model_subdirectory else ""
    required = (
        {f"{prefix}{name}" for name in ("config.json", "tokenizer.json", "tokenizer_config.json")}
        if kind == "model"
        else {f"data/splits/{split}.csv" for split in ("train", "validation", "test")}
    )
    if not required <= hashes.keys():
        raise ValueError(f"Missing required release paths: {sorted(required - hashes.keys())}")
    if kind == "model" and not any(n.endswith((".safetensors", ".onnx")) for n in hashes):
        raise ValueError("Missing model weights path")
    files = {}
    for name, expected in hashes.items():
        if not allowed_path(kind, name):
            raise ValueError(f"Disallowed release path: {name}")
        base = artifacts if kind == "model" else root
        source_name = (
            "generation/factors/languages.yaml" if name == "metadata/languages.yaml" else name
        )
        path = checked_file(base, source_name)
        actual = sha256(path)
        if verify_hashes and actual != expected:
            raise ValueError(
                f"Release hash mismatch: {name}; review inputs before refreshing hashes"
            )
        files[name] = path
    cards = root / "releases/cards" / kind
    files["README.md"] = checked_file(cards, "README.md")
    files["LICENSE"] = checked_file(root, "LICENSE")
    if (cards / "NOTICE").exists():
        files["NOTICE"] = checked_file(cards, "NOTICE")
    files[MANIFEST_NAMES[kind]] = manifest_path
    return Release(kind, REPOS[kind], metadata, files)


def verify_source(release, root):
    """Require source publication and exact dataset bytes at the source pin."""
    revision = release.metadata["code_revision"]
    result = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/main"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.split()[0]
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, head], cwd=root, check=True)
    if release.kind == "dataset":
        for name in release.metadata[HASH_KEYS["dataset"]]:
            source = (
                "generation/factors/languages.yaml" if name == "metadata/languages.yaml" else name
            )
            content = subprocess.run(
                ["git", "show", f"{revision}:{source}"], cwd=root, check=True, capture_output=True
            ).stdout
            if hashlib.sha256(content).hexdigest() != sha256(release.files[name]):
                raise ValueError(f"Dataset bytes differ from code_revision: {source}")


def publish_release(release, *, api=None):
    from huggingface_hub import CommitOperationAdd, HfApi

    api = api or HfApi()
    parent = api.repo_info(repo_id=release.repo_id, repo_type=release.kind, revision="main").sha
    if release.kind == "model":
        api.repo_info(
            repo_id=release.metadata["dataset_repo"],
            repo_type="dataset",
            revision=release.metadata["dataset_revision"],
        )
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
        for name, path in sorted(release.files.items())
    ]
    result = api.create_commit(
        repo_id=release.repo_id,
        repo_type=release.kind,
        revision="main",
        parent_commit=parent,
        operations=operations,
        commit_message=f"Release from source {release.metadata['code_revision']}",
    )
    return result.oid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=REPOS)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="versioned release manifest under releases/"
    )
    parser.add_argument("--artifacts", type=Path, help="model export directory, e.g. models/run2")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--publish", action="store_true", help="upload to HF; default is offline dry run"
    )
    mode.add_argument(
        "--refresh-hashes",
        action="store_true",
        help="update selected file hashes locally, without uploading",
    )
    args = parser.parse_args(argv)
    if args.kind == "model" and args.artifacts is None:
        parser.error("model releases require --artifacts")
    try:
        release = build_release(
            args.kind,
            root=ROOT,
            manifest_path=args.manifest,
            artifacts=args.artifacts,
            verify_hashes=not args.refresh_hashes,
        )
        if args.refresh_hashes:
            release.metadata[HASH_KEYS[args.kind]] = {
                name: sha256(release.files[name]) for name in release.metadata[HASH_KEYS[args.kind]]
            }
            args.manifest.write_text(json.dumps(release.metadata, indent=2) + "\n")
            print(
                f"Updated {args.manifest}; review metadata, counts, provenance and hashes before publishing."
            )
            return 0
        print(f"{release.kind}: {release.repo_id}")
        for name, path in sorted(release.files.items()):
            print(f"  {sha256(path)}  {path.stat().st_size:>12}  {name}")
        if args.publish:
            verify_source(release, ROOT)
            print(f"Published revision: {publish_release(release)}")
        else:
            print("Offline dry run; no API calls or writes. Add --publish to upload.")
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
