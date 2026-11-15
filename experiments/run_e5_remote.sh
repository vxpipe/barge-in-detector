#!/usr/bin/env bash
# Start this script inside the Vast.ai tmux session gpubox.
set -Eeuo pipefail
mkdir -p /workspace/barge-in-detector
cd /workspace/barge-in-detector
exec > >(tee -a setup.log) 2>&1
trap 'rc=$?; echo "$rc" > run.exit' EXIT
tar -xzf /tmp/barge-e5-source.tar.gz
python3 - <<'PY'
import hashlib, json
from pathlib import Path
m=json.loads(Path("reports/e5-phrase-run1/source-manifest.json").read_text())
for name, expected in m["sha256"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected, name
print("Uploaded source/data hashes verified", flush=True)
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
fi
uv sync --frozen --python 3.14 --group train --group baseline --group export
echo "Running fixed E5 fine-tuning protocol"
uv run --no-sync python -m experiments.finetune_e5_supplement > training.log 2>&1
echo "Training and GPU analyses completed; exporting CPU models"
uv run --no-sync python -m barge_in_detector.training.onnx_export \
    --checkpoint models/e5-phrase-run1/checkpoint --out models/e5-phrase-run1/onnx > export.log 2>&1
echo "Verifying exports against GPU predictions"
uv run --no-sync python -m experiments.verify_e5_export > verification.log 2>&1
uv run --no-sync python - <<'PY'
import hashlib, json, shutil
from pathlib import Path
root=Path("models/e5-phrase-run1")
shutil.copyfile("reports/e5-phrase-run1/source-manifest.json",root/"reports/source-manifest.json")
(root/"logs").mkdir(exist_ok=True)
for name in ("training.log","export.log","verification.log"):
    shutil.copyfile(name,root/"logs"/name)
files=[p for sub in ("checkpoint","onnx","reports","logs") for p in (root/sub).rglob("*")
       if p.is_file() and p.name!="model-raw.onnx"]
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(root/"artifacts.sha256.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("Ready to download",len(files),"files",sum(p.stat().st_size for p in files),"bytes",flush=True)
PY
