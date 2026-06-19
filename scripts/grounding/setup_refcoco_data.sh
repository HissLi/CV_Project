#!/bin/bash
set -euo pipefail

BASE="$HOME/cv_project/datasets/refer/data"
PADT="$HOME/cv_project/datasets/refer/padt"
mkdir -p "$BASE/images" "$PADT"
cd "$PADT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python - <<'PY'
from huggingface_hub import hf_hub_download
files = ["refcoco_val.json", "refcoco+_val.json", "refcocog_val.json"]
for filename in files:
    path = hf_hub_download(
        repo_id="PaDT-MLLM/RefCOCO",
        repo_type="dataset",
        filename=filename,
        local_dir="/home/turing_lab/cse12210210/cv_project/datasets/refer/padt",
    )
    print("DOWN", path)
PY

ln -sfn "$HOME/cv_project/datasets/coco" "$BASE/images/mscoco"
ls -la "$PADT"
echo "REF_SETUP_DONE_PADT"
