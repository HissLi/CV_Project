#!/bin/bash
# One-time local setup for RefCOCO small-sample eval (Mac M-series friendly).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="${VENV:-$PROJECT/.venv-refcoco}"
SMALL_N="${SMALL_N:-20}"
DATASETS="${DATASETS:-refcoco,refcoco+,refcocog}"
export HF_ENDPOINT="${HF_ENDPOINT:-}"
if [[ -n "$HF_ENDPOINT" ]]; then
  export HUGGINGFACE_HUB_ENDPOINT="$HF_ENDPOINT"
fi

PADT_DIR="$PROJECT/datasets/refer/padt"
COCO_ROOT="$PROJECT/datasets/coco"
GDINO_DIR="$PROJECT/models/gdino"
OWLVIT_DIR="$PROJECT/models/owlvit"

echo "=== Local RefCOCO setup ==="
echo "PROJECT=$PROJECT"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip
python -m pip install -r "$PROJECT/requirements-local-refcoco.txt"

mkdir -p "$PADT_DIR" "$COCO_ROOT/val2017" "$COCO_ROOT/train2017" "$GDINO_DIR" "$OWLVIT_DIR" "$PROJECT/logs"

echo "=== Download PaDT RefCOCO val jsonl (~40 MB) ==="
python - <<PY
from huggingface_hub import hf_hub_download
files = ["refcoco_val.json", "refcoco+_val.json", "refcocog_val.json"]
for filename in files:
    path = hf_hub_download(
        repo_id="PaDT-MLLM/RefCOCO",
        repo_type="dataset",
        filename=filename,
        local_dir="${PADT_DIR}",
        local_dir_use_symlinks=False,
    )
    print("DOWN", path)
PY

echo "=== Download COCO images for small subset only ==="
python "$PROJECT/scripts/grounding/fetch_coco_images.py" \
  --padt_dir "$PADT_DIR" \
  --coco2017_root "$COCO_ROOT" \
  --datasets "$DATASETS" \
  --max_samples "$SMALL_N"

echo "=== Download / cache models (GDINO ~1.5 GB, OWL-ViT ~600 MB) ==="
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("IDEA-Research/grounding-dino-base", local_dir="${GDINO_DIR}")
snapshot_download("google/owlvit-base-patch32", local_dir="${OWLVIT_DIR}")
print("MODELS_OK")
PY

echo ""
echo "Setup done. Run eval with:"
echo "  bash scripts/grounding/run_local_small.sh"
