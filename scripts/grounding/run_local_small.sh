#!/bin/bash
# Small-sample RefCOCO eval on local machine (Mac MPS / CPU).
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
export PYTHONUNBUFFERED=1

if [[ ! -d "$VENV" ]]; then
  echo "Virtualenv not found. Run first:"
  echo "  bash scripts/grounding/setup_local_refcoco.sh"
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJECT"
LOG="$PROJECT/logs/refcoco_local_small_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJECT/logs" "$PROJECT/results"

COMMON=(
  --datasets "$DATASETS"
  --split val
  --refer_data_root "$PROJECT/datasets/refer/data"
  --refer_repo_root "$PROJECT/datasets/refer/refer"
  --coco2014_root "$PROJECT/datasets/mscoco2014"
  --coco2017_root "$PROJECT/datasets/coco"
  --max_samples "$SMALL_N"
)

{
  echo "=== Local RefCOCO small-sample eval ==="
  echo "Started: $(date)"
  echo "SMALL_N=$SMALL_N DEVICE check:"
  python - <<'PY'
import torch
from scripts.grounding.device_utils import get_torch_device
print("torch", torch.__version__)
print("device", get_torch_device())
PY

  echo "=== GDINO ==="
  python "$PROJECT/scripts/gdino/eval_refcoco.py" \
    "${COMMON[@]}" \
    --model_dir "$PROJECT/models/gdino" \
    --hf_model_id "IDEA-Research/grounding-dino-base" \
    --output_dir "$PROJECT/results/gdino_refcoco_small"

  echo "=== OWL-ViT ==="
  python "$PROJECT/scripts/owlvit/eval_refcoco.py" \
    "${COMMON[@]}" \
    --model_dir "$PROJECT/models/owlvit" \
    --hf_model_id "google/owlvit-base-patch32" \
    --output_dir "$PROJECT/results/owlvit_refcoco_small" \
    --batch_size 1

  echo "Finished: $(date)"
  echo "GDINO:  $PROJECT/results/gdino_refcoco_small/params.json"
  echo "OWLViT: $PROJECT/results/owlvit_refcoco_small/params.json"
} 2>&1 | tee "$LOG"

echo "Log: $LOG"
