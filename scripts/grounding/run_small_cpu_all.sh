#!/bin/bash
# Small-sample RefCOCO eval on login-node CPU (no GPU / no sbatch).
# 20 samples per dataset × 3 datasets × 2 models.
set -euo pipefail

PROJECT="${PROJECT:-$HOME/cv_project}"
LOG="$PROJECT/logs/refcoco_small_cpu_$(date +%Y%m%d_%H%M%S).log"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export MODE=small
export SMALL_N="${SMALL_N:-20}"
export DATASETS="${DATASETS:-refcoco,refcoco+,refcocog}"

cd "$PROJECT"
mkdir -p "$PROJECT/logs"

{
  echo "=== RefCOCO small-sample CPU eval ==="
  echo "Started: $(date)"
  echo "SMALL_N=$SMALL_N DATASETS=$DATASETS"
  MODEL=gdino bash scripts/grounding/run_refcoco_eval_interactive.sh
  MODEL=owlvit bash scripts/grounding/run_refcoco_eval_interactive.sh
  echo "Finished: $(date)"
  echo "GDINO: $PROJECT/results/gdino_refcoco_small/params.json"
  echo "OWLViT: $PROJECT/results/owlvit_refcoco_small/params.json"
} 2>&1 | tee "$LOG"

echo "Log: $LOG"
