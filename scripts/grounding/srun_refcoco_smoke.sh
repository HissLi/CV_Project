#!/bin/bash
# Launch smoke tests via srun (no sbatch). Runs GDINO then OWL-ViT sequentially.
set -euo pipefail

PROJECT="${PROJECT:-$HOME/cv_project}"
PARTITION="${PARTITION:-gpulab01}"
TIME="${TIME:-01:00:00}"
MEM="${MEM:-32G}"
LOG="$PROJECT/logs/refcoco_srun_smoke_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$PROJECT/logs"

run_block() {
  local model="$1"
  srun -p "$PARTITION" --gres=gpu:1 -n 1 --cpus-per-task=8 --mem="$MEM" -t "$TIME" \
    bash -lc "cd '$PROJECT' && MODE=smoke MODEL=$model bash scripts/grounding/run_refcoco_eval_interactive.sh"
}

{
  echo "=== RefCOCO smoke via srun partition=$PARTITION ==="
  echo "Started: $(date)"
  run_block gdino
  run_block owlvit
  echo "Finished: $(date)"
} 2>&1 | tee "$LOG"

echo "Log: $LOG"
