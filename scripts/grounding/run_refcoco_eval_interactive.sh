#!/bin/bash
# Run RefCOCO grounding eval without sbatch (interactive GPU shell or srun --pty).
#
# Usage (inside GPU shell):
#   cd ~/cv_project
#   MODE=small MODEL=gdino bash scripts/grounding/run_refcoco_eval_interactive.sh
#   MODE=small MODEL=owlvit bash scripts/grounding/run_refcoco_eval_interactive.sh
#   MODE=full  MODEL=gdino bash scripts/grounding/run_refcoco_eval_interactive.sh
#   MODE=full  MODEL=owlvit bash scripts/grounding/run_refcoco_eval_interactive.sh
#
# Or launch GPU shell from login node:
#   PARTITION=a100 TIME=02:00:00 bash scripts/grounding/run_refcoco_eval_interactive.sh shell

set -euo pipefail

PROJECT="${PROJECT:-$HOME/cv_project}"
MODE="${MODE:-smoke}"       # smoke | small | full
MODEL="${MODEL:-gdino}"     # gdino | owlvit
SMALL_N="${SMALL_N:-20}"    # samples per dataset when MODE=small
PARTITION="${PARTITION:-gpulab01}"
TIME="${TIME:-02:00:00}"
MEM="${MEM:-32G}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

mkdir -p "$PROJECT/logs"

run_gdino() {
  source /opt/ohpc/pub/apps/anaconda3/bin/activate gdino
  local max_arg=""
  local out="$PROJECT/results/gdino_refcoco_zeroshot"
  if [[ "$MODE" == "smoke" ]]; then
    max_arg="--max_samples 100"
    out="$PROJECT/results/gdino_refcoco_zeroshot_smoke"
  elif [[ "$MODE" == "small" ]]; then
    max_arg="--max_samples ${SMALL_N}"
    out="$PROJECT/results/gdino_refcoco_small"
  fi
  echo "=== GDINO eval MODE=$MODE OUT=$out ==="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  else
    echo "WARN: no GPU visible (nvidia-smi missing); running on CPU will be very slow"
  fi
  python "$PROJECT/scripts/gdino/eval_refcoco.py" \
    --datasets "${DATASETS:-refcoco,refcoco+,refcocog}" \
    --split "${SPLIT:-val}" \
    --model_dir "${MODEL_DIR:-$PROJECT/models/gdino}" \
    --refer_data_root "${REFER_DATA_ROOT:-$PROJECT/datasets/refer/data}" \
    --refer_repo_root "${REFER_REPO_ROOT:-$PROJECT/datasets/refer/refer}" \
    --coco2014_root "${COCO2014_ROOT:-$PROJECT/datasets/mscoco2014}" \
    --coco2017_root "${COCO2017_ROOT:-$PROJECT/datasets/coco}" \
    --output_dir "$out" \
    --threshold "${THRESHOLD:-0.25}" \
    $max_arg
}

run_owlvit() {
  source /opt/ohpc/pub/apps/anaconda3/bin/activate owlvit
  local max_arg=""
  local out="$PROJECT/results/owlvit_refcoco_zeroshot"
  if [[ "$MODE" == "smoke" ]]; then
    max_arg="--max_samples 100"
    out="$PROJECT/results/owlvit_refcoco_zeroshot_smoke"
  elif [[ "$MODE" == "small" ]]; then
    max_arg="--max_samples ${SMALL_N}"
    out="$PROJECT/results/owlvit_refcoco_small"
  fi
  echo "=== OWL-ViT eval MODE=$MODE OUT=$out ==="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  else
    echo "WARN: no GPU visible (nvidia-smi missing); running on CPU will be very slow"
  fi
  python "$PROJECT/scripts/owlvit/eval_refcoco.py" \
    --datasets "${DATASETS:-refcoco,refcoco+,refcocog}" \
    --split "${SPLIT:-val}" \
    --model_dir "${MODEL_DIR:-$PROJECT/models/owlvit}" \
    --refer_data_root "${REFER_DATA_ROOT:-$PROJECT/datasets/refer/data}" \
    --refer_repo_root "${REFER_REPO_ROOT:-$PROJECT/datasets/refer/refer}" \
    --coco2014_root "${COCO2014_ROOT:-$PROJECT/datasets/mscoco2014}" \
    --coco2017_root "${COCO2017_ROOT:-$PROJECT/datasets/coco}" \
    --output_dir "$out" \
    --threshold "${THRESHOLD:-0.1}" \
    --batch_size "${BATCH_SIZE:-$([[ "$MODE" == "small" ]] && echo 2 || echo 8)}" \
    $max_arg
}

if [[ "${1:-}" == "shell" ]] || [[ "$MODE" == "shell" ]]; then
  echo "Requesting interactive GPU: partition=$PARTITION time=$TIME mem=$MEM"
  exec srun -p "$PARTITION" --gres=gpu:1 -n 1 --cpus-per-task=8 --mem="$MEM" -t "$TIME" --pty bash -l
fi

cd "$PROJECT"
case "$MODEL" in
  gdino) run_gdino ;;
  owlvit) run_owlvit ;;
  *)
    echo "Unknown MODEL=$MODEL (use gdino or owlvit)"
    exit 1
    ;;
esac

echo "Done: MODEL=$MODEL MODE=$MODE"
