#!/bin/bash
set -euo pipefail
#SBATCH --job-name=owlvit_refcoco_eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/turing_lab/cse12210210/cv_project/logs/%j_owlvit_refeval.out
#SBATCH --error=/home/turing_lab/cse12210210/cv_project/logs/%j_owlvit_refeval.err

mkdir -p /home/turing_lab/cse12210210/cv_project/logs
source /opt/ohpc/pub/apps/anaconda3/bin/activate owlvit
export PYTHONUNBUFFERED=1

echo "=== Job: $SLURM_JOB_ID ==="
echo "Node: $SLURM_NODELIST | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "DATASETS=${DATASETS:-refcoco,refcoco+,refcocog} SPLIT=${SPLIT:-val} MAX_SAMPLES=${MAX_SAMPLES:-all} THRESHOLD=${THRESHOLD:-0.1} BS=${BATCH_SIZE:-8}"

MAX_SAMPLES_ARG=""
if [ -n "${MAX_SAMPLES}" ]; then
  MAX_SAMPLES_ARG="--max_samples ${MAX_SAMPLES}"
fi

python ~/cv_project/scripts/owlvit/eval_refcoco.py \
  --datasets "${DATASETS:-refcoco,refcoco+,refcocog}" \
  --split "${SPLIT:-val}" \
  --model_dir "${MODEL_DIR:-~/cv_project/models/owlvit}" \
  --refer_data_root "${REFER_DATA_ROOT:-~/cv_project/datasets/refer/data}" \
  --refer_repo_root "${REFER_REPO_ROOT:-~/cv_project/datasets/refer/refer}" \
  --coco2014_root "${COCO2014_ROOT:-~/cv_project/datasets/mscoco2014}" \
  --coco2017_root "${COCO2017_ROOT:-~/cv_project/datasets/coco}" \
  --output_dir "${OUTPUT_DIR:-~/cv_project/results/owlvit_refcoco_zeroshot}" \
  --threshold "${THRESHOLD:-0.1}" \
  --batch_size "${BATCH_SIZE:-8}" \
  ${MAX_SAMPLES_ARG}

echo "Done: $SLURM_JOB_ID"
