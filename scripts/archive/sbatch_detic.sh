#!/bin/bash
#SBATCH --job-name=detic_train
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/turing_lab/cse12210210/cv_project/logs/%j_detic.out
#SBATCH --error=/home/turing_lab/cse12210210/cv_project/logs/%j_detic.err

mkdir -p /home/turing_lab/cse12210210/cv_project/logs
source /opt/ohpc/pub/apps/anaconda3/bin/activate detic
export PYTHONUNBUFFERED=1

echo "=== Job: $SLURM_JOB_ID ==="
echo "Node: $SLURM_NODELIST | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "LR=${LR:-2e-4} BS=${BS:-8} EPOCHS=${EPOCHS:-12} FREEZE=${FREEZE:-1} FREEZE_DECODER=${FREEZE_DECODER:-0}"

FREEZE_FLAG=""
if [ "${FREEZE:-1}" = "1" ]; then
    FREEZE_FLAG="--freeze_backbone"
    if [ "${FREEZE_DECODER:-0}" = "1" ]; then
        FREEZE_FLAG="$FREEZE_FLAG --freeze_decoder"
        echo "Backbone + decoder frozen (ultra-fast mode)"
    else
        echo "Backbone frozen (fast mode)"
    fi
fi

python ~/cv_project/scripts/train_detic.py \
    --lr ${LR:-2e-4} --batch ${BS:-8} --epochs ${EPOCHS:-12} \
    --warmup ${WARMUP:-1000} \
    --output_dir "results/${NAME:-detic_baseline}" \
    $FREEZE_FLAG

echo "Done: $SLURM_JOB_ID"
