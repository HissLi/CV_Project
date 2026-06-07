#!/bin/bash
#SBATCH --job-name=gdino_train
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/home/turing_lab/cse12210210/cv_project/logs/%j_gdino.out
#SBATCH --error=/home/turing_lab/cse12210210/cv_project/logs/%j_gdino.err

mkdir -p /home/turing_lab/cse12210210/cv_project/logs
source /opt/ohpc/pub/apps/anaconda3/bin/activate gdino
export PYTHONUNBUFFERED=1

echo "=== Job: $SLURM_JOB_ID ==="
echo "Node: $SLURM_NODELIST | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "LR=${LR:-1e-4} BS=${BS:-4} EPOCHS=${EPOCHS:-12} RESUME=${RESUME:-0}"

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_FLAG="--resume"
    echo "Resuming from checkpoint..."
fi

python ~/cv_project/scripts/train_gdino.py \
    --lr ${LR:-1e-4} --batch ${BS:-4} --epochs ${EPOCHS:-12} \
    --warmup ${WARMUP:-1000} \
    --output_dir "results/${NAME:-gdino_baseline}" \
    $RESUME_FLAG

echo "Done: $SLURM_JOB_ID"

if [ "${RESUME:-0}" != "1" ]; then
    echo "=============="
    echo "To resume: RESUME=1 NAME=${NAME:-gdino_baseline} sbatch scripts/sbatch_gdino.sh"
    echo "=============="
fi
