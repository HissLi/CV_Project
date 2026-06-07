#!/bin/bash
#SBATCH --job-name=yolow_train
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/turing_lab/cse12210210/cv_project/logs/%j_yolow.out
#SBATCH --error=/home/turing_lab/cse12210210/cv_project/logs/%j_yolow.err

mkdir -p /home/turing_lab/cse12210210/cv_project/logs
source /opt/ohpc/pub/apps/anaconda3/bin/activate yolow
export PYTHONUNBUFFERED=1

echo "=== Job: $SLURM_JOB_ID ==="
echo "Node: $SLURM_NODELIST | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "LR=${LR:-2e-4} BS=${BS:-8} EPOCHS=${EPOCHS:-12}"

python ~/cv_project/scripts/train_yolow.py \
    --lr ${LR:-2e-4} --batch ${BS:-8} --epochs ${EPOCHS:-12} \
    --warmup ${WARMUP:-1000} \
    --name "${NAME:-baseline}"

echo "Done: $SLURM_JOB_ID"
