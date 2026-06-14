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
echo "LR=${LR:-2e-4} BS=${BS:-8} EPOCHS=${EPOCHS:-12} OPTIMIZER=${OPTIMIZER:-AdamW} WD=${WD:-0.05} IMGSZ=${IMGSZ:-640} WARMUP=${WARMUP:-1000} MODEL=${MODEL:-yolov8l-worldv2.pt} FREEZE=${FREEZE:-0} MOSAIC=${MOSAIC:-1.0} AA=${AUTO_AUGMENT:-randaugment} ERASING=${ERASING:-0.4} SCALE=${SCALE:-0.5}"

MODEL_FLAG="--model ~/cv_project/models/${MODEL:-yolov8l-worldv2.pt}"
FREEZE_FLAG=""
if [ "${FREEZE:-0}" = "1" ]; then
    FREEZE_FLAG="--freeze"
    echo "Freeze backbone enabled"
fi

python ~/cv_project/scripts/yolow/train.py \
    --lr ${LR:-2e-4} --batch ${BS:-8} --epochs ${EPOCHS:-12} \
    --warmup ${WARMUP:-1000} --optimizer ${OPTIMIZER:-AdamW} \
    --weight_decay ${WD:-0.05} --imgsz ${IMGSZ:-640} \
    --name "${NAME:-baseline}" $MODEL_FLAG $FREEZE_FLAG \
    --mosaic ${MOSAIC:-1.0} --auto_augment ${AUTO_AUGMENT:-randaugment} \
    --erasing ${ERASING:-0.4} --scale ${SCALE:-0.5}

echo "Done: $SLURM_JOB_ID"
