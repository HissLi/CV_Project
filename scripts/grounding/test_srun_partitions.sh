#!/bin/bash
for p in v100 rtx2080ti a100; do
  echo "=== PARTITION $p ==="
  srun -p "$p" --gres=gpu:1 -n 1 --mem=8G -t 00:01:00 bash -lc "hostname && nvidia-smi -L" 2>&1 | head -8
done
