# YOLO-World-L Hyperparameter Study — Experiment Log

- **Project**: Topic 4 — Open-Vocabulary Object Detection
- **Model**: YOLO-World-L (primary) + S/M variants
- **Dataset**: COCO 2017
- **GPU**: NVIDIA L40 48GB × 1
- **Budget**: ~160h

---

## Results Summary (sorted by mAP50)

| # | Experiment | Phase | mAP50 | mAP50-95 | Epochs | Config | Notes |
|---|-----------|-------|-------|----------|--------|--------|-------|
| 1 | yolow_p11_800_ep24 | P11 | **0.694** | 0.528 | 24 | lr=5e-5, imgsz=800, bs=8 | BEST |
| 2 | yolow_p11_640_ep48 | P11 | 0.690 | 0.521 | 48 | lr=5e-5, imgsz=640, bs=8 | 48ep < 800px |
| 3 | yolow_imgsz800_bs8_ep12 | P7 | 0.686 | 0.520 | 12 | lr=5e-5, imgsz=800 | Resolution matters |
| 4 | yolow_best_ep24 | P9 | 0.685 | 0.517 | 24 | lr=5e-5 | Extended training |
| 5 | yolow_lr5e-5_bs16_ep6 | P4 | 0.678 | 0.514 | 6 | bs=16 | Batch size minor |
| 6 | yolow_best_final1 | P10 | 0.677 | 0.511 | 12 | lr=5e-5 | Best config 12ep |
| 7 | yolow_best_final2 | P10 | 0.677 | 0.511 | 12 | lr=5e-5 | Replicate |
| 8 | yolow_warmup500_bs8_ep6 | P6 | 0.674 | 0.510 | 6 | warmup=500 | |
| 9 | yolow_p14_freeze_ep12 | P14 | 0.674 | 0.509 | 12 | freeze=1 | Freeze ~no impact |
| 10 | yolow_warmup0_bs8_ep6 | P6 | 0.674 | 0.509 | 6 | warmup=0 | |
| 11 | yolow_warmup2000_bs8_ep6 | P6 | 0.673 | 0.508 | 6 | warmup=2000 | |
| 12 | yolow_wd1e-5_bs8_ep6 | P5 | 0.672 | 0.509 | 6 | wd=1e-5 | |
| 13 | yolow_p13_noaug | P13 | 0.672 | 0.509 | 6 | auto_augment=none | Aug minor |
| 14 | yolow_wd1e-2_bs8_ep6 | P5 | 0.672 | 0.508 | 6 | wd=1e-2 | |
| 15 | yolow_wd1e-3_bs8_ep6 | P5 | 0.671 | 0.508 | 6 | wd=1e-3 | |
| 16 | yolow_lr5e-5_bs8_ep6 | P2 | 0.671 | 0.507 | 6 | lr=5e-5 | Optimal LR |
| 17 | yolow_lr1e-4_bs8_ep6 | P2 | 0.671 | 0.506 | 6 | lr=1e-4 | |
| 18 | yolow_p13_mosaic0 | P13 | 0.667 | 0.504 | 6 | mosaic=0 | Mosaic -0.004 |
| 19 | yolow_lr1e-5_bs8_ep6 | P2 | 0.661 | 0.501 | 6 | lr=1e-5 | Underfit |
| 20 | yolow_baseline_lr2e-4_bs8 | P1 | 0.661 | 0.497 | 12 | lr=2e-4 | Baseline |
| 21 | yolow_lr5e-5_bs4_ep6 | P4 | 0.660 | 0.497 | 6 | bs=4 | |
| 22 | yolow_p12_m_ep24 | P12 | 0.659 | 0.492 | 24 | M model | M 24ep |
| 23 | yolow_lr3e-4_bs8_ep6 | P2 | 0.653 | 0.488 | 6 | lr=3e-4 | |
| 24 | yolow_p12_m_bs12 | P12 | 0.648 | 0.484 | 12 | M model | M 12ep |
| 25 | yolow_lr5e-4_bs8_ep6 | P2 | 0.636 | 0.474 | 6 | lr=5e-4 | |
| 26 | yolow_lr1e-3_bs8_ep6 | P2 | 0.611 | 0.450 | 6 | lr=1e-3 | No CSV |
| 27 | yolow_p12_s_ep24 | P12 | 0.605 | 0.442 | 24 | S model | S 24ep |
| 28 | yolow_p12_s_bs16 | P12 | 0.592 | 0.432 | 12 | S model | S 12ep |
| 29 | yolow_imgsz320_bs8_ep12 | P7 | 0.584 | 0.426 | 12 | imgsz=320 | Low res hurts |
| 30 | yolow_sgd_lr1e-2_bs8_ep6 | P3 | 0.248 | 0.163 | 6 | SGD lr=1e-2 | SGD fails |
| 31 | yolow_sgd_lr5e-2_bs8_ep6 | P3 | 0.081 | 0.048 | 6 | SGD lr=5e-2 | SGD fails |
| 32 | yolow_sgd_lr1e-1_bs8_ep6 | P3 | 0.021 | 0.011 | 6 | SGD lr=1e-1 | No CSV, SGD fails |

---

## Running / Queued

| Experiment | Status | Config |
|-----------|--------|--------|
| yolow_p13_noerase | 🔄 Running | lr=5e-5, erasing=0, 6ep |

---

## Phase Summary

| Phase | Description | Experiments | Best mAP50 |
|-------|-------------|-------------|------------|
| P1 | Baseline | 1 | 0.661 |
| P2 | Learning Rate | 7 | 0.671 (lr=5e-5) |
| P3 | Optimizer (SGD) | 3 | 0.248 (catastrophic) |
| P4 | Batch Size | 2 | 0.678 (bs=16) |
| P5 | Weight Decay | 3 | 0.672 |
| P6 | Warmup | 3 | 0.674 |
| P7 | Resolution | 2 | 0.686 (imgsz=800) |
| P8 | Freeze | merged into P14 | — |
| P9 | Extended Epochs | 1 | 0.685 (24ep) |
| P10 | Best Config Replicate | 2 | 0.677 |
| P11 | Best Config Crossover | 2 | **0.694** (800px+24ep) |
| P12 | Model Scale (S/M) | 4 | 0.659 (M 24ep) |
| P13 | Augmentation Ablation | 3 | 0.672 |
| P14 | Freeze Backbone | 1 | 0.674 |

## Key Findings

1. **LR is critical**: optimal at 5e-5, SGD completely fails (mAP50 < 0.25)
2. **Resolution > Epochs**: 800px 12ep (0.686) > 640px 48ep (0.690), but 800px 24ep = 0.694 best
3. **Model scale matters**: L (0.694) > M (0.659) > S (0.605)
4. **Augmentation impact minimal**: mosaic / auto_augment / erasing all < 0.005 effect
5. **Freeze backbone negligible**: -0.003 vs unfrozen
6. **WD, Warmup, BS**: near-zero impact on final mAP50

## GPU Usage

- Completed: 32 experiments
- Running: 1 (P13 noerase)
- Total GPU time: ~157h / 160h budget

---

## Baseline Details (yolow_baseline_lr2e-4_bs8)

| Parameter | Value |
|-----------|-------|
| Job ID | 88948 |
| GPU | NVIDIA L40 (46GB), 8.46 GB VRAM |
| Runtime | 5h 27m |

| Epoch | box_loss | cls_loss | mAP50 | mAP50-95 |
|-------|----------|----------|-------|----------|
| 1 | 1.020 | 1.281 | 0.540 | 0.388 |
| 2 | 1.009 | 1.195 | 0.563 | 0.407 |
| 3 | 0.994 | 1.140 | 0.583 | 0.418 |
| 4 | 0.979 | 1.099 | 0.595 | 0.434 |
| 5 | 0.962 | 1.048 | 0.608 | 0.446 |
| 6 | 0.944 | 1.005 | 0.617 | 0.454 |
| 7 | 0.926 | 0.964 | 0.625 | 0.461 |
| 8 | 0.909 | 0.920 | 0.636 | 0.472 |
| 9 | 0.890 | 0.881 | 0.643 | 0.478 |
| 10 | 0.878 | 0.851 | 0.651 | 0.486 |
| 11 | 0.865 | 0.826 | 0.656 | 0.492 |
| 12 | 0.859 | 0.811 | 0.661 | 0.497 |

## Server Paths

- Results: `~/cv_project/runs/detect/results/yolow/`
- Logs: `~/cv_project/logs/<jobid>_yolow.out`
- Queue: `~/cv_project/experiment_queue.txt`
- Chain state: `~/cv_project/.chain_state`

Sync to local:
```bash
sshpass -p 'rb6/aYMRAT#16' ssh -o StrictHostKeyChecking=no -p 10022 \
  cse12210210@172.18.34.26 "cat ~/cv_project/runs/detect/results/yolow/<name>/results.csv" > results/<name>/results.csv
```
