# 实验记录

## 实验总览

| ID | 日期 | 模型 | 阶段 | lr | bs | epochs | 状态 | mAP50 | mAP50-95 |
|----|------|------|------|-----|----|--------|------|-------|----------|
| yolow_lr2e-4_bs8_ep12 | 2026-06-06 | YOLO-World-L | Phase 1 Baseline | 2e-4 | 8 | 12 | 完成 | 0.661 | 0.497 |
| owlvit_zeroshot | 2026-06-06 | OWL-ViT-B/32 | Zero-shot Eval | — | 8 (eval) | — | 完成 | 0.064 | 0.040 |
| yolow_lr1e-5_bs8_ep6 | 2026-06-07 | YOLO-World-L | Phase 2 LR=1e-5 | 1e-5 | 8 | 6 | 完成 | 0.661 | 0.501 |
| yolow_lr5e-5_bs8_ep6 | 2026-06-07 | YOLO-World-L | Phase 2 LR=5e-5 | 5e-5 | 8 | 6 | 完成 | 0.671 | 0.507 |
| yolow_lr1e-4_bs8_ep6 | 2026-06-07 | YOLO-World-L | Phase 2 LR=1e-4 | 1e-4 | 8 | 6 | 完成 | 0.671 | 0.506 |
| yolow_lr3e-4_bs8_ep6 | 2026-06-07 | YOLO-World-L | Phase 2 LR=3e-4 | 3e-4 | 8 | 6 | 完成 | 0.653 | 0.488 |
| yolow_lr5e-4_bs8_ep6 | 2026-06-08 | YOLO-World-L | Phase 2 LR=5e-4 | 5e-4 | 8 | 6 | 完成 | 0.636 | 0.474 |
| yolow_lr1e-3_bs8_ep6 | 2026-06-08 | YOLO-World-L | Phase 2 LR=1e-3 | 1e-3 | 8 | 6 | 完成 | 0.611 | 0.450 |
| yolow_lr5e-5_bs8_ep12 | 2026-06-08 | YOLO-World-L | Phase 2b Best LR | 5e-5 | 8 | 12 | 完成 | 0.677 | 0.511 |
| yolow_sgd_lr1e-2_bs8_ep6 | 2026-06-08 | YOLO-World-L | Phase 3 SGD lr=1e-2 | 1e-2 | 8 | 6 | 完成 | 0.248 | 0.163 |
| yolow_sgd_lr5e-2_bs8_ep6 | 2026-06-08 | YOLO-World-L | Phase 3 SGD lr=5e-2 | 5e-2 | 8 | 6 | 完成 | 0.081 | 0.048 |
| yolow_sgd_lr1e-1_bs8_ep6 | 2026-06-08 | YOLO-World-L | Phase 3 SGD lr=1e-1 | 1e-1 | 8 | 6 | 完成 | 0.021 | 0.011 |
| yolow_lr5e-5_bs4_ep6 | 2026-06-09 | YOLO-World-L | Phase 4 BS=4 | 5e-5 | 4 | 6 | 完成 | 0.660 | 0.497 |
| yolow_lr5e-5_bs16_ep6 | 2026-06-09 | YOLO-World-L | Phase 4 BS=16 | 5e-5 | 16 | 6 | 完成 | 0.678 | 0.514 |
| yolow_wd1e-5_bs8_ep6 | 2026-06-09 | YOLO-World-L | Phase 5 WD=1e-5 | 5e-5 | 8 | 6 | 完成 | 0.672 | 0.509 |
| yolow_wd1e-3_bs8_ep6 | 2026-06-09 | YOLO-World-L | Phase 5 WD=1e-3 | 5e-5 | 8 | 6 | 完成 | 0.671 | 0.508 |
| yolow_wd1e-2_bs8_ep6 | 2026-06-09 | YOLO-World-L | Phase 5 WD=1e-2 | 5e-5 | 8 | 6 | 完成 | 0.672 | 0.508 |
| yolow_warmup0_bs8_ep6 | 2026-06-09 | YOLO-World-L | Phase 6 Warmup=0 | 5e-5 | 8 | 6 | 完成 | 0.674 | 0.509 |
| yolow_warmup500_bs8_ep6 | 2026-06-09 | YOLO-World-L | Phase 6 Warmup=500 | 5e-5 | 8 | 6 | 完成 | 0.674 | 0.510 |
| yolow_warmup2000_bs8_ep6 | 2026-06-09 | YOLO-World-L | Phase 6 Warmup=2000 | 5e-5 | 8 | 6 | 完成 | 0.673 | 0.508 |
| yolow_imgsz320_bs8_ep12 | 2026-06-10 | YOLO-World-L | Phase 7 imgsz=320 | 5e-5 | 8 | 12 | 完成 | 0.584 | 0.426 |
| yolow_imgsz800_bs8_ep12 | 2026-06-10 | YOLO-World-L | Phase 7 imgsz=800 | 5e-5 | 8 | 12 | 完成 | 0.686 | 0.520 |
| yolow_best_ep24 | 2026-06-10 | YOLO-World-L | Phase 9 24 epoch | 5e-5 | 8 | 24 | 完成 | 0.685 | 0.517 |
| yolow_best_final1 | 2026-06-11 | YOLO-World-L | Phase 10 Final 1 | 5e-5 | 8 | 12 | 完成 | 0.67688 | 0.51137 |
| yolow_best_final2 | 2026-06-11 | YOLO-World-L | Phase 10 Final 2 | 5e-5 | 8 | 12 | 完成 | 0.677 | 0.511 |
| yolow_p11_800_ep24 | 2026-06-12 | YOLO-World-L | Phase 11 800px 24ep | 5e-5 | 8 | 24 | 完成 | 0.69432 | 0.52762 |

---

## 实验详情

### yolow_lr2e-4_bs8_ep12 — YOLO-World-L Phase 1 Baseline

| 参数 | 值 |
|------|-----|
| 作业 ID | 88948 |
| 开始时间 | 2026-06-06 12:47 |
| 结束时间 | 2026-06-06 19:07 |
| 运行时长 | 5h 27m |
| GPU | NVIDIA L40 (46GB) |
| GPU 内存 | 8.46 GB |

**超参数：**

| 参数 | 值 |
|------|-----|
| Model | yolov8l-worldv2.pt |
| Epochs | 12 |
| Batch Size | 8 |
| Learning Rate | 2e-4 |
| Warmup | 1000 steps |
| Optimizer | AdamW |
| Weight Decay | 0.05 |
| Image Size | 640 |
| Scheduler | cosine |

**最终结果 (COCO val2017)：**

| 指标 | 值 |
|------|-----|
| mAP50 | 0.661 |
| mAP50-95 | 0.497 |
| Precision | 0.704 |
| Recall | 0.607 |
| box_loss (train) | 0.859 |
| cls_loss (train) | 0.811 |

**每 epoch 结果：**

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

**文件位置：**
- 本地: `results/yolow_lr2e-4_bs8_ep12_20260606_1907/`
- 服务器: `~/cv_project/runs/detect/results/yolow/yolow_lr2e-4_bs8_ep12_lr2e-4_bs8/`

---

## 服务器路径

训练结果保存在服务器 `~/cv_project/results/` 下，通过以下命令同步到本地：

```bash
sshpass -p 'rb6/aYMRAT#16' scp -o StrictHostKeyChecking=no -rP 10022 \
  cse12210210@172.18.34.26:~/cv_project/results/<exp_name>/ \
  results/
```
