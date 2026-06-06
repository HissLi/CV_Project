# 实验记录

## 实验总览

| ID | 日期 | 模型 | 阶段 | lr | bs | epochs | 状态 | mAP50 | mAP50-95 |
|----|------|------|------|-----|----|--------|------|-------|----------|
| yolow_baseline | 2026-06-06 | YOLO-World-L | Phase 1 Baseline | 2e-4 | 8 | 12 | 完成 | 0.661 | 0.497 |
| owlvit_zeroshot | 2026-06-06 | OWL-ViT-B/32 | Zero-shot Eval | — | 8 (eval) | — | 完成 | 0.064 | 0.040 |

---

## 实验详情

### yolow_baseline — YOLO-World-L Phase 1 Baseline

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
- 本地: `results/yolow_baseline_20260606_1907/`
- 服务器: `~/cv_project/runs/detect/results/yolow/yolow_baseline_lr2e-4_bs8/`

---

## 服务器路径

训练结果保存在服务器 `~/cv_project/results/` 下，通过以下命令同步到本地：

```bash
sshpass -p 'rb6/aYMRAT#16' scp -o StrictHostKeyChecking=no -rP 10022 \
  cse12210210@172.18.34.26:~/cv_project/results/<exp_name>/ \
  results/
```
