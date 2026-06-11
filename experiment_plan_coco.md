# YOLO-World 超参数敏感性研究 — COCO 2017

## 课题信息

- **课题**：Topic 4 — Open-Vocabulary Object Detection and Visual Grounding
- **方向**：YOLO-World-L 单模型超参数系统研究 + OWL-ViT 零样本对照
- **数据集**：COCO 2017
- **算力**：NVIDIA L40 48GB × 1
- **已用时**：5.5h（baseline）
- **预算**：~160h GPU 时间（含 buffer）
- **截止日期**：2026.6.21

---

## 1. 研究动机

当前 OVOD 论文各自报告最优结果，但缺乏对**同一模型在不同超参数下行为差异的系统分析**。本实验以 YOLO-World-L 为对象，在 COCO 上做全面超参数扫描，回答：

1. **哪些超参数对最终性能影响最大？**（lr >> warmup > weight_decay > bs ...？）
2. **不同超参数如何影响训练动力学？**（收敛速度、过拟合程度、loss 曲线形态）
3. **CNN 单阶段检测器的最优超参数配置是否有普适规律？**

> 最后用 OWL-ViT 零样本评估作为微小补充对照，展示微调的价值量级（不占训练时间）。

---

## 2. 实验总览

### 超参数维度

| 维度 | 含义 | 为何重要 |
|------|------|---------|
| Learning Rate | 优化步长 | 最核心参数，决定收敛速度和最终性能 |
| Optimizer | 优化算法 | AdamW vs SGD，CNN 偏好不同 |
| Batch Size | 每步样本数 | 影响 BN 统计量和泛化 |
| Weight Decay | L2 正则化 | 控制过拟合 |
| Warmup Steps | 学习率预热 | 稳定训练初期 |
| Image Resolution | 输入分辨率 | 小物体 vs 速度的 tradeoff |
| Epochs | 训练时长 | 是否充分收敛 |
| Freeze Backbone | 迁移学习策略 | 微调 vs 全量训练 |

### 实验矩阵

| Phase | 维度 | 值 | epochs | 耗时/组 | 组数 | 小计 |
|-------|------|-----|--------|---------|------|------|
| 1 | **Baseline** | lr=2e-4, bs=8, AdamW, wd=0.05, warmup=1000, 640px | 12 | 5.5h | 1 | **5.5h** ✅ |
| 2 | **Learning Rate** | 1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3 | 6 | 2.7h | 6 | **16.2h** |
| 2b | LR best full | 选出最优 lr | 12 | 5.5h | 1 | **5.5h** |
| 3 | **Optimizer** | SGD (lr=1e-2, 5e-2, 1e-1) + AdamW(2e-4) | 6 | 2.7h | 3 | **8.1h** |
| 4 | **Batch Size** | 4, 16 | 6 | 2.7h | 2 | **5.4h** |
| 5 | **Weight Decay** | 1e-5, 1e-3, 1e-2 | 6 | 2.7h | 3 | **8.1h** |
| 6 | **Warmup** | 0, 500, 2000 | 6 | 2.7h | 3 | **8.1h** |
| 7 | **Image Size** | 320, 800 | 12 | 5.5h | 2 | **11.0h** |
| 8 | **Freeze Backbone** | frozen / trainable | 12 | 5.5h | 1 | **5.5h** |
| 9 | **Extended Epochs** | best config × 24 epochs | 24 | 11.0h | 1 | **11.0h** |
| 10 | **Best Final** | best config × 2 runs | 12 | 5.5h | 2 | **11.0h** |
| — | **OWL-ViT** | 零样本评估 | — | — | 1 | **0h** ✅ |
| — | **LR+BS grid** | 2 lr × 2 bs 交叉 | 6 | 2.7h | 4 | **10.8h** |
| — | **Ablation: no mosaic** | mosaic=0 | 12 | 5.5h | 1 | **5.5h** |
| — | **Buffer** | 预留 | — | — | — | **~20h** |
| **总计** | | | | | ~40 组 | **~132h** |

> 加上已完成的 baseline 5.5h，总计约 **138h**，在 160h 预算内有 ~22h buffer。

---

## 3. 每阶段详细设计

### Phase 1：Baseline ✅

| 参数 | 值 |
|------|-----|
| lr | 2e-4 |
| bs | 8 |
| epochs | 12 |
| optimizer | AdamW |
| weight_decay | 0.05 |
| warmup | 1000 |
| imgsz | 640 |

**结果**：mAP50=0.661, mAP=0.497，5.5h

### Phase 2：Learning Rate 扫描（最核心）

7 个 lr 值覆盖 3 个数量级，6 epoch 快速扫描。每个 epoch ~27min，单组 ~2.7h。

```
1e-5  ─── 极低 lr，预期欠拟合，收敛慢
5e-5  ─── 低 lr
1e-4  ─── 中等偏低
2e-4  ─── baseline（已完成）
3e-4  ─── 中等偏高
5e-4  ─── 高 lr
1e-3  ─── 极高 lr，预期不稳定/发散
```

**分析产出**：lr-mAP 曲线，最优 lr 区间，lr 鲁棒性范围。

**额外**：选最优 lr 跑完整 12 epoch（5.5h）。

### Phase 3：Optimizer 对比

SGD 是 CNN 检测器的传统选择，AdamW 是现代默认。对比二者。

| Optimizer | lr 范围 | 理由 |
|------|------|------|
| AdamW | 2e-4 | baseline |
| SGD | 1e-2, 5e-2, 1e-1 | SGD 需要更高 lr |

**分析产出**：CNN+BN 对 optimizer 的偏好，收敛速度差异。

### Phase 4：Batch Size

CNN 的 BatchNorm 对 batch size 敏感。测试 BS 下限（4）和上限（16，如果显存允许）。

| BS | 预期 |
|------|------|
| 4 | BN 统计量噪声大，可能影响收敛 |
| 16 | BN 更稳定，但泛化可能略差 |

### Phase 5：Weight Decay

AdamW 解耦了 weight decay，测试不同强度。

| wd | 预期 |
|------|------|
| 1e-5 | 几乎无正则化，可能过拟合 |
| 1e-3 | 强正则化 |
| 1e-2 | 极强，可能欠拟合 |

### Phase 6：Warmup

测试不同 warmup 长度对训练稳定性的影响。

| warmup | 预期 |
|------|------|
| 0 | lr 从 2e-4 起步，可能早期不稳定 |
| 500 | 短 warmup |
| 2000 | 长 warmup，初期收敛更慢 |

### Phase 7：Image Resolution

| 分辨率 | 预期 |
|------|------|
| 320 | 快 4x，小物体 AP 低 |
| 800 | 慢 ~2.5x，小物体 AP 可能提升 |

### Phase 8：Freeze Backbone

冻结 YOLOv8 骨干，仅训练 neck + head。测试迁移学习程度对性能的影响。

### Phase 9：Extended Training

最优配置跑 24 epoch，观察是否进一步收敛。

### Phase 10：LR × BS 网格

2 lr × 2 bs 交叉验证，分析交互效应。

---

## 4. 时间估算

| 类别 | 组数 | 耗时 |
|------|------|------|
| 6 epoch 速扫 | 25 组 | ~67.5h |
| 12 epoch 全训 | 9 组 | ~49.5h |
| 24 epoch | 1 组 | ~11.0h |
| OWL-ViT eval | 1 组 | 已完成 |
| Buffer | — | ~22h |
| **总计** | **~40 组** | **~150h** |

YOLO 12 epoch 约 5.5h，6 epoch 约 2.7h，单 epoch 约 27min（BS=8，L40 实测）。

---

## 5. 分析框架

### 5.1 超参数重要性排序

用 ANOVA 或简单的 max-min 差距，对每个维度的 best-worst mAP 差异排序：

```
LR     ████████████████████  最大影响
WD     ██████████████
Opt    ██████████
BS     ████████
Warmup ██████
```

### 5.2 交互效应

lr × bs 网格分析：大 lr 是否必须搭配大 bs？小 bs 是否对 lr 更敏感？

### 5.3 训练动力学

同一参数不同取值下的 loss 曲线对比：
- 收敛速度（多少 epoch 达到 plateau）
- 过拟合程度（train/val gap）
- 稳定性（loss 波动幅度）

### 5.4 补充：零样本对照

用 OWL-ViT-B/32 在 COCO val2017 上做一次零样本评估（mAP50=0.064），作为**附录中的一个小参考点**，展示不训练的检测模型与微调后 YOLO 的差距（0.064 vs 0.661），量化微调的价值。不参与任何对比分析。

---

## 6. 提交命令

```bash
# 6 epoch 快速扫描
LR=5e-4 EPOCHS=6 NAME=yolow_lr5e4_ep6 sbatch scripts/sbatch_yolow.sh

# 12 epoch 完整训练
LR=3e-4 EPOCHS=12 NAME=yolow_lr3e4_ep12 sbatch scripts/sbatch_yolow.sh

# 改变 bs
BS=16 LR=2e-4 EPOCHS=6 NAME=yolow_bs16 sbatch scripts/sbatch_yolow.sh

# 改变 optimizer（需修改脚本或添加参数）
OPTIMIZER=SGD LR=1e-2 EPOCHS=6 NAME=yolow_sgd_lr1e2 sbatch scripts/sbatch_yolow.sh
```

---

## 7. 备用/扩展

- 如果 160h 预算有余：增加 lr-bs-wd 三因素交叉实验
- 如果时间紧张：削减 Phase 4-8 中各减 1 个值
- NMS/conf threshold 调优：不需要训练，可在评估阶段进行

---

## 8. 扩展方案：Phase 11-14（~72h）

已完成 Phase 1-10（85h），剩余 ~75h。关键发现：最优 lr=5e-5，最佳分辨率=800 (0.686)，SGD 失败，WD/Warmup/BS 不敏感。

### Phase 11：最佳配置交叉（~28h）

| 实验 | 配置 | epochs | 预计 | 目的 |
|------|------|------|------|------|
| P11-1 | lr=5e-5, imgsz=800, bs=8 | 24 | ~11h | 最优组合上限 |
| P11-2 | lr=5e-5, imgsz=640, bs=8 | 48 | ~22h | epoch 数 vs 分辨率 |
| P11-3 | lr=5e-5, imgsz=800, bs=16 | 12 | ~3h | 大 bs 高分辨率测试 |

### Phase 12：模型规模（~20h）

| 实验 | 模型 | 配置 | epochs | 预计 |
|------|------|------|------|------|
| P12-1 | YOLO-World-S | lr=5e-5, bs=16 | 12 | ~3h |
| P12-2 | YOLO-World-M | lr=5e-5, bs=12 | 12 | ~4h |
| P12-3 | YOLO-World-S | best config | 24 | ~6h |
| P12-4 | YOLO-World-M | best config | 24 | ~8h |

### Phase 13：数据增强消融（~10h）

| 实验 | 配置 | epochs | 预计 |
|------|------|------|------|
| P13-1 | mosaic=0 | 6 | ~2.5h |
| P13-2 | auto_augment=False | 6 | ~2.5h |
| P13-3 | erasing=0 | 6 | ~2.5h |
| P13-4 | 更强多尺度 scale=0.9 | 6 | ~2.5h |

### Phase 14：Freeze Backbone（~10h）

| 实验 | 配置 | epochs | 预计 |
|------|------|------|------|
| P14-1 | freeze backbone, best lr | 12 | ~5.5h |
| P14-2 | freeze backbone, imgsz=800 | 12 | ~5.5h |

### 时间总览

| Phase | 实验数 | 时间 |
|------|------|------|
| 11 | 3 | ~28h |
| 12 | 4 | ~21h |
| 13 | 4 | ~10h |
| 14 | 2 | ~11h |
| **总计** | **13** | **~70h** |

累计 ~155h / 160h。
