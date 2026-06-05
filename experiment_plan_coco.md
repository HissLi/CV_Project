# OVOD 双模型对比实验方案 — COCO 2017

## 课题信息

- **课题**：Topic 4 — Open-Vocabulary Object Detection and Visual Grounding
- **方向**：双架构对比研究（CNN late-fusion vs Transformer deep-fusion）
- **数据集**：COCO 2017
- **算力**：L40 48GB × 1（集群二 a100 分区，l40gpu001/l40gpu002）
- **账号**：cse12210210 @ 172.18.34.26:10022
- **时间限制**：10 天
- **截止日期**：2026.6.21

---

## 1. 研究动机

开放词汇目标检测（OVOD）目前在架构选型上存在两条主流路线：

- **CNN 单阶段 + Late Fusion**：文本仅在分类头与视觉特征做轻量交互（如 YOLO-World）
- **Transformer Query-based + Deep Fusion**：文本在 decoder 每一层通过 cross-attention 持续参与视觉特征构建（如 Grounding DINO）

两种范式的核心差异在于**文本-视觉融合深度**——这决定了梯度流动路径、对超参数的敏感性、以及最终检测性能的天花板。目前没有人在控制变量的条件下对两者进行过系统对比。

### 核心研究问题

1. **为什么两种不同架构在同一个数据集上需要不同的最优超参数？**
2. **架构差异导致的超参数敏感性不同，根源是什么？**
3. **两种架构的最终性能差距由哪些因素构成（定位 vs 分类 vs 漏检）？**

---

## 2. 模型选择

| 属性 | YOLO-World-L | Grounding DINO-B |
|------|-------------|-------------------|
| **范式** | CNN 单阶段 | Transformer Query-based |
| **视觉骨干** | YOLOv8-L | Swin-B |
| **语言编码器** | CLIP-L (frozen) | BERT-B (fine-tuned) |
| **融合方式** | Late fusion — 分类头 text embedding 点积 | Deep fusion — 6 层 decoder cross-attention |
| **参数量** | ~100M | ~200M |
| **推荐 bs (L40)** | 8–16 | 4–8 |
| **每 epoch 训练时间** | ~1.5h | ~3.5h |
| **官方预训练权重** | 有 | 有 |

### 选择理由

- **架构对比度最大**：两者在文本-视觉融合谱系上分别占据浅端和深端，差异最显著
- **训练成本可接受**：YOLO-World 训练很快（~1.5h/epoch），Grounding DINO 在单 L40 上也可完成
- **都有成熟的预训练权重和代码库**：减少复现中的不确定性
- **两者在 COCO 上均有优秀表现**，可进行有意义的性能对比

### 核心差异：文本-视觉融合

```
文本参与深度：

YOLO-World-L                      Grounding DINO-B
     |                                    |
     浅 ←←←←←←←←←←←←←←←←←←←←←←←←→ 深
     |                                    |
 仅分类头用 CLIP                  6 层 decoder 每层
 embedding 做点积                都有 cross-attention
 文本梯度不回传                   文本梯度全量回传到视觉 backbone
```

文本参与越深，视觉 backbone 的优化景观受语言信号的扰动越大 → 超参数敏感性更高，尤其体现在 learning rate 和 warmup 设置上。

---

## 3. 数据集：COCO 2017

| 属性 | 值 |
|------|-----|
| 类别数 | 80 |
| 训练图像 | ~118K |
| 验证图像 | ~5K |
| 标注格式 | COCO JSON |
| 路径（服务器） | `~/cv_project/datasets/coco/` |

选择 COCO 而非 LVIS 的理由：
- COCO 的 80 类更均衡，训练更稳定，适合在有限时间内完成
- COCO 是 OVOD 的基准评测集，结果可直接与论文对比
- 训练速度快于 LVIS（类别少，每张图标注框少）

---

## 4. 实验设计

### 总览

```
Day 1      Day 2–4           Day 5–7            Day 8–10
  |           |                  |                   |
环境配置   Baseline × 2      lr 敏感性扫描      最优配置全训练
           (12 epochs)      (4 lr × 2 模型)     + 对比分析
                              (6 epochs)
```

### 4.1 Phase 0：环境配置（Day 1）

**目标**：在新账号 cse12210210 下搭建训练环境

| 任务 | 内容 | 预计耗时 |
|------|------|----------|
| Conda 环境 | 创建 yolow + gdino 两个 conda env | 2h |
| YOLO-World 环境 | `pip install ultralytics` + pycocotools, wandb | 0.5h |
| GDINO 环境 | transformers, accelerate, timm, opencv, pycocotools | 0.5h |
| COCO 数据集 | 如果服务器已有，直接软链接；否则需下载（~19GB） | 0–4h |
| 模型权重 | 下载 YOLO-World-L (90MB) + GDINO-B (895MB) 预训练权重 | 0.5h |
| 验证 | 各跑 1 个 mini-batch 确认 pipeline 通过 | 0.5h |

**预计**：4–8 小时（含可能的 COCO 下载）

### 4.2 Phase 1：Baseline 复现（Day 2–4）

两个模型各跑 1 次完整训练，使用官方推荐超参数。

| 参数 | YOLO-World-L | Grounding DINO-B |
|------|-------------|-------------------|
| Learning rate | 2e-4 | 1e-4 |
| Batch size | 8 | 4 |
| Epochs | 12 | 12 |
| Warmup steps | 1000 | 1000 |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.05 | 1e-4 |
| Image size | 640 | 800 |
| Scheduler | cosine | cosine |
| **预计训练时长** | **~18h** | **~42h** |

**目的**：
- 建立性能基线（AP, AP50, AP75, AP_s, AP_m, AP_l）
- 验证训练流程的正确性
- 确认 L40 48GB 显存下的最大可行 bs

### 4.3 Phase 2：Learning Rate 敏感性扫描（Day 5–7）

**核心实验**。对每个模型扫描 4 个 lr 值，使用 6 epochs（足以看出收敛趋势和相对排序），固定其他参数为 baseline 值。

| 模型 | lr 扫描范围 | 单次训练时间 | 4 组总时间 |
|------|-----------|-------------|-----------|
| YOLO-World-L | [5e-5, 1e-4, 2e-4 (baseline), 5e-4] | ~9h | ~36h |
| Grounding DINO-B | [1e-5, 5e-5, 1e-4 (baseline), 5e-4] | ~21h | ~84h |

> **注意**：Phase 2 优先跑 YOLO-World（快），再跑 GDINO。YOLO 的 4 组 lr 中有一组复用 Phase 1 baseline 结果（lr=2e-4），实际只需跑 3 组新的。

**预期发现**：
- YOLO-World 的最优 lr 区间预期较宽（CNN 的批归一化提供了隐式正则化），可能在 1e-4 ~ 5e-4 范围内表现平稳
- Grounding DINO 的最优 lr 区间预期较窄（深层 cross-attention 放大梯度方差），偏离最优 lr 后性能下降更快
- 两者的最优 lr 值可能有数量级差异

### 4.4 Phase 3：最优配置全训练 + 最终评估（Day 8–10）

Phase 2 中每模型选出最优 lr，然后跑完整的 12 epochs：

| 模型 | 训练时间 | 任务 |
|------|---------|------|
| YOLO-World-L (best lr) | ~18h | 完整 12 epoch 训练 + COCO val 评估 |
| Grounding DINO-B (best lr) | ~42h | 完整 12 epoch 训练 + COCO val 评估 |

评估指标：
- mAP, AP50, AP75
- AP_s, AP_m, AP_l（按目标大小分层）
- 每类 AP（分析类别间差异）

---

## 5. 时间线

```
         D1    D2    D3    D4    D5    D6    D7    D8    D9    D10
         ──    ──    ──    ──    ──    ──    ──    ──    ──    ──
Phase 0  ████
Phase 1  ██    ██████████████
        (YW)  (YW 18h)      (GD 42h)
Phase 2               ████████████████████████████████████████
                     (YW lr sweep 36h)    (GD lr sweep 84h)
Phase 3                                                     ████████████
                                                           (best full train)
Analysis         ←←←←←←←←←←←← 贯穿全程，训练间歇进行 →→→→→→→→→→→→
```

**时间总结**：

| 阶段 | 内容 | 时长 | 累计 |
|------|------|------|------|
| Phase 0 | 环境配置 | 1 天 | 1 天 |
| Phase 1 | YOLO baseline | 0.75 天 | 1.75 天 |
| Phase 1 | GDINO baseline | 1.75 天 | 3.5 天 |
| Phase 2 | YOLO lr sweep (3 新 lr) | 1.5 天 | 5 天 |
| Phase 2 | GDINO lr sweep (4 lr) | 3.5 天 | 8.5 天 |
| Phase 3 | Best 配置各训 12 epoch | 视时间而定 | — |

**风险缓解**：
- 如果时间紧张，Phase 3 可跳过——Phase 2 的 6-epoch 结果已足以分析 lr 敏感性
- 如果 GDINO 训练太慢，可减少为 3 个 lr 值（省 ~21h）
- 环境配置可复用服务器上已有的数据集和模型权重（如另一个账号 ~cse12212230 下的文件）

---

## 6. 分析框架

### 6.1 lr 敏感性曲线对比

对每个模型绘制 lr-AP 曲线，分析：
- **最优 lr 的绝对值差异** → 反映优化景观的"陡峭程度"
- **曲线的宽度（robustness）** → CNN 的 BN 提供隐式正则化，预期 YOLO-World 曲线更平坦
- **高 lr 区谁崩得更快** → 深层融合模型（GDINO）预期在高 lr 下更快发散

### 6.2 训练动力学分析

同一模型在不同 lr 下的训练曲线对比：
- Loss 下降速度 vs 最终收敛水平
- 训练集 vs 验证集的过拟合 gap
- GDINO 的 text encoder 梯度范数 vs lr 的关系

### 6.3 Error Breakdown

对最优模型进行错误分类（TIDE 框架）：

| 错误类型 | 含义 | 预期差异 |
|---------|------|---------|
| Cls | 框对、类错 | YOLO 更多（late fusion 语义辨别弱） |
| Loc | 类对、框偏 | GDINO 更少（Transformer 定位天然更准） |
| Miss | 漏检 | YOLO 可能更多小物体漏检 |
| Bkg | 背景误检 | GDINO 更多（query 机制可能过度关注背景） |

### 6.4 按目标大小分层分析

COCO 的 AP_s / AP_m / AP_l 天然允许按目标大小分析：
- **小物体 (AP_s)**：GDINO 的多尺度 deformable attention 在定位小物体上预期有优势
- **大物体 (AP_l)**：两者预期接近，大物体的检测更多由语义决定而非定位精度

### 6.5 文本编码器角色分析

| 维度 | CLIP (YOLO-World) | BERT (Grounding DINO) |
|------|-------------------|----------------------|
| 预训练方式 | 图文配对（语义天然对齐视觉） | 纯文本（语义丰富但与视觉不对齐） |
| Fine-tune 策略 | 冻结 | 微调 |
| 梯度影响 | 文本不产生视觉梯度 | 文本梯度全量回传 |
| 对 lr 的敏感性 | 间接（通过分类头 loss 权重） | 直接（BERT 参数直接被优化） |

这个差异是理解"为什么 GDINO 对 lr 更敏感"的关键——GDINO 的 BERT 在训练中需要学习跨模态对齐，lr 太大会破坏 BERT 的语义结构，lr 太小则对齐速度慢。

---

## 7. 训练命令参考

### YOLO-World Baseline

```bash
# sbatch 提交
python train_yolow.py \
    --data coco.yaml \
    --model yolov8l-worldv2.pt \
    --epochs 12 \
    --batch 8 \
    --lr 2e-4 \
    --warmup 1000 \
    --optimizer AdamW \
    --weight_decay 0.05 \
    --imgsz 640 \
    --device 0
```

### Grounding DINO Baseline

```bash
python train_gdino.py \
    --dataset coco \
    --model groundingdino-base \
    --epochs 12 \
    --batch 4 \
    --lr 1e-4 \
    --warmup 1000 \
    --optimizer AdamW \
    --weight_decay 1e-4 \
    --imgsz 800 \
    --device 0
```

---

## 8. 预期实验组数汇总

| 阶段 | 模型 | 组数 | epochs/组 | 总 GPU 时间 |
|------|------|------|-----------|-------------|
| Phase 1 | YOLO baseline | 1 | 12 | 18h |
| Phase 1 | GDINO baseline | 1 | 12 | 42h |
| Phase 2 | YOLO lr sweep | 3 (新) | 6 | 27h |
| Phase 2 | GDINO lr sweep | 4 | 6 | 84h |
| Phase 3 | YOLO best | 1 | 12 | 18h |
| Phase 3 | GDINO best | 1 | 12 | 42h |
| **总计** | | **11 组** | | **~231h (~9.6 天)** |

> Phase 3 视剩余时间灵活调整优先级。如果时间不足，Phase 2 的 6-epoch 结果已支撑完整分析。

---

## 9. 备用方案

如果 Grounding DINO 训练时间超出预期，可启用：

**Backup A**：GDINO lr 扫描减少为 3 个 lr 值（省 21h）

**Backup B**：YOLO-World-L + YOLO-World-M（同架构不同规模，分析 scaling 效应，训练都很快）

**Backup C**：全部使用 6 epoch 训练 + 延长分析时间，放弃 12 epoch 全训练
