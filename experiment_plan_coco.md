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
- **CNN 两阶段 Proposal-based**：RPN 先提取区域提议，分类头用 CLIP embedding 做开集分类（如 Detic）

两种范式的核心差异在于**文本-视觉融合深度**——这决定了梯度流动路径、对超参数的敏感性、以及最终检测性能的天花板。目前没有人在控制变量的条件下对两者进行过系统对比。

### 核心研究问题

1. **为什么两种不同架构在同一个数据集上需要不同的最优超参数？**
2. **架构差异导致的超参数敏感性不同，根源是什么？**
3. **两种架构的最终性能差距由哪些因素构成（定位 vs 分类 vs 漏检）？**

---

## 2. 模型选择

| 属性 | YOLO-World-L | Detic (Faster R-CNN) |
|------|-------------|----------------------|
| **范式** | CNN 单阶段 dense prediction | CNN 两阶段 proposal-based |
| **视觉骨干** | YOLOv8-L | ResNet-50 |
| **语言编码器** | CLIP-L (frozen) | CLIP (frozen, 仅生成分类器权重) |
| **融合方式** | Late fusion — 分类头 text embedding 点积 | 解耦：RPN 纯视觉 + 分类头用 CLIP embedding 替换权重 |
| **参数量** | ~100M | ~50M |
| **推荐 bs (L40)** | 8–16 | 8–16 |
| **每 epoch 训练时间（实测）** | ~0.45h (27min) | ~0.3h（预估） |
| **官方预训练权重** | 有 | 有 |
| **训练框架** | Ultralytics | Detectron2 |

### 选择理由

- **检测范式对比度最大**：单阶段 dense prediction（YOLO-World）vs 两阶段 proposal-based（Detic），这是目标检测最核心的架构分水岭
- **文本使用方式截然不同**：两者都用 CLIP，但 YOLO-World 在分类头做点积，Detic 直接用 CLIP embedding 替换分类器权重矩阵——相同的文本编码器，完全不同的利用策略
- **训练成本低**：YOLO-World ~0.45h/epoch，Detic 基于 ResNet-50 预计更快，两项实验在 10 天内可完成
- **都有成熟代码库**：Ultralytics + Detectron2 都是工业级框架

### 核心差异：检测范式 + 文本利用策略

```
检测范式：

YOLO-World-L                          Detic
     |                                    |
  Dense Prediction ←←←←←←←←→ Proposal-based
     |                                    |
 一次 forward 出所有框              RPN 先提 proposal
 + 分类分数                        RoI pooling 后分类
     |                                    |
 文本用于：分类头点积                文本用于：生成分类器权重
 (每个位置都和 text emb 算相似度)    (CLIP emb 直接替换 FC 权重)
```

两个关键分析维度：
1. **单阶段 vs 两阶段**如何影响 lr、bs 的选择
2. **相同的 CLIP 编码器，不同的利用方式**（点积 vs 权重替换）如何影响开集泛化

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
| Conda 环境 | 创建 yolow + detic 两个 conda env | 2h |
| YOLO-World 环境 | `pip install ultralytics` + pycocotools, wandb | 0.5h |
| Detic 环境 | detectron2, torch, CLIP, pycocotools | 1h |
| COCO 数据集 | 已有，无需重新下载 | 0h |
| 模型权重 | 下载 YOLO-World-L (90MB) + Detic ResNet-50 预训练权重 | 0.5h |
| 验证 | 各跑 1 个 mini-batch 确认 pipeline 通过 | 0.5h |

**预计**：4–8 小时（含可能的 COCO 下载）

### 4.2 Phase 1：Baseline 复现（Day 2–4）

两个模型各跑 1 次完整训练，使用官方推荐超参数。

| 参数 | YOLO-World-L | Detic (ResNet-50) |
|------|-------------|-------------------|
| Learning rate | 2e-4 | 2e-4（待确认官方默认值） |
| Batch size | 8 | 8 |
| Epochs | 12 | 12 |
| Warmup steps | 1000 | 1000 |
| Optimizer | AdamW | SGD（待确认） |
| Weight decay | 0.05 | 1e-4 |
| Image size | 640 | 640 |
| Scheduler | cosine | cosine |
| **预计训练时长** | **~5.5h** | **~3.6h（预估）** |

**目的**：
- 建立性能基线（AP, AP50, AP75, AP_s, AP_m, AP_l）
- 验证训练流程的正确性
- 确认 L40 48GB 显存下的最大可行 bs

### 4.3 Phase 2：Learning Rate 敏感性扫描（Day 5–7）

**核心实验**。对每个模型扫描 4 个 lr 值，使用 6 epochs（足以看出收敛趋势和相对排序），固定其他参数为 baseline 值。

| 模型 | lr 扫描范围 | 单次训练时间 | 4 组总时间 |
|------|-----------|-------------|-----------|
| YOLO-World-L | [5e-5, 1e-4, 2e-4 (baseline), 5e-4] | ~2.7h (6 epoch) | ~10.8h |
| Detic | [5e-5, 1e-4, 2e-4, 5e-4] | ~1.8h（预估） | ~7.2h（预估） |

> **注意**：YOLO 的 4 组 lr 中有一组复用 Phase 1 baseline 结果（lr=2e-4），实际只需跑 3 组新的。

**预期发现**：
- YOLO-World（单阶段 dense）的 lr 敏感性受 PAN neck 的 batch norm 影响，预期最优 lr 区间较宽
- Detic（两阶段 proposal-based）的 RPN 和分类头可能需要不同 lr 尺度，RPN lr 过高会降低 proposal 质量

### 4.4 Phase 3：最优配置全训练 + 最终评估（Day 8–10）

Phase 2 中每模型选出最优 lr，然后跑完整的 12 epochs：

| 模型 | 训练时间 | 任务 |
|------|---------|------|
| YOLO-World-L (best lr) | ~5.5h | 完整 12 epoch 训练 + COCO val 评估 |
| Detic (best lr) | ~3.6h（预估） | 完整 12 epoch 训练 + COCO val 评估 |

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
Phase 1  ██    ████████████
        (YW)  (YW 5.5h)  (Detic 3.6h)
Phase 2          ████████████████████████████
                (YW lr sweep 10.8h)  (Detic lr sweep 7.2h)
Phase 3                                     ██████████████████
                                           (best full train)
Analysis         ←←←←←←←←←←←← 贯穿全程，训练间歇进行 →→→→→→→→→→→→
```

**时间总结**：

| 阶段 | 内容 | 时长 | 累计 |
|------|------|------|------|
| Phase 0 | 环境配置 + Detic 搭建 | 1 天 | 1 天 |
| Phase 1 | YOLO baseline（已完成） | 0.25 天 | 1.25 天 |
| Phase 1 | Detic baseline | 0.15 天 | 1.4 天 |
| Phase 2 | YOLO lr sweep (3 新 lr) | 0.45 天 | ~2 天 |
| Phase 2 | Detic lr sweep (4 lr) | 0.3 天 | ~2.3 天 |
| Phase 3 | Best 配置各训 12 epoch | 0.4 天 | ~3 天 |

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
- **高 lr 区谁崩得更快** → 两阶段模型的 RPN 可能在高 lr 下 proposal 质量下降明显

### 6.2 训练动力学分析

同一模型在不同 lr 下的训练曲线对比：
- Loss 下降速度 vs 最终收敛水平
- 训练集 vs 验证集的过拟合 gap
- Detic 的 RPN loss vs 分类 loss 的收敛差异

### 6.3 Error Breakdown

对最优模型进行错误分类（TIDE 框架）：

| 错误类型 | 含义 | 预期差异 |
|---------|------|---------|
| Cls | 框对、类错 | YOLO 可能更多（dense prediction 类别歧义大） |
| Loc | 类对、框偏 | Detic 可能更少（RoI pooling 精准定位） |
| Miss | 漏检 | Detic 可能更少（RPN recall 高），YOLO 可能漏小物体 |
| Bkg | 背景误检 | YOLO 可能更多（dense prediction 输出密集） |

### 6.4 按目标大小分层分析

COCO 的 AP_s / AP_m / AP_l：
- **小物体 (AP_s)**：Detic 的 RPN+RoI 对高分辨率特征处理较好，YOLO 的跨尺度 PAN 也有优势，需要实测对比
- **大物体 (AP_l)**：两者预期接近

### 6.5 文本编码器策略对比

| 维度 | YOLO-World | Detic |
|------|-----------|-------|
| CLIP 使用方式 | 分类头与 text embedding 做点积 | CLIP embedding 直接替换 FC 权重 |
| 文本参与梯度 | 无（CLIP 冻结，点积不产生文本梯度） | 无（CLIP 仅用来生成固定权重） |
| 开集能力来源 | 可替换 text prompt | 可替换 CLIP embedding |
| 训练学什么 | 视觉特征到 CLIP 空间的映射 | 区域特征对齐到 CLIP 空间 |

两者都冻结 CLIP，差异在于视觉特征如何与 CLIP 空间交互——点积 vs 权重替换。

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

### Detic Baseline（Detectron2）

```bash
python train_detic.py \
    --dataset coco \
    --config detic_swin_r50 \
    --epochs 12 \
    --batch 8 \
    --lr 2e-4 \
    --warmup 1000 \
    --imgsz 640 \
    --device 0
```

---

## 8. 预期实验组数汇总

| 阶段 | 模型 | 组数 | epochs/组 | 总 GPU 时间 |
|------|------|------|-----------|-------------|
| Phase 1 | YOLO baseline | 1 | 12 | 5.5h（已完成） |
| Phase 1 | Detic baseline | 1 | 12 | ~3.6h（预估） |
| Phase 2 | YOLO lr sweep | 3 (新) | 6 | ~8h |
| Phase 2 | Detic lr sweep | 4 | 6 | ~7.2h（预估） |
| Phase 3 | YOLO best | 1 | 12 | 5.5h |
| Phase 3 | Detic best | 1 | 12 | ~3.6h（预估） |
| **总计** | | **11 组** | | **~33h (~1.4 天)** |

> Phase 3 视剩余时间灵活调整优先级。如果时间不足，Phase 2 的 6-epoch 结果已支撑完整分析。

---

## 9. 备用方案

**Backup A**：如 Detectron2 环境问题无法解决，回退到 GroundingDINO-Tiny（更小的 Swin-T 骨干）

**Backup B**：全部使用 6 epoch 训练 + 延长分析时间
