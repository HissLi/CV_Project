# Open-Vocabulary Object Detection: 多架构对比研究

## 课题信息

- **课题**：课题4 — Open-Vocabulary Object Detection and Visual Grounding
- **方向**：多模型跨架构对比研究
- **算力**：L40（48GB 显存），集群（二）a100 分区（l40gpu001, l40gpu002）
- **账号**：cse12212230 @ 172.18.34.26:10022
- **截止日期**：2026.6.21（报告 + 展示）

---

## 研究动机

开放词汇目标检测（Open-Vocabulary Object Detection, OVOD）旨在利用任意文本描述检测图像中的目标，而非仅限固定类别集合。当前主流 OVOD 模型在架构选型上存在显著分歧：CNN 单阶段、Transformer query-based、两阶段检测解耦等。不同架构的核心差异在于**文本-视觉融合机制**——这直接影响了模型对超参数的敏感性、对长尾类别的泛化能力以及开放词汇潜力。

### 核心研究问题

1. **为什么不同架构的模型在同一数据集上需要不同的最优训练参数？**
2. **架构差异如何导致性能差异？**
3. **文本-视觉融合机制的深浅如何影响模型行为？**

目前每篇 OVOD 论文都在自己的 setting 下报告最优结果，但：
- 用的数据集不同（COCO vs LVIS vs Objects365）
- 超参数扫描范围不同（有的只扫 lr，有的固定 bs=总 batch）
- 评估协议不同（zero-shot vs fine-tune）
- **没有人在控制变量的条件下横向比较过不同范式的超参数敏感性**

本课题选取三种不同范式的主流 OVOD 模型，在 LVIS 长尾分布数据集上进行统一的 fine-tune 和超参数扫描，从文本-视觉融合机制的角度进行归因分析。

---

## 模型选择与技术分析

### 总览

| 范式 | 模型 | 骨干 | 语言编码器 | 融合方式 | 参数量 | L40 推荐 bs |
|------|------|------|------------|----------|--------|-------------|
| CNN 单阶段 | **YOLO-World-L** | YOLOv8-L | CLIP-L | Late fusion（分类头 text embedding 替换） | ~100M | 16-32 |
| Transformer | **Grounding DINO-B** | Swin-B | BERT-B | Deep fusion（decoder cross-attention 持续注入） | ~200M | 4-8 |
| 两阶段检测 | **Detic (Faster R-CNN)** | ResNet-101 | CLIP | 解耦：分类器权重替换为 CLIP embeddings | ~80M | 8-16 |

三者的文本-视觉融合机制构成一个**从浅到深的谱系**：

```
文本参与程度：

YOLO-World          Detic              Grounding DINO
  |                   |                     |
  浅 ←←←←←←←←←←←←←←←←←←←←←←←←←←←→ 深
  |                   |                     |
 仅在分类头替换      分类头用 CLIP         每一层 decoder
 文本 embedding      做开集分类映射        都有 cross-attention
```

文本参与越深，模型对超参数的依赖性越强——深层融合意味着文本编码器的梯度会通过 cross-attention 反向传播到视觉 backbone，改变整个优化景观。

---

### 2.1 YOLO-World — Late Fusion 代表

**推理流程**：

```
图像 → YOLOv8 Backbone → PAN Neck → 检测头 → 边界框回归
                                              ↓
文本 → CLIP Text Encoder → Text Embeddings → 分类分数（点积）
```

**关键细节**：

- **RepVL-PAN**：YOLO-World 在 neck 中使用跨模态卷积做轻量文本-视觉交互，文本 embedding 仅作为卷积核权重参与，不是真正的 attention
- **文本不产生梯度回传到 backbone**：fine-tune 时通常冻结 CLIP text encoder，只训视觉部分 + 检测头
- **NMS 是后处理但影响大**：不同 bs 下模型输出的置信度分布会偏移，NMS threshold 的敏感度反应了模型校准（calibration）质量

**为什么 bs 对它影响更大**：
YOLOv8 的 PAN neck 使用 FPN 结构，不同尺度的特征金字塔对 batch statistics 敏感。更大的 bs → 更稳定的 batch norm → 小物体检测更好（PAN 的高分辨率层受益更多）

**为什么最优 lr 可能更高**：
Late fusion 下梯度不经过文本编码器，视觉部分优化更独立，可以用更激进的学习率

---

### 2.2 Grounding DINO — Deep Fusion 代表

**推理流程**：

```
图像 → Swin Transformer Backbone → Multi-scale Feature Maps
                                          ↓
文本 → BERT Text Encoder ──────────→ Feature Enhancer (Deformable Self-Attn)
                                          ↓
                                    Cross-Modality Decoder × 6
                                    (每一层都有 text-to-vision cross-attn)
                                          ↓
                                    检测头 → 边界框 + 分类
```

**关键细节**：

- **Deformable Attention**：在参考点周围稀疏采样少量 key，让 query 专注于局部区域。query 数量越多，每层采样的 key 越多 → 对文本描述的定位越精细
- **Feature Enhancer**：进入 decoder 之前，文本特征与视觉特征先做一层交互（GLIP 遗留设计）
- **BERT vs CLIP**：BERT 是纯文本预训练，其 embedding 空间和视觉没有对齐。Grounding DINO 必须在训练中学习跨模态对齐，而 YOLO-World 的 CLIP 是天然对齐的

**为什么 warmup 对它更关键**：
6 层 decoder 的 cross-attention 在初始化时是随机的。没有 warmup 时，早期梯度大且方向随机 → text encoder embedding 被快速拉扯 → BERT 的语义结构被破坏。warmup 让视觉分支先稳定，再逐步引入文本梯度

**为什么 query number 是特有参数**：
YOLO-World 的检测头是 dense prediction，天然有大量候选。Grounding DINO 用固定数量的 object query（如 900 个）来 cover 整张图。在 LVIS 1203 类场景下，900 个 query 可能不够（一张图可能有 50+ 个标注框）→ query 数影响 recall 上限

---

### 2.3 Detic — Decoupled Two-Stage 代表

**推理流程**：

```
Stage 1 (RPN, 纯视觉):
  图像 → ResNet-101 → FPN → RPN → Region Proposals (类别无关)

Stage 2 (分类, 文本参与):
  Region Features → RoI Pooling → 分类头
                                    ↓
  文本 → CLIP Text Encoder → Classifier Weights (CLIP embeddings)
```

**关键细节**：

- **RPN 和分类头完全解耦**：RPN 只做"这里有没有物体"，分类头只做"这是什么物体"
- **分类器权重不是学出来的，是生成的**：用 CLIP 对每类文本做 embedding，这些 embedding 直接作为分类器的权重矩阵。这就是 "open-vocabulary" 的来源——推理时可加入新类别的 CLIP embedding，模型就能检测新类别
- **训练时实际在学的是"图像特征到 CLIP 空间的映射"**，而非传统的分类边界

**为什么 RPN 和分类头需要不同 lr**：
- RPN 是标准 Faster R-CNN 预训练的，已收敛良好，只需小 lr 微调
- 分类头需学习将 RoI 特征映射到 CLIP 空间，这是全新任务 → 需要更大 lr
- 如果 RPN lr 太大，区域提议质量会退化；如果分类头 lr 太小，映射学不好

**Detic 在稀有类别上的潜在优势**：
LVIS 的 1203 类中很多是细粒度类别（如不同品种的狗），CLIP 的 text embedding 天然能区分它们（因为 CLIP 在 internet-scale image-text pairs 上训练），而 Grounding DINO 的 BERT 需要在训练中重新学习这些语义边界。这意味着 Detic 在稀有类别上的 AP_r 可能反而比 Grounding DINO 高，尽管后者整体 AP 更高——这是一个非常有价值的分析维度

---

## 数据集

### 主数据集：LVIS v1

| 属性 | 值 |
|------|-----|
| 类别数 | 1203 |
| 训练图像 | ~100K |
| 分布特征 | 长尾分布（稀有类别可能仅 1 张图） |
| 类别划分 | frequent (>100), common (10-100), rare (1-10) |

**频率分布细节**：

| 类别 | 图像数阈值 | 类别数 | 典型例子 |
|------|-----------|--------|----------|
| Frequent | > 100 张 | ~120 类 | person, car, chair |
| Common | 10-100 张 | ~460 类 | backpack, skateboard |
| Rare | 1-10 张 | ~620 类 | 具体品种、特殊工具、罕见动物 |

Rare 类占类别数的一半以上，但总样本量极小。这对三个模型的影响不同：

| 模型 | 稀有类瓶颈 | 预期表现 |
|------|-----------|----------|
| YOLO-World | CLIP 对细粒度类别的 text embedding 区分度不足 + 视觉样本少 | AP_r 可能最低 |
| Grounding DINO | BERT 不对齐视觉，样本少导致 cross-attention 学不好 | AP_r 受限但整体 AP 可能最高 |
| Detic | CLIP zero-shot 能力 + RoI 精准定位，样本少不影响分类器 | AP_r 可能反而最高 |

**选择 LVIS 的理由**：
- 1203 类比 COCO 的 80 类更接近真正的"开放词汇"场景
- 长尾分布天然考验模型的文本辅助泛化能力——稀有类别几乎只能靠语言描述来识别
- head/mid/tail 类别分层分析提供丰富的分析维度
- LVIS 上的 OVOD 对比研究仍相对稀缺，有探索价值

### 额外评测：Zero-shot Cross-Dataset Evaluation on COCO

在 LVIS 上完成 fine-tune 和超参数调优后，**不做任何 fine-tune**，直接在 COCO 2017 val 上推理：

LVIS 和 COCO 共享约 80 个类别，但存在差异：
- **标注风格不同**：两个数据集的标注团队不同 → 框的松紧程度可能有差异
- **类别定义有细微差异**：如 "couch" vs "sofa"，"cup" vs "mug" → 对语言编码器是真正的泛化测试
- **图像风格不同**：COCO 更"生活化"，LVIS 有更多专业摄影图

**分析维度**：
- 三个模型在跨数据集场景下的**性能绝对值和衰减率**
- 特别关注 LVIS-COCO 共有的 80 个类别在"见过但不同分布"上的表现差异
- 文本编码器（CLIP vs BERT）在跨数据集泛化中的角色

此部分为独立分析章节，属于加分项。

---

## 实验设计

### Phase 1：Baseline 复现

- 三个模型分别用官方建议的超参数在 LVIS 上 fine-tune
- 各 1 个 run，建立性能基线
- 验证环境正确性和训练流程

**预计**：3-4 天

### Phase 2：超参数扫描（核心实验）

采用**控制变量法**（每次只变一个参数），避免网格搜索的组合爆炸。

**通用参数扫描**（3 个模型均适用）：

| 超参数 | 扫描范围 | 备注 |
|--------|----------|------|
| Learning rate | [1e-5, 5e-5, 1e-4, 5e-4, 1e-3] | 单参数扫描，固定其他为 baseline 值 |
| Batch size | [4, 8, 16, 32] | 利用 L40 48GB 显存，探索上限 |
| Warmup steps | [0, 500, 1000, 2000] | 重点分析 Grounding DINO |
| Scheduler | cosine / step / constant | 分析收敛行为差异 |

**模型特定参数**：

| 模型 | 参数 | 扫描范围 | 是否需要重新训练 |
|------|------|----------|-------------------|
| YOLO-World | NMS threshold | [0.45, 0.50, 0.55, 0.60, 0.65] | 否（仅改变推理参数） |
| Grounding DINO | Query number | [300, 600, 900] | 否（可仅改推理参数） |
| Grounding DINO | Decoder layers | [3, 6] | 是（需重新训练） |
| Detic | RPN:分类头 lr ratio | [1:1, 1:2, 1:5, 2:1] | 是（需重新训练） |

**实验组数估算**：

通用部分（每模型）：5 (lr) + 3 (bs, 不含 baseline) + 3 (warmup) + 2 (scheduler) = **13 组**

加上模型特定 + baseline：
- YOLO-World: 13 组
- Grounding DINO: 13 + 2 (decoder layers) = 15 组
- Detic: 13 + 4 (lr ratio) = 17 组
- Baseline: 3 组

**总计约 48 组训练 run**（含不可训练的推理参数变化则为 54 组）

**L40 每次训练预估**：

| 模型 | 推荐 bs | 预计显存 | 训练时间/epoch | 12 epochs 总时间 |
|------|--------|---------|---------------|-------------------|
| YOLO-World-L | 16-32 | ~30-40GB | ~2h | ~24h |
| Grounding DINO-B | 4-8 | ~35-45GB | ~4h | ~48h（需分两次提交） |
| Detic | 8-16 | ~25-35GB | ~3h | ~36h（需分两次提交） |

> **注意**：每次 sbatch 作业最长 24h。Grounding DINO 和 Detic 的完整训练需要拆分为多个作业或减少 epoch。

**预计**：5-7 天

### Phase 3：归因分析

#### 3.1 lr-bs 热力图（核心可视化）

对每个模型，在 lr-bs 平面上采 6-9 个点（部分析因设计），用颜色表示 AP 值：

```
YOLO-World                    Grounding DINO
bs                           bs
32 |  .    .    .            32 |  .    .    .
16 |  .    .    .            16 |  .    .    .
 8 |  .    .    .             8 |  .    .    .
 4 |  .    .    .             4 |  .    .    .
    ----------------- lr         ----------------- lr
    1e-4 5e-4 1e-3              1e-5 5e-5 1e-4
```

**预期发现**：
- YOLO-World 最优区域在**高 lr + 大 bs** 区（CNN 对 batch norm 友好）
- Grounding DINO 最优区域在**低 lr 区**且范围更窄（Transformer 对 lr 更敏感）
- Detic 热力图形状更扁平（RPN 和分类头分离）

#### 3.2 head/mid/tail 分层评估

分析各模型在 LVIS 三类频率类别上的 AP 差异：
- YOLO-World 的 rare AP_r 可能受限于 CLIP 文本编码质量
- Grounding DINO 的 rare AP_r 可能受限于训练样本量
- Detic 的 rare AP_r 可能反而最高（CLIP + RoI 精准定位）

#### 3.3 文本编码器分析

CLIP（YOLO-World, Detic）vs BERT（Grounding DINO）对稀有类别的泛化差异：
- CLIP 是 vision-language 联合训练的，对视觉相关的语义有天然对齐
- BERT 是纯文本预训练，语义结构更丰富但可能和视觉不对齐
- 在细粒度类别上的表现差异可归因到文本编码器

#### 3.4 Error Analysis（TIDE 分类）

将所有 FP 归类为：
- **Cls（分类错误）**：正确框 + 错误类别 → 语义混淆 vs 非语义混淆
- **Loc（定位错误）**：类别正确 + IoU 不够
- **Both（分类+定位都错）**
- **Dupe（重复检测）**
- **Bkg（背景误检）**
- **Miss（漏检）**

**预期发现**：
- Grounding DINO 的 Loc 错误比例更低（Transformer 框更准）
- YOLO-World 的 Cls 错误更多（late fusion 文本区分能力弱）
- Detic 的 Miss 更多（RPN 在长尾类别上 recall 不足）

#### 3.5 Zero-shot COCO 评估

- 三个模型在 COCO val 上的 AP（不做 fine-tune）
- 性能衰减率 = (COCO_AP / LVIS_AP_on_shared_classes) × 100%
- Per-class AP 差异热力图

**预计**：3-4 天

---

## 评估指标

### 检测指标
- **AP**（Average Precision, IoU=0.50:0.95）
- **AP50 / AP75**
- **AP_r / AP_c / AP_f**（rare / common / frequent，LVIS 特有）
- **AR**（Average Recall）
- **AP on LVIS-COCO shared 80 classes**

### Zero-shot 指标
- COCO AP（不做任何 fine-tune）
- 性能衰减率
- Per-class AP 差异分析

---

## 工作量估算

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 环境搭建 | 三模型配置 + LVIS/COCO 数据预处理 | 2-3 天 |
| Phase 1 | Baseline fine-tune（3 模型 × 1 run） | 3-4 天 |
| Phase 2 | 超参数扫描（3 模型 × 约 16 组配置） | 5-7 天 |
| Phase 3 | Zero-shot 测试 + 归因分析 + 可视化 | 3-4 天 |
| 报告撰写 | 分析 + 图表 + 写作 | 4-5 天 |
| **总计** | | **17-23 天** |

---

## 报告结构

| 章节 | 建议页数 | 核心内容 | 占分 |
|------|---------|----------|------|
| **1. Introduction** | 1-2 页 | OVOD 背景 + 为什么比较不同架构有意义 + 研究问题 | 10 |
| **2. Related Work** | 1-2 页 | OVOD 三大范式 + LVIS benchmark + 超参数研究现状 | 5 |
| **3. Method** | 3-4 页 | 三个模型的架构细节 + 融合机制对比 + 关键差异分析（用自己话重新组织，聚焦"文本如何参与检测"的统一视角） | 10 |
| **4. Experiments** | 5-7 页 | 见下文 | 20 |
| **5. Conclusion** | 0.5-1 页 | 发现总结 + 局限性 + 未来方向 | 5 |
| **References** | 1 页 | 所有引用 | 5 |
| **Clarity** | — | 图表清晰 + 逻辑连贯 | 5 |

**实验章节（4. Experiments）细分**：
- 4.1 Datasets：LVIS + COCO 介绍，频率分布可视化
- 4.2 Implementation Details：每个模型的超参数、框架、训练 trick
- 4.3 Metrics：AP / AP_r / AP_c / AP_f / AR 的定义
- 4.4 Results：
  - 4.4.1 Baseline 对比表（三模型 × 官方参数）
  - 4.4.2 超参数扫描结果（lr-bs 热力图 × 3 + 各参数趋势折线图）
  - 4.4.3 模型特定参数分析（NMS/query/lr ratio）
  - 4.4.4 Head/Mid/Tail 分层分析
  - 4.4.5 Error Analysis（TIDE 分类饼图 × 3）
  - 4.4.6 Zero-shot COCO 结果
- 4.5 Discussion：连接架构差异和实验观察

---

## 执行策略

### 优先级排序

| 优先级 | 实验 | 原因 |
|--------|------|------|
| **P0 必须** | Baseline fine-tune × 3 | 没有 baseline 就没有一切 |
| **P0 必须** | lr 扫描 × 3 | 最核心的超参数对比 |
| **P1 重要** | bs 扫描 × 3 | 和 lr 一起支撑热力图 |
| **P1 重要** | LVIS 分层评估 | 支撑"长尾泛化"分析 |
| **P1 重要** | Zero-shot COCO | 独立加分章节 |
| **P2 增强** | Warmup 扫描 | 主要对 Grounding DINO 有意义 |
| **P2 增强** | 模型特定参数 | 加深各模型的分析深度 |
| **P3 锦上添花** | Scheduler 对比 | 影响通常小于 lr/bs |
| **P3 锦上添花** | TIDE Error Analysis | 烧时间但效果好的可视化 |

### 并行执行策略

集群每个用户最多提交 2 个作业，同时运行 1 个：
1. 前台跑一个模型的实验
2. 另一个模型的数据预处理/调试在登录节点做
3. 用 `squeue` 监控，一旦结束立即提交下一个

---

## 归因分析的因果关联（核心贡献）

| 观察 | 架构解释 |
|------|----------|
| YOLO-World 最优 lr > Grounding DINO | Late fusion 梯度不经过文本编码器，可用更激进的 lr |
| Grounding DINO 需要更长 warmup | Cross-attention 需视觉先稳定，文本才能学会"在哪里 attend" |
| Detic AP_r 可能最高 | CLIP zero-shot 能力 + RoI 精准定位，稀有样本少不影响分类器 |
| Grounding DINO bs=4 最优 vs YOLO bs=32 最优 | Transformer decoder 处理 object query，对 bs 不敏感；CNN 受益于大 bs 的 batch norm |
| YOLO-World Cls 错误多 | Late fusion 下文本 embedding 无法根据视觉上下文动态调整 |
| Grounding DINO Loc 错误少 | Cross-attention 使每个 query 能精准 attend 到文本描述的区域 |

---

## 风险与应对

| 风险 | 应对 |
|------|------|
| 某模型官方代码不可用或 bug 多 | 优先用 HuggingFace 版本，或替换为同范式其他模型（如 OWL-ViT 替代 Grounding DINO） |
| LVIS 1203 类训练过慢 | 先用 LVIS mini（~5K 图）跑通流程，再上全量 |
| L40 被抢占或排队过久 | 利用 rtx2080ti 分区先跑 YOLO-World 等轻量实验 |
| 24h 作业不够训练完完整 epoch | 拆分为多阶段训练，每阶段 checkpoint 续训 |
| 48 组实验太多时间不够 | 砍掉 scheduler 和部分 warmup 扫描，降到 ~30 组 |

---

## 参考资源

### 模型
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- YOLO-World: https://github.com/AILab-CVC/YOLO-World
- Detic: https://github.com/facebookresearch/Detic
- GLIP: https://github.com/microsoft/GLIP
- OWL-ViT: https://huggingface.co/docs/transformers/model_doc/owlvit

### 数据集
- LVIS: https://www.lvisdataset.org/
- COCO: https://cocodataset.org/
- ODinW: https://github.com/IDEA-Research/ODinW

### 评估工具
- COCO API: https://github.com/cocodataset/cocoapi
- LVIS API: https://github.com/lvis-dataset/lvis-api
- pycocotools: https://github.com/ppwwyyxx/cocoapi
- TIDE (Error Analysis): https://github.com/dbolya/tide
