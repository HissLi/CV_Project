# OVOD 多架构对比研究 — 精简版执行计划

## 课题信息

- **课题**：课题 4 — Open-Vocabulary Object Detection and Visual Grounding
- **方向**：双模型跨架构对比研究（Late Fusion vs Deep Fusion）
- **算力**：L40（48GB 显存），a100 分区（l40gpu001: 8×GPU, l40gpu002: 3×GPU）
- **账号**：cse12212230 @ 172.18.34.26:10022
- **作业时限**：48h（a100 分区），训练均可完整完成
- **截止日期**：2026.6.21（报告 + 展示，只剩 16 天）

---

## 研究动机与核心问题

开放词汇目标检测（OVOD）的主流模型在架构上存在根本分歧：CNN 单阶段 v.s. Transformer query-based。两种范式的**文本-视觉融合深度**不同，决定了它们对训练超参数的敏感性、对长尾类别的泛化行为以及跨数据集迁移能力。

本课题选择两个最具代表性的模型——**YOLO-World（浅融合）** 和 **Grounding DINO（深度融合）**——在 LVIS 长尾分布数据集上进行控制变量实验，回答以下问题：

1. **融合深度如何影响最优学习率和批次大小？**
2. **在稀有类别上，浅融合（依赖 CLIP 先验）和深度融合（依赖 cross-attention 学习）哪种更有效？**
3. **架构差异在跨数据集泛化（LVIS → COCO）中如何体现？**

### 融合深度谱系

```
YOLO-World                              Grounding DINO
  │                                         │
  浅 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←→ 深
  │                                         │
  文本仅在分类头参与                      每层 decoder 都有
  (Late Fusion)                          cross-attention (Deep Fusion)
```

---

## 模型选择

### YOLO-World-L — Late Fusion（浅融合）

| 属性 | 值 |
|------|-----|
| 骨干 | YOLOv8-L |
| 语言编码器 | CLIP-L |
| 参数量 | ~100M |
| L40 推荐 bs | 16-32 |
| 预估训练时间 | ~18h（12 epochs） |
| GitHub | https://github.com/AILab-CVC/YOLO-World |

**文本参与方式**：CLIP text encoder 将类别描述编码为 text embeddings，仅在分类头做点积替换分类权重。文本梯度不反向传播到视觉 backbone。RepVL-PAN 中有轻量跨模态卷积，但不是真正的 attention。

### Grounding DINO-B — Deep Fusion（深度融合）

| 属性 | 值 |
|------|-----|
| 骨干 | Swin-B |
| 语言编码器 | BERT-B |
| 参数量 | ~200M |
| L40 推荐 bs | 4-8 |
| 预估训练时间 | ~36h（12 epochs） |
| GitHub | https://github.com/IDEA-Research/GroundingDINO |

**文本参与方式**：6 层 Cross-Modality Decoder，每一层都有 text-to-vision cross-attention。BERT 文本特征持续注入检测过程，梯度通过 cross-attention 反向传播到视觉 backbone。BERT 是纯文本预训练，需要训练中学习跨模态对齐。

### 为什么选这两个

- **对比最鲜明**：融合深度从最浅（仅分类头）到最深（每层 decoder）
- **CLIP vs BERT**：文本编码器的差异是分析重点——CLIP 天然视觉对齐，BERT 需要从零学对齐
- **CNN vs Transformer**：检测架构的根本差异
- **Detic 被砍掉**：作为"中间"方案，在两模型对比中增量价值有限，且节省约 36h 训练时间

---

## 数据集

### 主数据集：LVIS v1

| 属性 | 值 |
|------|-----|
| 类别数 | 1203 |
| 训练图像 | ~100K |
| 分布特征 | 长尾分布 |
| 类别划分 | frequent (>100), common (10-100), rare (1-10) |

**频率分布**：

| 类别 | 图像数阈值 | 类别数 | 典型例子 |
|------|-----------|--------|----------|
| Frequent | > 100 张 | ~120 类 | person, car, chair |
| Common | 10-100 张 | ~460 类 | backpack, skateboard |
| Rare | 1-10 张 | ~620 类 | 具体品种、特殊工具 |

**为什么 LVIS 而非 COCO**：
- 1203 类远多于 COCO 的 80 类，更接近"开放词汇"场景
- 长尾分布天然测试语言辅助泛化——稀有类别几乎只能靠文本描述来识别
- head/mid/tail 分层评估提供天然的分析维度
- LVIS 上的 OVOD 系统对比研究仍较稀缺

### 跨数据集评测：COCO 2017 val（Zero-shot）

LVIS fine-tune 完成后，不做任何调整，直接在 COCO 2017 val 上推理。分析维度：
- 两个模型在跨数据集场景下的性能衰减率
- LVIS-COCO 共享 ~80 类的 per-class 差异
- CLIP vs BERT 在跨数据集泛化中的角色

---

## 实验设计

### 总览

| 阶段 | 实验 | 训练次数 | 优先级 |
|------|------|---------|--------|
| Phase 1 | 环境搭建 + 数据准备 | 0 | P0 |
| Phase 2 | Baseline 复现 | 2 | P0 |
| Phase 3 | lr 扫描 | 6 | P0 |
| Phase 4 | bs 扫描 + lr-bs 热力图 | 4 | P1 |
| Phase 5 | 模型特定参数 | 4 | P2 |
| Phase 6 | 评估 + 分析 | 0 | P0 |
| **总计** | | **最多 16 次训练** | |

### Phase 1：环境搭建与数据准备（2-3 天）

#### 1.1 服务器环境
```
~/cv_project/
├── envs/           # conda 环境
│   ├── yolow/      # YOLO-World 环境
│   └── gdino/      # Grounding DINO 环境
├── repos/          # 模型代码
│   ├── YOLO-World/
│   └── GroundingDINO/
├── datasets/
│   ├── lvis/       # LVIS v1
│   └── coco/       # COCO 2017
├── scripts/        # sbatch 训练脚本
├── results/        # 训练日志 + checkpoint
└── eval/           # 评估脚本
```

#### 1.2 步骤清单
1. 创建 conda 环境（Python 3.10 + PyTorch 2.x + CUDA 12.x）
2. Clone YOLO-World 和 Grounding DINO 代码仓库
3. 下载 LVIS v1 数据集（~20GB 图像 + 标注）
4. 下载 COCO 2017 val（~1GB 图像 + 标注，仅需 val）
5. 下载预训练权重（YOLO-World-L, Grounding DINO-B）
6. 验证推理 pipeline 可运行
7. 编写统一格式的 sbatch 训练脚本
8. 编写 LVIS/COCO 评估脚本（基于 pycocotools + lvis-api）

### Phase 2：Baseline 复现（2-3 天）

两个模型各跑 1 次 baseline，使用官方推荐超参数。

| 参数 | YOLO-World-L | Grounding DINO-B |
|------|-------------|-------------------|
| Learning rate | 5e-4 (det) / 5e-5 (backbone) | 1e-4 |
| Batch size | 16 | 4 |
| Epochs | 12 | 12 |
| Warmup | 1000 steps | 1000 steps |
| Scheduler | cosine | cosine |
| Optimizer | AdamW | AdamW |
| Weight decay | 1e-4 | 1e-4 |
| 预估训练时间 | ~18h | ~36h |

**产出**：Baseline AP / AP_r / AP_c / AP_f / AP50 / AP75 表格

### Phase 3：Learning Rate 扫描（3-4 天）— P0 核心实验

**每个模型跑 3 个额外 lr 值**（baseline 作为第 4 个点）。

| 模型 | lr 扫描范围 |
|------|-----------|
| YOLO-World-L | 1e-4, 2e-4, **5e-4 (baseline)**, 1e-3 |
| Grounding DINO-B | 5e-5, **1e-4 (baseline)**, 2e-4, 5e-4 |

**预期发现**：
- YOLO-World 最优 lr 更高（Late fusion 梯度不经过文本编码器，优化更独立）
- Grounding DINO 最优 lr 范围更窄（Deep fusion 的 cross-attention 对 lr 更敏感）

**产出**：lr vs AP 折线图 × 2，包含 head/mid/tail 分层曲线

### Phase 4：Batch Size 扫描 + lr-bs 热力图（2-3 天）— P1

在 Phase 3 找到的最优 lr 基础上，各跑 2 个额外 bs 值（baseline bs 作为第 3 个点）。

| 模型 | bs 扫描范围 |
|------|-----------|
| YOLO-World-L | 8, **16 (baseline)**, 32 |
| Grounding DINO-B | **4 (baseline)**, 8, 16 (L40 显存允许) |

结合 Phase 3 的 lr 数据，不再跑完整网格。用已有 3 lr + 2 bs 绘制近似的 lr-bs 热力图（每个模型 6-9 个数据点）。

**预期发现**：
- YOLO-World 在大 bs 下持续受益（batch norm 统计更稳定）
- Grounding DINO 对 bs 相对不敏感（Transformer decoder 处理 object query，batch 维度影响小）
- 两个模型的最优 (lr, bs) 区域位置和形状有本质差异

**产出**：lr-bs 热力图 × 2

### Phase 5：模型特定参数（2 天）— P2（时间不够可砍）

| 模型 | 参数 | 扫描范围 | 跑几次 |
|------|------|----------|--------|
| YOLO-World | NMS threshold | [0.45, 0.50, 0.55, 0.60, 0.65] | 0（仅推理参数调整） |
| YOLO-World | Warmup steps | [0, 500, 1000] | 2（额外训练） |
| Grounding DINO | Query number | [300, 600, 900] | 0（仅推理参数调整） |
| Grounding DINO | Warmup steps | [0, 500, 1000, 2000] | 3（额外训练） |

**NMS/Query 调整不需要重新训练**，只需在最优 checkpoint 上跑多次推理。

**Warmup 扫描**需要重新训练，其中 Grounding DINO 的 warmup 分析是重点（6 层 cross-attention 在初始化时是随机的，没有 warmup 可能破坏 BERT 语义结构）。

### Phase 6：综合评估与分析（3 天）

#### 6.1 LVIS 分层评估
- 在最优超参数下，报告 AP / AP_r / AP_c / AP_f / AP50 / AP75
- 重点分析 AP_r（稀有类别）：比较两个模型在仅 1-10 张训练样本的类别上的表现
- **核心假设**：YOLO-World 的 CLIP 对稀有类别有 zero-shot 先验优势；Grounding DINO 的 BERT 需要足够样本学习跨模态对齐，稀有类别样本不足

#### 6.2 Zero-shot COCO 评估
- 两个模型在 LVIS 上 fine-tune 后，直接在 COCO 2017 val 上推理
- 计算性能衰减率 = COCO_AP / LVIS_AP_shared × 100%
- 分析 LVIS-COCO 共享 ~80 类的 per-class AP 差异
- **核心假设**：YOLO-World 的 CLIP 在跨数据集场景泛化更好（视觉-语言天然对齐）；Grounding DINO 的 BERT 可能过拟合 LVIS 的文本分布

#### 6.3 Error Analysis（TIDE，时间允许）
使用 TIDE 工具将错误分为 6 类：Cls / Loc / Both / Dupe / Bkg / Miss

**预期发现**：
- YOLO-World Cls 错误更多（late fusion 文本区分能力弱）
- Grounding DINO Loc 错误更少（cross-attention 使 query 精准定位到文本描述区域）
- Grounding DINO Miss 更少（Transformer 对密集场景的全图建模更好）

#### 6.4 文本编码器归因分析
- CLIP（YOLO-World）：vision-language 联合训练，语义与视觉天然对齐
- BERT（Grounding DINO）：纯文本预训练，需要训练中学习对齐
- 分析两者在细粒度类别（如"border collie" vs "australian shepherd"）上的区分能力
- 分析 LVIS 类别名在 CLIP 和 BERT embedding 空间中的几何结构差异

---

## 评估指标

| 指标 | 说明 |
|------|------|
| AP | Average Precision, IoU=0.50:0.95 |
| AP50 / AP75 | 单 IoU 阈值下的 AP |
| AP_r / AP_c / AP_f | LVIS 特有：rare / common / frequent |
| AR | Average Recall |
| COCO AP (zero-shot) | 不 fine-tune，直接推理 |
| 性能衰减率 | (COCO_AP / LVIS_AP_shared) × 100% |

---

## 归因分析框架（核心贡献）

| 观察 | 架构解释 |
|------|----------|
| YOLO-World 最优 lr > Grounding DINO | Late fusion 梯度不经过文本编码器，视觉部分优化更独立 |
| Grounding DINO 最优 lr 范围更窄 | Deep fusion 梯度路径长，lr 太大会破坏 BERT 语义 |
| YOLO-World 大 bs 持续受益 | CNN + batch norm 依赖 batch statistics |
| Grounding DINO bs 不敏感 | Transformer decoder 处理 object query，与 batch 维度解耦 |
| YOLO-World AP_r 可能更高 | CLIP 对稀有类别有 zero-shot 先验，1-10 张图足够 |
| Grounding DINO AP_f 更高 | 充裕样本让 cross-attention 学好细粒度对齐 |
| YOLO-World Cls 错误多 | 文本 embedding 无法根据视觉上下文动态调整 |
| Grounding DINO Loc 错误少 | cross-attention 持续注入文本信息，定位更准 |

---

## 工作量与时间线

| 日期 | 天数 | 阶段 | 关键任务 | 并行策略 |
|------|------|------|---------|---------|
| 6.5-6.7 | 1-3 | 环境搭建 | 数据下载 + 环境配置 + 代码调试 | 下载时写脚本 |
| 6.7-6.9 | 3-5 | Baseline | YOLO-World baseline（18h）+ GDINO baseline（36h） | YOLO 跑完立即开始 GDINO |
| 6.9-6.13 | 5-9 | 超参数扫描 | lr 扫描 × 6 + bs 扫描 × 4 | 先跑 YOLO（快），后跑 GDINO |
| 6.13-6.15 | 9-11 | 特定参数 | Warmup/query 等 | 此阶段可砍 |
| 6.15-6.18 | 11-14 | 评估分析 | LVIS 分层 + COCO zero-shot + TIDE + 可视化 | 在本地写报告 |
| 6.18-6.21 | 14-16 | 报告撰写 | 文字 + 图表 + 排版 | 全力冲刺 |

### 训练时间估算

| 实验 | 单次时长 | 次数 | 总时长 |
|------|---------|------|--------|
| YOLO-World baseline | ~18h | 1 | 18h |
| GDINO baseline | ~36h | 1 | 36h |
| YOLO-World lr 扫描 | ~18h | 3 | 54h |
| GDINO lr 扫描 | ~36h | 3 | 108h |
| YOLO-World bs 扫描 | ~18h | 2 | 36h |
| GDINO bs 扫描 | ~36h | 2 | 72h |
| Warmup 扫描（可选） | ~18-36h | 2-3 | 36-108h |
| **合计（不含 warmup）** | | **12 次** | **~324h ≈ 13.5 天** |

**关键策略**：
- YOLO-World 训练快（18h），GDINO 慢（36h），先跑 YOLO 让它在白天完成，晚上提交 GDINO
- 同一模型的不同 lr 可以先后提交，利用 48h 窗口
- l40gpu001 有 8 个 GPU，目前 5 个被占用。如果有多 GPU 空闲，可以考虑单作业多 GPU 加速（如 GDINO 用 2 GPU，训练时间减半为 ~18h）。但优先保证能跑起来，不强依赖多 GPU
- 如果发现 l40gpu001/002 排队严重，可考虑 rtx2080ti 分区跑 YOLO-World（模型更轻，显存够用）

---

## 执行策略

### 优先级排序

| 优先级 | 实验 | 不做的后果 |
|--------|------|-----------|
| **P0 必须** | 环境搭建 + 数据准备 | 什么都做不了 |
| **P0 必须** | Baseline × 2 | 没有 baseline 归因分析无从谈起 |
| **P0 必须** | lr 扫描 × 6 | 最核心的超参数对比，支撑"融合深度→lr 敏感性"分析 |
| **P0 必须** | LVIS 分层评估 | 支撑"长尾泛化"核心论点 |
| **P1 重要** | bs 扫描 × 4 + lr-bs 热力图 | 支撑"CNN vs Transformer 对 bs 敏感性"分析 |
| **P1 重要** | Zero-shot COCO | 独立加分章节，跨数据集泛化分析 |
| **P2 增强** | Warmup 扫描 | 主要对 GDINO 有意义，加强分析深度 |
| **P3 锦上添花** | NMS/Query 调整 | 不训练，仅改推理参数，成本极低 |
| **P3 锦上添花** | TIDE Error Analysis | 烧时间但可视化效果好 |

### 如果时间不够的砍法

1. 先砍 Phase 5 warmup 训练（节省 2-3 天）
2. 再砍 Phase 4 bs 扫描的一半（YOLO 只跑 2 个 bs，GDINO 只跑 2 个 bs）
3. 再不行，COCO zero-shot 只做推理不做深度分析
4. 底线保证：Baseline × 2 + lr 扫描 × 6 + LVIS 分层评估（约 7 天训练 + 3 天分析）

### 如果时间充裕的扩展方向

以下按**单人投入产出比**排序。每个扩展都标注了预计耗时和加分潜力，方便按剩余时间灵活选择。

#### E1：加入 Detic（两阶段解耦）作为第三模型 — 约 2 天训练 + 1 天分析

Detic 的融合方式位于 YOLO-World 和 Grounding DINO 之间——RPN 纯视觉，分类头用 CLIP embeddings 替换。加入 Detic 后，三个模型的融合深度形成完整谱系（浅→中→深），报告论点的说服力大幅提升。

**不需要从头训练 12 epochs**：Detic 的 RPN 基于 Faster R-CNN 预训练，收敛很快，6 epochs 即可达到可用效果（约 18h）。

**额外分析价值**：
- Detic 的 RPN/分类头 lr ratio 是最独特的分析维度（其他两个模型没有这个解耦设计）
- 对"融合深度→超参数敏感性"假说的中间点验证
- AP_r 可能最高（CLIP zero-shot + RoI 精准定位），形成三条差异化曲线

**前提**：核心 12 组实验已跑完，有至少 2 天 GPU 时间。

#### E2：TIDE 完整 Error Analysis — 约 0.5 天（纯推理+分析，不需训练）

在最优 checkpoint 上跑 TIDE 工具，将错误分为 Cls / Loc / Both / Dupe / Bkg / Miss 六类。不需要任何重新训练。

**报告价值**：
- 三个饼图（或两模型并排）直观展示错误类型分布的架构差异
- 验证核心假说：YOLO-World Cls 错多、GDINO Loc 错少
- 可独立成 4.4.7 小节，增强实验章节的完整感

#### E3：文本编码器 Embedding 空间可视化 — 约 0.5 天（纯分析，不需训练）

用 t-SNE/UMAP 对 LVIS 1203 类别的 CLIP embedding 和 BERT embedding 分别降维可视化。

**分析内容**：
- CLIP 的类别 embedding 按语义（动物/车辆/食物等）自然聚类，且聚类紧致（视觉联合训练的结果）
- BERT 的类别 embedding 聚类更松散，某些细粒度类别在语义空间中靠得太近（纯文本训练的局限）
- 在两张图上标注稀有类别位置：哪些稀有类别在 CLIP 空间中是"孤岛"（容易被漏检），哪些在 BERT 空间中被常见类别"吞没"（容易被误分类）
- 将 TIDE 的 Cls 混淆矩阵与 embedding 空间距离做关联分析

**报告价值**：为"为什么 CLIP 对稀有类别更好 / BERT 对细粒度混淆更多"提供 embedding 级别的解释，比纯数字分析更有洞察。

#### E4：Scheduler 对比实验 — 约 2 天训练

两个模型各跑 cosine vs step vs constant 三种 scheduler（共 4 组额外训练）。

**分析价值**：
- 验证融合深度与 scheduler 选择的关系假设
- cosine 对 deep fusion 更重要（平稳衰减避免 cross-attention 震荡）
- 但 scheduler 的影响通常小于 lr/bs，属于锦上添花

#### E5：ODinW 多域泛化测试 — 约 1 天（纯推理，不需训练）

在 ODinW（Object Detection in the Wild）基准上评测 LVIS-tuned 模型。ODinW 包含 35 个不同域的数据集（医学影像、航拍、水下、卡通等），直接测试开放词汇检测的"真·泛化能力"。

**分析内容**：
- 哪类域偏移对哪种融合方式影响最大
- CLIP（YOLO-World）在视觉风格变化大的域上可能更鲁棒（CLIP 训练数据覆盖广）
- BERT（GDINO）在需要精确文本理解的域上可能更好

**前提**：ODinW 数据量较大（~30 个数据集），下载和预处理需提前规划。

#### E6：训练效率对比 — 约 0.2 天（已有数据）

从训练日志中提取：收敛速度（AP vs wall-clock time）、显存利用率、每 epoch 时间。作为独立小节讨论"实际落地时该选哪种架构"。

#### E7：Presentation Demo 视频 — 约 1 天

录制两个模型在相同图片上的推理演示（并排显示），直观展示检测差异。尤其关注：
- 同时正确检测的类别
- 一个检测到另一个漏掉的类别（尤其是稀有类别）
- 框的质量差异（GDINO 框更紧？YOLO 框更松？）

**展示价值**：15 分钟 presentation 的开场素材，比表格更能抓住注意力。

#### 扩展优先级汇总

| 优先级 | 扩展 | 额外耗时 | 需训练 | 加分潜力 | 何时启动 |
|--------|------|---------|--------|---------|---------|
| **高** | E2 TIDE Error Analysis | 0.5 天 | 否 | 高 | Phase 6 自然做 |
| **高** | E3 Embedding 可视化 | 0.5 天 | 否 | 高 | Phase 6 自然做 |
| **中** | E1 加入 Detic | 3 天 | 是 (18h) | 很高 | 核心实验完成后 |
| **中** | E6 训练效率对比 | 0.2 天 | 否 | 中 | Phase 6 顺手做 |
| **中** | E7 Demo 视频 | 1 天 | 否 | 中 | 报告写完后 |
| **低** | E4 Scheduler 对比 | 2 天 | 是 (72h) | 低 | E1 之后 |
| **低** | E5 ODinW 泛化测试 | 1 天 | 否 | 中 | E1 之后 |

**建议节奏**：
- Phase 6 分析阶段顺手把 E2 + E3 + E6 做了（总共约 1 天，零训练成本）
- 如果 6.15 前核心实验全跑完且有 GPU 空闲，启动 E1（Detic）
- E4/E5 除非提前 5 天以上完成所有核心实验，否则不建议启动
- E7 在报告初稿完成后、做 PPT 时顺手做

---

## 报告结构

| 章节 | 建议页数 | 核心内容 | 占分 |
|------|---------|----------|------|
| **1. Introduction** | 1-2 页 | OVOD 背景 + 融合深度谱系 + 三个研究问题 | 10 |
| **2. Related Work** | 1-2 页 | OVOD 三大范式 + LVIS benchmark + 超参数研究现状 | 5 |
| **3. Method** | 3-4 页 | 两个模型的架构细节 + 融合机制对比（统一视角：文本如何参与检测）+ 关键差异分析 | 10 |
| **4. Experiments** | 5-7 页 | 见下文 | 20 |
| **5. Conclusion** | 0.5-1 页 | 核心发现 + 局限性 + 未来方向 | 5 |
| **References** | 1 页 | 所有引用 | 5 |
| **Clarity** | — | 图表清晰 + 逻辑连贯 | 5 |

**实验章节细分**：

- **4.1 Datasets**：LVIS + COCO 介绍，频率分布可视化
- **4.2 Implementation Details**：超参数、框架、训练 trick
- **4.3 Metrics**：AP / AP_r / AP_c / AP_f / AR
- **4.4 Results**：
  - 4.4.1 Baseline 对比表
  - 4.4.2 lr 扫描结果（折线图 × 2，含 head/mid/tail 分层）
  - 4.4.3 lr-bs 热力图 × 2
  - 4.4.4 Head/Mid/Tail 分层分析（核心分析章节）
  - 4.4.5 Model-Specific 参数分析（NMS, Query）
  - 4.4.6 Zero-shot COCO 评估
  - 4.4.7 Error Analysis（TIDE，时间允许）
- **4.5 Discussion**：连接架构差异和实验观察，归因分析

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| L40 排队严重（5 个 GPU 已被占） | 中 | 训练延迟 | 先用 rtx2080ti 跑 YOLO-World；监控 l40 空闲情况 |
| GDINO 官方代码跑不通 | 中 | 卡在 Phase 1 | 用 HuggingFace `transformers` 版本或 MMDetection-GroundingDINO |
| GDINO 36h 训练 OOM 或发散 | 中 | 浪费 1.5 天 | 先跑 3 epochs 验证 loss 下降，确认没问题再跑完整 12 epochs |
| YOLO-World 在大 bs=32 时 OOM | 低 | 少一个 bs 数据点 | 降 bs 或用 gradient accumulation 模拟大 bs |
| 48h 不够 GDINO 完整训练 | 低 | 作业被杀 | checkpoint 续训机制（每 epoch 保存，resume 时自动检测） |
| l40gpu002 只有 3 GPU 且 2 个已被占 | 高 | 只能等 | 优先使用 l40gpu001，它 8 GPU 中 5 被占，还有 3 个 |
| 时间不够 | 高 | 砍实验 | 严格按 P0→P1→P2 顺序执行，随时准备砍 |

---

## 当前服务器状态（2026-06-05）

| 项目 | 状态 |
|------|------|
| L40 GPU | l40gpu001（3/8 空闲）、l40gpu002（1/3 空闲） |
| 作业时限 | 48h（a100 分区） |
| CUDA | 登录节点无 CUDA，GPU 节点有（需确认版本） |
| Conda | 可用（系统 `/opt/ohpc/pub/apps/anaconda3` + 用户 `~/miniconda3`） |
| 磁盘 | 28TB 可用（/home 网络存储） |
| 数据集 | `~/datasets/lvis/` 和 `~/datasets/coco/` 目录已建，数据未下载 |
| 模型代码 | `~/cv_project/repos/` 目录已建，未 clone |
| 现有 conda 环境 | opentslm, py39, yolo（旧环境，建议重建） |

---

## 下一步行动（今天 6.5 立即开始）

1. SSH 到服务器，开始下载 LVIS + COCO 数据集（下载的同时做第 2 步）
2. 创建 conda 环境，确认 CUDA/PyTorch 版本
3. Clone YOLO-World 和 Grounding DINO 代码
4. 验证推理 pipeline 能跑通（用预训练权重 + 单张图片）
5. 编写 sbatch 脚本模板
6. 提交第一个 baseline 训练作业
