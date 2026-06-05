# 实验跟踪记录

## 实验总览

| 阶段 | 实验 | 状态 | 训练数 |
|------|------|------|--------|
| Phase 2 | Baseline × 2 | 进行中 | 2 |
| Phase 3 | lr 扫描 × 6 | 待开始 | 6 |
| Phase 4 | bs 扫描 × 4 | 待开始 | 4 |
| Phase 5 | 模型特定参数 | 待开始 | 0-4 |
| Phase 6 | 评估分析 | 待开始 | 0 |

---

## Phase 2: Baseline 复现

### 实验设计

两个模型各跑 1 次 baseline，使用官方推荐超参数：
- YOLO-World-L: lr=2e-4, bs=8, epochs=12, warmup=1000 steps, optimizer=AdamW
- Grounding DINO-B: lr=1e-4, bs=4, epochs=12, warmup=1000 steps, optimizer=AdamW

---

### 训练记录

#### Run #1: YOLO-World-L Baseline

| 项目 | 详情 |
|------|------|
| **作业 ID** | - |
| **开始时间** | - |
| **结束时间** | - |
| **状态** | 待提交 |

**超参数：**
| 参数 | 值 |
|------|-----|
| Model | yolov8l-worldv2.pt (ultralytics pretrained) |
| Epochs | 12 |
| Batch Size | 8 |
| Learning Rate | 2e-4 |
| Warmup | 1000 steps |
| Optimizer | AdamW |
| Weight Decay | 0.05 |
| Image Size | 640 |
| Scheduler | cosine |

**结果：**
| 指标 | 值 |
|------|-----|
| AP | - |
| AP_r | - |
| AP_c | - |
| AP_f | - |
| AP50 | - |
| AP75 | - |

**文件位置：**
- 输出: `results/yolow_baseline/`
- 权重: `results/yolow_baseline/weights/best.pt`

---

#### Run #2: Grounding DINO-B Baseline

| 项目 | 详情 |
|------|------|
| **作业 ID** | - |
| **开始时间** | - |
| **结束时间** | - |
| **状态** | 待提交 |

**超参数：**
| 参数 | 值 |
|------|-----|
| Model | IDEA-Research/GroundingDINO (Swin-B) |
| Epochs | 12 |
| Batch Size | 4 |
| Learning Rate | 1e-4 |
| Warmup | 1000 steps |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| Image Size | 800 |
| Scheduler | cosine |

**结果：**
| 指标 | 值 |
|------|-----|
| AP | - |
| AP_r | - |
| AP_c | - |
| AP_f | - |
| AP50 | - |
| AP75 | - |

**文件位置：**
- 输出: `results/gdino_baseline/`
- 权重: `results/gdino_baseline/`
