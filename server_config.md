# 服务器配置信息

## 连接方式

| 项目 | 详情 |
|------|------|
| 登录节点 | `172.18.34.26` |
| 端口 | `10022` |
| 用户名 | `cse12210210` |
| 密码 | `rb6/aYMRAT#16` |
| 集群 | 集群（二） |

```bash
ssh -p 10022 cse12210210@172.18.34.26
# 密码: rb6/aYMRAT#16
```

> 注意：端口号前需要加 `-p`。备用节点 `172.18.34.25` 当前不可用。

## 硬件资源

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA L40 × 1（46GB 显存） |
| GPU 分区 | `a100`（l40gpu001, l40gpu002） |
| QOS | `a100` |
| 作业时限 | 单次作业 24h，分区 48h |
| 并发作业 | 最多 1 个 |
| 共享存储 | 84TB（~28TB 可用） |
| Home 目录 | `/home/turing_lab/cse12210210` |
| CUDA Driver | 575.57.08 / CUDA 12.9 |
| PyTorch CUDA | 12.1（通过 `cu121` wheel 安装） |

## Conda 环境

### yolow — YOLO-World-L 训练环境

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| TorchVision | 0.20.1+cu121 |
| Ultralytics | 8.4.60 |
| OpenCV | 4.13.0.92 |
| NumPy | 2.2.6 |
| pycocotools | 2.0.11 |
| wandb | 0.27.2 |
| tensorboard | 2.20.0 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |

激活方式：
```bash
source /opt/ohpc/pub/apps/anaconda3/bin/activate yolow
# 或
module load anaconda/4.12.0 && conda activate yolow
```

### detic — Detic 训练环境（Detectron2）

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| TorchVision | 0.20.1+cu121 |
| Detectron2 | 待安装 |
| CLIP | 待安装 |
| pycocotools | 2.0.11 |

激活方式：
```bash
source /opt/ohpc/pub/apps/anaconda3/bin/activate detic
# 或
module load anaconda/4.12.0 && conda activate detic
```

> Detic 基于 Facebook Detectron2 框架，需要从源码编译或使用预编译 wheel。

## 数据集

| 组件 | 位置 | 数量 | 大小 |
|------|------|------|------|
| COCO 标注 | `~/cv_project/datasets/coco/annotations/annotations/` | 6 个 JSON | 242 MB |
| COCO val2017 | `~/cv_project/datasets/coco/val2017/` | 5,000 张 | 778 MB |
| COCO train2017 | `~/cv_project/datasets/coco/train2017/` | 118,287 张 | 19 GB |
| COCO 总大小 | `~/cv_project/datasets/coco/` | — | 39 GB |
| COCO yaml (YOLO) | `~/cv_project/datasets/coco/coco.yaml` | 训练脚本自动生成 | — |

数据集目录结构：
```
~/cv_project/datasets/coco/
├── annotations/
│   ├── annotations_trainval2017.zip
│   └── annotations/
│       ├── instances_train2017.json
│       ├── instances_val2017.json
│       ├── captions_train2017.json
│       ├── captions_val2017.json
│       ├── person_keypoints_train2017.json
│       └── person_keypoints_val2017.json
├── train2017/
│   ├── train2017.zip (19 GB)
│   └── *.jpg (118,287 张，图片直接在 train2017/ 下)
├── val2017/
│   ├── val2017.zip (778 MB)
│   └── *.jpg (5,000 张，图片直接在 val2017/ 下)
└── coco.yaml
```

> **路径注意**：图片直接在 `train2017/` 和 `val2017/` 目录下（无额外嵌套）。训练脚本中 YOLO 的 data yaml 使用 `train: train2017` 和 `val: val2017`；GDINO 的 train_root 和 val_root 同理。

## 模型权重

| 模型 | 位置 | 大小 |
|------|------|------|
| YOLO-World-L | `~/cv_project/models/yolov8l-worldv2.pt` | 90 MB |
| Detic (ResNet-50) | `~/cv_project/models/detic/` | 待下载 |

Detic 模型文件（来自 Facebook Research）：
```
~/cv_project/models/detic/
├── Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
└── (CLIP 权重 + Detectron2 config)
```

## 训练脚本

| 脚本 | 位置 | 用途 |
|------|------|------|
| `train_yolow.py` | `~/cv_project/scripts/` | YOLO-World COCO 训练 |
| `train_detic.py` | `~/cv_project/scripts/` | Detic COCO 训练（Detectron2） |
| `sbatch_yolow.sh` | `~/cv_project/scripts/` | YOLO sbatch 提交（24h 时限） |
| `sbatch_detic.sh` | `~/cv_project/scripts/` | Detic sbatch 提交 |
| `test_env.py` | `~/cv_project/` | GPU 环境验证脚本 |

## 训练基线超参数

| 参数 | YOLO-World-L | Detic (ResNet-50) |
|------|-------------|-------------------|
| Learning rate | 2e-4 | 2e-4（待确认） |
| Batch size | 8 | 8 |
| Epochs | 12 | 12 |
| Warmup steps | 1000 | 1000 |
| Optimizer | AdamW | SGD（待确认） |
| Weight decay | 0.05 | 1e-4 |
| Image size | 640 | 640 |
| Scheduler | cosine | cosine |
| Gradient accumulation | — | — |
| 预计每 epoch 时间 | ~0.45h (27min，实测) | ~0.3h（预估） |
| 预计总训练时间 | ~5.5h | ~3.6h（预估） |
| 作业策略 | 单次完成 | 单次完成 |

## 提交作业

```bash
# YOLO-World（~5.5h，单次完成）
sbatch ~/cv_project/scripts/sbatch_yolow.sh
LR=5e-4 BS=8 NAME=yolow_lr5e4 sbatch ~/cv_project/scripts/sbatch_yolow.sh

# Detic（~3.6h，单次完成）
sbatch ~/cv_project/scripts/sbatch_detic.sh
LR=5e-4 NAME=detic_lr5e4 sbatch ~/cv_project/scripts/sbatch_detic.sh

# 交互式 GPU 作业（测试用）
srun -p a100 --qos=a100 --gres=gpu:1 -n 1 --mem=16G -t 00:30:00 --pty bash
```

## 已知问题

1. **transformers 版本**：gdino 环境必须使用 `transformers==4.47.1`，加载模型需用 `GroundingDinoForObjectDetection.from_pretrained()` 而非 `AutoModelForObjectDetection`
2. **备用登录节点**：`172.18.34.25` 当前不可用（Connection closed）
3. **a100 分区名称**：该分区实际 GPU 为 L40，不是 A100
4. **conda 模块**：系统仅提供 `anaconda/4.12.0`，conda 版本较老，推荐主要用 pip 装包
5. **pip 镜像**：已配置 tsinghua 镜像（`https://pypi.tuna.tsinghua.edu.cn/simple`）
