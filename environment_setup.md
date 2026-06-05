# 环境配置记录

## 服务器信息

| 项目 | 详情 |
|------|------|
| 登录节点 | login01 @ 172.18.34.26:10022 |
| 账号 | cse12212230 |
| GPU 分区 | a100 (l40gpu001: 8×L40 48GB, l40gpu002: 3×L40 48GB) |
| 作业时限 | 48h (a100 分区) |
| CUDA 版本 | 12.9 (驱动) / 12.1 (PyTorch) |
| 磁盘 | 84TB 网络存储，~28TB 可用 |
| 作业队列 | --qos=a100, 最多同时运行 1 个作业 |
| 包管理器 | conda (miniconda3) + pip (tsinghua mirror) |

## Conda 环境

### yolow (YOLO-World)

| 组件 | 版本 |
|------|------|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| TorchVision | 0.20.1+cu121 |
| Ultralytics | 8.4.60 |
| OpenCV | 4.13.0.92 |
| NumPy | 2.2.6 |
| wandb | 0.27.1 |
| tensorboard | 2.20.0 |
| pycocotools | 2.0.11 |
| lvis | 0.5.3 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |

### gdino (Grounding DINO)

| 组件 | 版本 |
|------|------|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| TorchVision | 0.20.1+cu121 |
| Transformers | 4.57.6 |
| Accelerate | 1.13.0 |
| Timm | 1.0.27 |
| OpenCV | 4.13.0.92 |
| pycocotools | 2.0.11 |
| lvis | 0.5.3 |

## 模型权重

| 模型 | 文件 | 大小 | 来源 |
|------|------|------|------|
| YOLO-World-L | `l_stage1-7d280586.pth` | 422 MB | HuggingFace wondervictor/YOLO-World-V2.1 |
| YOLO-World-L (ultralytics) | `yolov8l-worldv2.pt` | 90 MB | Ultralytics GitHub Releases |
| Grounding DINO-B | `groundingdino_swinb_cogcoor.pth` | 895 MB | GitHub IDEA-Research Releases v0.1.0-alpha2 |
| CLIP ViT-B-32 | `ViT-B-32.pt` | 338 MB | OpenAI Azure (本地下载后上传到 ~/.cache/clip/) |

## 数据集

| 数据集 | 图像数 | 路径 |
|--------|--------|------|
| COCO train2017 | 118,287 | `~/cv_project/datasets/coco/train2017/train2017/` |
| COCO val2017 | 5,000 | `~/cv_project/datasets/coco/val2017/val2017/` |
| LVIS v1 train | 100,170 | 标注: `~/cv_project/datasets/coco/lvis/lvis_v1_train.json` |
| LVIS v1 val | 19,626 | 标注: `~/cv_project/datasets/coco/lvis/lvis_v1_val.json` |
| LVIS YOLO labels | train: 99,388 / val: 19,626 | `~/cv_project/datasets/coco/lvis_yolo/labels/` |

## 数据目录结构

```
~/cv_project/
├── repos/
│   ├── YOLO-World/          # 官方代码仓库
│   ├── GroundingDINO/       # 官方代码仓库 (推理用)
│   ├── l_stage1-7d280586.pth    # YOLO-World-L 官方权重
│   └── groundingdino_swinb_cogcoor.pth  # GDINO-B 官方权重
├── datasets/
│   └── coco/
│       ├── train2017/train2017/  # 118,287 张训练图
│       ├── val2017/val2017/      # 5,000 张验证图
│       ├── annotations/          # COCO 标注
│       ├── lvis/                 # LVIS 标注
│       └── lvis_yolo/            # YOLO 格式 LVIS 标注 + dataset.yaml
├── scripts/                 # sbatch 训练脚本
├── results/                 # 训练输出
└── envs/                    # (空，使用 miniconda3 envs)
```

## CLIP 缓存修复

服务器从 GitHub 下载 CLIP 模型速度极慢（<500KB/s）。通过本地下载（338MB, ~2min）后 scp 上传到 `~/.cache/clip/ViT-B-32.pt` 解决。
