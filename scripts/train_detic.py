"""Detic fine-tuning on COCO 2017 using Detectron2.

Reference: https://github.com/facebookresearch/Detic
"""

import argparse, os, sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="~/cv_project/datasets/coco")
    parser.add_argument("--output_dir", default="results/detic")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    print("Detic training with Detectron2")
    print(f"Config: epochs={args.epochs}, bs={args.batch}, lr={args.lr}")
    # TODO: Implement Detectron2-based training after environment setup
    print("Training not yet implemented - need Detectron2 environment first")


if __name__ == "__main__":
    main()
