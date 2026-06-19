"""Grounding DINO zero-shot RefCOCO-family evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, GroundingDinoForObjectDetection


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scripts.grounding.metrics import box_iou, evaluate_predictions
from scripts.grounding.refcoco_dataset import load_refcoco_samples, SPLIT_BY_MAP
from scripts.grounding.device_utils import get_torch_device, load_hf_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="refcoco,refcoco+,refcocog")
    parser.add_argument("--split", default="val")
    parser.add_argument("--model_dir", default="~/cv_project/models/gdino")
    parser.add_argument("--hf_model_id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--refer_data_root", default="~/cv_project/datasets/refer/data")
    parser.add_argument("--refer_repo_root", default="~/cv_project/datasets/refer/refer")
    parser.add_argument("--coco2014_root", default="~/cv_project/datasets/mscoco2014")
    parser.add_argument("--coco2017_root", default="~/cv_project/datasets/coco")
    parser.add_argument("--output_dir", default="~/cv_project/results/gdino_refcoco_zeroshot")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def infer_top1_box(model, processor, image: Image.Image, phrase: str, threshold: float, device: str):
    inputs = processor(images=image, text=phrase, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=device)
    pred = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    if pred["scores"].numel() == 0:
        return None, 0.0
    idx = int(torch.argmax(pred["scores"]).item())
    box = pred["boxes"][idx].detach().cpu().tolist()
    score = float(pred["scores"][idx].detach().cpu().item())
    return [float(v) for v in box], score


def run_dataset(
    model,
    processor,
    dataset: str,
    split: str,
    threshold: float,
    device: str,
    args,
) -> Dict:
    split_by = SPLIT_BY_MAP[dataset]
    samples = load_refcoco_samples(
        dataset=dataset,
        split=split,
        split_by=split_by,
        refer_data_root=args.refer_data_root,
        refer_repo_root=args.refer_repo_root,
        coco2014_root=args.coco2014_root,
        coco2017_root=args.coco2017_root,
        max_samples=args.max_samples,
    )
    print(f"[{dataset}] Loaded {len(samples)} samples")

    rows: List[Dict] = []
    for sample in tqdm(samples, desc=f"GDINO {dataset}/{split}"):
        if not os.path.exists(sample.image_path):
            rows.append(
                {
                    "dataset": sample.dataset,
                    "split": split,
                    "image_id": sample.image_id,
                    "ref_id": sample.ref_id,
                    "sent_id": sample.sent_id,
                    "sentence": sample.sentence,
                    "gt_box_xyxy": sample.gt_box_xyxy,
                    "pred_box_xyxy": None,
                    "score": 0.0,
                    "iou": 0.0,
                    "missing_image": True,
                }
            )
            continue

        image = Image.open(sample.image_path).convert("RGB")
        pred_box, score = infer_top1_box(
            model=model,
            processor=processor,
            image=image,
            phrase=sample.sentence.lower().strip(),
            threshold=threshold,
            device=device,
        )
        iou = 0.0 if pred_box is None else box_iou(pred_box, sample.gt_box_xyxy)
        rows.append(
            {
                "dataset": sample.dataset,
                "split": split,
                "image_id": sample.image_id,
                "ref_id": sample.ref_id,
                "sent_id": sample.sent_id,
                "sentence": sample.sentence,
                "gt_box_xyxy": sample.gt_box_xyxy,
                "pred_box_xyxy": pred_box,
                "score": score,
                "iou": iou,
                "hit_at_05": iou >= 0.5,
                "hit_at_075": iou >= 0.75,
            }
        )

    summary = evaluate_predictions(rows)
    summary["missing_images"] = sum(1 for row in rows if row.get("missing_image", False))
    return {"rows": rows, "summary": summary}


def main():
    args = parse_args()
    device = get_torch_device()
    model_dir = os.path.expanduser(args.model_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading model from {model_dir}")
    model, processor = load_hf_model(
        GroundingDinoForObjectDetection,
        AutoProcessor,
        model_dir,
        args.hf_model_id,
    )
    model = model.to(device)
    model.eval()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    aggregated = {
        "model": "grounding-dino-base",
        "mode": "zero-shot",
        "task": "referring expression grounding",
        "split": args.split,
        "threshold": args.threshold,
        "datasets": {},
    }

    all_rows = []
    for dataset in datasets:
        result = run_dataset(
            model=model,
            processor=processor,
            dataset=dataset,
            split=args.split,
            threshold=args.threshold,
            device=device,
            args=args,
        )
        dataset_rows = result["rows"]
        all_rows.extend(dataset_rows)

        out_json = os.path.join(output_dir, f"{dataset}_{args.split}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(dataset_rows, f, ensure_ascii=False)
        aggregated["datasets"][dataset] = result["summary"]
        print(f"[{dataset}] {result['summary']}")

    aggregated["overall"] = evaluate_predictions(all_rows)
    params_path = os.path.join(output_dir, "params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(aggregated, indent=2, ensure_ascii=False))

    print(f"Saved summary to {params_path}")
    print(f"Overall: {aggregated['overall']}")


if __name__ == "__main__":
    main()
