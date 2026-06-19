"""OWL-ViT zero-shot phrase grounding on RefCOCO-family datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import OwlViTForObjectDetection, OwlViTProcessor


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scripts.grounding.metrics import box_iou, evaluate_predictions
from scripts.grounding.refcoco_dataset import SPLIT_BY_MAP, load_refcoco_samples
from scripts.grounding.device_utils import get_torch_device, load_hf_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="refcoco,refcoco+,refcocog")
    parser.add_argument("--split", default="val")
    parser.add_argument("--model_dir", default="~/cv_project/models/owlvit")
    parser.add_argument("--hf_model_id", default="google/owlvit-base-patch32")
    parser.add_argument("--refer_data_root", default="~/cv_project/datasets/refer/data")
    parser.add_argument("--refer_repo_root", default="~/cv_project/datasets/refer/refer")
    parser.add_argument("--coco2014_root", default="~/cv_project/datasets/mscoco2014")
    parser.add_argument("--coco2017_root", default="~/cv_project/datasets/coco")
    parser.add_argument("--output_dir", default="~/cv_project/results/owlvit_refcoco_zeroshot")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def run_dataset(model, processor, dataset: str, split: str, args, device: str) -> Dict:
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
    bsz = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(samples), bsz), desc=f"OWLViT {dataset}/{split}"):
        batch = samples[start : start + bsz]
        images = []
        texts = []
        kept = []
        for sample in batch:
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
            images.append(Image.open(sample.image_path).convert("RGB"))
            texts.append(sample.sentence.strip())
            kept.append(sample)

        if not kept:
            continue

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([im.size[::-1] for im in images], device=device)
        processed = processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=args.threshold,
            target_sizes=target_sizes,
            text_labels=[[t] for t in texts],
        )

        for sample, pred in zip(kept, processed):
            if pred["scores"].numel() == 0:
                pred_box, score, iou = None, 0.0, 0.0
            else:
                idx = int(torch.argmax(pred["scores"]).item())
                pred_box = [float(v) for v in pred["boxes"][idx].detach().cpu().tolist()]
                score = float(pred["scores"][idx].detach().cpu().item())
                iou = box_iou(pred_box, sample.gt_box_xyxy)

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
    print(f"Loading OWL-ViT from {model_dir}")
    model, processor = load_hf_model(
        OwlViTForObjectDetection,
        OwlViTProcessor,
        model_dir,
        args.hf_model_id,
    )
    model = model.to(device)
    model.eval()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    aggregated = {
        "model": "owlvit-base-patch32",
        "mode": "zero-shot",
        "task": "referring expression grounding",
        "split": args.split,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "datasets": {},
    }

    all_rows = []
    for dataset in datasets:
        result = run_dataset(model, processor, dataset, args.split, args, device)
        dataset_rows = result["rows"]
        all_rows.extend(dataset_rows)
        with open(os.path.join(output_dir, f"{dataset}_{args.split}.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_rows, f, ensure_ascii=False)
        aggregated["datasets"][dataset] = result["summary"]
        print(f"[{dataset}] {result['summary']}")

    aggregated["overall"] = evaluate_predictions(all_rows)
    with open(os.path.join(output_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {os.path.join(output_dir, 'params.json')}")
    print(f"Overall: {aggregated['overall']}")


if __name__ == "__main__":
    main()
