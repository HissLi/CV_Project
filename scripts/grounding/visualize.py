"""Visualize grounding predictions against GT boxes."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from PIL import Image, ImageDraw


def _draw_box(draw: ImageDraw.ImageDraw, box_xyxy, color: str, width: int = 3):
    x1, y1, x2, y2 = box_xyxy
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def _safe_name(row: Dict) -> str:
    dataset = row["dataset"]
    ref_id = row["ref_id"]
    sent_id = row["sent_id"]
    return f"{dataset}_ref{ref_id}_sent{sent_id}.png"


def load_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_rows(rows: List[Dict]) -> Dict:
    return {(row["dataset"], row["ref_id"], row["sent_id"]): row for row in rows}


def pick_examples(gdino_rows: List[Dict], owl_rows: List[Dict], topk_success: int, topk_fail: int):
    owl_index = index_rows(owl_rows)
    success = []
    fail = []
    for row in gdino_rows:
        key = (row["dataset"], row["ref_id"], row["sent_id"])
        owl = owl_index.get(key)
        if not owl:
            continue
        gd_hit = float(row.get("iou", 0.0)) >= 0.5
        ow_hit = float(owl.get("iou", 0.0)) >= 0.5
        if gd_hit and not ow_hit and len(success) < topk_success:
            success.append((row, owl))
        elif (not gd_hit) and (not ow_hit) and len(fail) < topk_fail:
            fail.append((row, owl))
        if len(success) >= topk_success and len(fail) >= topk_fail:
            break
    return success, fail


def draw_and_save(row: Dict, out_path: str):
    if not os.path.exists(row["image_path"]):
        return False
    image = Image.open(row["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    _draw_box(draw, row["gt_box_xyxy"], color="green")
    pred = row.get("pred_box_xyxy")
    if pred:
        _draw_box(draw, pred, color="red")
    text = f'{row["sentence"]}\nIoU={row.get("iou", 0.0):.3f}'
    draw.text((8, 8), text, fill="yellow")
    image.save(out_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdino_json", required=True)
    parser.add_argument("--owlvit_json", required=True)
    parser.add_argument("--output_dir", default="~/cv_project/results/grounding_vis")
    parser.add_argument("--topk_success", type=int, default=5)
    parser.add_argument("--topk_fail", type=int, default=5)
    args = parser.parse_args()

    gdino_rows = load_rows(os.path.expanduser(args.gdino_json))
    owl_rows = load_rows(os.path.expanduser(args.owlvit_json))
    output_dir = os.path.expanduser(args.output_dir)
    success_dir = os.path.join(output_dir, "success")
    fail_dir = os.path.join(output_dir, "fail")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    success, fail = pick_examples(gdino_rows, owl_rows, args.topk_success, args.topk_fail)
    saved = 0
    for gdino_row, _ in success:
        out_path = os.path.join(success_dir, _safe_name(gdino_row))
        saved += int(draw_and_save(gdino_row, out_path))
    for gdino_row, _ in fail:
        out_path = os.path.join(fail_dir, _safe_name(gdino_row))
        saved += int(draw_and_save(gdino_row, out_path))

    print(f"Saved {saved} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
