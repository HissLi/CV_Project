"""Bounding-box metrics for visual grounding."""

from __future__ import annotations

from typing import Dict, Iterable, List


def box_iou(box1_xyxy: List[float], box2_xyxy: List[float]) -> float:
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area1 = max(0.0, box1_xyxy[2] - box1_xyxy[0]) * max(0.0, box1_xyxy[3] - box1_xyxy[1])
    area2 = max(0.0, box2_xyxy[2] - box2_xyxy[0]) * max(0.0, box2_xyxy[3] - box2_xyxy[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def evaluate_predictions(pairs: Iterable[Dict]) -> Dict:
    rows = list(pairs)
    total = len(rows)
    if total == 0:
        return {"samples": 0, "acc_at_05": 0.0, "acc_at_075": 0.0, "mean_iou": 0.0}

    ious = [float(row["iou"]) for row in rows]
    hit_05 = sum(1 for x in ious if x >= 0.5)
    hit_075 = sum(1 for x in ious if x >= 0.75)
    return {
        "samples": total,
        "acc_at_05": hit_05 / total,
        "acc_at_075": hit_075 / total,
        "mean_iou": sum(ious) / total,
    }
