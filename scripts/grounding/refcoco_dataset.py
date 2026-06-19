"""RefCOCO family dataset loader for sentence-level grounding evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional


SPLIT_BY_MAP = {
    "refcoco": "unc",
    "refcoco+": "unc",
    "refcocog": "umd",
}


@dataclass
class RefSample:
    dataset: str
    split_by: str
    split: str
    image_id: int
    ref_id: int
    sent_id: int
    sentence: str
    image_path: str
    gt_box_xyxy: List[float]

    def to_json(self) -> Dict:
        return asdict(self)


def _xywh_to_xyxy(box_xywh: List[float]) -> List[float]:
    x, y, w, h = box_xywh
    return [float(x), float(y), float(x + w), float(y + h)]


def _resolve_split_by(dataset: str, split_by: Optional[str]) -> str:
    if split_by:
        return split_by
    if dataset not in SPLIT_BY_MAP:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return SPLIT_BY_MAP[dataset]


def _extract_phrase_from_prompt(text: str) -> str:
    # PaDT format: Please ... describes: "phrase".
    match = re.search(r'"(.+?)"', text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _resolve_image_path_from_name(image_name: str, coco2014_root: str, coco2017_root: str) -> str:
    candidates = [
        os.path.join(coco2014_root, image_name),
        os.path.join(coco2014_root, "val2014", image_name),
        os.path.join(coco2014_root, "train2014", image_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # COCO_train2014_000000376848.jpg -> 000000376848.jpg
    maybe_id = re.findall(r"(\d{12})\.jpg$", image_name)
    if maybe_id:
        name_2017 = f"{maybe_id[0]}.jpg"
        train2017 = os.path.join(coco2017_root, "train2017", name_2017)
        val2017 = os.path.join(coco2017_root, "val2017", name_2017)
        if os.path.exists(train2017):
            return train2017
        if os.path.exists(val2017):
            return val2017

    return os.path.join(coco2017_root, "train2017", image_name)


def _iter_padt_samples(
    dataset: str,
    split: str,
    split_by: str,
    refer_data_root: str,
    coco2014_root: str,
    coco2017_root: str,
    max_samples: Optional[int],
) -> Iterable[RefSample]:
    file_map = {
        "refcoco": "refcoco_val.json",
        "refcoco+": "refcoco+_val.json",
        "refcocog": "refcocog_val.json",
    }
    if split != "val":
        raise ValueError("PaDT fallback currently supports only val split.")
    padt_candidates = [
        os.path.join(refer_data_root, "padt", file_map[dataset]),
        os.path.join(os.path.dirname(refer_data_root), "padt", file_map[dataset]),
    ]
    jsonl_path = next((p for p in padt_candidates if os.path.exists(p)), padt_candidates[0])
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"PaDT fallback file not found. Tried: {padt_candidates}"
        )

    yielded = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            phrase = _extract_phrase_from_prompt(
                row.get("conversations", [{}])[0].get("value", "")
            )
            objects = row.get("objects", [])
            if not objects:
                continue
            obj = objects[0]
            gt = [float(v) for v in obj.get("bbox", [0.0, 0.0, 1.0, 1.0])]

            image_name = row.get("image", "")
            image_path = _resolve_image_path_from_name(image_name, coco2014_root, coco2017_root)
            image_id_digits = re.findall(r"(\d{12})\.jpg$", image_name)
            image_id = int(image_id_digits[0]) if image_id_digits else int(row.get("id", yielded))

            if gt[2] <= 1.001 and gt[3] <= 1.001:
                # Normalized xyxy -> absolute xyxy
                try:
                    from PIL import Image
                    with Image.open(image_path) as im:
                        w, h = im.size
                    gt = [gt[0] * w, gt[1] * h, gt[2] * w, gt[3] * h]
                except Exception:
                    pass

            yield RefSample(
                dataset=dataset,
                split_by=split_by,
                split=split,
                image_id=image_id,
                ref_id=int(row.get("id", yielded)),
                sent_id=int(row.get("id", yielded)),
                sentence=phrase,
                image_path=image_path,
                gt_box_xyxy=gt,
            )
            yielded += 1
            if max_samples is not None and yielded >= max_samples:
                return


def _get_refer_cls(refer_repo_root: str):
    external_path = os.path.join(refer_repo_root, "external")
    python_refer_path = os.path.join(refer_repo_root, "refer")
    for p in (external_path, python_refer_path):
        if p not in sys.path:
            sys.path.append(p)

    try:
        from refer import REFER  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Cannot import REFER. Ensure the refer repository exists and "
            "python path includes <refer_repo_root>/refer and external."
        ) from exc
    return REFER


def iter_refcoco_samples(
    dataset: str,
    split: str = "val",
    split_by: Optional[str] = None,
    refer_data_root: str = "~/cv_project/datasets/refer/data",
    refer_repo_root: str = "~/cv_project/datasets/refer/refer",
    coco2014_root: str = "~/cv_project/datasets/mscoco2014",
    coco2017_root: str = "~/cv_project/datasets/coco",
    max_samples: Optional[int] = None,
) -> Iterable[RefSample]:
    """Yield sentence-level RefCOCO samples with absolute image path and GT box."""

    refer_data_root = os.path.expanduser(refer_data_root)
    refer_repo_root = os.path.expanduser(refer_repo_root)
    coco2014_root = os.path.expanduser(coco2014_root)
    coco2017_root = os.path.expanduser(coco2017_root)
    split_by = _resolve_split_by(dataset, split_by)

    refer_data_dir = os.path.join(refer_data_root, dataset)
    if not os.path.isdir(refer_data_dir):
        # Fallback to PaDT jsonl format downloaded from HF mirror.
        yield from _iter_padt_samples(
            dataset=dataset,
            split=split,
            split_by=split_by,
            refer_data_root=refer_data_root,
            coco2014_root=coco2014_root,
            coco2017_root=coco2017_root,
            max_samples=max_samples,
        )
        return

    REFER = _get_refer_cls(refer_repo_root)
    refer_api = REFER(refer_data_root, dataset, split_by)

    ref_ids = refer_api.getRefIds(split=split)
    refs = refer_api.loadRefs(ref_ids)

    yielded = 0
    for ref in refs:
        ann = refer_api.Anns[ref["ann_id"]]
        image_id = ref["image_id"]
        image_info = refer_api.Imgs[image_id]
        image_path = os.path.join(coco2014_root, image_info["file_name"])
        if not os.path.exists(image_path):
            image_path = os.path.join(coco2014_root, "val2014", image_info["file_name"])
        if not os.path.exists(image_path):
            image_path = os.path.join(coco2014_root, "train2014", image_info["file_name"])
        if not os.path.exists(image_path):
            # RefCOCO comes from COCO2014; your server already has COCO2017.
            # Reuse the same image_id by switching to 12-digit jpg names.
            img_name_2017 = f"{int(image_id):012d}.jpg"
            train2017 = os.path.join(coco2017_root, "train2017", img_name_2017)
            val2017 = os.path.join(coco2017_root, "val2017", img_name_2017)
            image_path = train2017 if os.path.exists(train2017) else val2017
        gt_xyxy = _xywh_to_xyxy(ann["bbox"])

        for sent in ref["sentences"]:
            sample = RefSample(
                dataset=dataset,
                split_by=split_by,
                split=split,
                image_id=image_id,
                ref_id=ref["ref_id"],
                sent_id=sent["sent_id"],
                sentence=sent["sent"],
                image_path=image_path,
                gt_box_xyxy=gt_xyxy,
            )
            yield sample
            yielded += 1
            if max_samples is not None and yielded >= max_samples:
                return


def load_refcoco_samples(**kwargs) -> List[RefSample]:
    return list(iter_refcoco_samples(**kwargs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="refcoco", choices=["refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--split_by", default=None)
    parser.add_argument("--refer_data_root", default="~/cv_project/datasets/refer/data")
    parser.add_argument("--refer_repo_root", default="~/cv_project/datasets/refer/refer")
    parser.add_argument("--coco2014_root", default="~/cv_project/datasets/mscoco2014")
    parser.add_argument("--coco2017_root", default="~/cv_project/datasets/coco")
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()

    samples = load_refcoco_samples(
        dataset=args.dataset,
        split=args.split,
        split_by=args.split_by,
        refer_data_root=args.refer_data_root,
        refer_repo_root=args.refer_repo_root,
        coco2014_root=args.coco2014_root,
        coco2017_root=args.coco2017_root,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(samples)} sentence-level samples.")
    for sample in samples[: min(3, len(samples))]:
        print(sample.to_json())


if __name__ == "__main__":
    main()
