"""Download only the COCO2017 images needed for a small RefCOCO eval subset."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import Iterable, Set


PADT_FILES = {
    "refcoco": "refcoco_val.json",
    "refcoco+": "refcoco+_val.json",
    "refcocog": "refcocog_val.json",
}

COCO_BASE = "http://images.cocodataset.org"


def _image_id_from_row(row: dict) -> int | None:
    image_name = row.get("image", "")
    digits = re.findall(r"(\d{12})\.jpg$", image_name)
    if digits:
        return int(digits[0])
    row_id = row.get("id")
    return int(row_id) if row_id is not None else None


def collect_image_ids(padt_dir: str, datasets: Iterable[str], max_samples: int) -> Set[int]:
    ids: Set[int] = set()
    for dataset in datasets:
        path = os.path.join(padt_dir, PADT_FILES[dataset])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing PaDT file: {path}")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                image_id = _image_id_from_row(row)
                if image_id is None:
                    continue
                ids.add(image_id)
                count += 1
                if count >= max_samples:
                    break
        print(f"[{dataset}] collected {count} sample image ids")
    return ids


def download_image(image_id: int, coco_root: str) -> str:
    name = f"{image_id:012d}.jpg"
    for split in ("val2017", "train2017"):
        dest = os.path.join(coco_root, split, name)
        if os.path.isfile(dest):
            return dest
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"{COCO_BASE}/{split}/{name}"
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except urllib.error.HTTPError:
            continue
    raise FileNotFoundError(f"Could not download {name} from val2017 or train2017")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--padt_dir", required=True)
    parser.add_argument("--coco2017_root", required=True)
    parser.add_argument("--datasets", default="refcoco,refcoco+,refcocog")
    parser.add_argument("--max_samples", type=int, default=20)
    args = parser.parse_args()

    padt_dir = os.path.expanduser(args.padt_dir)
    coco_root = os.path.expanduser(args.coco2017_root)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    image_ids = collect_image_ids(padt_dir, datasets, args.max_samples)
    print(f"Need {len(image_ids)} unique COCO images")

    ok = 0
    failed = []
    for image_id in sorted(image_ids):
        try:
            download_image(image_id, coco_root)
            ok += 1
        except Exception as exc:
            failed.append((image_id, str(exc)))
            print(f"FAIL {image_id:012d}: {exc}")

    print(f"Downloaded/found {ok}/{len(image_ids)} images under {coco_root}")
    if failed:
        print(f"Failed {len(failed)} images (eval will skip missing_image for those)")


if __name__ == "__main__":
    main()
