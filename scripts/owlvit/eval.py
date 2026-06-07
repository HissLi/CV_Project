"""OWL-ViT zero-shot evaluation on COCO val2017."""
import os, json, argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="~/cv_project/models/owlvit")
    parser.add_argument("--data_dir", default="~/cv_project/datasets/coco")
    parser.add_argument("--output", default="~/cv_project/results/owlvit_zero_eval.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = os.path.expanduser(args.model_dir)
    data_dir = os.path.expanduser(args.data_dir)

    print("Loading model...")
    model = OwlViTForObjectDetection.from_pretrained(model_dir, local_files_only=True).to(device)
    processor = OwlViTProcessor.from_pretrained(model_dir, local_files_only=True)

    # Load COCO val annotations
    from pycocotools.coco import COCO
    ann_file = os.path.join(data_dir, "annotations", "annotations", "instances_val2017.json")
    coco = COCO(ann_file)
    img_ids = sorted(coco.imgs.keys())
    if args.max_samples:
        img_ids = img_ids[:args.max_samples]

    # Build text queries for all 80 classes
    text_queries = [[f"a photo of a {c}"] for c in COCO_CLASSES]

    text_labels = COCO_CLASSES
    text_inputs = processor(text=text_labels, return_tensors="pt")

    results = []
    for i in tqdm(range(0, len(img_ids), args.batch_size), desc="Evaluating"):
        batch_ids = img_ids[i:i + args.batch_size]
        images = []
        for img_id in batch_ids:
            img_info = coco.imgs[img_id]
            img_path = os.path.join(data_dir, "val2017", img_info["file_name"])
            images.append(Image.open(img_path).convert("RGB"))

        img_inputs = processor(images=images, return_tensors="pt")
        bsz = img_inputs["pixel_values"].shape[0]
        inputs = {
            "pixel_values": img_inputs["pixel_values"].to(device),
            "input_ids": text_inputs["input_ids"].repeat(bsz, 1).to(device),
            "attention_mask": text_inputs["attention_mask"].repeat(bsz, 1).to(device),
        }

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        target_sizes = torch.tensor([img.size[::-1] for img in images]).to(device)
        processed = processor.post_process_object_detection(
            outputs, threshold=0.1, target_sizes=target_sizes
        )

        for j, (img_id, pred) in enumerate(zip(batch_ids, processed)):
            boxes = pred["boxes"].cpu().tolist()
            scores = pred["scores"].cpu().tolist()
            labels = pred["labels"].cpu().tolist()
            for b, s, l in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b
                results.append({
                    "image_id": img_id,
                    "category_id": int(l) + 1,  # COCO uses 1-indexed
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(s),
                })

    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} predictions to {output_path}")

    # Run COCO evaluation
    from pycocotools.cocoeval import COCOeval
    coco_dt = coco.loadRes(output_path)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(f"mAP: {coco_eval.stats[0]:.4f}, mAP50: {coco_eval.stats[1]:.4f}, mAP75: {coco_eval.stats[2]:.4f}")


if __name__ == "__main__":
    main()
