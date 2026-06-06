"""Grounding DINO-B fine-tuning on COCO 2017."""

import argparse, os
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    GroundingDinoForObjectDetection,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)

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


class CocoGDinoDataset(Dataset):
    """Returns raw images + annotations; processor runs in collate_fn."""

    def __init__(self, root, ann_file, max_samples=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(ann_file)
        # Build COCO category_id -> 0-79 index mapping
        self.cat_id_to_idx = {
            cat["id"]: i for i, cat in enumerate(self.coco.loadCats(self.coco.getCatIds()))
        }
        self.img_ids = sorted(self.coco.imgs.keys())
        if max_samples:
            self.img_ids = self.img_ids[:max_samples]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        w, h = img_info["width"], img_info["height"]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes, text_labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([x, y, x + bw, y + h])
            idx = self.cat_id_to_idx[ann["category_id"]]
            text_labels.append(COCO_CLASSES[idx])

        if len(boxes) == 0:
            boxes = [[0, 0, float(w), float(h)]]
            text_labels = ["person"]

        scale = max(w, h)
        return {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32) / scale,
            "class_labels": torch.tensor(
                [COCO_CLASSES.index(l) for l in text_labels], dtype=torch.long
            ),
            "text_labels": text_labels,
        }


class CollateFn:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        texts = [". ".join(item["text_labels"]) + "." for item in batch]

        inputs = self.processor(
            images=images, text=texts, return_tensors="pt", padding=True
        )

        # Labels: list of dicts, one per image (format model expects)
        inputs["labels"] = [
            {
                "boxes": item["boxes"],
                "class_labels": item["class_labels"],
                "size": torch.tensor(item["boxes"].shape[0]),
            }
            for item in batch
        ]
        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="~/cv_project/models/gdino")
    parser.add_argument("--data_dir", default="~/cv_project/datasets/coco")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="results/gdino")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    model_dir = os.path.expanduser(args.model_dir)
    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(f"~/{args.output_dir}")

    ckpt_dir = None
    if args.resume:
        for d in sorted(os.listdir(output_dir)):
            if d.startswith("checkpoint-"):
                ckpt_dir = os.path.join(output_dir, d)
        print(f"Resuming from {ckpt_dir}" if ckpt_dir else "No checkpoint, fresh start")

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

    if ckpt_dir:
        model = GroundingDinoForObjectDetection.from_pretrained(ckpt_dir)
    else:
        model = GroundingDinoForObjectDetection.from_pretrained(
            model_dir, local_files_only=True
        )

    train_ann = os.path.join(data_dir, "annotations", "annotations",
                              "instances_train2017.json")
    val_ann = os.path.join(data_dir, "annotations", "annotations",
                            "instances_val2017.json")
    train_root = os.path.join(data_dir, "train2017")
    val_root = os.path.join(data_dir, "val2017")

    train_dataset = CocoGDinoDataset(train_root, train_ann, max_samples=args.max_samples)
    val_dataset = CocoGDinoDataset(
        val_root, val_ann,
        max_samples=args.max_samples // 4 if args.max_samples else None
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CollateFn(processor),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=ckpt_dir)
    trainer.save_model(os.path.join(output_dir, "final"))
    print("Training complete.")


if __name__ == "__main__":
    main()
