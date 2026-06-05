"""Grounding DINO-B fine-tuning on COCO 2017."""
import argparse, os, json
import torch
import numpy as np
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
    def __init__(self, root, ann_file, processor, max_samples=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(ann_file)
        self.processor = processor
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

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(COCO_CLASSES[ann["category_id"]])

        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = ["person"]

        inputs = self.processor(
            images=image,
            text=[labels],
            return_tensors="pt",
            padding=True,
        )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.squeeze(0)

        inputs["labels"] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) / max(img_info["width"], img_info["height"]),
            "class_labels": torch.tensor(
                [COCO_CLASSES.index(l) for l in labels], dtype=torch.long
            ),
        }
        return inputs


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = {
        "boxes": [b["labels"]["boxes"] for b in batch],
        "class_labels": [b["labels"]["class_labels"] for b in batch],
    }
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="~/cv_project/models/gdino")
    parser.add_argument("--data_dir", default="~/cv_project/datasets/coco")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="results/gdino")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    model_dir = os.path.expanduser(args.model_dir)
    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(f"~/{args.output_dir}")

    if args.resume:
        # Find latest checkpoint
        ckpt_dir = None
        for d in sorted(os.listdir(output_dir)):
            if d.startswith("checkpoint-"):
                ckpt_dir = os.path.join(output_dir, d)
        if ckpt_dir:
            print(f"Resuming from {ckpt_dir}")
        else:
            print("No checkpoint found, starting fresh")
    else:
        ckpt_dir = None

    if ckpt_dir:
        print("Loading model from checkpoint...")
        model = GroundingDinoForObjectDetection.from_pretrained(ckpt_dir)
        processor = AutoProcessor.from_pretrained(ckpt_dir)
    else:
        print("Loading processor and model from pretrained...")
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = GroundingDinoForObjectDetection.from_pretrained(
            model_dir, local_files_only=True
        )

    train_ann = os.path.join(data_dir, "annotations", "annotations", "instances_train2017.json")
    val_ann = os.path.join(data_dir, "annotations", "annotations", "instances_val2017.json")
    train_root = os.path.join(data_dir, "train2017")
    val_root = os.path.join(data_dir, "val2017")

    train_dataset = CocoGDinoDataset(
        train_root, train_ann, processor, max_samples=args.max_samples
    )
    val_dataset = CocoGDinoDataset(
        val_root, val_ann, processor,
        max_samples=args.max_samples // 4 if args.max_samples else None
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=os.path.expanduser(f"~/{args.output_dir}"),
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
        data_collator=collate_fn,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=ckpt_dir if args.resume else None)
    trainer.save_model(os.path.join(output_dir, "final"))
    print("Training complete.")


if __name__ == "__main__":
    main()
