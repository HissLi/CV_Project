"""OWL-ViT fine-tuning on COCO 2017 via HuggingFace."""

import argparse, os
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    OwlViTForObjectDetection,
    OwlViTImageProcessor,
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


class CocoOwlVitDataset(Dataset):
    def __init__(self, root, ann_file, max_samples=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(ann_file)
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
        boxes, class_labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([x / w, y / h, (x + bw) / w, (y + bh) / h])
            class_labels.append(ann["category_id"])

        if len(boxes) == 0:
            boxes = [[0.0, 0.0, 0.01, 0.01]]
            class_labels = [1]

        return {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "class_labels": self._map_labels(class_labels),
            "orig_size": torch.tensor([h, w]),
        }

    def _map_labels(self, labels):
        cats = sorted([c["id"] for c in self.coco.loadCats(self.coco.getCatIds())])
        id_to_idx = {cid: i for i, cid in enumerate(cats)}
        return torch.tensor([id_to_idx[l] for l in labels], dtype=torch.long)


class CollateFn:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs["labels"] = [
            {
                "boxes": item["boxes"],
                "class_labels": item["class_labels"],
                "image_id": torch.zeros(1, dtype=torch.long),
                "area": torch.ones(item["boxes"].shape[0]),
                "iscrowd": torch.zeros(item["boxes"].shape[0], dtype=torch.long),
                "orig_size": item["orig_size"],
                "size": item["orig_size"],
            }
            for item in batch
        ]
        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="~/cv_project/models/owlvit")
    parser.add_argument("--data_dir", default="~/cv_project/datasets/coco")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="results/owlvit")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gradient_accum", type=int, default=1)
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
    processor = OwlViTImageProcessor.from_pretrained(model_dir, local_files_only=True)

    if ckpt_dir:
        model = OwlViTForObjectDetection.from_pretrained(ckpt_dir)
    else:
        model = OwlViTForObjectDetection.from_pretrained(
            model_dir, local_files_only=True, ignore_mismatched_sizes=True
        )

    train_ann = os.path.join(data_dir, "annotations", "annotations",
                              "instances_train2017.json")
    val_ann = os.path.join(data_dir, "annotations", "annotations",
                            "instances_val2017.json")
    train_root = os.path.join(data_dir, "train2017")
    val_root = os.path.join(data_dir, "val2017")

    train_dataset = CocoOwlVitDataset(train_root, train_ann, max_samples=args.max_samples)
    val_dataset = CocoOwlVitDataset(val_root, val_ann,
                                    max_samples=args.max_samples // 4 if args.max_samples else None)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.gradient_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        save_safetensors=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
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
