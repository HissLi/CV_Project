import argparse, os

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


def create_coco_yaml(path):
    names_str = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO_CLASSES))
    content = f"""path: {os.path.dirname(path)}
train: train2017
val: val2017
nc: 80
names:
{names_str}
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/cv_project/models/yolov8l-worldv2.pt")
    parser.add_argument("--data_yaml", default="coco.yaml")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="results/yolow")
    parser.add_argument("--name", default="train")
    parser.add_argument("--cos_lr", action="store_true", default=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    data_yaml = os.path.expanduser(f"~/cv_project/datasets/coco/{args.data_yaml}")
    if not os.path.exists(data_yaml):
        data_yaml = create_coco_yaml(data_yaml)
        print(f"Created COCO data yaml at {data_yaml}")

    model_path = os.path.expanduser(args.model)
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, trying ultralytics auto-download")
        model_path = os.path.basename(args.model)

    model = YOLO(model_path)
    print(f"Model loaded. Starting training...")

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr,
        warmup_epochs=0,
        warmup_momentum=0.8,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        cos_lr=args.cos_lr,
        amp=True,
        workers=4,
        val=True,
        save=True,
        save_period=1,
        exist_ok=True,
        plots=True,
        close_mosaic=12,  # disable mosaic for last epoch
        nbs=64,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
