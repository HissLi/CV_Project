"""Generate compact, report-friendly figures for Final_Report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.facecolor": "white",
    }
)

COLORS = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "accent": "#059669",
    "muted": "#6B7280",
    "gdino": "#F59E0B",
    "owlvit": "#8B5CF6",
}


def save(fig, name: str):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print("Wrote", path)


def fig_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(6.8, 1.55))
    ax.axis("off")
    boxes = [
        (0.02, 0.35, 0.16, 0.35, "COCO 2017\n118K images", "#DBEAFE"),
        (0.22, 0.35, 0.18, 0.35, "YOLO-World-L\nFine-tuning", "#BFDBFE"),
        (0.44, 0.35, 0.16, 0.35, "COCO val\nmAP metrics", "#93C5FD"),
        (0.64, 0.55, 0.16, 0.35, "RefCOCO jsonl\n+ COCO images", "#D1FAE5"),
        (0.64, 0.05, 0.18, 0.35, "GDINO / OWL-ViT\nZero-shot", "#A7F3D0"),
        (0.86, 0.30, 0.12, 0.40, "Acc@0.5\nIoU viz", "#6EE7B7"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.0, edgecolor="#374151", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)
    arrows = [
        (0.18, 0.52, 0.22, 0.52), (0.40, 0.52, 0.44, 0.52),
        (0.80, 0.72, 0.86, 0.58), (0.80, 0.22, 0.86, 0.42),
        (0.60, 0.52, 0.64, 0.72), (0.60, 0.52, 0.64, 0.22),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "fig00_pipeline.png")


def fig_hyperparam_importance():
    names = ["Optimizer", "Model scale", "Resolution", "Learning rate",
             "Epochs", "Augment.", "Freeze", "BS/WD/WU"]
    scores = [0.656, 0.102, 0.102, 0.060, 0.017, 0.005, 0.003, 0.005]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.barh(y, scores, color=COLORS["primary"], alpha=0.85, height=0.62)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Approx. mAP50 impact")
    ax.set_title("Hyperparameter Importance Ranking", pad=6)
    ax.grid(axis="x", alpha=0.25)
    save(fig, "fig05_hyperparam_importance.png")


def fig_hyperparam_grid():
    """2x2 compact panel for the main YOLO-World sweeps."""
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    # (a) LR sweep
    ax = axes[0, 0]
    lrs = ["1e-5", "5e-5", "1e-4", "2e-4", "3e-4", "5e-4", "1e-3"]
    x = np.arange(len(lrs))
    ax.plot(x, [0.661, 0.671, 0.671, 0.661, 0.653, 0.636, 0.611],
            "o-", color=COLORS["primary"], linewidth=1.6, markersize=4, label="mAP50")
    ax.plot(x, [0.501, 0.507, 0.506, 0.497, 0.488, 0.474, 0.450],
            "s--", color=COLORS["secondary"], linewidth=1.4, markersize=3.5, label="mAP50-95")
    ax.axvspan(1, 3.2, alpha=0.10, color=COLORS["accent"])
    ax.set_xticks(x, lrs, rotation=28, ha="right")
    ax.set_title("(a) Learning rate", pad=4)
    ax.set_ylabel("COCO val")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=False)

    # (b) Optimizer
    ax = axes[0, 1]
    names = ["AdamW", "SGD\n1e-2", "SGD\n5e-2", "SGD\n1e-1"]
    vals = [0.677, 0.248, 0.081, 0.021]
    colors = [COLORS["accent"], COLORS["muted"], COLORS["muted"], COLORS["muted"]]
    bars = ax.bar(names, vals, color=colors, width=0.62)
    bars[0].set_color(COLORS["accent"])
    ax.set_ylim(0, 0.75)
    ax.set_title("(b) Optimizer", pad=4)
    ax.set_ylabel("mAP50")
    ax.grid(axis="y", alpha=0.25)

    # (c) Resolution vs epochs
    ax = axes[1, 0]
    labels = ["320\n12", "640\n12", "800\n12", "640\n24", "800\n24", "640\n48"]
    map50 = [0.584, 0.677, 0.686, 0.685, 0.694, 0.690]
    bar_colors = [COLORS["muted"]] * len(labels)
    bar_colors[int(np.argmax(map50))] = COLORS["primary"]
    ax.bar(labels, map50, color=bar_colors, width=0.62)
    ax.set_ylim(0.55, 0.72)
    ax.set_title("(c) Resolution vs epochs", pad=4)
    ax.set_ylabel("mAP50")
    ax.grid(axis="y", alpha=0.25)

    # (d) Model scale
    ax = axes[1, 1]
    models = ["S", "M", "L"]
    x = np.arange(len(models))
    w = 0.34
    ax.bar(x - w / 2, [0.592, 0.648, 0.677], w, label="12 ep", color=COLORS["muted"])
    ax.bar(x + w / 2, [0.605, 0.659, 0.694], w, label="24 ep", color=COLORS["primary"])
    ax.set_xticks(x, models)
    ax.set_title("(d) Model scale", pad=4)
    ax.set_ylabel("mAP50")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    save(fig, "fig_hyperparam_grid.png")


def fig_results_pair():
    """Detection vs grounding side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    fig.subplots_adjust(wspace=0.38)

    ax = axes[0]
    names = ["YOLO-World-L", "OWL-ViT"]
    x = np.arange(2)
    w = 0.32
    ax.bar(x - w / 2, [0.694, 0.064], w, label="mAP50", color=COLORS["primary"])
    ax.bar(x + w / 2, [0.528, 0.040], w, label="mAP50-95", color=COLORS["secondary"])
    ax.set_xticks(x, names)
    ax.set_ylabel("COCO val score")
    ax.set_title("(a) Detection: fine-tuned vs zero-shot", pad=4)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    datasets = ["refcoco", "refcoco+", "refcocog", "All"]
    gdino = [0.085, 0.100, 0.260, 0.148]
    owlvit = [0.210, 0.300, 0.305, 0.272]
    x = np.arange(len(datasets))
    w = 0.32
    ax.bar(x - w / 2, gdino, w, label="GDINO", color=COLORS["gdino"])
    ax.bar(x + w / 2, owlvit, w, label="OWL-ViT", color=COLORS["owlvit"])
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 0.35)
    ax.set_ylabel("Acc@0.5")
    ax.set_title("(b) Grounding on RefCOCO subset (n=200)", pad=4)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    save(fig, "fig_results_pair.png")


def _resolve_image_path(image_id: int) -> str | None:
    name = f"{int(image_id):012d}.jpg"
    for split in ("val2017", "train2017"):
        path = ROOT / "datasets/coco" / split / name
        if path.exists():
            return str(path)
    return None


def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resize(img: Image.Image, max_w: int = 240) -> Image.Image:
    w, h = img.size
    if w <= max_w:
        return img
    scale = max_w / w
    return img.resize((max_w, int(h * scale)), Image.Resampling.LANCZOS)


def _draw_panel(row: dict, title: str, color: str) -> Image.Image | None:
    image_path = row.get("image_path") or _resolve_image_path(row["image_id"])
    if not image_path or not os.path.exists(image_path):
        return None
    img = _resize(Image.open(image_path).convert("RGB"), max_w=240)
    draw = ImageDraw.Draw(img)
    scale = img.width / Image.open(image_path).convert("RGB").width
    gt = [v * scale for v in row["gt_box_xyxy"]]
    draw.rectangle(gt, outline="#22C55E", width=2)
    pred = row.get("pred_box_xyxy")
    if pred:
        pred = [v * scale for v in pred]
        draw.rectangle(pred, outline=color, width=2)
    sent = row.get("sentence", "")[:42]
    caption = f'{title} | IoU={row.get("iou", 0.0):.2f}\n"{sent}..."'
    draw.rectangle([0, 0, img.width, 36], fill=(0, 0, 0))
    draw.text((4, 4), caption, fill="white")
    return img


def fig_grounding_qualitative():
    gd_dir = ROOT / "results/gdino_refcoco_n200"
    ow_dir = ROOT / "results/owlvit_refcoco_n200"
    if not gd_dir.exists():
        print("Skip qualitative: missing n200 results")
        return

    gd_rows, ow_rows = [], []
    for ds in ["refcoco", "refcoco+", "refcocog"]:
        gd_rows.extend(_load_json(gd_dir / f"{ds}_val.json"))
        ow_rows.extend(_load_json(ow_dir / f"{ds}_val.json"))
    ow_index = {(r["dataset"], r["ref_id"], r["sent_id"]): r for r in ow_rows}

    picks = []
    for row in gd_rows:
        key = (row["dataset"], row["ref_id"], row["sent_id"])
        ow = ow_index.get(key)
        if not ow or _resolve_image_path(row["image_id"]) is None:
            continue
        gd_hit = float(row.get("iou", 0)) >= 0.5
        ow_hit = float(ow.get("iou", 0)) >= 0.5
        if ow_hit and not gd_hit:
            picks.append((row, ow))
        elif gd_hit and not ow_hit:
            picks.append((row, ow))
        if len(picks) >= 2:
            break
    if len(picks) < 2:
        for row in gd_rows[:2]:
            key = (row["dataset"], row["ref_id"], row["sent_id"])
            ow = ow_index.get(key)
            if ow and _resolve_image_path(row["image_id"]):
                picks.append((row, ow))

    n = min(2, len(picks))
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(5.2, 2.1 * n))
    if n == 1:
        axes = np.array([axes])
    for i, (gd, ow) in enumerate(picks[:n]):
        gd_img = _draw_panel(gd, "GDINO", "#F59E0B")
        ow_img = _draw_panel(ow, "OWL-ViT", "#8B5CF6")
        if gd_img is None or ow_img is None:
            continue
        axes[i, 0].imshow(gd_img)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Grounding DINO" if i == 0 else "", fontsize=8, pad=2)
        axes[i, 1].imshow(ow_img)
        axes[i, 1].axis("off")
        axes[i, 1].set_title("OWL-ViT" if i == 0 else "", fontsize=8, pad=2)

    legend = [
        mpatches.Patch(color="#22C55E", label="GT"),
        mpatches.Patch(color="#F59E0B", label="GDINO"),
        mpatches.Patch(color="#8B5CF6", label="OWL-ViT"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "fig08_grounding_qualitative.png")


def main():
    fig_pipeline_diagram()
    fig_hyperparam_importance()
    fig_hyperparam_grid()
    fig_results_pair()
    fig_grounding_qualitative()
    print(f"All figures saved to {OUT}")


if __name__ == "__main__":
    main()
