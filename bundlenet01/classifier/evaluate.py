#!/usr/bin/env python3
"""Evaluate a trained PointNet model on the test set.

Produces:
  - confusion_matrix.png   visual heatmap
  - confusion_matrix.csv   raw counts
  - per_class_accuracy.csv per-class accuracy + support
  - overall accuracy printed to stdout

Usage:
    # Evaluate the best model from a run
    python classifier/evaluate.py --run-dir experiments/run_20240101_120000

    # Override dataset or checkpoint path
    python classifier/evaluate.py --run-dir experiments/run_20240101_120000 \\
        --data-dir data/ModelNet10 --checkpoint path/to/model.pth
"""

import argparse
import csv
import os
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from src.dataset import PointCloudDataset, CLASSES
from src.model import PointNet
from src.transforms import NormalizePointCloud

_DEFAULT_DATA = os.path.normpath(os.path.join(_HERE, "..", "data", "ModelNet10"))
_DEFAULT_RUNS = os.path.normpath(os.path.join(_HERE, "..", "experiments"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PointNet model.")
    p.add_argument("--run-dir",    required=True,
                   help="Path to the experiment directory (contains best_model.pth)")
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path (default: <run-dir>/best_model.pth)")
    p.add_argument("--data-dir",   default=_DEFAULT_DATA, help="Dataset root")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device",     default=None,
                   help="Force device: 'cpu', 'cuda', 'mps' (default: auto)")
    return p.parse_args()


def select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                          save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix image saved to: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"Device: {device}")

    # --- Load checkpoint ---
    ckpt_path = args.checkpoint or os.path.join(args.run_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    class_list  = ckpt.get("classes", CLASSES)
    num_classes = len(class_list)
    saved_epoch = ckpt.get("epoch", "?")
    saved_acc   = ckpt.get("val_acc", float("nan"))
    print(f"Checkpoint: epoch={saved_epoch}  val_acc={saved_acc*100:.1f}%")

    # --- Dataset ---
    transform = NormalizePointCloud()
    test_ds   = PointCloudDataset(args.data_dir, split="test",
                                  transform=transform, classes=class_list)
    loader    = DataLoader(test_ds, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}\n")

    # --- Model ---
    in_features = ckpt.get("in_features", 3)
    model = PointNet(num_classes=num_classes, in_features=in_features).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- Inference ---
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits, _ = model(batch.pos, batch.batch, batch.x)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Overall accuracy ---
    overall_acc = (all_preds == all_labels).mean()
    print(f"Overall accuracy: {overall_acc*100:.2f}%\n")

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(num_classes)))

    cm_img_path = os.path.join(args.run_dir, "confusion_matrix.png")
    cm_csv_path = os.path.join(args.run_dir, "confusion_matrix.csv")
    plot_confusion_matrix(cm, class_list, cm_img_path)

    with open(cm_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_list)
        for i, row in enumerate(cm):
            writer.writerow([class_list[i]] + row.tolist())
    print(f"Confusion matrix CSV saved to: {cm_csv_path}")

    # --- Per-class accuracy ---
    pca_path = os.path.join(args.run_dir, "per_class_accuracy.csv")
    print(f"\nPer-class accuracy:")
    print(f"  {'class':<14} {'accuracy':>10}  {'correct':>8}  {'support':>8}")
    print(f"  {'-'*46}")

    with open(pca_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "accuracy", "correct", "support"])
        for i, class_name in enumerate(class_list):
            mask    = all_labels == i
            support = mask.sum()
            correct = (all_preds[mask] == i).sum() if support > 0 else 0
            acc     = correct / support if support > 0 else 0.0
            writer.writerow([class_name, f"{acc:.6f}", correct, support])
            print(f"  {class_name:<14} {acc*100:>9.1f}%  {correct:>8}  {support:>8}")

    print(f"\nPer-class accuracy saved to: {pca_path}")


if __name__ == "__main__":
    main()
