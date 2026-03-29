#!/usr/bin/env python3
"""Train a PointNet classifier on the generated point-cloud dataset.

Usage examples:
    python classifier/train.py

    python classifier/train.py --run-name exp01 --epochs 150 --batch-size 32 --lr 1e-3

    python classifier/train.py --data-dir data/ModelNet10 --seed 42 --device cpu
"""

import argparse
import csv
import os
import sys
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from src.dataset import PointCloudDataset
from src.model import PointNet
from src.transforms import NormalizePointCloud

_DEFAULT_DATA = os.path.normpath(os.path.join(_HERE, "..", "data", "ModelNet10"))
_DEFAULT_RUNS = os.path.normpath(os.path.join(_HERE, "..", "experiments"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train PointNet on point clouds.")
    p.add_argument("--data-dir",    default=_DEFAULT_DATA,  help="Dataset root")
    p.add_argument("--runs-dir",    default=_DEFAULT_RUNS,  help="Experiments root")
    p.add_argument("--run-name",    default=None,
                   help="Run name (default: auto timestamp)")
    p.add_argument("--epochs",      type=int,   default=150)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--num-workers", type=int,   default=0,
                   help="DataLoader workers (0 = main process, safe on MPS)")
    p.add_argument("--device",      default=None,
                   help="Force device: 'cpu', 'cuda', 'mps' (default: auto)")
    p.add_argument("--seed",        type=int,   default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None):
    """Run one epoch. If optimizer is None, run in eval mode."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            logits, feat_t = model(batch.pos, batch.batch, batch.x)
            loss = model.loss(logits, batch.y, feat_t)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds       = logits.argmax(dim=1)
            correct    += (preds == batch.y).sum().item()
            total      += batch.num_graphs

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = select_device(args.device)
    print(f"Device : {device}")

    # --- Dataset & loaders ---
    transform = NormalizePointCloud()
    train_ds = PointCloudDataset(args.data_dir, split="train", transform=transform)
    val_ds   = PointCloudDataset(args.data_dir, split="test",  transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    num_classes = len(train_ds.class_list)
    print(f"Classes: {num_classes}  |  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # --- Model ---
    in_features = 6 if train_ds[0].x is not None else 3
    model = PointNet(num_classes=num_classes, in_features=in_features,
                     dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # --- Run directory ---
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir  = os.path.join(args.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}\n")

    metrics_path = os.path.join(run_dir, "metrics.csv")
    best_model_path = os.path.join(run_dir, "best_model.pth")

    # CSV header
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss,   val_acc   = run_epoch(model, val_loader,   device)
        scheduler.step()

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:>4}/{args.epochs} | "
            f"train loss {train_loss:.4f}  acc {train_acc*100:.1f}% | "
            f"val loss {val_loss:.4f}  acc {val_acc*100:.1f}% | "
            f"{elapsed:.1f}s"
        )

        # Save metrics
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                             f"{val_loss:.6f}", f"{val_acc:.6f}"])

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "in_features": in_features,
                "args":        vars(args),
                "classes":     train_ds.class_list,
            }, best_model_path)
            print(f"           ↳ new best val acc: {val_acc*100:.1f}%  (saved)")

    print(f"\nTraining complete. Best val acc: {best_val_acc*100:.1f}%")
    print(f"Model saved to: {best_model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
