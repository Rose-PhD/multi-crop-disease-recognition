#!/usr/bin/env python3
"""
TRAINING: CUSTOM HIERARCHICAL RESNET-18 FROM SCRATCH (CONCATENATED HEADS)

Training loss = CrossEntropy(crop) + mean CrossEntropy(disease slice)
Two-stage inference: predict crop → restrict disease logits to that crop's slice.

- StratifiedKFold (5 folds) stratified on joint class labels
- Adam, early stopping (patience=7)
- Separate train / val transforms

Outputs (under SAVE_ROOT):
  label_maps.json
  fold{n}/best_model.pth
  fold{n}/fold_summary.csv
  fold{n}/cm_*.{png,csv}
  summary_all_folds.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from utils import (
    SEED, g, seed_worker, safe_collate,
    make_train_transform, make_val_transform,
    build_index, HierDataset,
    build_global_index, run_epoch_loop, evaluate_hier,
)

from models.hier.custom_hier_model import CustomHierResNet18

# paths / config
DATASET   = "/deepstore/datasets/dmb/ComputerVision/biology/train-D"
SAVE_ROOT = "/home/nalwangar/wilfred/logs_customHierY"

N_FOLDS     = 5
BATCH_SIZE  = 32
NUM_WORKERS = 4
MAX_EPOCHS  = 50
PATIENCE    = 7
LR          = 1e-4

_criterion = nn.CrossEntropyLoss()


def _hier_loss(model, batch, device):
    imgs, yc, yd, _yg = batch
    imgs, yc, yd = imgs.to(device), yc.to(device), yd.to(device)

    out_crop, out_dis = model(imgs)
    loss_crop = _criterion(out_crop, yc)

    loss_dis   = 0.0
    batch_size = imgs.size(0)
    for i in range(batch_size):
        ci_true = int(yc[i].item())
        start, end = model.crop_slices[ci_true]
        loss_dis += _criterion(out_dis[i, start:end].unsqueeze(0), yd[i].unsqueeze(0))

    return loss_crop + loss_dis / batch_size


def train_fold(fold, model, train_loader, val_loader, device, fold_dir,
               crops, global_labels):
    best_val_loss, model_path = run_epoch_loop(
        fold, model, train_loader, val_loader, device, fold_dir,
        _hier_loss, patience=PATIENCE, max_epochs=MAX_EPOCHS, lr=LR,
    )
    summary = evaluate_hier(
        model, model_path, val_loader, device, fold_dir, crops, global_labels,
    )
    summary.pop("per_crop_acc")
    row = {"fold": fold, "best_val_loss": best_val_loss, **summary}
    pd.DataFrame([row]).to_csv(fold_dir / "fold_summary.csv", index=False)
    return row


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    crops, diseases_by_crop, items = build_index(DATASET)

    with open(f"{SAVE_ROOT}/label_maps.json", "w") as f:
        json.dump({"crops": crops, "diseases_within_crop": diseases_by_crop}, f, indent=4)

    global_index, global_labels = build_global_index(crops, diseases_by_crop)

    joint_labels = [global_index[(c, d)] for _, c, d in items]

    train_transform = make_train_transform()
    val_transform   = make_val_transform()
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skf          = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    paths        = [p for p, _, _ in items]

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, joint_labels), 1):
        print(f"\n===== FOLD {fold} =====")

        train_items = [items[i] for i in train_idx]
        val_items   = [items[i] for i in val_idx]

        train_loader = DataLoader(
            HierDataset(train_items, train_transform, global_index),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
            collate_fn=safe_collate, worker_init_fn=seed_worker, generator=g,
        )
        val_loader = DataLoader(
            HierDataset(val_items, val_transform, global_index),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            collate_fn=safe_collate, worker_init_fn=seed_worker, generator=g,
        )

        model    = CustomHierResNet18(crops, diseases_by_crop).to(device)
        fold_dir = Path(SAVE_ROOT) / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_results.append(train_fold(
            fold, model, train_loader, val_loader, device,
            fold_dir, crops, global_labels,
        ))

    pd.DataFrame(fold_results).to_csv(f"{SAVE_ROOT}/summary_all_folds.csv", index=False)
    print("\n=== TRAINING COMPLETE: CUSTOM HIERARCHICAL RESNET-18 FROM SCRATCH ===")


if __name__ == "__main__":
    main()
