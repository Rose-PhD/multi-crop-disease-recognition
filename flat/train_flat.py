#!/usr/bin/env python3
"""
TRAINING: FLAT RESNET-18 BASELINE (JOINT CROP+DISEASE LABELS)

- Same dataset and CV protocol as hierarchical ResNet-18
- Treat each (crop, disease) pair as a single atomic class
- Single linear classifier over K joint classes
- 5-fold CV
- Confusion matrices + CSV summaries

Improvements:
- StratifiedKFold now uses JOINT class labels, not crop labels only
- Separate train and validation transforms
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from utils import (
    SEED, g, seed_worker,
    safe_collate,
    make_train_transform, make_val_transform,
    build_index,
    plot_cm,
    HierDataset,
    FlatResNet18,
)


# evaluation
def evaluate(model, model_path, val_loader, device, fold_dir,
             crops, global_labels, global_to_crop_dis, crop_to_global_ids):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    true_crop, pred_crop = [], []
    true_global, pred_global = [], []
    pred_global_true_crop = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            imgs, yc, yd, yg = batch
            imgs = imgs.to(device)
            yc = yc.to(device)
            yg = yg.to(device)

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            for i in range(imgs.size(0)):
                gi_true = int(yg[i].item())
                gi_pred = int(preds[i].item())

                ci_true, di_true = global_to_crop_dis[gi_true]
                ci_pred, di_pred = global_to_crop_dis[gi_pred]

                true_global.append(gi_true)
                pred_global.append(gi_pred)
                true_crop.append(ci_true)
                pred_crop.append(ci_pred)

                crop_global_ids = crop_to_global_ids[ci_true]
                logits_i = logits[i]
                crop_logits = torch.stack([logits_i[g_id] for g_id in crop_global_ids], dim=0)
                local_pred_idx = int(crop_logits.argmax().item())
                pred_global_true_crop.append(int(crop_global_ids[local_pred_idx]))

    crop_acc = accuracy_score(true_crop, pred_crop)
    disease_acc_pred_crop = accuracy_score(true_global, pred_global)
    disease_acc_true_crop = accuracy_score(true_global, pred_global_true_crop)

    cm_crop = confusion_matrix(true_crop, pred_crop, labels=range(len(crops)))
    plot_cm(cm_crop, crops, fold_dir / "cm_crop.png", "Crop Confusion Matrix")
    pd.DataFrame(cm_crop, index=crops, columns=crops).to_csv(fold_dir / "cm_crop.csv")

    cm_dis = confusion_matrix(true_global, pred_global, labels=range(len(global_labels)))
    plot_cm(cm_dis, global_labels, fold_dir / "cm_disease_pred_crop.png",
            "Disease Confusion Matrix (Pred Joint Class)")
    pd.DataFrame(cm_dis, index=global_labels, columns=global_labels).to_csv(
        fold_dir / "cm_disease_pred_crop.csv"
    )

    cm_dis_oracle = confusion_matrix(true_global, pred_global_true_crop,
                                     labels=range(len(global_labels)))
    plot_cm(cm_dis_oracle, global_labels, fold_dir / "cm_disease_true_crop.png",
            "Disease Confusion Matrix (Oracle Crop Mask)")
    pd.DataFrame(cm_dis_oracle, index=global_labels, columns=global_labels).to_csv(
        fold_dir / "cm_disease_true_crop.csv"
    )

    summary = {
        "crop_acc": crop_acc,
        "disease_acc_pred_crop": disease_acc_pred_crop,
        "disease_acc_true_crop": disease_acc_true_crop,
        "n_val_samples": len(true_crop),
        "f1_crop_macro": f1_score(true_crop, pred_crop, average='macro', zero_division=0),
        "f1_crop_weighted": f1_score(true_crop, pred_crop, average='weighted', zero_division=0),
        "f1_disease_pred_crop_macro": f1_score(true_global, pred_global, average='macro', zero_division=0),
        "f1_disease_pred_crop_weighted": f1_score(true_global, pred_global, average='weighted', zero_division=0),
        "f1_disease_true_crop_macro": f1_score(true_global, pred_global_true_crop, average='macro', zero_division=0),
        "f1_disease_true_crop_weighted": f1_score(true_global, pred_global_true_crop, average='weighted', zero_division=0),
    }
    return summary


# training
def train_fold(fold, model, train_loader, val_loader, device, fold_dir,
               crops, global_labels, global_to_crop_dis, crop_to_global_ids):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    best_loss = float("inf")
    wait = 0
    patience = 7
    model_path = fold_dir / "best_model.pth"

    for epoch in range(1, 50 + 1):
        model.train()
        total_train = 0.0

        for batch in train_loader:
            if batch is None:
                continue
            imgs, yc, yd, yg = batch
            imgs = imgs.to(device)
            yg = yg.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = nn.CrossEntropyLoss()(logits, yg)
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, yc, yd, yg = batch
                imgs = imgs.to(device)
                yg = yg.to(device)
                logits = model(imgs)
                total_val += criterion(logits, yg).item()

        print(f"[Fold {fold}] Epoch {epoch} | Train: {total_train:.4f} | Val: {total_val:.4f}")

        if total_val < best_loss:
            best_loss = total_val
            wait = 0
            torch.save(model.state_dict(), model_path)
            print("   Best model updated!")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping!")
                break

    eval_summary = evaluate(
        model, model_path, val_loader, device, fold_dir,
        crops, global_labels, global_to_crop_dis, crop_to_global_ids
    )

    summary = {"fold": fold, "best_val_loss": best_loss, **eval_summary}
    pd.DataFrame([summary]).to_csv(fold_dir / "fold_summary.csv", index=False)
    return summary


# main
def main():
    DATASET = "/deepstore/datasets/dmb/ComputerVision/biology/train-D"
    SAVE_ROOT = "/home/nalwangar/wilfred/logs_hierY"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    crops, diseases_by_crop, items = build_index(DATASET)

    with open(f"{SAVE_ROOT}/label_maps.json", "w") as f:
        json.dump({"crops": crops, "diseases_within_crop": diseases_by_crop}, f, indent=4)

    global_index = {}
    labels = []
    idx = 0
    for ci, crop in enumerate(crops):
        for di, dis in enumerate(diseases_by_crop[crop]):
            global_index[(ci, di)] = idx
            labels.append(f"{crop}:{dis}")
            idx += 1

    num_joint_classes = idx
    print(f"Total joint (crop, disease) classes: {num_joint_classes}")

    global_to_crop_dis = {gid: (ci, di) for (ci, di), gid in global_index.items()}
    crop_to_global_ids = {}
    for (ci, di), gid in global_index.items():
        crop_to_global_ids.setdefault(ci, []).append(gid)

    global_labels = labels
    joint_labels = [global_index[(c, d)] for _, c, d in items]

    train_transform = make_train_transform()
    val_transform = make_val_transform()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fold_results = []
    paths = [p for p, _, _ in items]

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, joint_labels), 1):
        print(f"\n===== FOLD {fold} =====")

        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]

        train_loader = DataLoader(
            HierDataset(train_items, train_transform, global_index),
            batch_size=32, shuffle=True, num_workers=4,
            collate_fn=safe_collate, worker_init_fn=seed_worker, generator=g
        )
        val_loader = DataLoader(
            HierDataset(val_items, val_transform, global_index),
            batch_size=32, shuffle=False, num_workers=4,
            collate_fn=safe_collate, worker_init_fn=seed_worker, generator=g
        )

        model = FlatResNet18(num_joint_classes=num_joint_classes).to(device)

        fold_dir = Path(SAVE_ROOT) / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        summary = train_fold(
            fold, model, train_loader, val_loader, device,
            fold_dir, crops, global_labels, global_to_crop_dis, crop_to_global_ids
        )
        fold_results.append(summary)

    pd.DataFrame(fold_results).to_csv(f"{SAVE_ROOT}/summary_all_folds.csv", index=False)
    print("\n=== TRAINING COMPLETE: FLAT RESNET-18 BASELINE ===")


if __name__ == "__main__":
    main()
