#!/usr/bin/env python3
"""
TRAINING: FULLY HIERARCHICAL RESNET-18 (OPTION C CONCATENATED HEADS)
- Fully hierarchical + concatenated disease heads
- Two-stage (crop -> disease slice) in BOTH training and evaluation
- Fine-tune last block (layer4)
- 5-fold CV
- Confusion matrices + CSV summaries + label maps for testing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json

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
    HierResNet18Concat,
)


# evaluation
def evaluate(model, model_path, val_loader, device, fold_dir, crops, global_labels,
             global_index):
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
            yd = yd.to(device)
            yg = yg.to(device)

            out_crop, out_dis = model(imgs)
            pred_c = out_crop.argmax(1)

            for i in range(imgs.size(0)):
                ci_pred = int(pred_c[i].item())
                ci_true = int(yc[i].item())
                di_true = int(yd[i].item())

                start_pred, end_pred = model.crop_slices[ci_pred]
                local_pred_pred_crop = int(out_dis[i, start_pred:end_pred].argmax().item())
                global_pred_pred_crop = global_index[(ci_pred, local_pred_pred_crop)]

                start_true, end_true = model.crop_slices[ci_true]
                local_pred_true_crop = int(out_dis[i, start_true:end_true].argmax().item())
                global_pred_true_crop = global_index[(ci_true, local_pred_true_crop)]

                true_crop.append(ci_true)
                pred_crop.append(ci_pred)
                true_global.append(int(yg[i].item()))
                pred_global.append(global_pred_pred_crop)
                pred_global_true_crop.append(global_pred_true_crop)

    crop_acc = accuracy_score(true_crop, pred_crop)
    disease_acc_pred_crop = accuracy_score(true_global, pred_global)
    disease_acc_true_crop = accuracy_score(true_global, pred_global_true_crop)

    cm_crop = confusion_matrix(true_crop, pred_crop, labels=range(len(crops)))
    plot_cm(cm_crop, crops, fold_dir / "cm_crop.png", "Crop Confusion Matrix")
    pd.DataFrame(cm_crop, index=crops, columns=crops).to_csv(fold_dir / "cm_crop.csv")

    cm_dis = confusion_matrix(true_global, pred_global, labels=range(len(global_labels)))
    plot_cm(cm_dis, global_labels, fold_dir / "cm_disease_pred_crop.png",
            "Disease Confusion Matrix (Pred Crop Slice)")
    pd.DataFrame(cm_dis, index=global_labels, columns=global_labels).to_csv(
        fold_dir / "cm_disease_pred_crop.csv"
    )

    cm_dis_oracle = confusion_matrix(true_global, pred_global_true_crop,
                                     labels=range(len(global_labels)))
    plot_cm(cm_dis_oracle, global_labels, fold_dir / "cm_disease_true_crop.png",
            "Disease Confusion Matrix (True Crop Slice)")
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
def train_fold(fold, model, train_loader, val_loader, device, fold_dir, crops,
               global_labels, global_index):
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
            yc = yc.to(device)
            yd = yd.to(device)

            optimizer.zero_grad()
            out_crop, out_dis = model(imgs)

            loss_crop = criterion(out_crop, yc)

            loss_dis = 0.0
            batch_size = imgs.size(0)
            for i in range(batch_size):
                ci_true = int(yc[i].item())
                di_true = yd[i].unsqueeze(0)
                start, end = model.crop_slices[ci_true]
                slice_logits = out_dis[i, start:end].unsqueeze(0)
                loss_dis = loss_dis + criterion(slice_logits, di_true)
            loss_dis = loss_dis / batch_size

            loss = loss_crop + loss_dis
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
                yc = yc.to(device)
                yd = yd.to(device)

                out_crop, out_dis = model(imgs)
                loss_crop_val = criterion(out_crop, yc)

                loss_dis_val = 0.0
                batch_size = imgs.size(0)
                for i in range(batch_size):
                    ci_true = int(yc[i].item())
                    di_true = yd[i].unsqueeze(0)
                    start, end = model.crop_slices[ci_true]
                    slice_logits = out_dis[i, start:end].unsqueeze(0)
                    loss_dis_val = loss_dis_val + criterion(slice_logits, di_true)
                loss_dis_val = loss_dis_val / batch_size

                total_val += (loss_crop_val + loss_dis_val).item()

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
        crops, global_labels, global_index
    )

    summary = {"fold": fold, "best_val_loss": best_loss, **eval_summary}
    pd.DataFrame([summary]).to_csv(fold_dir / "fold_summary.csv", index=False)
    return summary


# main
def main():
    DATASET = "/deepstore/datasets/dmb/ComputerVision/biology/train-V"
    SAVE_ROOT = "/home/nalwangar/finally/logs_hierX"
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

        model = HierResNet18Concat(crops, diseases_by_crop).to(device)

        fold_dir = Path(SAVE_ROOT) / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        summary = train_fold(
            fold, model, train_loader, val_loader, device,
            fold_dir, crops, global_labels, global_index
        )
        fold_results.append(summary)

    pd.DataFrame(fold_results).to_csv(f"{SAVE_ROOT}/summary_all_folds.csv", index=False)
    print("\n=== TRAINING COMPLETE: FULLY HIERARCHICAL OPTION C (CONCATENATED HEADS) ===")


if __name__ == "__main__":
    main()
