#!/usr/bin/env python3
"""
TRAINING: FULLY HIERARCHICAL RESNET-18 (OPTION C — CONCATENATED HEADS)

- Crop head + per-crop disease heads, concatenated
- Two-stage loss: crop CE + sliced disease CE
- Two unfreezing variants per run (layer4-only vs full backbone)
- 5-fold CV
- Confusion matrices + CSV summaries
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from configs.paths import HIER_DATASET, HIER_SAVE_ROOT, N_FOLDS, BATCH_SIZE, NUM_WORKERS
from utils import (
    SEED, g, seed_worker,
    safe_collate,
    make_train_transform, make_val_transform,
    build_index,
    plot_cm,
    HierDataset,
    HierResNet18Concat,
    setup_logger,
)

logger = logging.getLogger("hier.train")


# evaluation
def evaluate(model, model_path, val_loader, device, fold_dir, crops,
             global_labels, global_index):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    true_crop, pred_crop = [], []
    true_global, pred_global, pred_global_true_crop = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            imgs, yc, yd, yg = batch
            imgs = imgs.to(device)
            yc   = yc.to(device)
            yd   = yd.to(device)
            yg   = yg.to(device)

            out_crop, out_dis = model(imgs)
            pred_c = out_crop.argmax(1)

            for i in range(imgs.size(0)):
                ci_pred = int(pred_c[i].item())
                ci_true = int(yc[i].item())

                s_pred, e_pred = model.crop_slices[ci_pred]
                gp_pred = global_index[(ci_pred, int(out_dis[i, s_pred:e_pred].argmax().item()))]

                s_true, e_true = model.crop_slices[ci_true]
                gp_true = global_index[(ci_true, int(out_dis[i, s_true:e_true].argmax().item()))]

                true_crop.append(ci_true)
                pred_crop.append(ci_pred)
                true_global.append(int(yg[i].item()))
                pred_global.append(gp_pred)
                pred_global_true_crop.append(gp_true)

    crop_acc              = accuracy_score(true_crop, pred_crop)
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

    cm_oracle = confusion_matrix(true_global, pred_global_true_crop,
                                 labels=range(len(global_labels)))
    plot_cm(cm_oracle, global_labels, fold_dir / "cm_disease_true_crop.png",
            "Disease Confusion Matrix (True Crop Slice)")
    pd.DataFrame(cm_oracle, index=global_labels, columns=global_labels).to_csv(
        fold_dir / "cm_disease_true_crop.csv"
    )

    return {
        "crop_acc":                      crop_acc,
        "disease_acc_pred_crop":         disease_acc_pred_crop,
        "disease_acc_true_crop":         disease_acc_true_crop,
        "n_val_samples":                 len(true_crop),
        "f1_crop_macro":                 f1_score(true_crop, pred_crop, average="macro",    zero_division=0),
        "f1_crop_weighted":              f1_score(true_crop, pred_crop, average="weighted", zero_division=0),
        "f1_disease_pred_crop_macro":    f1_score(true_global, pred_global,           average="macro",    zero_division=0),
        "f1_disease_pred_crop_weighted": f1_score(true_global, pred_global,           average="weighted", zero_division=0),
        "f1_disease_true_crop_macro":    f1_score(true_global, pred_global_true_crop, average="macro",    zero_division=0),
        "f1_disease_true_crop_weighted": f1_score(true_global, pred_global_true_crop, average="weighted", zero_division=0),
    }


# training
def train_fold(fold, model, train_loader, val_loader, device, fold_dir,
               crops, global_labels, global_index):
    criterion  = nn.CrossEntropyLoss()
    optimizer  = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    best_loss  = float("inf")
    wait       = 0
    model_path = fold_dir / "best_model.pth"

    for epoch in range(1, 50 + 1):
        model.train()
        total_train = 0.0
        for batch in train_loader:
            if batch is None:
                continue
            imgs, yc, yd, _ = batch
            imgs = imgs.to(device)
            yc   = yc.to(device)
            yd   = yd.to(device)

            optimizer.zero_grad()
            out_crop, out_dis = model(imgs)
            loss_crop = criterion(out_crop, yc)

            loss_dis   = 0.0
            batch_size = imgs.size(0)
            for i in range(batch_size):
                ci = int(yc[i].item())
                s, e = model.crop_slices[ci]
                loss_dis += criterion(out_dis[i, s:e].unsqueeze(0), yd[i].unsqueeze(0))
            loss_dis /= batch_size

            (loss_crop + loss_dis).backward()
            optimizer.step()
            total_train += (loss_crop + loss_dis).item()

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, yc, yd, _ = batch
                imgs = imgs.to(device)
                yc   = yc.to(device)
                yd   = yd.to(device)

                out_crop, out_dis = model(imgs)
                loss_crop_val = criterion(out_crop, yc)

                loss_dis_val = 0.0
                batch_size   = imgs.size(0)
                for i in range(batch_size):
                    ci = int(yc[i].item())
                    s, e = model.crop_slices[ci]
                    loss_dis_val += criterion(out_dis[i, s:e].unsqueeze(0), yd[i].unsqueeze(0))
                loss_dis_val /= batch_size

                total_val += (loss_crop_val + loss_dis_val).item()

        logger.info("Fold %d | Epoch %3d | Train: %.4f | Val: %.4f",
                    fold, epoch, total_train, total_val)

        if total_val < best_loss:
            best_loss = total_val
            wait = 0
            torch.save(model.state_dict(), model_path)
            logger.info("  -> best model saved (val_loss=%.4f)", best_loss)
        else:
            wait += 1
            if wait >= 7:
                logger.info("  -> early stopping at epoch %d", epoch)
                break

    eval_summary = evaluate(
        model, model_path, val_loader, device, fold_dir,
        crops, global_labels, global_index,
    )

    summary = {"fold": fold, "best_val_loss": best_loss, **eval_summary}
    pd.DataFrame([summary]).to_csv(fold_dir / "fold_summary.csv", index=False)
    return summary


# per-config CV loop
def run_config(name, unfreeze_from, config_dir, items, global_index, global_labels,
               crops, diseases_by_crop, train_transform, val_transform, device):
    """Run 5-fold CV for one unfreezing configuration and return all fold summaries."""
    with open(config_dir / "config.json", "w") as f:
        json.dump({"name": name, "unfreeze_from": unfreeze_from}, f, indent=2)

    joint_labels = [global_index[(c, d)] for _, c, d in items]
    paths        = [p for p, _, _ in items]
    skf          = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, joint_labels), 1):
        logger.info("[%s] ===== FOLD %d =====", name, fold)

        train_loader = DataLoader(
            HierDataset([items[i] for i in train_idx], train_transform, global_index),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
            collate_fn=safe_collate, worker_init_fn=seed_worker, generator=g,
        )
        val_loader = DataLoader(
            HierDataset([items[i] for i in val_idx], val_transform, global_index),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            collate_fn=safe_collate, worker_init_fn=seed_worker, generator=g,
        )

        model    = HierResNet18Concat(crops, diseases_by_crop,
                                      unfreeze_from=unfreeze_from).to(device)
        fold_dir = config_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        summary = train_fold(
            fold, model, train_loader, val_loader, device,
            fold_dir, crops, global_labels, global_index,
        )
        fold_results.append({"variant": name, **summary})

    pd.DataFrame(fold_results).to_csv(config_dir / "summary_all_folds.csv", index=False)
    logger.info("[%s] training complete", name)
    return fold_results


# entry point
def main():
    CONFIGS = [
        {"name": "hier_layer4_only",   "unfreeze_from": "layer4"},
        {"name": "hier_full_unfreeze", "unfreeze_from": "layer1"},
    ]

    save_root = Path(HIER_SAVE_ROOT)
    save_root.mkdir(parents=True, exist_ok=True)

    setup_logger("hier.train", save_root / "train.log")
    logger.info("Starting hierarchical training — save root: %s", save_root)

    crops, diseases_by_crop, items = build_index(HIER_DATASET)
    logger.info("Dataset: %s | crops: %d | images: %d",
                HIER_DATASET, len(crops), len(items))

    global_index = {}
    labels       = []
    idx          = 0
    for ci, crop in enumerate(crops):
        for di, dis in enumerate(diseases_by_crop[crop]):
            global_index[(ci, di)] = idx
            labels.append(f"{crop}:{dis}")
            idx += 1

    logger.info("Total joint (crop, disease) classes: %d", idx)

    with open(save_root / "label_maps.json", "w") as f:
        json.dump({"crops": crops, "diseases_within_crop": diseases_by_crop}, f, indent=4)

    train_transform = make_train_transform()
    val_transform   = make_val_transform()
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    all_results = []
    for cfg in CONFIGS:
        config_dir = save_root / cfg["name"]
        config_dir.mkdir(parents=True, exist_ok=True)
        results = run_config(
            name=cfg["name"],
            unfreeze_from=cfg["unfreeze_from"],
            config_dir=config_dir,
            items=items,
            global_index=global_index,
            global_labels=labels,
            crops=crops,
            diseases_by_crop=diseases_by_crop,
            train_transform=train_transform,
            val_transform=val_transform,
            device=device,
        )
        all_results.extend(results)

    pd.DataFrame(all_results).to_csv(save_root / "comparison_all_variants.csv", index=False)
    logger.info("All variants complete.")


if __name__ == "__main__":
    main()
