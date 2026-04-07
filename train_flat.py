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

import os
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import default_collate

# ============================================================
#                  GLOBAL SEEDING / WORKERS
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# ============================================================
#                         MODEL
#                    FLAT RESNET-18
# ============================================================
class FlatResNet18(nn.Module):
    """
    Flat baseline:
    - Shared ResNet-18 backbone
    - Single linear classifier over all (crop, disease) joint classes
    - Fine-tune only layer4 + final classifier layer
    """

    def __init__(self, num_joint_classes):
        super().__init__()

        try:
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("Loaded ResNet18 pretrained weights.")
        except Exception:
            print("Offline mode: loading local ResNet-18 weights.")
            backbone = models.resnet18(weights=None)
            local_path = "/home/nalwangar/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
            backbone.load_state_dict(torch.load(local_path, map_location="cpu"))

        for name, p in backbone.named_parameters():
            p.requires_grad = False

        for name, p in backbone.named_parameters():
            if "layer4" in name:
                p.requires_grad = True

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_joint_classes)

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


# ============================================================
#                       DATASET
# ============================================================
class HierDataset(Dataset):
    """
    items: list of (img_path, crop_id, dis_id)
    Returns: img, crop_id, dis_id, global_joint_id
    """

    def __init__(self, items, transform):
        self.items = items
        self.t = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, crop_id, dis_id = self.items[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            print(f"[CORRUPTED] Skipping {path}", flush=True)
            return None

        img = self.t(img)

        global items_global_map
        global_dis_id = items_global_map[(crop_id, dis_id)]
        return img, crop_id, dis_id, global_dis_id


# ============================================================
#                         TRANSFORMS
# ============================================================
def make_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def make_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ============================================================
#                     BUILD DATA INDEX
# ============================================================
def build_index(dataset_root):
    """
    Expects a hierarchical folder structure:

    dataset_root/
        crop1/
            disease1/
            disease2/
        crop2/
            disease1/
            ...

    Returns:
        crops: list[str] of crop names, sorted
        diseases_by_crop: dict[crop_name] -> list[str] of diseases (sorted)
        items: list of (img_path, crop_idx, disease_idx)
    """
    root = Path(dataset_root)
    crops = sorted([d.name for d in root.iterdir() if d.is_dir()])

    diseases_by_crop = {}
    items = []

    for ci, crop in enumerate(crops):
        ddir = root / crop
        dis_list = sorted([d.name for d in ddir.iterdir() if d.is_dir()])
        diseases_by_crop[crop] = dis_list

        for di, dis in enumerate(dis_list):
            for img in (ddir / dis).glob("*"):
                if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                    continue
                items.append((str(img), ci, di))

    return crops, diseases_by_crop, items


# ============================================================
#                   CONFUSION MATRIX UTILS
# ============================================================
def plot_confusion_matrix(cm, labels, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    plt.colorbar(im)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ============================================================
#                       EVALUATION
#       (FLAT JOINT PREDICTION + HIERARCHICAL METRICS)
# ============================================================
def evaluate(model, model_path, val_loader, device, fold_dir,
             crops, global_labels, global_to_crop_dis, crop_to_global_ids):
    """
    model: FlatResNet18
    global_to_crop_dis: dict[global_id] -> (crop_id, dis_id)
    crop_to_global_ids: dict[crop_id] -> list[global_ids belonging to that crop]
    """
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
                gi_pred_true_crop = int(crop_global_ids[local_pred_idx])
                pred_global_true_crop.append(gi_pred_true_crop)

    crop_acc = accuracy_score(true_crop, pred_crop)
    disease_acc_pred_crop = accuracy_score(true_global, pred_global)
    disease_acc_true_crop = accuracy_score(true_global, pred_global_true_crop)

    cm_crop = confusion_matrix(true_crop, pred_crop, labels=range(len(crops)))
    plot_confusion_matrix(cm_crop, crops, fold_dir / "cm_crop.png", "Crop Confusion Matrix")
    pd.DataFrame(cm_crop, index=crops, columns=crops).to_csv(fold_dir / "cm_crop.csv")

    cm_dis = confusion_matrix(true_global, pred_global, labels=range(len(global_labels)))
    plot_confusion_matrix(cm_dis, global_labels, fold_dir / "cm_disease_pred_crop.png",
                          "Disease Confusion Matrix (Pred Joint Class)")
    pd.DataFrame(cm_dis, index=global_labels, columns=global_labels).to_csv(
        fold_dir / "cm_disease_pred_crop.csv"
    )

    cm_dis_oracle = confusion_matrix(true_global, pred_global_true_crop,
                                     labels=range(len(global_labels)))
    plot_confusion_matrix(cm_dis_oracle, global_labels, fold_dir / "cm_disease_true_crop.png",
                          "Disease Confusion Matrix (Oracle Crop Mask)")
    pd.DataFrame(cm_dis_oracle, index=global_labels, columns=global_labels).to_csv(
        fold_dir / "cm_disease_true_crop.csv"
    )

    f1_disease_pred_crop_macro = f1_score(true_global, pred_global,
                                          average='macro', zero_division=0)
    f1_disease_pred_crop_weighted = f1_score(true_global, pred_global,
                                             average='weighted', zero_division=0)

    f1_disease_true_crop_macro = f1_score(true_global, pred_global_true_crop,
                                          average='macro', zero_division=0)
    f1_disease_true_crop_weighted = f1_score(true_global, pred_global_true_crop,
                                             average='weighted', zero_division=0)

    f1_crop_macro = f1_score(true_crop, pred_crop, average='macro', zero_division=0)
    f1_crop_weighted = f1_score(true_crop, pred_crop, average='weighted', zero_division=0)

    summary = {
        "crop_acc": crop_acc,
        "disease_acc_pred_crop": disease_acc_pred_crop,
        "disease_acc_true_crop": disease_acc_true_crop,
        "n_val_samples": len(true_crop),
        "f1_crop_macro": f1_crop_macro,
        "f1_crop_weighted": f1_crop_weighted,
        "f1_disease_pred_crop_macro": f1_disease_pred_crop_macro,
        "f1_disease_pred_crop_weighted": f1_disease_pred_crop_weighted,
        "f1_disease_true_crop_macro": f1_disease_true_crop_macro,
        "f1_disease_true_crop_weighted": f1_disease_true_crop_weighted,
    }
    return summary


# ============================================================
#                        TRAINING
# ============================================================
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
            loss = criterion(logits, yg)
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
                loss_val = criterion(logits, yg)
                total_val += loss_val.item()

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

    summary = {
        "fold": fold,
        "crop_acc": eval_summary["crop_acc"],
        "disease_acc_pred_crop": eval_summary["disease_acc_pred_crop"],
        "disease_acc_true_crop": eval_summary["disease_acc_true_crop"],
        "f1_crop_macro": eval_summary["f1_crop_macro"],
        "f1_crop_weighted": eval_summary["f1_crop_weighted"],
        "f1_disease_pred_crop_macro": eval_summary["f1_disease_pred_crop_macro"],
        "f1_disease_pred_crop_weighted": eval_summary["f1_disease_pred_crop_weighted"],
        "f1_disease_true_crop_macro": eval_summary["f1_disease_true_crop_macro"],
        "f1_disease_true_crop_weighted": eval_summary["f1_disease_true_crop_weighted"],
        "best_val_loss": best_loss,
    }

    pd.DataFrame([summary]).to_csv(fold_dir / "fold_summary.csv", index=False)
    return summary


# ============================================================
#                           MAIN
# ============================================================
def main():
    DATASET = "/deepstore/datasets/dmb/ComputerVision/biology/training7"
    SAVE_ROOT = "/home/nalwangar/finally/logs_flatM"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    crops, diseases_by_crop, items = build_index(DATASET)

    with open(f"{SAVE_ROOT}/label_maps.json", "w") as f:
        json.dump({
            "crops": crops,
            "diseases_within_crop": diseases_by_crop
        }, f, indent=4)

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

    global items_global_map
    items_global_map = global_index
    global_labels = labels

    paths = [p for p, _, _ in items]

    # CHANGED: stratify by JOINT class instead of crop only
    joint_labels = [items_global_map[(c, d)] for _, c, d in items]

    # CHANGED: separate transforms
    train_transform = make_train_transform()
    val_transform = make_val_transform()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, joint_labels), 1):
        print(f"\n===== FOLD {fold} =====")

        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]

        train_loader = DataLoader(
            HierDataset(train_items, train_transform),
            batch_size=32,
            shuffle=True,
            num_workers=4,
            collate_fn=safe_collate,
            worker_init_fn=seed_worker,
            generator=g
        )
        val_loader = DataLoader(
            HierDataset(val_items, val_transform),
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=safe_collate,
            worker_init_fn=seed_worker,
            generator=g
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
