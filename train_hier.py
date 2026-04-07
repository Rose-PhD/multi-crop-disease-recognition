#!/usr/bin/env python3
"""
TRAINING: FULLY HIERARCHICAL RESNET-18 (OPTION C CONCATENATED HEADS)
- Fully hierarchical + concatenated disease heads
- Two-stage (crop -> disease slice) in BOTH training and evaluation
- Fine-tune last block (layer4)
- 5-fold CV
- Confusion matrices + CSV summaries + label maps for testing
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
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import default_collate

# ------------------------------------------------------------
#                    REPRODUCIBILITY
# ------------------------------------------------------------
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
#                 MODEL  OPTION C (CONCATENATED HEADS)
#              FULLY HIERARCHICAL TWO-STAGE VERSION
# ============================================================
class HierResNet18Concat(nn.Module):
    """
    Option C (fully hierarchical):
    - Backbone features
    - Parallel per-crop disease heads
    - Concatenate all disease head outputs into ONE long vector
    - Training: disease loss computed ONLY on the correct crop slice
    - Inference: crop predicted first, then disease predicted within crop slice
    """
    def __init__(self, crops, diseases_by_crop):
        super().__init__()

        self.crops = crops
        self.diseases_by_crop = diseases_by_crop

        # 1. Load pretrained ResNet-18 with offline fallback
        try:
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("Loaded ResNet18 pretrained weights.")
        except Exception:
            print("Offline mode: loading local ResNet-18 weights.")
            backbone = models.resnet18(weights=None)
            local_path = "/home/nalwangar/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
            backbone.load_state_dict(torch.load(local_path, map_location="cpu"))

        # 2. Fine-tune ONLY layer4
        for name, p in backbone.named_parameters():
            p.requires_grad = ("layer4" in name)

        # 3. Replace FC with Identity
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # 4. Crop classifier
        self.crop_head = nn.Linear(in_features, len(crops))

        # 5. Disease heads for each crop
        self.crop_names = list(crops)
        self.heads = nn.ModuleList([
            nn.Linear(in_features, len(diseases_by_crop[c]))
            for c in self.crop_names
        ])

        # 6. Build indexing map for concatenation
        #    global index = concat(head_0, head_1, ...)
        self.global_labels = []
        self.offsets = {}   # maps (crop_idx, dis_idx) -> global index
        index = 0
        for ci, crop in enumerate(crops):
            for di, dis in enumerate(diseases_by_crop[crop]):
                self.offsets[(ci, di)] = index
                self.global_labels.append(f"{crop}:{dis}")
                index += 1
        self.total_diseases = len(self.global_labels)

        # 7. Build crop slices over the concatenated disease vector
        #    crop_slices[ci] = (start, end) for that crop in concat logits
        self.crop_slices = {}
        start = 0
        for ci, crop in enumerate(crops):
            n_dis = len(diseases_by_crop[crop])
            self.crop_slices[ci] = (start, start + n_dis)
            start += n_dis

    def forward(self, x):
        feats = self.backbone(x)

        # Crop prediction
        crop_logits = self.crop_head(feats)

        # Concatenate all disease heads
        all_dis_logits = []
        for head in self.heads:
            all_dis_logits.append(head(feats))  # shape (B, n_dis_for_that_crop)

        concat_logits = torch.cat(all_dis_logits, dim=1)  # (B, total_diseases)

        return crop_logits, concat_logits


# ============================================================
#                       DATASET
# ============================================================
class HierDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.t = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):   # ← inside class
        path, crop_id, dis_id = self.items[idx]

        # ---- SAFE LOAD ----
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
    """Train-time transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def make_eval_transform():
    """Eval-time transforms (no random augmentation)."""
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
#         (TWO-STAGE: crop -> disease slice -> global ID)
# ============================================================
def evaluate(model, model_path, val_loader, device, fold_dir, crops, global_labels):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    true_crop, pred_crop = [], []
    true_global, pred_global = [], []

    # For oracle-crop metric
    pred_global_true_crop = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:   # <<< added to handle safe_collate(None)
                continue

            imgs, yc, yd, yg = batch
            imgs = imgs.to(device)
            yc = yc.to(device)
            yd = yd.to(device)
            yg = yg.to(device)

            out_crop, out_dis = model(imgs)

            # --- Stage 1: predicted crop ---
            pred_c = out_crop.argmax(1)

            for i in range(imgs.size(0)):
                ci_pred = int(pred_c[i].item())
                ci_true = int(yc[i].item())
                di_true = int(yd[i].item())

                # --- Metric 1: Disease prediction using *predicted crop slice* ---
                start_pred, end_pred = model.crop_slices[ci_pred]
                local_logits_pred = out_dis[i, start_pred:end_pred]
                local_pred_pred_crop = int(local_logits_pred.argmax().item())
                global_pred_pred_crop = items_global_map[(ci_pred, local_pred_pred_crop)]

                # --- Metric 2: Disease prediction using *true crop slice* (oracle crop) ---
                start_true, end_true = model.crop_slices[ci_true]
                local_logits_true = out_dis[i, start_true:end_true]
                local_pred_true_crop = int(local_logits_true.argmax().item())
                global_pred_true_crop = items_global_map[(ci_true, local_pred_true_crop)]

                # --- Save ground-truth and predictions ---
                true_crop.append(ci_true)
                pred_crop.append(ci_pred)
                true_global.append(int(yg[i].item()))
                pred_global.append(global_pred_pred_crop)
                pred_global_true_crop.append(global_pred_true_crop)

    # --- Accuracies ---
    crop_acc = accuracy_score(true_crop, pred_crop)
    disease_acc_pred_crop = accuracy_score(true_global, pred_global)
    disease_acc_true_crop = accuracy_score(true_global, pred_global_true_crop)

    # --- Confusion matrices (use disease_acc_pred_crop for the main confusion matrix) ---
    cm_crop = confusion_matrix(true_crop, pred_crop, labels=range(len(crops)))
    plot_confusion_matrix(cm_crop, crops, fold_dir / "cm_crop.png", "Crop Confusion Matrix")
    pd.DataFrame(cm_crop, index=crops, columns=crops).to_csv(fold_dir / "cm_crop.csv")

    cm_dis = confusion_matrix(true_global, pred_global, labels=range(len(global_labels)))
    plot_confusion_matrix(cm_dis, global_labels, fold_dir / "cm_disease_pred_crop.png",
                          "Disease Confusion Matrix (Pred Crop Slice)")
    pd.DataFrame(cm_dis, index=global_labels, columns=global_labels).to_csv(
        fold_dir / "cm_disease_pred_crop.csv"
    )

    # Also confusion matrix for oracle crop
    cm_dis_oracle = confusion_matrix(true_global, pred_global_true_crop, labels=range(len(global_labels)))
    plot_confusion_matrix(cm_dis_oracle, global_labels, fold_dir / "cm_disease_true_crop.png",
                          "Disease Confusion Matrix (True Crop Slice)")
    pd.DataFrame(cm_dis_oracle, index=global_labels, columns=global_labels).to_csv(
        fold_dir / "cm_disease_true_crop.csv"
    )

    # --- Summary ---
    from sklearn.metrics import f1_score

    # ---- F1-score computations ----
    f1_disease_pred_crop_macro = f1_score(true_global, pred_global, average='macro', zero_division=0)
    f1_disease_pred_crop_weighted = f1_score(true_global, pred_global, average='weighted', zero_division=0)

    f1_disease_true_crop_macro = f1_score(true_global, pred_global_true_crop, average='macro', zero_division=0)
    f1_disease_true_crop_weighted = f1_score(true_global, pred_global_true_crop, average='weighted', zero_division=0)

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
#       (TWO-STAGE HIERARCHICAL LOSS WITH CONCATENATED HEADS)
# ============================================================
def train_fold(fold, model, train_loader, val_loader, device, fold_dir, crops, global_labels):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    best_loss = float("inf")
    wait = 0
    patience = 7

    model_path = fold_dir / "best_model.pth"

    for epoch in range(1, 50 + 1):
        model.train()
        total_train = 0.0

        # ------------------ TRAINING LOOP ------------------
        for batch in train_loader:
            if batch is None:
                continue
            imgs, yc, yd, yg = batch

            imgs = imgs.to(device)
            yc = yc.to(device)      # crop labels (local)
            yd = yd.to(device)      # disease labels within crop (local)
            # yg = yg.to(device)    # global disease index (not needed for loss)

            optimizer.zero_grad()
            out_crop, out_dis = model(imgs)

            # Crop loss (normal)
            loss_crop = criterion(out_crop, yc)

            # Disease loss (fully hierarchical, two-stage):
            # For each sample, only the slice corresponding to the TRUE crop is trained.
            loss_dis = 0.0
            batch_size = imgs.size(0)

            for i in range(batch_size):
                ci_true = int(yc[i].item())         # true crop index
                di_true = yd[i].unsqueeze(0)        # true disease index within crop
                start, end = model.crop_slices[ci_true]
                slice_logits = out_dis[i, start:end].unsqueeze(0)  # shape (1, n_dis_for_crop)
                loss_dis = loss_dis + criterion(slice_logits, di_true)

            loss_dis = loss_dis / batch_size

            loss = loss_crop + loss_dis
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        # ------------------ VALIDATION LOOP ------------------
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

                # Validation crop loss
                loss_crop_val = criterion(out_crop, yc)

                # Validation disease loss (same hierarchical logic)
                loss_dis_val = 0.0
                batch_size = imgs.size(0)
                for i in range(batch_size):
                    ci_true = int(yc[i].item())
                    di_true = yd[i].unsqueeze(0)
                    start, end = model.crop_slices[ci_true]
                    slice_logits = out_dis[i, start:end].unsqueeze(0)
                    loss_dis_val = loss_dis_val + criterion(slice_logits, di_true)

                loss_dis_val = loss_dis_val / batch_size
                loss_val = loss_crop_val + loss_dis_val
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

    # Final evaluation on validation set using two-stage hierarchical prediction
    eval_summary = evaluate(
        model, model_path, val_loader, device, fold_dir, crops, global_labels
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
    SAVE_ROOT = "/home/nalwangar/finally/logs_hierM"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # Load dataset structure
    crops, diseases_by_crop, items = build_index(DATASET)

    # Save label maps for test script
    with open(f"{SAVE_ROOT}/label_maps.json", "w") as f:
        json.dump({
            "crops": crops,
            "diseases_within_crop": diseases_by_crop
        }, f, indent=4)

    # Global map for Option C (crop,disease) -> global index
    global items_global_map
    global_index = {}
    labels = []
    idx = 0
    for ci, crop in enumerate(crops):
        for di, dis in enumerate(diseases_by_crop[crop]):
            global_index[(ci, di)] = idx
            labels.append(f"{crop}:{dis}")
            idx += 1

    items_global_map = global_index
    global_labels = labels

    paths = [p for p, _, _ in items]

    # CHANGED: stratify by joint (crop, disease) class
    joint_labels = [items_global_map[(c, d)] for _, c, d in items]

    # separate train/eval transforms
    train_transform = make_train_transform()
    val_transform = make_eval_transform()

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

        model = HierResNet18Concat(crops, diseases_by_crop).to(device)

        fold_dir = Path(SAVE_ROOT) / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        summary = train_fold(
            fold, model, train_loader, val_loader, device,
            fold_dir, crops, global_labels
        )
        fold_results.append(summary)

    # Global summary
    pd.DataFrame(fold_results).to_csv(f"{SAVE_ROOT}/summary_all_folds.csv", index=False)
    print("\n=== TRAINING COMPLETE: FULLY HIERARCHICAL OPTION C (CONCATENATED HEADS) ===")


if __name__ == "__main__":
    main()
