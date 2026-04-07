#!/usr/bin/env python3
"""
utils.py — Shared utilities for flat and hierarchical crop disease recognition.

Imported by: train_flat.py, train_hier.py, test_flat.py, test_hier.py
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def make_eval_transform():
    """Shared eval/val/test transform — no random augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# Aliases for clarity at call sites
make_val_transform = make_eval_transform
make_test_transform = make_eval_transform


# ============================================================
#                     BUILD DATA INDEX
# ============================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_index(dataset_root):
    """
    Expects dataset_root/<crop>/<disease>/<images>.
    Returns crops, diseases_by_crop, items=(img_path, crop_idx, disease_idx).
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
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                items.append((str(img), ci, di))

    return crops, diseases_by_crop, items


# ============================================================
#                   CONFUSION MATRIX PLOTTER
# ============================================================
def plot_cm(cm, labels, save_path, title):
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
#                   DATASET — TRAINING
# ============================================================
class HierDataset(Dataset):
    """
    Training dataset.
    items: list of (img_path, crop_id, dis_id)
    global_map: dict[(crop_id, dis_id)] -> global_joint_id
    Returns: img, crop_id, dis_id, global_joint_id
    """

    def __init__(self, items, transform, global_map):
        self.items = items
        self.t = transform
        self.global_map = global_map

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
        global_dis_id = self.global_map[(crop_id, dis_id)]
        return img, crop_id, dis_id, global_dis_id


# ============================================================
#                   DATASET — REGION / TEST
# ============================================================
class RegionDataset(Dataset):
    """
    Test/region dataset (flat and hierarchical).
    items: list of (img_path, crop_id, dis_local, global_joint_id)
           global_joint_id = -1 if disease is unknown.
    Returns: img, crop_id, dis_local, global_joint_id
    """

    def __init__(self, items, transform):
        self.items = items
        self.t = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, crop_id, dis_local, global_joint_id = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            print(f"[WARNING] Skipping corrupted image: {path}")
            return None
        img = self.t(img)
        return img, crop_id, dis_local, global_joint_id


# ============================================================
#                    MODEL — FLAT RESNET-18
# ============================================================
class FlatResNet18(nn.Module):
    """
    Flat baseline:
    - Shared ResNet-18 backbone
    - Single linear classifier over all (crop, disease) joint classes
    - Fine-tune only layer4 + final classifier
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
#           MODEL — HIERARCHICAL RESNET-18 (OPTION C)
# ============================================================
class HierResNet18Concat(nn.Module):
    """
    Option C (fully hierarchical):
    - Backbone features
    - Parallel per-crop disease heads concatenated into one vector
    - Training: disease loss only on the correct crop slice
    - Inference: crop predicted first, then disease within crop slice
    """

    def __init__(self, crops, diseases_by_crop):
        super().__init__()

        self.crops = crops
        self.diseases_by_crop = diseases_by_crop

        try:
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("Loaded ResNet18 pretrained weights.")
        except Exception:
            print("Offline mode: loading local ResNet-18 weights.")
            backbone = models.resnet18(weights=None)
            local_path = "/home/nalwangar/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
            backbone.load_state_dict(torch.load(local_path, map_location="cpu"))

        for name, p in backbone.named_parameters():
            p.requires_grad = ("layer4" in name)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.crop_head = nn.Linear(in_features, len(crops))

        self.crop_names = list(crops)
        self.heads = nn.ModuleList([
            nn.Linear(in_features, len(diseases_by_crop[c]))
            for c in self.crop_names
        ])

        self.global_labels = []
        self.offsets = {}
        index = 0
        for ci, crop in enumerate(crops):
            for di, dis in enumerate(diseases_by_crop[crop]):
                self.offsets[(ci, di)] = index
                self.global_labels.append(f"{crop}:{dis}")
                index += 1
        self.total_diseases = len(self.global_labels)

        self.crop_slices = {}
        start = 0
        for ci, crop in enumerate(crops):
            n_dis = len(diseases_by_crop[crop])
            self.crop_slices[ci] = (start, start + n_dis)
            start += n_dis

    def forward(self, x):
        feats = self.backbone(x)
        crop_logits = self.crop_head(feats)
        concat_logits = torch.cat([head(feats) for head in self.heads], dim=1)
        return crop_logits, concat_logits


# ============================================================
#            LABEL MAP LOADING (test scripts)
# ============================================================
def load_label_maps(model_root):
    """
    Load label_maps.json saved during training.
    Returns:
        crops, diseases_by_crop, global_index,
        global_labels, global_to_crop_dis, crop_to_global_ids
    """
    with open(Path(model_root) / "label_maps.json", "r") as f:
        lm = json.load(f)

    crops = lm["crops"]
    diseases_by_crop = lm["diseases_within_crop"]

    global_index = {}
    global_labels = []
    idx = 0
    for ci, crop in enumerate(crops):
        for di, dis in enumerate(diseases_by_crop[crop]):
            global_index[(ci, di)] = idx
            global_labels.append(f"{crop}:{dis}")
            idx += 1

    global_to_crop_dis = {gid: (ci, di) for (ci, di), gid in global_index.items()}

    crop_to_global_ids = {}
    for (ci, di), gid in global_index.items():
        crop_to_global_ids.setdefault(ci, []).append(gid)

    return crops, diseases_by_crop, global_index, global_labels, global_to_crop_dis, crop_to_global_ids


# ============================================================
#             BUILD REGION ITEMS (test scripts)
# ============================================================
def build_region_items(region_root, train_crops, train_dis, global_index):
    """
    Scan a test-region directory and match images to training labels.
    Returns items list and stats dict.
    """
    items = []
    stats = {
        "total_images": 0,
        "skipped_unknown_crop": 0,
        "known_crop_known_disease": 0,
        "known_crop_unknown_disease": 0,
    }

    region_root = Path(region_root)
    if not region_root.exists():
        return items, stats

    for crop_dir in sorted([d for d in region_root.iterdir() if d.is_dir()]):
        crop_name = crop_dir.name

        if crop_name not in train_crops:
            for dis_dir in [d for d in crop_dir.iterdir() if d.is_dir()]:
                for img in dis_dir.glob("*"):
                    if img.suffix.lower() in IMG_EXTS:
                        stats["skipped_unknown_crop"] += 1
            continue

        ci = train_crops.index(crop_name)
        train_dis_list = train_dis[crop_name]

        for dis_dir in sorted([d for d in crop_dir.iterdir() if d.is_dir()]):
            dis_name = dis_dir.name

            if dis_name in train_dis_list:
                di = train_dis_list.index(dis_name)
                known = True
            else:
                di = -1
                known = False

            for img_path in dis_dir.glob("*"):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                stats["total_images"] += 1
                if known:
                    stats["known_crop_known_disease"] += 1
                    global_joint_id = global_index[(ci, di)]
                else:
                    stats["known_crop_unknown_disease"] += 1
                    global_joint_id = -1
                items.append((str(img_path), ci, di, global_joint_id))

    return items, stats


# ============================================================
#              REPORTING UTILS (test scripts)
# ============================================================
def fmt_mean_std(mean_val, std_val):
    if mean_val is None or pd.isna(mean_val):
        return None
    return f"{mean_val:.3f} ± {std_val:.3f}"


def compute_region_stats(region, crops, df_region, stats):
    """Aggregate fold-level metrics into region-level mean/std summary."""
    region_stats = {
        "region": region,
        "total_images": stats["total_images"],
        "skipped_unknown_crop": stats["skipped_unknown_crop"],
        "known_crop_known_disease": stats["known_crop_known_disease"],
        "known_crop_unknown_disease": stats["known_crop_unknown_disease"],
        "mean_crop_acc": df_region["crop_acc"].mean(),
        "std_crop_acc": df_region["crop_acc"].std(ddof=0),
        "mean_disease_acc_pred_crop": df_region["disease_acc_pred_crop"].mean(),
        "std_disease_acc_pred_crop": df_region["disease_acc_pred_crop"].std(ddof=0),
        "mean_disease_acc_true_crop": df_region["disease_acc_true_crop"].mean(),
        "std_disease_acc_true_crop": df_region["disease_acc_true_crop"].std(ddof=0),
        "mean_f1_crop_macro": df_region["f1_crop_macro"].mean(),
        "std_f1_crop_macro": df_region["f1_crop_macro"].std(ddof=0),
        "mean_f1_crop_weighted": df_region["f1_crop_weighted"].mean(),
        "std_f1_crop_weighted": df_region["f1_crop_weighted"].std(ddof=0),
        "mean_f1_disease_pred_crop_macro": df_region["f1_disease_pred_crop_macro"].mean(),
        "std_f1_disease_pred_crop_macro": df_region["f1_disease_pred_crop_macro"].std(ddof=0),
        "mean_f1_disease_pred_crop_weighted": df_region["f1_disease_pred_crop_weighted"].mean(),
        "std_f1_disease_pred_crop_weighted": df_region["f1_disease_pred_crop_weighted"].std(ddof=0),
        "mean_f1_disease_true_crop_macro": df_region["f1_disease_true_crop_macro"].mean(),
        "std_f1_disease_true_crop_macro": df_region["f1_disease_true_crop_macro"].std(ddof=0),
        "mean_f1_disease_true_crop_weighted": df_region["f1_disease_true_crop_weighted"].mean(),
        "std_f1_disease_true_crop_weighted": df_region["f1_disease_true_crop_weighted"].std(ddof=0),
    }

    for crop in crops:
        col = f"disease_{crop}"
        if col in df_region.columns:
            region_stats[f"mean_{col}"] = df_region[col].mean()
            region_stats[f"std_{col}"] = df_region[col].std(ddof=0)
        else:
            region_stats[f"mean_{col}"] = None
            region_stats[f"std_{col}"] = None

    return region_stats


def save_region_tables(region, crops, df_region, region_stats, region_save):
    """Write table1 (global metrics) and table2 (per-crop metrics) CSVs."""
    # --- Table 1 ---
    P1_global = fmt_mean_std(region_stats["mean_crop_acc"], region_stats["std_crop_acc"])

    P2_per_crop = {
        crop: fmt_mean_std(
            region_stats.get(f"mean_disease_{crop}"),
            region_stats.get(f"std_disease_{crop}"),
        )
        for crop in crops
    }

    P3_global = fmt_mean_std(
        region_stats["mean_disease_acc_pred_crop"],
        region_stats["std_disease_acc_pred_crop"],
    )

    table1_row = {"region": region, "P1_global": P1_global}
    for crop in crops:
        table1_row[f"P2_{crop}"] = P2_per_crop[crop]
    table1_row["P3_global"] = P3_global
    pd.DataFrame([table1_row]).to_csv(region_save / "table1_global_metrics.csv", index=False)

    # --- Table 2 ---
    per_crop_fold_vals = {crop: [] for crop in crops}
    for _, row in df_region.iterrows():
        cm_path = region_save / f"fold{int(row['fold'])}" / "cm_crop.csv"
        if cm_path.exists():
            cm = pd.read_csv(cm_path, index_col=0).values
            for ci, crop in enumerate(crops):
                total = cm[ci, :].sum()
                if total > 0:
                    per_crop_fold_vals[crop].append(cm[ci, ci] / total)

    per_crop_p1 = {}
    for crop in crops:
        vals = per_crop_fold_vals[crop]
        if vals:
            per_crop_p1[crop] = (np.mean(vals), np.std(vals, ddof=0))
        else:
            per_crop_p1[crop] = (None, None)

    per_crop_p2 = {
        crop: (
            region_stats.get(f"mean_disease_{crop}"),
            region_stats.get(f"std_disease_{crop}"),
        )
        for crop in crops
    }

    all_means = [v for c in crops for v in [per_crop_p2[c][0]] if v is not None]
    if all_means:
        P2_global = fmt_mean_std(np.mean(all_means), np.std(all_means, ddof=0))
    else:
        P2_global = None

    table2_rows = [
        {
            "crop": crop,
            "P1_per_crop": fmt_mean_std(*per_crop_p1[crop]),
            "P2_per_crop_pred": fmt_mean_std(*per_crop_p2[crop]),
            "P2_global": P2_global,
        }
        for crop in crops
    ]
    pd.DataFrame(table2_rows).to_csv(region_save / "table2_per_crop_metrics.csv", index=False)
