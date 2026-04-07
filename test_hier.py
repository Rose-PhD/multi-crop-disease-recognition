#!/usr/bin/env python3
"""
TESTING: FULLY HIERARCHICAL RESNET-18 (CONCATENATED HEADS)
Multi-region evaluation: testA, testB, testC, testD...

Includes:
 - Corrupted image skipping
 - Safe collate to remove None samples
 - Per-crop disease accuracy tracking
 - Region-level mean ± std per crop
 - Outputs fully ready for LaTeX table
 - Both:
    * disease prediction using predicted crop slice
    * disease prediction using true crop slice (oracle crop)
"""

import os
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import default_collate

from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
#                    REPRODUCIBILITY
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
#                   CONFIG
# ============================================================
MODEL_ROOT = "/home/nalwangar/finally/logs_hierM"
TEST_ROOT = "/deepstore/datasets/dmb/ComputerVision/biology/testsets7"
SAVE_ROOT = "/home/nalwangar/finally/logs_newh/testM"

N_FOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 4

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
#            SAFE COLLATE : SKIP CORRUPTED SAMPLES
# ============================================================
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# ============================================================
#     MODEL: ResNet-18 (Option C, concatenated heads)
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

        all_dis_logits = []
        for head in self.heads:
            all_dis_logits.append(head(feats))

        concat_logits = torch.cat(all_dis_logits, dim=1)
        return crop_logits, concat_logits


# ============================================================
#                 TRANSFORMS
# ============================================================
def make_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
#                     DATASET
# ============================================================
class RegionHierDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.t = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, crop_id, dis_local, dis_global = self.items[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            print(f"[WARNING] Skipping corrupted image: {path}")
            return None

        img = self.t(img)
        return img, crop_id, dis_local, dis_global


# ============================================================
#               LABEL MAP LOADING
# ============================================================
def load_label_maps(model_root):
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

    return crops, diseases_by_crop, global_index, global_labels


# ============================================================
#          BUILD REGION ITEMS (per test region)
# ============================================================
def build_region_items(region_root, train_crops, train_dis, global_index):
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
                    if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        stats["skipped_unknown_crop"] += 1
            continue

        ci = train_crops.index(crop_name)

        for dis_dir in sorted([d for d in crop_dir.iterdir() if d.is_dir()]):
            dis_name = dis_dir.name
            train_dis_list = train_dis[crop_name]

            if dis_name in train_dis_list:
                di = train_dis_list.index(dis_name)
                known = True
            else:
                di = -1
                known = False

            for img_path in dis_dir.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                stats["total_images"] += 1

                if known:
                    stats["known_crop_known_disease"] += 1
                    global_idx = global_index[(ci, di)]
                else:
                    stats["known_crop_unknown_disease"] += 1
                    global_idx = -1

                items.append((str(img_path), ci, di, global_idx))

    return items, stats


# ============================================================
#                CONFUSION MATRIX PLOTTER
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
#              EVALUATE ONE FOLD
# ============================================================
def evaluate_region_fold(model, model_path, loader, device, fold_dir, crops, global_labels):
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    true_crop_all, pred_crop_all = [], []
    true_global_all = []
    pred_global_pred_crop_all = []
    pred_global_true_crop_all = []

    per_crop_results = {ci: {"true": [], "pred": []} for ci in range(len(crops))}

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            imgs, yc, yd_local, yg_global = batch
            imgs = imgs.to(device)
            yc = yc.to(device)
            yd_local = yd_local.to(device)
            yg_global = yg_global.to(device)

            out_crop, out_dis = model(imgs)
            pred_crop = out_crop.argmax(1)

            batch_size = imgs.size(0)

            for i in range(batch_size):
                ci_true = int(yc[i].item())
                ci_pred = int(pred_crop[i].item())
                gi_true = int(yg_global[i].item())

                true_crop_all.append(ci_true)
                pred_crop_all.append(ci_pred)

                if gi_true >= 0:
                    # disease prediction using predicted crop slice
                    start_pred, end_pred = model.crop_slices[ci_pred]
                    logits_pred = out_dis[i, start_pred:end_pred]
                    local_pred_pred_crop = int(logits_pred.argmax().item())
                    gi_pred_pred_crop = start_pred + local_pred_pred_crop

                    # disease prediction using true crop slice (oracle crop)
                    start_true, end_true = model.crop_slices[ci_true]
                    logits_true = out_dis[i, start_true:end_true]
                    local_pred_true_crop = int(logits_true.argmax().item())
                    gi_pred_true_crop = start_true + local_pred_true_crop

                    true_global_all.append(gi_true)
                    pred_global_pred_crop_all.append(gi_pred_pred_crop)
                    pred_global_true_crop_all.append(gi_pred_true_crop)

                    per_crop_results[ci_true]["true"].append(gi_true)
                    per_crop_results[ci_true]["pred"].append(gi_pred_pred_crop)

    crop_acc = accuracy_score(true_crop_all, pred_crop_all) if true_crop_all else 0.0
    disease_acc_pred_crop = (
        accuracy_score(true_global_all, pred_global_pred_crop_all) if true_global_all else 0.0
    )
    disease_acc_true_crop = (
        accuracy_score(true_global_all, pred_global_true_crop_all) if true_global_all else 0.0
    )

    per_crop_acc = {}
    for ci in per_crop_results:
        t = per_crop_results[ci]["true"]
        p = per_crop_results[ci]["pred"]
        per_crop_acc[ci] = accuracy_score(t, p) if t else None

    cm_crop = confusion_matrix(true_crop_all, pred_crop_all, labels=list(range(len(crops))))
    plot_cm(cm_crop, crops, fold_dir / "cm_crop.png", "Crop Confusion Matrix")
    pd.DataFrame(cm_crop, index=crops, columns=crops).to_csv(fold_dir / "cm_crop.csv")

    if true_global_all:
        cm_dis_pred = confusion_matrix(
            true_global_all,
            pred_global_pred_crop_all,
            labels=list(range(len(global_labels)))
        )
        plot_cm(
            cm_dis_pred,
            global_labels,
            fold_dir / "cm_disease_pred_crop.png",
            "Disease Confusion Matrix (Pred Crop Slice)"
        )
        pd.DataFrame(cm_dis_pred, index=global_labels, columns=global_labels).to_csv(
            fold_dir / "cm_disease_pred_crop.csv"
        )

        cm_dis_true = confusion_matrix(
            true_global_all,
            pred_global_true_crop_all,
            labels=list(range(len(global_labels)))
        )
        plot_cm(
            cm_dis_true,
            global_labels,
            fold_dir / "cm_disease_true_crop.png",
            "Disease Confusion Matrix (True Crop Slice)"
        )
        pd.DataFrame(cm_dis_true, index=global_labels, columns=global_labels).to_csv(
            fold_dir / "cm_disease_true_crop.csv"
        )

    f1_crop_macro = f1_score(true_crop_all, pred_crop_all, average="macro", zero_division=0)
    f1_crop_weighted = f1_score(true_crop_all, pred_crop_all, average="weighted", zero_division=0)

    if true_global_all:
        f1_disease_pred_crop_macro = f1_score(
            true_global_all, pred_global_pred_crop_all, average="macro", zero_division=0
        )
        f1_disease_pred_crop_weighted = f1_score(
            true_global_all, pred_global_pred_crop_all, average="weighted", zero_division=0
        )

        f1_disease_true_crop_macro = f1_score(
            true_global_all, pred_global_true_crop_all, average="macro", zero_division=0
        )
        f1_disease_true_crop_weighted = f1_score(
            true_global_all, pred_global_true_crop_all, average="weighted", zero_division=0
        )
    else:
        f1_disease_pred_crop_macro = None
        f1_disease_pred_crop_weighted = None
        f1_disease_true_crop_macro = None
        f1_disease_true_crop_weighted = None

    summary = {
        "crop_acc": crop_acc,
        "disease_acc_pred_crop": disease_acc_pred_crop,
        "disease_acc_true_crop": disease_acc_true_crop,
        "n_crop_samples": len(true_crop_all),
        "n_disease_eval_samples": len(true_global_all),
        "f1_crop_macro": f1_crop_macro,
        "f1_crop_weighted": f1_crop_weighted,
        "f1_disease_pred_crop_macro": f1_disease_pred_crop_macro,
        "f1_disease_pred_crop_weighted": f1_disease_pred_crop_weighted,
        "f1_disease_true_crop_macro": f1_disease_true_crop_macro,
        "f1_disease_true_crop_weighted": f1_disease_true_crop_weighted,
        "per_crop_acc": per_crop_acc,
    }
    return summary


# ============================================================
#                         MAIN
# ============================================================
def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    crops, diseases_by_crop, global_index, global_labels = load_label_maps(MODEL_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tform = make_test_transforms()

    region_dirs = sorted(
        [d for d in Path(TEST_ROOT).iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    all_region_summaries = []

    for region_dir in region_dirs:
        region = region_dir.name
        print(f"\n=== Evaluating region: {region} ===")

        items, stats = build_region_items(region_dir, crops, diseases_by_crop, global_index)

        if not items:
            print(f"  No usable images in {region}. Skipping.")
            continue

        region_save = Path(SAVE_ROOT) / region
        region_save.mkdir(parents=True, exist_ok=True)

        ds = RegionHierDataset(items, tform)
        dl = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=safe_collate,
        )

        fold_summaries = []

        for fold in range(1, N_FOLDS + 1):
            print(f"  Fold {fold}...")

            fold_path = Path(MODEL_ROOT) / f"fold{fold}" / "best_model.pth"
            if not fold_path.exists():
                print(f"    Missing: {fold_path}")
                continue

            fold_dir = region_save / f"fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            model = HierResNet18Concat(crops, diseases_by_crop)
            summary = evaluate_region_fold(
                model, fold_path, dl, device, fold_dir, crops, global_labels
            )

            flat = {
                "region": region,
                "fold": fold,
                "crop_acc": summary["crop_acc"],
                "disease_acc_pred_crop": summary["disease_acc_pred_crop"],
                "disease_acc_true_crop": summary["disease_acc_true_crop"],
                "n_crop_samples": summary["n_crop_samples"],
                "n_disease_eval_samples": summary["n_disease_eval_samples"],
                "f1_crop_macro": summary["f1_crop_macro"],
                "f1_crop_weighted": summary["f1_crop_weighted"],
                "f1_disease_pred_crop_macro": summary["f1_disease_pred_crop_macro"],
                "f1_disease_pred_crop_weighted": summary["f1_disease_pred_crop_weighted"],
                "f1_disease_true_crop_macro": summary["f1_disease_true_crop_macro"],
                "f1_disease_true_crop_weighted": summary["f1_disease_true_crop_weighted"],
            }

            for ci, acc in summary["per_crop_acc"].items():
                crop_name = crops[ci]
                flat[f"disease_{crop_name}"] = acc

            fold_summaries.append(flat)
            pd.DataFrame([flat]).to_csv(fold_dir / "fold_summary.csv", index=False)

        if not fold_summaries:
            print(f"No folds evaluated for {region}.")
            continue

        df_region = pd.DataFrame(fold_summaries)
        df_region.to_csv(region_save / "summary_folds.csv", index=False)

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

        pd.DataFrame([region_stats]).to_csv(region_save / "summary_stats.csv", index=False)

        # ============================================================
        #                     TABLE 1: GLOBAL METRICS
        # ============================================================
        def fmt_mean_std(mean_val, std_val):
            if mean_val is None or pd.isna(mean_val):
                return None
            return f"{mean_val:.3f} ± {std_val:.3f}"

        P1_global = fmt_mean_std(region_stats["mean_crop_acc"], region_stats["std_crop_acc"])

        P2_per_crop = {}
        for crop in crops:
            mean_v = region_stats.get(f"mean_disease_{crop}", None)
            std_v = region_stats.get(f"std_disease_{crop}", None)
            P2_per_crop[crop] = fmt_mean_std(mean_v, std_v)

        P3_global = fmt_mean_std(
            region_stats["mean_disease_acc_pred_crop"],
            region_stats["std_disease_acc_pred_crop"],
        )

        table1_row = {"region": region, "P1_global": P1_global}
        for crop in crops:
            table1_row[f"P2_{crop}"] = P2_per_crop[crop]
        table1_row["P3_global"] = P3_global

        df_table1 = pd.DataFrame([table1_row])
        df_table1.to_csv(region_save / "table1_global_metrics.csv", index=False)

        # ============================================================
        #                     TABLE 2: PER-CROP METRICS
        # ============================================================
        per_crop_p1_mean = {}
        per_crop_p1_std = {}

        per_crop_fold_vals = {crop: [] for crop in crops}

        for _, row in df_region.iterrows():
            fold_dir = region_save / f"fold{int(row['fold'])}"
            cm_path = fold_dir / "cm_crop.csv"
            if cm_path.exists():
                cm = pd.read_csv(cm_path, index_col=0).values
                for ci, crop in enumerate(crops):
                    total = cm[ci, :].sum()
                    correct = cm[ci, ci]
                    if total > 0:
                        per_crop_fold_vals[crop].append(correct / total)

        for crop in crops:
            vals = per_crop_fold_vals[crop]
            if len(vals) > 0:
                per_crop_p1_mean[crop] = np.mean(vals)
                per_crop_p1_std[crop] = np.std(vals, ddof=0)
            else:
                per_crop_p1_mean[crop] = None
                per_crop_p1_std[crop] = None

        per_crop_p2_pred_mean = {}
        per_crop_p2_pred_std = {}
        for crop in crops:
            mean_v = region_stats.get(f"mean_disease_{crop}", None)
            std_v = region_stats.get(f"std_disease_{crop}", None)
            per_crop_p2_pred_mean[crop] = mean_v
            per_crop_p2_pred_std[crop] = std_v

        all_means = [
            per_crop_p2_pred_mean[c]
            for c in crops
            if per_crop_p2_pred_mean[c] is not None
        ]
        if len(all_means) > 0:
            P2_global_mean = np.mean(all_means)
            P2_global_std = np.std(all_means, ddof=0)
            P2_global = fmt_mean_std(P2_global_mean, P2_global_std)
        else:
            P2_global = None

        table2_rows = []
        for crop in crops:
            P1_val = fmt_mean_std(per_crop_p1_mean[crop], per_crop_p1_std[crop])
            P2_val = fmt_mean_std(per_crop_p2_pred_mean[crop], per_crop_p2_pred_std[crop])

            table2_rows.append({
                "crop": crop,
                "P1_per_crop": P1_val,
                "P2_per_crop_pred": P2_val,
                "P2_global": P2_global,
            })

        df_table2 = pd.DataFrame(table2_rows)
        df_table2.to_csv(region_save / "table2_per_crop_metrics.csv", index=False)

        all_region_summaries.append(region_stats)

    if all_region_summaries:
        pd.DataFrame(all_region_summaries).to_csv(
            Path(SAVE_ROOT) / "summary_all_regions.csv", index=False
        )
        print("\n=== DONE: Multi-region hierarchical evaluation complete. ===")
    else:
        print("\nNo regions evaluated.")


if __name__ == "__main__":
    main()

