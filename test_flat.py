
#!/usr/bin/env python3
"""
TESTING: FLAT RESNET-18 BASELINE (JOINT CROP+DISEASE LABELS)

Multi-region evaluation: testA, testB, testC, testD...

Matches the improved flat training script:
- Same FlatResNet18 architecture
- Same label_maps.json loading
- Same global joint-class mapping over (crop, disease)
- Reports both:
    * disease prediction from flat predicted joint class
    * oracle-crop disease prediction (restrict logits to true crop classes)

Outputs:
- Per-fold CSV summaries
- Region-level mean ± std CSV summaries
- Crop and disease confusion matrices
- Table-ready CSVs
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

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
#                    PATHS / CONFIG
# ============================================================
MODEL_ROOT = "/home/nalwangar/finally/logs_flatD"
TEST_ROOT = "/deepstore/datasets/dmb/ComputerVision/biology/testsets7"
SAVE_ROOT = "/home/nalwangar/finally/logs_newf/testd"

N_FOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 4

os.makedirs(SAVE_ROOT, exist_ok=True)


# ============================================================
#                         MODEL
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
#               TRANSFORMS & SAFE COLLATE
# ============================================================
def make_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# ============================================================
#                     DATASET
# ============================================================
class RegionFlatDataset(Dataset):
    """
    items: list of (img_path, crop_id, dis_local, global_joint_id)
           global_joint_id = -1 if disease is unknown
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

    global_to_crop_dis = {gid: (ci, di) for (ci, di), gid in global_index.items()}

    crop_to_global_ids = {}
    for (ci, di), gid in global_index.items():
        crop_to_global_ids.setdefault(ci, []).append(gid)

    return crops, diseases_by_crop, global_index, global_labels, global_to_crop_dis, crop_to_global_ids


# ============================================================
#          BUILD REGION ITEMS
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
                    if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
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
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
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
def evaluate_region_fold(
    model,
    model_path,
    loader,
    device,
    fold_dir,
    crops,
    global_labels,
    global_to_crop_dis,
    crop_to_global_ids,
):
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
            yg_global = yg_global.to(device)

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            for i in range(imgs.size(0)):
                ci_true = int(yc[i].item())
                gi_true = int(yg_global[i].item())

                # Crop prediction is always available from flat predicted joint class
                gi_pred = int(preds[i].item())
                ci_pred, di_pred = global_to_crop_dis[gi_pred]

                true_crop_all.append(ci_true)
                pred_crop_all.append(ci_pred)

                # Disease metrics only for known diseases
                if gi_true >= 0:
                    # predicted joint class
                    pred_global_pred_crop_all.append(gi_pred)
                    true_global_all.append(gi_true)

                    # oracle-crop disease prediction:
                    # restrict logits to classes belonging to the true crop
                    crop_global_ids = crop_to_global_ids[ci_true]
                    logits_i = logits[i]
                    crop_logits = torch.stack([logits_i[g_id] for g_id in crop_global_ids], dim=0)
                    local_pred_idx = int(crop_logits.argmax().item())
                    gi_pred_true_crop = int(crop_global_ids[local_pred_idx])
                    pred_global_true_crop_all.append(gi_pred_true_crop)

                    per_crop_results[ci_true]["true"].append(gi_true)
                    per_crop_results[ci_true]["pred"].append(gi_pred)

    crop_acc = accuracy_score(true_crop_all, pred_crop_all) if true_crop_all else 0.0
    disease_acc_pred_crop = (
        accuracy_score(true_global_all, pred_global_pred_crop_all) if true_global_all else 0.0
    )
    disease_acc_true_crop = (
        accuracy_score(true_global_all, pred_global_true_crop_all) if true_global_all else 0.0
    )

    cm_crop = confusion_matrix(true_crop_all, pred_crop_all, labels=list(range(len(crops))))
    plot_cm(cm_crop, crops, fold_dir / "cm_crop.png", "Crop Confusion Matrix")
    pd.DataFrame(cm_crop, index=crops, columns=crops).to_csv(fold_dir / "cm_crop.csv")

    if true_global_all:
        cm_dis_pred = confusion_matrix(
            true_global_all,
            pred_global_pred_crop_all,
            labels=list(range(len(global_labels))),
        )
        plot_cm(
            cm_dis_pred,
            global_labels,
            fold_dir / "cm_disease_pred_crop.png",
            "Disease Confusion Matrix (Pred Joint Class)",
        )
        pd.DataFrame(cm_dis_pred, index=global_labels, columns=global_labels).to_csv(
            fold_dir / "cm_disease_pred_crop.csv"
        )

        cm_dis_true = confusion_matrix(
            true_global_all,
            pred_global_true_crop_all,
            labels=list(range(len(global_labels))),
        )
        plot_cm(
            cm_dis_true,
            global_labels,
            fold_dir / "cm_disease_true_crop.png",
            "Disease Confusion Matrix (Oracle Crop Mask)",
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

    per_crop_acc = {}
    for ci in per_crop_results:
        t = per_crop_results[ci]["true"]
        p = per_crop_results[ci]["pred"]
        per_crop_acc[ci] = accuracy_score(t, p) if t else None

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
#                            MAIN
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        crops,
        diseases_by_crop,
        global_index,
        global_labels,
        global_to_crop_dis,
        crop_to_global_ids,
    ) = load_label_maps(MODEL_ROOT)

    num_joint_classes = len(global_labels)
    tform = make_test_transform()

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

        ds = RegionFlatDataset(items, tform)
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

            model = FlatResNet18(num_joint_classes=num_joint_classes)
            summary = evaluate_region_fold(
                model,
                fold_path,
                dl,
                device,
                fold_dir,
                crops,
                global_labels,
                global_to_crop_dis,
                crop_to_global_ids,
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

        def fmt_mean_std(mean_val, std_val):
            if mean_val is None or pd.isna(mean_val):
                return None
            return f"{mean_val:.3f} ± {std_val:.3f}"

        # Table 1
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

        pd.DataFrame([table1_row]).to_csv(region_save / "table1_global_metrics.csv", index=False)

        # Table 2
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

            table2_rows.append(
                {
                    "crop": crop,
                    "P1_per_crop": P1_val,
                    "P2_per_crop_pred": P2_val,
                    "P2_global": P2_global,
                }
            )

        pd.DataFrame(table2_rows).to_csv(region_save / "table2_per_crop_metrics.csv", index=False)

        all_region_summaries.append(region_stats)

    if all_region_summaries:
        pd.DataFrame(all_region_summaries).to_csv(
            Path(SAVE_ROOT) / "summary_all_regions.csv", index=False
        )
        print("\n=== DONE: Multi-region flat evaluation complete. ===")
    else:
        print("\nNo regions evaluated.")


if __name__ == "__main__":
    main()
