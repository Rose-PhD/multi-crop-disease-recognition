#!/usr/bin/env python3
"""
TESTING: FULLY HIERARCHICAL RESNET-18 (CONCATENATED HEADS)
Multi-region evaluation: testA, testB, testC, testD...

Includes:
 - Corrupted image skipping via safe_collate
 - Per-crop disease accuracy tracking
 - Region-level mean ± std per crop
 - Outputs fully ready for LaTeX table
 - Disease prediction using predicted crop slice and true crop slice (oracle)
"""

import os
from pathlib import Path

import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import (
    safe_collate,
    make_test_transform,
    plot_cm,
    RegionDataset,
    HierResNet18Concat,
    load_label_maps,
    build_region_items,
    compute_region_stats,
    save_region_tables,
)

# ============================================================
#                   CONFIG
# ============================================================
MODEL_ROOT = "/home/nalwangar/finally/logs_hierM"
TEST_ROOT = "/deepstore/datasets/dmb/ComputerVision/biology/testsets7"
SAVE_ROOT = "/home/nalwangar/finally/logs_newh/testM"

N_FOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 4

os.makedirs(SAVE_ROOT, exist_ok=True)


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

            for i in range(imgs.size(0)):
                ci_true = int(yc[i].item())
                ci_pred = int(pred_crop[i].item())
                gi_true = int(yg_global[i].item())

                true_crop_all.append(ci_true)
                pred_crop_all.append(ci_pred)

                if gi_true >= 0:
                    start_pred, end_pred = model.crop_slices[ci_pred]
                    logits_pred = out_dis[i, start_pred:end_pred]
                    local_pred_pred_crop = int(logits_pred.argmax().item())
                    gi_pred_pred_crop = start_pred + local_pred_pred_crop

                    start_true, end_true = model.crop_slices[ci_true]
                    logits_true = out_dis[i, start_true:end_true]
                    local_pred_true_crop = int(logits_true.argmax().item())
                    gi_pred_true_crop = start_true + local_pred_true_crop

                    true_global_all.append(gi_true)
                    pred_global_pred_crop_all.append(gi_pred_pred_crop)
                    pred_global_true_crop_all.append(gi_pred_true_crop)

                    per_crop_results[ci_true]["true"].append(gi_true)
                    per_crop_results[ci_true]["pred"].append(gi_pred_true_crop)

    crop_acc = accuracy_score(true_crop_all, pred_crop_all) if true_crop_all else 0.0
    disease_acc_pred_crop = accuracy_score(true_global_all, pred_global_pred_crop_all) if true_global_all else 0.0
    disease_acc_true_crop = accuracy_score(true_global_all, pred_global_true_crop_all) if true_global_all else 0.0

    per_crop_acc = {
        ci: (accuracy_score(v["true"], v["pred"]) if v["true"] else None)
        for ci, v in per_crop_results.items()
    }

    cm_crop = confusion_matrix(true_crop_all, pred_crop_all, labels=list(range(len(crops))))
    plot_cm(cm_crop, crops, fold_dir / "cm_crop.png", "Crop Confusion Matrix")
    pd.DataFrame(cm_crop, index=crops, columns=crops).to_csv(fold_dir / "cm_crop.csv")

    if true_global_all:
        cm_dis_pred = confusion_matrix(true_global_all, pred_global_pred_crop_all,
                                       labels=list(range(len(global_labels))))
        plot_cm(cm_dis_pred, global_labels, fold_dir / "cm_disease_pred_crop.png",
                "Disease Confusion Matrix (Pred Crop Slice)")
        pd.DataFrame(cm_dis_pred, index=global_labels, columns=global_labels).to_csv(
            fold_dir / "cm_disease_pred_crop.csv"
        )

        cm_dis_true = confusion_matrix(true_global_all, pred_global_true_crop_all,
                                       labels=list(range(len(global_labels))))
        plot_cm(cm_dis_true, global_labels, fold_dir / "cm_disease_true_crop.png",
                "Disease Confusion Matrix (True Crop Slice)")
        pd.DataFrame(cm_dis_true, index=global_labels, columns=global_labels).to_csv(
            fold_dir / "cm_disease_true_crop.csv"
        )

    f1_crop_macro = f1_score(true_crop_all, pred_crop_all, average="macro", zero_division=0)
    f1_crop_weighted = f1_score(true_crop_all, pred_crop_all, average="weighted", zero_division=0)

    if true_global_all:
        f1_disease_pred_crop_macro = f1_score(true_global_all, pred_global_pred_crop_all, average="macro", zero_division=0)
        f1_disease_pred_crop_weighted = f1_score(true_global_all, pred_global_pred_crop_all, average="weighted", zero_division=0)
        f1_disease_true_crop_macro = f1_score(true_global_all, pred_global_true_crop_all, average="macro", zero_division=0)
        f1_disease_true_crop_weighted = f1_score(true_global_all, pred_global_true_crop_all, average="weighted", zero_division=0)
    else:
        f1_disease_pred_crop_macro = f1_disease_pred_crop_weighted = None
        f1_disease_true_crop_macro = f1_disease_true_crop_weighted = None

    return {
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


# ============================================================
#                         MAIN
# ============================================================
def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    crops, diseases_by_crop, global_index, global_labels, _, _ = load_label_maps(MODEL_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        dl = DataLoader(
            RegionDataset(items, tform),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=safe_collate,
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
                flat[f"disease_{crops[ci]}"] = acc

            fold_summaries.append(flat)
            pd.DataFrame([flat]).to_csv(fold_dir / "fold_summary.csv", index=False)

        if not fold_summaries:
            print(f"No folds evaluated for {region}.")
            continue

        df_region = pd.DataFrame(fold_summaries)
        df_region.to_csv(region_save / "summary_folds.csv", index=False)

        region_stats = compute_region_stats(region, crops, df_region, stats)
        pd.DataFrame([region_stats]).to_csv(region_save / "summary_stats.csv", index=False)

        save_region_tables(region, crops, df_region, region_stats, region_save)
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
