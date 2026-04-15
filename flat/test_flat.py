#!/usr/bin/env python3
"""
TESTING: FLAT RESNET-18 BASELINE (JOINT CROP+DISEASE LABELS)

Multi-region evaluation across all trained variants discovered from MODEL_BASE.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from configs.paths import (
    FLAT_SAVE_ROOT as MODEL_BASE,
    FLAT_TEST_ROOT as TEST_ROOT,
    FLAT_TEST_SAVE_ROOT as SAVE_ROOT,
    N_FOLDS, BATCH_SIZE, NUM_WORKERS,
)
from utils import (
    safe_collate,
    make_test_transform,
    plot_cm,
    RegionDataset,
    FlatResNet18,
    load_label_maps,
    build_region_items,
    compute_region_stats,
    save_region_tables,
    setup_logger,
)

logger = logging.getLogger("flat.test")


def discover_configs(model_base):
    """Read config.json from every variant sub-folder written by train_flat.py."""
    configs = []
    for d in sorted(Path(model_base).iterdir()):
        cfg_path = d / "config.json"
        if d.is_dir() and cfg_path.exists():
            with open(cfg_path) as f:
                configs.append(json.load(f))
    if not configs:
        raise FileNotFoundError(
            f"No variant config.json files found under {model_base}. "
            "Run train_flat.py first."
        )
    return configs


# fold evaluation
def evaluate_region_fold(model, model_path, loader, device, fold_dir,
                         crops, global_labels, global_to_crop_dis, crop_to_global_ids):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    true_crop_all, pred_crop_all          = [], []
    true_global_all                       = []
    pred_global_pred_crop_all             = []
    pred_global_true_crop_all             = []
    per_crop_results = {ci: {"true": [], "pred": []} for ci in range(len(crops))}

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, yc, _, yg_global = batch
            imgs      = imgs.to(device)
            yc        = yc.to(device)
            yg_global = yg_global.to(device)

            logits = model(imgs)
            preds  = logits.argmax(dim=1)

            for i in range(imgs.size(0)):
                ci_true = int(yc[i].item())
                gi_true = int(yg_global[i].item())
                gi_pred = int(preds[i].item())
                ci_pred, _ = global_to_crop_dis[gi_pred]

                true_crop_all.append(ci_true)
                pred_crop_all.append(ci_pred)

                if gi_true >= 0:
                    true_global_all.append(gi_true)
                    pred_global_pred_crop_all.append(gi_pred)

                    crop_ids    = crop_to_global_ids[ci_true]
                    crop_logits = torch.stack([logits[i][g] for g in crop_ids])
                    gi_oracle   = int(crop_ids[crop_logits.argmax().item()])
                    pred_global_true_crop_all.append(gi_oracle)

                    per_crop_results[ci_true]["true"].append(gi_true)
                    per_crop_results[ci_true]["pred"].append(gi_oracle)

    crop_acc              = accuracy_score(true_crop_all, pred_crop_all)              if true_crop_all  else 0.0
    disease_acc_pred_crop = accuracy_score(true_global_all, pred_global_pred_crop_all) if true_global_all else 0.0
    disease_acc_true_crop = accuracy_score(true_global_all, pred_global_true_crop_all) if true_global_all else 0.0

    cm_crop = confusion_matrix(true_crop_all, pred_crop_all, labels=list(range(len(crops))))
    plot_cm(cm_crop, crops, fold_dir / "cm_crop.png", "Crop Confusion Matrix")
    pd.DataFrame(cm_crop, index=crops, columns=crops).to_csv(fold_dir / "cm_crop.csv")

    if true_global_all:
        cm_pred = confusion_matrix(true_global_all, pred_global_pred_crop_all,
                                   labels=list(range(len(global_labels))))
        plot_cm(cm_pred, global_labels, fold_dir / "cm_disease_pred_crop.png",
                "Disease Confusion Matrix (Pred Joint Class)")
        pd.DataFrame(cm_pred, index=global_labels, columns=global_labels).to_csv(
            fold_dir / "cm_disease_pred_crop.csv"
        )

        cm_true = confusion_matrix(true_global_all, pred_global_true_crop_all,
                                   labels=list(range(len(global_labels))))
        plot_cm(cm_true, global_labels, fold_dir / "cm_disease_true_crop.png",
                "Disease Confusion Matrix (Oracle Crop Mask)")
        pd.DataFrame(cm_true, index=global_labels, columns=global_labels).to_csv(
            fold_dir / "cm_disease_true_crop.csv"
        )

    f1_crop_macro    = f1_score(true_crop_all, pred_crop_all, average="macro",    zero_division=0)
    f1_crop_weighted = f1_score(true_crop_all, pred_crop_all, average="weighted", zero_division=0)

    if true_global_all:
        f1_dis_pred_macro    = f1_score(true_global_all, pred_global_pred_crop_all, average="macro",    zero_division=0)
        f1_dis_pred_weighted = f1_score(true_global_all, pred_global_pred_crop_all, average="weighted", zero_division=0)
        f1_dis_true_macro    = f1_score(true_global_all, pred_global_true_crop_all, average="macro",    zero_division=0)
        f1_dis_true_weighted = f1_score(true_global_all, pred_global_true_crop_all, average="weighted", zero_division=0)
    else:
        f1_dis_pred_macro = f1_dis_pred_weighted = None
        f1_dis_true_macro = f1_dis_true_weighted = None

    per_crop_acc = {
        ci: (accuracy_score(v["true"], v["pred"]) if v["true"] else None)
        for ci, v in per_crop_results.items()
    }

    return {
        "crop_acc":                      crop_acc,
        "disease_acc_pred_crop":         disease_acc_pred_crop,
        "disease_acc_true_crop":         disease_acc_true_crop,
        "n_crop_samples":                len(true_crop_all),
        "n_disease_eval_samples":        len(true_global_all),
        "f1_crop_macro":                 f1_crop_macro,
        "f1_crop_weighted":              f1_crop_weighted,
        "f1_disease_pred_crop_macro":    f1_dis_pred_macro,
        "f1_disease_pred_crop_weighted": f1_dis_pred_weighted,
        "f1_disease_true_crop_macro":    f1_dis_true_macro,
        "f1_disease_true_crop_weighted": f1_dis_true_weighted,
        "per_crop_acc":                  per_crop_acc,
    }


# per-variant region loop
def run_variant(name, unfreeze_from, model_root, num_joint_classes, crops, diseases_by_crop,
                global_index, global_labels, global_to_crop_dis, crop_to_global_ids,
                region_dirs, tform, device, variant_save):
    all_region_summaries = []

    for region_dir in region_dirs:
        region = region_dir.name
        logger.info("[%s] === Evaluating region: %s ===", name, region)

        items, stats = build_region_items(region_dir, crops, diseases_by_crop, global_index)
        if not items:
            logger.warning("[%s] No usable images in %s — skipping.", name, region)
            continue

        region_save = variant_save / region
        region_save.mkdir(parents=True, exist_ok=True)

        dl = DataLoader(
            RegionDataset(items, tform),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=safe_collate,
        )

        fold_summaries = []

        for fold in range(1, N_FOLDS + 1):
            fold_path = model_root / f"fold{fold}" / "best_model.pth"
            if not fold_path.exists():
                logger.warning("[%s] Missing checkpoint: %s", name, fold_path)
                continue

            logger.info("[%s] %s fold %d …", name, region, fold)
            fold_dir = region_save / f"fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            model   = FlatResNet18(num_joint_classes=num_joint_classes,
                                   unfreeze_from=unfreeze_from)
            summary = evaluate_region_fold(
                model, fold_path, dl, device, fold_dir,
                crops, global_labels, global_to_crop_dis, crop_to_global_ids,
            )

            row = {
                "variant": name, "region": region, "fold": fold,
                **{k: summary[k] for k in summary if k != "per_crop_acc"},
            }
            for ci, acc in summary["per_crop_acc"].items():
                row[f"disease_{crops[ci]}"] = acc

            fold_summaries.append(row)
            pd.DataFrame([row]).to_csv(fold_dir / "fold_summary.csv", index=False)

        if not fold_summaries:
            logger.warning("[%s] No folds evaluated for %s.", name, region)
            continue

        df_region    = pd.DataFrame(fold_summaries)
        df_region.to_csv(region_save / "summary_folds.csv", index=False)

        region_stats = compute_region_stats(region, crops, df_region, stats)
        pd.DataFrame([region_stats]).to_csv(region_save / "summary_stats.csv", index=False)
        save_region_tables(region, crops, df_region, region_stats, region_save)
        all_region_summaries.append({"variant": name, **region_stats})

    if all_region_summaries:
        pd.DataFrame(all_region_summaries).to_csv(
            variant_save / "summary_all_regions.csv", index=False
        )
    logger.info("[%s] done.", name)
    return all_region_summaries


# entry point
def main():
    save_root = Path(SAVE_ROOT)
    save_root.mkdir(parents=True, exist_ok=True)

    setup_logger("flat.test", save_root / "test.log")
    logger.info("Starting flat testing — save root: %s", save_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    crops, diseases_by_crop, global_index, global_labels, global_to_crop_dis, crop_to_global_ids = \
        load_label_maps(MODEL_BASE)

    num_joint_classes = len(global_labels)
    tform = make_test_transform()

    region_dirs = sorted(
        [d for d in Path(TEST_ROOT).iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    configs = discover_configs(MODEL_BASE)
    logger.info("Found %d trained variant(s): %s",
                len(configs), [c["name"] for c in configs])

    all_summaries = []
    for cfg in configs:
        variant_save = save_root / cfg["name"]
        variant_save.mkdir(parents=True, exist_ok=True)

        summaries = run_variant(
            name=cfg["name"],
            unfreeze_from=cfg["unfreeze_from"],
            model_root=Path(MODEL_BASE) / cfg["name"],
            num_joint_classes=num_joint_classes,
            crops=crops,
            diseases_by_crop=diseases_by_crop,
            global_index=global_index,
            global_labels=global_labels,
            global_to_crop_dis=global_to_crop_dis,
            crop_to_global_ids=crop_to_global_ids,
            region_dirs=region_dirs,
            tform=tform,
            device=device,
            variant_save=variant_save,
        )
        all_summaries.extend(summaries)

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(
            save_root / "comparison_all_variants.csv", index=False
        )
        logger.info("All variants evaluated.")
    else:
        logger.warning("No regions evaluated.")


if __name__ == "__main__":
    main()
