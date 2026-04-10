"""
utils/test_utils.py — Shared multi-region test loop.

Provides
--------
run_region_test_loop : iterate test regions, evaluate per fold, aggregate stats
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import RegionDataset
from .label_maps import build_region_items
from .metrics import compute_region_stats, save_region_tables
from .seeding import safe_collate
from .transforms import make_test_transform


def run_region_test_loop(make_model, model_root, test_root, save_root,
                         crops, diseases_by_crop, global_index, global_labels,
                         evaluate_fold_fn, *, n_folds=5, batch_size=32, num_workers=4):
    """
    Multi-region evaluation loop shared by all test scripts.

    Parameters
    ----------
    make_model       : callable() -> nn.Module
        Returns a fresh (weight-free) model instance for each fold.
    model_root       : str | Path
        Directory containing fold{n}/best_model.pth checkpoints.
    test_root        : str | Path
        Root directory whose subdirectories are per-region test sets.
    save_root        : str | Path
        Root directory for all output files.
    evaluate_fold_fn : callable(model, model_path, loader, device, fold_dir) -> dict
        Runs one fold's evaluation and returns a metrics dict that includes
        "per_crop_acc" and all standard accuracy/F1 keys.
        Use functools.partial to bind crops, global_labels, and any label maps.
    """
    model_root = Path(model_root)
    save_root  = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tform  = make_test_transform()

    region_dirs = sorted(
        [d for d in Path(test_root).iterdir() if d.is_dir()],
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

        region_save = save_root / region
        region_save.mkdir(parents=True, exist_ok=True)

        dl = DataLoader(
            RegionDataset(items, tform),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=safe_collate,
        )

        fold_summaries = []

        for fold in range(1, n_folds + 1):
            print(f"  Fold {fold}...")
            fold_path = model_root / f"fold{fold}" / "best_model.pth"
            if not fold_path.exists():
                print(f"    Missing: {fold_path}")
                continue

            fold_dir = region_save / f"fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            model   = make_model()
            summary = evaluate_fold_fn(model, fold_path, dl, device, fold_dir)

            flat = {
                "region":                        region,
                "fold":                          fold,
                "crop_acc":                      summary["crop_acc"],
                "disease_acc_pred_crop":         summary["disease_acc_pred_crop"],
                "disease_acc_true_crop":         summary["disease_acc_true_crop"],
                "n_crop_samples":                summary["n_crop_samples"],
                "n_disease_eval_samples":        summary["n_disease_eval_samples"],
                "f1_crop_macro":                 summary["f1_crop_macro"],
                "f1_crop_weighted":              summary["f1_crop_weighted"],
                "f1_disease_pred_crop_macro":    summary["f1_disease_pred_crop_macro"],
                "f1_disease_pred_crop_weighted": summary["f1_disease_pred_crop_weighted"],
                "f1_disease_true_crop_macro":    summary["f1_disease_true_crop_macro"],
                "f1_disease_true_crop_weighted": summary["f1_disease_true_crop_weighted"],
            }
            for ci, acc in summary["per_crop_acc"].items():
                flat[f"disease_{crops[ci]}"] = acc

            fold_summaries.append(flat)
            pd.DataFrame([flat]).to_csv(fold_dir / "fold_summary.csv", index=False)

        if not fold_summaries:
            print(f"  No folds evaluated for {region}.")
            continue

        df_region = pd.DataFrame(fold_summaries)
        df_region.to_csv(region_save / "summary_folds.csv", index=False)

        region_stats = compute_region_stats(region, crops, df_region, stats)
        pd.DataFrame([region_stats]).to_csv(region_save / "summary_stats.csv", index=False)

        save_region_tables(region, crops, df_region, region_stats, region_save)
        all_region_summaries.append(region_stats)

    if all_region_summaries:
        pd.DataFrame(all_region_summaries).to_csv(
            save_root / "summary_all_regions.csv", index=False
        )
        print("\n=== DONE: Multi-region evaluation complete. ===")
    else:
        print("\nNo regions evaluated.")
