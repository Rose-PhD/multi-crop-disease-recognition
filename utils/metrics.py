"""
utils/metrics.py — Evaluation reporting: confusion matrix plotting,
region-level statistics aggregation, and LaTeX-ready table generation.

Provides:
  - plot_cm              : save a confusion matrix PNG
  - fmt_mean_std         : format "mean ± std" strings for tables
  - compute_region_stats : aggregate fold results into region-level mean/std
  - save_region_tables   : write table1 and table2 CSVs from region stats
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_cm(cm, labels, save_path, title):
    """
    Save a colour-mapped confusion matrix image at 300 dpi.

    Args:
        cm       : square numpy array (true × predicted)
        labels   : tick labels for both axes
        save_path: output file path (.png)
        title    : figure title
    """
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


def fmt_mean_std(mean_val, std_val):
    """Return 'mean ± std' string, or None if mean_val is None or NaN."""
    if mean_val is None or pd.isna(mean_val):
        return None
    return f"{mean_val:.3f} ± {std_val:.3f}"


def compute_region_stats(region, crops, df_region, stats):
    """
    Aggregate per-fold metrics in df_region into region-level mean and std.

    Args:
        region   : region name string
        crops    : list of crop names
        df_region: DataFrame with one row per fold
        stats    : image count dict from build_region_items

    Returns a flat dict of mean_*/std_* values ready to write to CSV.
    """
    metric_cols = [
        "crop_acc",
        "disease_acc_pred_crop",
        "disease_acc_true_crop",
        "f1_crop_macro",
        "f1_crop_weighted",
        "f1_disease_pred_crop_macro",
        "f1_disease_pred_crop_weighted",
        "f1_disease_true_crop_macro",
        "f1_disease_true_crop_weighted",
    ]

    region_stats = {
        "region": region,
        "total_images": stats["total_images"],
        "skipped_unknown_crop": stats["skipped_unknown_crop"],
        "known_crop_known_disease": stats["known_crop_known_disease"],
        "known_crop_unknown_disease": stats["known_crop_unknown_disease"],
    }

    for col in metric_cols:
        region_stats[f"mean_{col}"] = df_region[col].mean()
        region_stats[f"std_{col}"] = df_region[col].std(ddof=0)

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
    """
    Write two LaTeX-ready CSV tables for a region.

    Table 1 — global metrics:
        P1_global   : mean ± std crop accuracy
        P2_{crop}   : mean ± std per-crop disease accuracy (oracle-crop)
        P3_global   : mean ± std global disease accuracy (pred-crop)

    Table 2 — per-crop breakdown:
        P1_per_crop     : per-crop classification accuracy (from cm_crop.csv diagonals)
        P2_per_crop_pred: per-crop oracle-crop disease accuracy
        P2_global       : macro average of per-crop oracle-crop disease accuracy
    """
    # --- Table 1 ---
    table1_row = {
        "region": region,
        "P1_global": fmt_mean_std(
            region_stats["mean_crop_acc"], region_stats["std_crop_acc"]
        ),
    }
    for crop in crops:
        table1_row[f"P2_{crop}"] = fmt_mean_std(
            region_stats.get(f"mean_disease_{crop}"),
            region_stats.get(f"std_disease_{crop}"),
        )
    table1_row["P3_global"] = fmt_mean_std(
        region_stats["mean_disease_acc_pred_crop"],
        region_stats["std_disease_acc_pred_crop"],
    )
    pd.DataFrame([table1_row]).to_csv(
        region_save / "table1_global_metrics.csv", index=False
    )

    # --- Table 2 ---
    # Compute per-crop P1 from confusion matrix diagonals across folds
    per_crop_fold_vals = {crop: [] for crop in crops}
    for _, row in df_region.iterrows():
        cm_path = region_save / f"fold{int(row['fold'])}" / "cm_crop.csv"
        if cm_path.exists():
            cm = pd.read_csv(cm_path, index_col=0).values
            for ci, crop in enumerate(crops):
                total = cm[ci, :].sum()
                if total > 0:
                    per_crop_fold_vals[crop].append(cm[ci, ci] / total)

    per_crop_p1 = {
        crop: (
            (np.mean(vals), np.std(vals, ddof=0))
            if vals else (None, None)
        )
        for crop, vals in per_crop_fold_vals.items()
    }

    per_crop_p2 = {
        crop: (
            region_stats.get(f"mean_disease_{crop}"),
            region_stats.get(f"std_disease_{crop}"),
        )
        for crop in crops
    }

    all_means = [per_crop_p2[c][0] for c in crops if per_crop_p2[c][0] is not None]
    P2_global = (
        fmt_mean_std(np.mean(all_means), np.std(all_means, ddof=0))
        if all_means else None
    )

    table2_rows = [
        {
            "crop": crop,
            "P1_per_crop": fmt_mean_std(*per_crop_p1[crop]),
            "P2_per_crop_pred": fmt_mean_std(*per_crop_p2[crop]),
            "P2_global": P2_global,
        }
        for crop in crops
    ]
    pd.DataFrame(table2_rows).to_csv(
        region_save / "table2_per_crop_metrics.csv", index=False
    )
