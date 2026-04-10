"""
utils/eval_utils.py — Shared evaluation utilities.

Provides
--------
save_fold_cms        : save crop + disease confusion matrices (PNG + CSV)
compute_fold_metrics : compute accuracy + F1 metric dict
evaluate_flat        : full flat-model evaluation  (train-time and test-time)
evaluate_hier        : full hier-model evaluation  (train-time and test-time)
"""

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .metrics import plot_cm


def save_fold_cms(fold_dir, true_crop, pred_crop, true_global,
                  pred_global_pred_crop, pred_global_true_crop,
                  crops, global_labels):
    """
    Save all three confusion matrices (PNG + CSV):
      cm_crop                — crop-level predictions
      cm_disease_pred_crop   — disease predictions using predicted crop slice
      cm_disease_true_crop   — disease predictions using oracle (true) crop slice
    Disease CMs are skipped when true_global is empty.
    """
    fold_dir = Path(fold_dir)

    cm_crop = confusion_matrix(true_crop, pred_crop, labels=list(range(len(crops))))
    plot_cm(cm_crop, crops, fold_dir / "cm_crop.png", "Crop Confusion Matrix")
    pd.DataFrame(cm_crop, index=crops, columns=crops).to_csv(fold_dir / "cm_crop.csv")

    if true_global:
        cm_pred = confusion_matrix(true_global, pred_global_pred_crop,
                                   labels=list(range(len(global_labels))))
        plot_cm(cm_pred, global_labels, fold_dir / "cm_disease_pred_crop.png",
                "Disease Confusion Matrix (Pred Crop)")
        pd.DataFrame(cm_pred, index=global_labels, columns=global_labels).to_csv(
            fold_dir / "cm_disease_pred_crop.csv"
        )

        cm_true = confusion_matrix(true_global, pred_global_true_crop,
                                   labels=list(range(len(global_labels))))
        plot_cm(cm_true, global_labels, fold_dir / "cm_disease_true_crop.png",
                "Disease Confusion Matrix (Oracle Crop)")
        pd.DataFrame(cm_true, index=global_labels, columns=global_labels).to_csv(
            fold_dir / "cm_disease_true_crop.csv"
        )


def compute_fold_metrics(true_crop, pred_crop, true_global,
                         pred_global_pred_crop, pred_global_true_crop,
                         per_crop_results):
    """
    Return a unified metrics dict including accuracy, F1, and per-crop disease accuracy.

    Parameters
    ----------
    per_crop_results : dict[int, {"true": list, "pred": list}]
        Keyed by crop index; "pred" uses the oracle (true) crop slice.
    """
    crop_acc              = accuracy_score(true_crop, pred_crop) if true_crop else 0.0
    disease_acc_pred_crop = accuracy_score(true_global, pred_global_pred_crop) if true_global else 0.0
    disease_acc_true_crop = accuracy_score(true_global, pred_global_true_crop) if true_global else 0.0

    f1_crop_macro    = f1_score(true_crop, pred_crop, average="macro",    zero_division=0)
    f1_crop_weighted = f1_score(true_crop, pred_crop, average="weighted", zero_division=0)

    if true_global:
        f1_dis_pred_macro    = f1_score(true_global, pred_global_pred_crop, average="macro",    zero_division=0)
        f1_dis_pred_weighted = f1_score(true_global, pred_global_pred_crop, average="weighted", zero_division=0)
        f1_dis_true_macro    = f1_score(true_global, pred_global_true_crop, average="macro",    zero_division=0)
        f1_dis_true_weighted = f1_score(true_global, pred_global_true_crop, average="weighted", zero_division=0)
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
        "n_crop_samples":                len(true_crop),
        "n_disease_eval_samples":        len(true_global),
        "f1_crop_macro":                 f1_crop_macro,
        "f1_crop_weighted":              f1_crop_weighted,
        "f1_disease_pred_crop_macro":    f1_dis_pred_macro,
        "f1_disease_pred_crop_weighted": f1_dis_pred_weighted,
        "f1_disease_true_crop_macro":    f1_dis_true_macro,
        "f1_disease_true_crop_weighted": f1_dis_true_weighted,
        "per_crop_acc":                  per_crop_acc,
    }


def evaluate_flat(model, model_path, loader, device, fold_dir,
                  crops, global_labels, global_to_crop_dis, crop_to_global_ids):
    """
    Flat model evaluation: load checkpoint, run inference, save CMs, return metrics.

    Works for both train-time validation and multi-region test evaluation.
    The gi_true >= 0 guard is always true during training (all labels valid) and
    handles unlabelled test images gracefully.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    true_crop, pred_crop       = [], []
    true_global                = []
    pred_global_pred_crop      = []
    pred_global_true_crop      = []
    per_crop_results = {ci: {"true": [], "pred": []} for ci in range(len(crops))}

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, yc, _yd, yg = batch
            imgs = imgs.to(device)
            yc   = yc.to(device)
            yg   = yg.to(device)

            logits = model(imgs)
            preds  = logits.argmax(dim=1)

            for i in range(imgs.size(0)):
                ci_true = int(yc[i].item())
                gi_true = int(yg[i].item())
                gi_pred = int(preds[i].item())
                ci_pred, _ = global_to_crop_dis[gi_pred]

                true_crop.append(ci_true)
                pred_crop.append(ci_pred)

                if gi_true >= 0:
                    true_global.append(gi_true)
                    pred_global_pred_crop.append(gi_pred)

                    crop_ids    = crop_to_global_ids[ci_true]
                    crop_logits = torch.stack([logits[i][g] for g in crop_ids])
                    gi_oracle   = int(crop_ids[crop_logits.argmax().item()])
                    pred_global_true_crop.append(gi_oracle)

                    per_crop_results[ci_true]["true"].append(gi_true)
                    per_crop_results[ci_true]["pred"].append(gi_oracle)

    save_fold_cms(fold_dir, true_crop, pred_crop, true_global,
                  pred_global_pred_crop, pred_global_true_crop, crops, global_labels)

    return compute_fold_metrics(true_crop, pred_crop, true_global,
                                pred_global_pred_crop, pred_global_true_crop,
                                per_crop_results)


def evaluate_hier(model, model_path, loader, device, fold_dir,
                  crops, global_labels):
    """
    Hierarchical model evaluation: load checkpoint, run inference, save CMs, return metrics.

    Works for both train-time validation and multi-region test evaluation.
    Disease global ids are derived from model.crop_slices offsets — no external
    global_index dict required.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    true_crop, pred_crop       = [], []
    true_global                = []
    pred_global_pred_crop      = []
    pred_global_true_crop      = []
    per_crop_results = {ci: {"true": [], "pred": []} for ci in range(len(crops))}

    with torch.no_grad():
        for batch in loader:
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
                ci_true = int(yc[i].item())
                ci_pred = int(pred_c[i].item())
                gi_true = int(yg[i].item())

                true_crop.append(ci_true)
                pred_crop.append(ci_pred)

                if gi_true >= 0:
                    start_pred, end_pred = model.crop_slices[ci_pred]
                    gi_pred_pred = start_pred + int(out_dis[i, start_pred:end_pred].argmax().item())

                    start_true, end_true = model.crop_slices[ci_true]
                    gi_pred_true = start_true + int(out_dis[i, start_true:end_true].argmax().item())

                    true_global.append(gi_true)
                    pred_global_pred_crop.append(gi_pred_pred)
                    pred_global_true_crop.append(gi_pred_true)

                    per_crop_results[ci_true]["true"].append(gi_true)
                    per_crop_results[ci_true]["pred"].append(gi_pred_true)

    save_fold_cms(fold_dir, true_crop, pred_crop, true_global,
                  pred_global_pred_crop, pred_global_true_crop, crops, global_labels)

    return compute_fold_metrics(true_crop, pred_crop, true_global,
                                pred_global_pred_crop, pred_global_true_crop,
                                per_crop_results)
