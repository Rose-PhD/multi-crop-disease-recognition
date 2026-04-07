# Multi-Crop Disease Recognition

This repository implements two CNN-based pipelines — a **flat classifier** and a **hierarchical classifier** — for jointly identifying crop type and disease from leaf images. Both pipelines use ResNet-18 as the backbone and are evaluated with 5-fold cross-validation.

---

## Repository Structure

```
multi-crop-disease-recognition/
├── utils/                  # Shared package — single source of truth for all pipelines
│   ├── __init__.py         #   re-exports every public symbol
│   ├── seeding.py          #   SEED, seed_worker, g, safe_collate
│   ├── transforms.py       #   make_train/eval/val/test_transform
│   ├── datasets.py         #   IMG_EXTS, build_index, HierDataset, RegionDataset
│   ├── models.py           #   FlatResNet18, HierResNet18Concat
│   ├── label_maps.py       #   load_label_maps, build_region_items
│   └── metrics.py          #   plot_cm, fmt_mean_std, compute_region_stats, save_region_tables
├── flat/                   # Flat (joint) classifier pipeline
│   ├── train_flat.py       #   training: 5-fold CV with FlatResNet18
│   └── test_flat.py        #   testing: multi-region evaluation
├── hier/                   # Hierarchical classifier pipeline
│   ├── train_hier.py       #   training: 5-fold CV with HierResNet18Concat
│   └── test_hier.py        #   testing: multi-region evaluation
└── train_hier.slurm        # SLURM job script for HPC submission
```

Scripts in `flat/` and `hier/` add the project root to `sys.path` at startup so
`from utils import ...` resolves correctly regardless of invocation directory:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Functions shared across pipelines are defined once in `utils/` and not repeated elsewhere.

---

## Pipeline Overview

| Concept                  | Flat Model                                                | Hierarchical Model                                                       |
| ------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Architecture**         | Single linear head over all (crop, disease) joint classes | Crop head + per-crop disease heads concatenated into one vector          |
| **Loss**                 | CrossEntropyLoss on joint class                           | CrossEntropyLoss (crop) + CrossEntropyLoss (per-crop disease slice)      |
| **Inference**            | `argmax` over all joint logits                            | Stage 1: predict crop → Stage 2: argmax within that crop's disease slice |
| **Backbone fine-tuning** | `layer4` + head (rest frozen)                             | `layer4` + both heads (rest frozen)                                      |

---

## Shared Utilities — `utils.py`

Everything listed here is defined once and imported by whichever scripts need it.

### Reproducibility

`SEED = 42` is set at module level. All scripts inherit it via `from utils import SEED`.

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

DataLoader workers are seeded via `seed_worker` and a `torch.Generator` (`g`) initialised with the same seed.

| Name            | Signature               | What It Does                                                                                          |
| --------------- | ----------------------- | ----------------------------------------------------------------------------------------------------- |
| `seed_worker`   | `seed_worker(worker_id)` | Sets NumPy and Python `random` seeds per DataLoader worker for reproducible data ordering.           |
| `g`             | `torch.Generator`       | A seeded generator passed to `DataLoader` so shuffle order is deterministic.                         |
| `safe_collate`  | `safe_collate(batch)`   | Filters `None` entries caused by corrupted/missing images before collating a batch.                  |

---

### Transforms

`make_val_transform` and `make_test_transform` are aliases of the same underlying `make_eval_transform` function.

| Name                  | Used by                        | What It Does                                                                          |
| --------------------- | ------------------------------ | ------------------------------------------------------------------------------------- |
| `make_train_transform` | `train_flat`, `train_hier`    | Resize to 224×224, `RandomHorizontalFlip`, `RandomVerticalFlip`, ToTensor, ImageNet normalise. |
| `make_eval_transform`  | —                             | Resize to 224×224, ToTensor, ImageNet normalise. No augmentation.                    |
| `make_val_transform`   | `train_flat`, `train_hier`    | Alias of `make_eval_transform`. Used at validation time during training.              |
| `make_test_transform`  | `test_flat`, `test_hier`      | Alias of `make_eval_transform`. Used at inference time during testing.                |

---

### Data Index Builder

| Function      | Signature                   | What It Does                                                                                                                              |
| ------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `build_index` | `build_index(dataset_root)` | Walks `dataset_root/<crop>/<disease>/<images>`. Returns `crops` (sorted list), `diseases_by_crop` (dict), and `items` list of `(img_path, crop_idx, disease_idx)`. |

---

### Confusion Matrix Plotter

| Function  | Signature                                  | What It Does                                                                                |
| --------- | ------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `plot_cm` | `plot_cm(cm, labels, save_path, title)`    | Saves a colour-mapped confusion matrix PNG at 300 dpi. Row = true label, column = predicted. |

---

### Datasets

| Class           | Used by                             | What It Does                                                                                                                                                                               |
| --------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `HierDataset`   | `train_flat`, `train_hier`          | Training dataset. Takes `items` of `(img_path, crop_id, dis_id)` and a `global_map` dict. Returns `(img, crop_id, dis_id, global_joint_id)`. Skips corrupted images by returning `None`. |
| `RegionDataset` | `test_flat`, `test_hier`            | Test/region dataset. Takes `items` of `(img_path, crop_id, dis_local, global_joint_id)` where `global_joint_id = -1` for unknown diseases. Returns the same 4-tuple with the image tensor. |

> `HierDataset` was refactored to accept `global_map` as a constructor argument instead of relying on a module-level global variable. Pass `global_index` at the `DataLoader` call site.

---

### Models

#### `FlatResNet18` — used by `train_flat`, `test_flat`

ResNet-18 with a single `nn.Linear(512, num_joint_classes)` head over all (crop, disease) joint classes. All layers except `layer4` and the head are frozen.

```
Input image → ResNet-18 backbone (layer4 trainable) → Linear(512, K) → joint logits
```

#### `HierResNet18Concat` — used by `train_hier`, `test_hier`

ResNet-18 backbone shared between two heads:

1. **`crop_head`**: `Linear(512, num_crops)` — predicts crop type.
2. **`heads`**: `nn.ModuleList` of per-crop `Linear(512, n_diseases_for_crop)` heads whose outputs are concatenated into a single disease logit vector of length `sum(n_diseases_per_crop)`.

`crop_slices` maps each `crop_id → (start, end)` into the concatenated disease vector, enabling the training and inference two-stage logic.

```
Input image → ResNet-18 backbone → crop_head  → crop logits
                                 → heads[0..C] → concat → disease logits
```

---

### Test-Script Utilities

| Function / Name          | Signature                                                          | What It Does                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_label_maps`        | `load_label_maps(model_root)`                                      | Reads `label_maps.json` saved during training. Returns `crops`, `diseases_by_crop`, `global_index`, `global_labels`, `global_to_crop_dis`, `crop_to_global_ids`.          |
| `build_region_items`     | `build_region_items(region_root, train_crops, train_dis, global_index)` | Walks a test-region directory. Skips unknown crops; marks unknown diseases with `global_joint_id = -1`. Returns `items` list and a `stats` dict of image counts.      |
| `fmt_mean_std`           | `fmt_mean_std(mean_val, std_val)`                                  | Formats a mean ± std pair as a string (e.g. `"0.874 ± 0.012"`). Returns `None` if the mean is `None` or NaN.                                                             |
| `compute_region_stats`   | `compute_region_stats(region, crops, df_region, stats)`            | Aggregates per-fold metrics in `df_region` into region-level mean/std for crop accuracy, disease accuracy (pred-crop and oracle-crop), F1 scores, and per-crop disease accuracy. |
| `save_region_tables`     | `save_region_tables(region, crops, df_region, region_stats, region_save)` | Writes `table1_global_metrics.csv` (P1 crop acc, P2 per-crop disease acc, P3 global disease acc) and `table2_per_crop_metrics.csv` (per-crop breakdown) to `region_save`. |

---

## Per-File Function Reference

Each file below only lists functions **unique** to that file. All shared logic lives in `utils.py`.

---

### `train_flat.py` — Train Flat Classifier

**Purpose**: Trains `FlatResNet18` with 5-fold cross-validation, stratified on the joint (crop, disease) label.

| Function     | Signature                                                                                                                           | What It Does                                                                                                                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `evaluate`   | `evaluate(model, model_path, val_loader, device, fold_dir, crops, global_labels, global_to_crop_dis, crop_to_global_ids)`           | Loads best weights, runs flat inference. Computes: (1) crop accuracy, (2) joint disease accuracy, (3) oracle-crop disease accuracy (logits restricted to the true crop's classes). Saves confusion matrices. |
| `train_fold` | `train_fold(fold, model, train_loader, val_loader, device, fold_dir, crops, global_labels, global_to_crop_dis, crop_to_global_ids)` | Trains for up to 50 epochs with Adam (lr=1e-4) and early stopping (patience=7) on validation loss. Saves `best_model.pth` and calls `evaluate` at fold end.                       |
| `main`       | `main()`                                                                                                                            | Builds the dataset index, creates 5-fold splits, instantiates a fresh `FlatResNet18` per fold, runs `train_fold`, and writes `summary_all_folds.csv` and `label_maps.json`.       |

---

### `train_hier.py` — Train Hierarchical Classifier

**Purpose**: Trains `HierResNet18Concat` with a two-stage hierarchical loss: crop head loss plus a per-sample disease loss computed only on the true crop's slice of the concatenated disease logits.

| Function     | Signature                                                                                        | What It Does                                                                                                                                                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `evaluate`   | `evaluate(model, model_path, val_loader, device, fold_dir, crops, global_labels, global_index)` | Loads best weights, runs two-stage inference. Computes crop accuracy, predicted-crop disease accuracy, and oracle-crop disease accuracy. Saves three confusion matrices per fold.                                                          |
| `train_fold` | `train_fold(fold, model, train_loader, val_loader, device, fold_dir, crops, global_labels, global_index)` | Hierarchical loss: `total = crop_loss + disease_loss`. For `disease_loss`, only the slice of concatenated logits for the **true crop** is passed to `CrossEntropyLoss`, preventing gradient leakage between crop-specific heads. Early stopping (patience=7). |
| `main`       | `main()`                                                                                         | Same structure as `train_flat.main()` but instantiates `HierResNet18Concat` per fold.                                                                                                                                                    |

---

### `test_flat.py` — Test Flat Classifier

**Purpose**: Loads trained flat checkpoints (one per fold) and evaluates them across held-out geographic test regions.

| Function               | Signature                                                                                                                         | What It Does                                                                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `evaluate_region_fold` | `evaluate_region_fold(model, model_path, loader, device, fold_dir, crops, global_labels, global_to_crop_dis, crop_to_global_ids)` | Flat inference: predicts joint class, derives crop from that. Also computes oracle-crop disease accuracy by restricting logits to the true crop's joint classes. Saves confusion matrices. |
| `main`                 | `main()`                                                                                                                          | Iterates all test regions × folds, builds `RegionDataset`, calls `evaluate_region_fold`, aggregates via `compute_region_stats` and `save_region_tables`, writes `summary_all_regions.csv`. |

---

### `test_hier.py` — Test Hierarchical Classifier

**Purpose**: Loads trained hierarchical checkpoints and evaluates using the same two-stage inference as training.

| Function               | Signature                                                                               | What It Does                                                                                                                                                                                                            |
| ---------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `evaluate_region_fold` | `evaluate_region_fold(model, model_path, loader, device, fold_dir, crops, global_labels)` | Two-stage inference: Stage 1 predicts crop; Stage 2 predicts disease from (a) predicted-crop slice and (b) true-crop slice (oracle). Computes per-crop disease accuracies and saves confusion matrices per fold. |
| `main`                 | `main()`                                                                                | Same structure as `test_flat.main()` but instantiates `HierResNet18Concat` and uses 2-argument `load_label_maps` unpacking (`_, _` for unused `global_to_crop_dis` / `crop_to_global_ids`).                           |

---

## Possible Errors

> Both errors below have been corrected. They are documented here for traceability.

### ERROR-01 — Per-Crop Disease Accuracy Uses Unconstrained Flat Prediction (FIXED)

**File**: [test_flat.py](test_flat.py)

**Buggy code**:

```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred)       # <-- BUG
```

**What goes wrong**: `gi_pred` is the flat model's argmax over **all** joint classes — it can be a joint class from any crop. Storing it under `ci_true` mixes prediction spaces, making every `disease_{crop_name}` metric in the output CSV meaningless.

**Fix**:

```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred_true_crop)   # oracle-crop prediction (already computed)
```

---

### ERROR-02 — Per-Crop Disease Accuracy Uses Predicted-Crop Slice Instead of True-Crop Slice (FIXED)

**File**: [test_hier.py](test_hier.py)

**Buggy code**:

```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred_pred_crop)   # <-- inconsistency
```

**What goes wrong**: `gi_pred_pred_crop` is from the **predicted crop's slice**, but `gi_true` is within the **true crop's slice**. When the crop prediction is wrong, the indices refer to different disease spaces.

**Fix**:

```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred_true_crop)   # oracle-crop prediction (already computed)
```

---

## Output Files (per fold)

| File                              | Description                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| `best_model.pth`                  | Checkpoint with the lowest validation loss                                         |
| `fold_summary.csv`                | Crop acc, disease acc (pred-crop and oracle-crop), F1 scores for the fold          |
| `cm_crop.png / .csv`              | Crop confusion matrix                                                              |
| `cm_disease_pred_crop.png / .csv` | Disease confusion matrix using the model's predicted crop                          |
| `cm_disease_true_crop.png / .csv` | Disease confusion matrix using the true (oracle) crop                              |
| `label_maps.json`                 | Serialised label→index mappings, read by test scripts via `load_label_maps()`      |
| `summary_all_folds.csv`           | Aggregated results across all 5 folds (training scripts)                           |
| `summary_all_regions.csv`         | Aggregated mean ± std across all test regions (test scripts)                       |
| `table1_global_metrics.csv`       | P1 (crop acc), P2 (per-crop disease acc), P3 (global disease acc) — LaTeX-ready   |
| `table2_per_crop_metrics.csv`     | Per-crop P1 and P2 breakdown — LaTeX-ready                                         |
