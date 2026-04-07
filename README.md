# Multi-Crop Disease Recognition

This repository implements two CNN-based pipelines — a **flat classifier** and a **hierarchical classifier** — for jointly identifying crop type and disease from leaf images. Both pipelines use ResNet-18 as the backbone and are evaluated with 5-fold cross-validation.

---

## Repository Structure

```
multi-crop-disease-recognition/
├── train_flat.py       # Train the flat (joint) classifier
├── test_flat.py        # Test the flat classifier on held-out regions
├── train_hier.py       # Train the hierarchical classifier
├── test_hier.py        # Test the hierarchical classifier on held-out regions
└── train_hier.slurm    # SLURM job script for HPC submission
```

---

## Pipeline Overview

| Concept                  | Flat Model                                                | Hierarchical Model                                                       |
| ------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Architecture**         | Single linear head over all (crop, disease) joint classes | Crop head + per-crop disease heads concatenated into one vector          |
| **Loss**                 | CrossEntropyLoss on joint class                           | CrossEntropyLoss (crop) + CrossEntropyLoss (per-crop disease slice)      |
| **Inference**            | `argmax` over all joint logits                            | Stage 1: predict crop → Stage 2: argmax within that crop's disease slice |
| **Backbone fine-tuning** | `layer4` + head (rest frozen)                             | `layer4` + both heads (rest frozen)                                      |

---

## File-by-File Function Reference

---

### `train_flat.py` — Train Flat Classifier

**Purpose**: Treats every (crop, disease) pair as a single atomic class and trains a ResNet-18 with a single linear head over all joint classes, using 5-fold cross-validation.

| Function / Class        | Signature                                                                                                                           | What It Does                                                                                                                                                                                                                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `seed_worker`           | `seed_worker(worker_id)`                                                                                                            | Sets NumPy and Python `random` seeds per DataLoader worker to ensure reproducibility across folds.                                                                                                                                                                                             |
| `safe_collate`          | `safe_collate(batch)`                                                                                                               | Filters out `None` samples (caused by corrupt/missing images) before collating a batch.                                                                                                                                                                                                        |
| `FlatResNet18`          | `class FlatResNet18(nn.Module)`                                                                                                     | ResNet-18 with a single `nn.Linear(512, num_classes)` head. Freezes all layers except `layer4` and the head. `num_classes` = total number of unique (crop, disease) pairs.                                                                                                                     |
| `HierDataset`           | `class HierDataset(Dataset)`                                                                                                        | PyTorch Dataset that reads images from a `crop/disease/image` directory tree and returns `(img_tensor, crop_id, local_disease_id, global_joint_id)`.                                                                                                                                           |
| `make_train_transform`  | `make_train_transform()`                                                                                                            | Returns a `torchvision.transforms` pipeline with `RandomHorizontalFlip`, `RandomVerticalFlip`, resize to 224×224, and ImageNet normalisation. Used during training.                                                                                                                            |
| `make_val_transform`    | `make_val_transform()`                                                                                                              | Same as above but without any random augmentations. Used during validation/evaluation.                                                                                                                                                                                                         |
| `build_index`           | `build_index(dataset_root)`                                                                                                         | Walks the dataset directory tree to collect `(img_path, crop_id, local_dis_id, global_joint_id)` for every image. Builds and returns dictionaries: `crops`, `global_labels`, `global_to_crop_dis`, `crop_to_global_ids`, and the flat `items` list.                                            |
| `plot_confusion_matrix` | `plot_confusion_matrix(cm, labels, save_path, title)`                                                                               | Saves a colour-mapped confusion matrix PNG using Matplotlib. Row = true label, column = predicted label.                                                                                                                                                                                       |
| `evaluate`              | `evaluate(model, model_path, val_loader, device, fold_dir, crops, global_labels, global_to_crop_dis, crop_to_global_ids)`           | Loads model weights from `model_path`, runs inference on `val_loader`, computes: (1) crop accuracy, (2) joint disease accuracy using the predicted joint class, (3) oracle-crop disease accuracy (restricts logits to the true crop's classes). Saves confusion matrices and a `summary.json`. |
| `train_fold`            | `train_fold(fold, model, train_loader, val_loader, device, fold_dir, crops, global_labels, global_to_crop_dis, crop_to_global_ids)` | Trains for up to 50 epochs with Adam (lr=1e-4, weight_decay=1e-4) and StepLR scheduler (step=10, gamma=0.5). Uses early stopping with patience=7 on validation loss. Saves `best_model.pt` and calls `evaluate` at the end of the fold.                                                        |
| `main`                  | `main()`                                                                                                                            | Entry point. Calls `build_index`, creates stratified 5-fold splits (stratified on global joint label), instantiates a fresh `FlatResNet18` per fold, runs `train_fold` for each, and writes aggregated results CSV and label maps JSON.                                                        |

---

### `test_flat.py` — Test Flat Classifier

**Purpose**: Loads the trained flat model checkpoints (one per fold) and evaluates them on held-out geographic regions (testA, testB, testC, testD, …).

| Function / Class       | Signature                                                                                                                         | What It Does                                                                                                                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FlatResNet18`         | `class FlatResNet18(nn.Module)`                                                                                                   | Identical architecture to `train_flat.py`. Must match exactly to load weights.                                                                                                                                         |
| `make_test_transform`  | `make_test_transform()`                                                                                                           | Returns no-augmentation transform (resize + normalise).                                                                                                                                                                |
| `safe_collate`         | `safe_collate(batch)`                                                                                                             | Filters `None` samples from a batch (same as training).                                                                                                                                                                |
| `RegionFlatDataset`    | `class RegionFlatDataset(Dataset)`                                                                                                | Dataset for a single test region. Assigns `global_joint_id = -1` for diseases not seen during training (unknown diseases). Returns `(img_tensor, crop_id, local_disease_id, global_joint_id)`.                         |
| `load_label_maps`      | `load_label_maps(model_root)`                                                                                                     | Reads `label_maps.json` saved during training to recover `crops`, `disease_labels`, and the `global_index` mapping. This ensures test label encoding matches training.                                                 |
| `build_region_items`   | `build_region_items(region_root, train_crops, train_dis, global_index)`                                                           | Walks the region directory tree. Skips images whose crop or disease was not seen in training. Returns flat items list and a set of unknown (unseen) disease names for diagnostics.                                     |
| `plot_cm`              | `plot_cm(cm, labels, save_path, title)`                                                                                           | Saves confusion matrix PNG (same as training equivalent).                                                                                                                                                              |
| `evaluate_region_fold` | `evaluate_region_fold(model, model_path, loader, device, fold_dir, crops, global_labels, global_to_crop_dis, crop_to_global_ids)` | Loads fold checkpoint, runs inference, computes crop accuracy, joint disease accuracy (pred-crop), oracle-crop disease accuracy, and per-crop disease accuracies. Saves confusion matrices and returns a results dict. |
| `main`                 | `main()`                                                                                                                          | Iterates over every test region × every fold. Aggregates per-region statistics (mean/std across folds) and writes them to a combined CSV.                                                                              |

---

### `train_hier.py` — Train Hierarchical Classifier

**Purpose**: Trains a two-stage hierarchical model. The backbone produces a feature vector that feeds both a crop classification head and a set of per-crop disease heads whose logits are concatenated into a single vector.

| Function / Class        | Signature                                                                                   | What It Does                                                                                                                                                                                                                                                                                                                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `seed_worker`           | `seed_worker(worker_id)`                                                                    | Same as `train_flat.py`.                                                                                                                                                                                                                                                                                                                                                                                    |
| `safe_collate`          | `safe_collate(batch)`                                                                       | Same as `train_flat.py`.                                                                                                                                                                                                                                                                                                                                                                                    |
| `HierResNet18Concat`    | `class HierResNet18Concat(nn.Module)`                                                       | ResNet-18 backbone shared between two heads: (1) **crop_head**: `Linear(512, num_crops)` — predicts crop type; (2) **dis_heads**: `nn.ModuleList` of per-crop `Linear(512, num_diseases_for_crop)` heads whose outputs are concatenated. `crop_slices` dict maps each `crop_id` to its `(start, end)` index range in the concatenated disease logits. Returns `(crop_logits, concatenated_disease_logits)`. |
| `HierDataset`           | `class HierDataset(Dataset)`                                                                | Same as `train_flat.py`.                                                                                                                                                                                                                                                                                                                                                                                    |
| `make_train_transform`  | `make_train_transform()`                                                                    | Same as `train_flat.py`.                                                                                                                                                                                                                                                                                                                                                                                    |
| `make_eval_transform`   | `make_eval_transform()`                                                                     | Same as `make_val_transform` in `train_flat.py`.                                                                                                                                                                                                                                                                                                                                                            |
| `build_index`           | `build_index(dataset_root)`                                                                 | Same as `train_flat.py`.                                                                                                                                                                                                                                                                                                                                                                                    |
| `plot_confusion_matrix` | `plot_confusion_matrix(cm, labels, save_path, title)`                                       | Same as `train_flat.py`.                                                                                                                                                                                                                                                                                                                                                                                    |
| `evaluate`              | `evaluate(model, model_path, val_loader, device, fold_dir, crops, global_labels)`           | Two-stage inference: (1) crop logits → predicted crop; (2) predicted-crop slice of disease logits → predicted disease. Also computes oracle-crop disease accuracy (uses true crop slice instead of predicted crop slice). Saves confusion matrices and summary JSON.                                                                                                                                        |
| `train_fold`            | `train_fold(fold, model, train_loader, val_loader, device, fold_dir, crops, global_labels)` | Hierarchical training loss: `total_loss = crop_loss + disease_loss`. For `disease_loss`, only the slice of the concatenated logits belonging to the **true crop** is passed through `CrossEntropyLoss` against the local disease label. This prevents gradients from leaking between crop-specific heads. Uses the same Adam + StepLR + early stopping scheme as `train_flat.py`.                           |
| `main`                  | `main()`                                                                                    | Same structure as `train_flat.py` but instantiates `HierResNet18Concat` and saves label maps + `crop_slices` for the test script.                                                                                                                                                                                                                                                                           |

---

### `test_hier.py` — Test Hierarchical Classifier

**Purpose**: Loads trained hierarchical model checkpoints and evaluates on held-out regions using the same two-stage inference as training.

| Function / Class       | Signature                                                                                 | What It Does                                                                                                                                                                                                          |
| ---------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `safe_collate`         | `safe_collate(batch)`                                                                     | Same as other files.                                                                                                                                                                                                  |
| `HierResNet18Concat`   | `class HierResNet18Concat(nn.Module)`                                                     | Identical architecture to `train_hier.py`. Must match exactly to load weights.                                                                                                                                        |
| `make_test_transforms` | `make_test_transforms()`                                                                  | No-augmentation transform (resize + normalise).                                                                                                                                                                       |
| `RegionHierDataset`    | `class RegionHierDataset(Dataset)`                                                        | Same as `RegionFlatDataset` in `test_flat.py` but used with the hierarchical model.                                                                                                                                   |
| `load_label_maps`      | `load_label_maps(model_root)`                                                             | Reads `label_maps.json` and also recovers `crop_slices` so the test-time two-stage inference uses the same index ranges as training.                                                                                  |
| `build_region_items`   | `build_region_items(region_root, train_crops, train_dis, global_index)`                   | Same as `test_flat.py`.                                                                                                                                                                                               |
| `plot_cm`              | `plot_cm(cm, labels, save_path, title)`                                                   | Same as `test_flat.py`.                                                                                                                                                                                               |
| `evaluate_region_fold` | `evaluate_region_fold(model, model_path, loader, device, fold_dir, crops, global_labels)` | Two-stage test inference: Stage 1 predicts crop; Stage 2 predicts disease from (a) predicted-crop slice and (b) true-crop slice (oracle). Computes per-crop disease accuracies from the predicted-crop slice results. |
| `main`                 | `main()`                                                                                  | Same structure as `test_flat.py` — iterates all regions × folds, aggregates statistics, writes CSV.                                                                                                                   |

---

## Possible Errors

### ERROR-01 — Per-Crop Disease Accuracy Uses Unconstrained Flat Prediction (CRITICAL)

**File**: [test_flat.py:328](test_flat.py#L328)

**Buggy code**:

```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred)       # <-- BUG
```

**What goes wrong**: `gi_pred` is the flat model's argmax over **all** joint classes — it can be a joint class from any crop (e.g., predicting "wheat*rust" when the true crop is "maize"). Storing it under `ci_true` and then computing per-crop disease accuracy mixes prediction spaces: the true label is constrained to crop A's disease space, but the prediction can be from crop B, C, etc. This makes every `disease*{crop_name}` metric in the output CSV meaningless and artificially deflated.

**Why results look "the same"**: Because `gi_pred` often falls into a dominant class regardless of the true crop, causing per-crop accuracies to collapse toward a common wrong value across different crops.

**Fix**:

```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred_true_crop)   # oracle-crop prediction (already computed on line 324)
```

`gi_pred_true_crop` is already computed two lines above (line 324) by restricting the flat logits to only the classes that belong to `ci_true`. Use that instead.

---

### ERROR-02 — Per-Crop Disease Accuracy Uses Predicted-Crop Slice Instead of True-Crop Slice (MODERATE)

**File**: [test_hier.py:340](test_hier.py#L340)

**Buggy code**:

```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred_pred_crop)   # <-- inconsistency
```

**What goes wrong**: `gi_pred_pred_crop` is the disease prediction taken from the **predicted crop's slice** of the concatenated disease logits. But `gi_true` is a global index within the **true crop's slice**. When the crop prediction is wrong, the two indices refer to entirely different disease spaces, making the comparison nonsensical. The per-crop `disease_{crop_name}` metrics in the CSV therefore reflect a blend of "model was right about crop AND disease" rather than "given the true crop, how well did the disease head perform".

**Fix**:

```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred_true_crop)   # oracle-crop prediction (already computed on line 333)
```

`gi_pred_true_crop` is already computed on line 333 by reading the true-crop slice. Swapping to it makes the per-crop metric measure pure disease-head performance within the correct crop space, matching the `disease_acc_true_crop` (oracle) column in the summary.

---

## Reproducibility Settings

All scripts use `SEED = 42` with the following pattern:

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

DataLoader workers are seeded via `seed_worker` + a `torch.Generator` with the same seed to ensure deterministic data ordering across runs.

---

## Output Files (per fold)

| File                              | Description                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| `best_model.pt`                   | Checkpoint with the lowest validation loss                                         |
| `summary.json`                    | Crop acc, disease acc (pred-crop), disease acc (oracle-crop), per-crop disease acc |
| `cm_crop.png / .csv`              | Crop confusion matrix                                                              |
| `cm_disease_pred_crop.png / .csv` | Disease confusion matrix using the model's predicted crop                          |
| `cm_disease_true_crop.png / .csv` | Disease confusion matrix using the true (oracle) crop                              |
| `label_maps.json`                 | Serialised label→index mappings, used by test scripts                              |
