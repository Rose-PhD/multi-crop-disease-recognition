# Changes: Redundancy Elimination

## Summary

All repeated functions across `train_flat.py`, `train_hier.py`, `test_flat.py`, and `test_hier.py` have been consolidated into a single shared module `utils.py`. Each file now imports what it needs and defines only what is unique to its own logic.

---

## New File: `utils.py`

Created as the single source of truth for all shared code.

---

## Removed Duplications

### 1. Seeding block (4 → 1)

Removed from: `train_flat.py`, `train_hier.py`, `test_flat.py`, `test_hier.py`

The global seed setup (`random.seed`, `np.random.seed`, `torch.manual_seed`, etc.) and the `seed_worker` function and `g = torch.Generator()` appeared identically in all four files. Moved to `utils.py`; scripts import `SEED`, `g`, `seed_worker`.

---

### 2. `safe_collate` (4 → 1)

Removed from: `train_flat.py`, `train_hier.py`, `test_flat.py`, `test_hier.py`

Identical implementation in all four files. Moved to `utils.py`.

---

### 3. `make_train_transform` (2 → 1)

Removed from: `train_hier.py`

Identical body in `train_flat.py` and `train_hier.py`. Kept in `utils.py`.

---

### 4. `make_val_transform` / `make_eval_transform` / `make_test_transform` / `make_test_transforms` (4 → 1)

Removed from: `train_flat.py` (`make_val_transform`), `train_hier.py` (`make_eval_transform`), `test_flat.py` (`make_test_transform`), `test_hier.py` (`make_test_transforms`)

All four functions had the same body: resize to 224×224, ToTensor, ImageNet normalise. Unified into a single `make_eval_transform()` in `utils.py` with `make_val_transform` and `make_test_transform` as aliases.

---

### 5. `build_index` (2 → 1)

Removed from: `train_hier.py`

Identical implementation in `train_flat.py` and `train_hier.py`. Moved to `utils.py`.

---

### 6. `plot_confusion_matrix` / `plot_cm` (4 → 1)

Removed from: `train_flat.py` (`plot_confusion_matrix`), `train_hier.py` (`plot_confusion_matrix`), `test_flat.py` (`plot_cm`), `test_hier.py` (`plot_cm`)

Identical implementation across all four files under two different names. Unified as `plot_cm` in `utils.py`. All call sites updated to use `plot_cm`.

---

### 7. `HierDataset` (2 → 1)

Removed from: `train_hier.py`

Identical class in `train_flat.py` and `train_hier.py`. Moved to `utils.py`.

**Refactor**: Removed the implicit `global items_global_map` dependency from `__getitem__`. The class now accepts `global_map` as a constructor parameter (`HierDataset(items, transform, global_map)`). Both training scripts pass `global_index` at instantiation time.

---

### 8. `RegionFlatDataset` / `RegionHierDataset` (2 → 1)

Removed from: `test_flat.py` (`RegionFlatDataset`), `test_hier.py` (`RegionHierDataset`)

Identical `__init__`, `__len__`, and `__getitem__` implementations. Merged into a single `RegionDataset` class in `utils.py`. Both test scripts updated to use `RegionDataset`.

---

### 9. `FlatResNet18` (2 → 1)

Removed from: `test_flat.py`

Identical class in `train_flat.py` and `test_flat.py`. Moved to `utils.py`; both files import it from there.

---

### 10. `HierResNet18Concat` (2 → 1)

Removed from: `test_hier.py`

Identical class in `train_hier.py` and `test_hier.py`. Moved to `utils.py`; both files import it from there.

---

### 11. `load_label_maps` (2 → 1)

Removed from: `test_hier.py`

Present in both `test_flat.py` and `test_hier.py` with different return signatures:
- `test_flat.py` returned 6 values (including `global_to_crop_dis`, `crop_to_global_ids`)
- `test_hier.py` returned 4 values

Unified to always return all 6 values (the fuller `test_flat.py` version). `test_hier.py` unpacks the extra two as `_, _` since it uses `model.crop_slices` directly instead.

---

### 12. `build_region_items` (2 → 1)

Removed from: `test_hier.py`

Present in both test files. Minor difference: `test_hier.py` only matched `.jpg`, `.jpeg`, `.png`; `test_flat.py` also matched `.bmp`, `.tif`, `.tiff`. Unified to the fuller extension set (`IMG_EXTS`) from `test_flat.py`, consistent with `build_index`.

---

### 13. `fmt_mean_std` (2 → 1)

Removed from: `test_flat.py` (inline nested function), `test_hier.py` (inline nested function)

Identical inline function defined inside `main()` in both test files. Extracted to a module-level function in `utils.py`.

---

### 14. Region stats computation block (2 → 1)

Removed from: `test_flat.py`, `test_hier.py`

The large `region_stats` dictionary construction (~25 lines, identical in both test files) extracted to `compute_region_stats(region, crops, df_region, stats)` in `utils.py`.

---

### 15. Table 1 and Table 2 generation block (2 → 1)

Removed from: `test_flat.py`, `test_hier.py`

The table1 / table2 CSV generation code (~60 lines, identical in both test files) extracted to `save_region_tables(region, crops, df_region, region_stats, region_save)` in `utils.py`.

---

## Bug Fixes

### FIX-01 — `test_flat.py`: Per-crop disease accuracy now uses oracle-crop prediction (ERROR-01)

**File**: `test_flat.py` — `evaluate_region_fold`

**Before**:
```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred)          # unconstrained flat prediction
```

**After**:
```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred_true_crop) # logits restricted to true crop
```

**Why it matters**: `gi_pred` is the flat model's argmax over *all* joint classes, so when the flat model predicts the wrong crop, `gi_pred` belongs to a different disease space than `gi_true`. The per-crop `disease_{crop_name}` columns in the output CSV were therefore incoherent — comparing a global index from crop A against a global index from crop B. The fix uses `gi_pred_true_crop`, which is already computed in the same loop by restricting the flat logits to only the joint-class indices belonging to `ci_true`. This makes per-crop disease accuracy measure "given the correct crop, how accurately does the flat model distinguish diseases?" — a meaningful metric that will now differ from the global `disease_acc_pred_crop` column whenever the flat model frequently mis-predicts crop.

---

### FIX-02 — `test_hier.py`: Per-crop disease accuracy now uses oracle-crop prediction (ERROR-02)

**File**: `test_hier.py` — `evaluate_region_fold`

**Before**:
```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred_pred_crop) # predicted-crop slice index
```

**After**:
```python
per_crop_results[ci_true]["true"].append(gi_true)
per_crop_results[ci_true]["pred"].append(gi_pred_true_crop) # true-crop slice index
```

**Why it matters**: `gi_pred_pred_crop = start_pred + local_pred_pred_crop` is an offset into the *predicted* crop's slice of the concatenated disease vector. When crop prediction is wrong, `start_pred` is for a different crop than `start_true`, so the integer value of `gi_pred_pred_crop` falls in a completely different region of the label space than `gi_true`. Comparing them gives a nonsensical accuracy. The fix uses `gi_pred_true_crop = start_true + local_pred_true_crop`, which is computed from the same true-crop slice as `gi_true` — making the comparison well-defined and the metric interpretable as pure disease-head performance within the correct crop.

**Effect on results**: The two pipelines will now produce different `disease_{crop_name}` columns. Previously both accidentally converged toward similar (wrong) values because both were comparing misaligned label spaces. After the fix: the flat model's per-crop disease accuracy reflects its oracle-constrained joint-class performance; the hierarchical model's reflects its disease-head performance given the true crop — a fairer comparison between the two architectures.

---

## Impact Summary

| Metric                        | Before | After |
| ----------------------------- | ------ | ----- |
| Files with shared code        | 4      | 1 (`utils.py`) |
| Copies of `safe_collate`      | 4      | 1     |
| Copies of eval transform      | 4      | 1     |
| Copies of `plot_cm`           | 4      | 1     |
| Copies of `HierDataset`       | 2      | 1     |
| Copies of `RegionDataset`     | 2      | 1     |
| Copies of `FlatResNet18`      | 2      | 1     |
| Copies of `HierResNet18Concat`| 2      | 1     |
| Copies of `build_index`       | 2      | 1     |
| Copies of `load_label_maps`   | 2      | 1     |
| Copies of `build_region_items`| 2      | 1     |
| Copies of region stats block  | 2      | 1     |
| Copies of table gen block     | 2      | 1     |
