"""
utils — Shared package for multi-crop disease recognition pipelines.

Submodules
----------
seeding      : SEED constant, seed_worker, generator g, safe_collate
transforms   : make_train_transform, make_eval_transform and aliases
datasets     : IMG_EXTS, build_index, HierDataset, RegionDataset
models       : FlatResNet18, HierResNet18Concat
label_maps   : load_label_maps, build_region_items
metrics      : plot_cm, fmt_mean_std, compute_region_stats, save_region_tables
train_utils  : build_global_index, run_epoch_loop
eval_utils   : save_fold_cms, compute_fold_metrics, evaluate_flat, evaluate_hier
test_utils   : run_region_test_loop

All public symbols are re-exported here so existing scripts can continue to
use  `from utils import <name>`  without referencing submodules directly.
"""

from .seeding import SEED, g, seed_worker, safe_collate
from .transforms import (
    make_train_transform,
    make_eval_transform,
    make_val_transform,
    make_test_transform,
)
from .datasets import IMG_EXTS, build_index, HierDataset, RegionDataset
from .models import FlatResNet18, HierResNet18Concat
from .label_maps import load_label_maps, build_region_items
from .metrics import plot_cm, fmt_mean_std, compute_region_stats, save_region_tables
from .train_utils import build_global_index, run_epoch_loop
from .eval_utils import save_fold_cms, compute_fold_metrics, evaluate_flat, evaluate_hier
from .test_utils import run_region_test_loop

__all__ = [
    # seeding
    "SEED", "g", "seed_worker", "safe_collate",
    # transforms
    "make_train_transform", "make_eval_transform",
    "make_val_transform", "make_test_transform",
    # datasets
    "IMG_EXTS", "build_index", "HierDataset", "RegionDataset",
    # models
    "FlatResNet18", "HierResNet18Concat",
    # label maps
    "load_label_maps", "build_region_items",
    # metrics
    "plot_cm", "fmt_mean_std", "compute_region_stats", "save_region_tables",
    # training
    "build_global_index", "run_epoch_loop",
    # evaluation
    "save_fold_cms", "compute_fold_metrics", "evaluate_flat", "evaluate_hier",
    # testing
    "run_region_test_loop",
]
