"""
utils — Shared package for multi-crop disease recognition pipelines.

Submodules
----------
seeding    : SEED constant, seed_worker, generator g, safe_collate
transforms : make_train_transform, make_eval_transform and aliases
datasets   : IMG_EXTS, build_index, HierDataset, RegionDataset
models     : FlatResNet18, HierResNet18Concat
label_maps : load_label_maps, build_region_items
metrics    : plot_cm, fmt_mean_std, compute_region_stats, save_region_tables

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
from .logging_utils import setup_logger

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
    # logging
    "setup_logger",
]
