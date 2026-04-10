#!/usr/bin/env python3
"""
TESTING: CUSTOM HIERARCHICAL RESNET-18 FROM SCRATCH (CONCATENATED HEADS)
Multi-region evaluation: testA, testB, testC, testD...
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
from functools import partial

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import load_label_maps, evaluate_hier, run_region_test_loop
from models.hier.custom_hier_model import CustomHierResNet18

# config
MODEL_ROOT = "/home/nalwangar/finally/logs_customHierY"
TEST_ROOT  = "/deepstore/datasets/dmb/ComputerVision/biology/testsets7"
SAVE_ROOT  = "/home/nalwangar/finally/logs_customHier/test"

os.makedirs(SAVE_ROOT, exist_ok=True)


def main():
    crops, diseases_by_crop, global_index, global_labels, _, _ = load_label_maps(MODEL_ROOT)

    make_model  = lambda: CustomHierResNet18(crops, diseases_by_crop)
    evaluate_fn = partial(
        evaluate_hier,
        crops=crops,
        global_labels=global_labels,
    )

    run_region_test_loop(
        make_model, MODEL_ROOT, TEST_ROOT, SAVE_ROOT,
        crops, diseases_by_crop, global_index, global_labels,
        evaluate_fn,
    )


if __name__ == "__main__":
    main()
