#!/usr/bin/env python3
"""
TESTING: CUSTOM RESNET-18 FROM SCRATCH (JOINT CROP+DISEASE LABELS)
Multi-region evaluation: testA, testB, testC, testD...
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
from functools import partial

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import load_label_maps, evaluate_flat, run_region_test_loop
from models.flat.custom_model import ResNet18

# config
MODEL_ROOT = "/home/nalwangar/finally/logs_customY"
TEST_ROOT  = "/deepstore/datasets/dmb/ComputerVision/biology/testsets7"
SAVE_ROOT  = "/home/nalwangar/finally/logs_customFlat/test"

os.makedirs(SAVE_ROOT, exist_ok=True)


def main():
    crops, diseases_by_crop, global_index, global_labels, \
        global_to_crop_dis, crop_to_global_ids = load_label_maps(MODEL_ROOT)

    num_joint_classes = len(global_labels)

    make_model  = lambda: ResNet18(num_classes=num_joint_classes)
    evaluate_fn = partial(
        evaluate_flat,
        crops=crops,
        global_labels=global_labels,
        global_to_crop_dis=global_to_crop_dis,
        crop_to_global_ids=crop_to_global_ids,
    )

    run_region_test_loop(
        make_model, MODEL_ROOT, TEST_ROOT, SAVE_ROOT,
        crops, diseases_by_crop, global_index, global_labels,
        evaluate_fn,
    )


if __name__ == "__main__":
    main()
