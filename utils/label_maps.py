"""
utils/label_maps.py — Label map loading and test-region item builder.

Provides:
  - load_label_maps   : reads label_maps.json written during training
  - build_region_items: walks a test-region directory and maps images to
                        training labels, marking unknown diseases as -1
"""

import json
from pathlib import Path

from .datasets import IMG_EXTS


def load_label_maps(model_root):
    """
    Load label_maps.json saved during training to recover the full label
    encoding used at training time.

    Returns:
        crops             : list of crop names
        diseases_by_crop  : dict[crop_name] -> list of disease names
        global_index      : dict[(crop_id, dis_id)] -> global_joint_id
        global_labels     : list of "crop:disease" strings, indexed by global_joint_id
        global_to_crop_dis: dict[global_joint_id] -> (crop_id, dis_id)
        crop_to_global_ids: dict[crop_id] -> list of global_joint_ids for that crop
    """
    with open(Path(model_root) / "label_maps.json", "r") as f:
        lm = json.load(f)

    crops = lm["crops"]
    diseases_by_crop = lm["diseases_within_crop"]

    global_index = {}
    global_labels = []
    idx = 0
    for ci, crop in enumerate(crops):
        for di, dis in enumerate(diseases_by_crop[crop]):
            global_index[(ci, di)] = idx
            global_labels.append(f"{crop}:{dis}")
            idx += 1

    global_to_crop_dis = {gid: (ci, di) for (ci, di), gid in global_index.items()}

    crop_to_global_ids = {}
    for (ci, di), gid in global_index.items():
        crop_to_global_ids.setdefault(ci, []).append(gid)

    return (
        crops,
        diseases_by_crop,
        global_index,
        global_labels,
        global_to_crop_dis,
        crop_to_global_ids,
    )


def build_region_items(region_root, train_crops, train_dis, global_index):
    """
    Walk a test-region directory and build an items list aligned to training labels.

    Images from crops not seen during training are skipped entirely.
    Images from known crops but unknown diseases receive global_joint_id = -1
    and are excluded from disease accuracy metrics (but included in crop metrics).

    Returns:
        items: list of (img_path, crop_id, dis_local, global_joint_id)
        stats: dict with counts for total_images, skipped_unknown_crop,
               known_crop_known_disease, known_crop_unknown_disease
    """
    items = []
    stats = {
        "total_images": 0,
        "skipped_unknown_crop": 0,
        "known_crop_known_disease": 0,
        "known_crop_unknown_disease": 0,
    }

    region_root = Path(region_root)
    if not region_root.exists():
        return items, stats

    for crop_dir in sorted([d for d in region_root.iterdir() if d.is_dir()]):
        crop_name = crop_dir.name

        if crop_name not in train_crops:
            for dis_dir in [d for d in crop_dir.iterdir() if d.is_dir()]:
                for img in dis_dir.glob("*"):
                    if img.suffix.lower() in IMG_EXTS:
                        stats["skipped_unknown_crop"] += 1
            continue

        ci = train_crops.index(crop_name)
        train_dis_list = train_dis[crop_name]

        for dis_dir in sorted([d for d in crop_dir.iterdir() if d.is_dir()]):
            dis_name = dis_dir.name

            if dis_name in train_dis_list:
                di = train_dis_list.index(dis_name)
                known = True
            else:
                di = -1
                known = False

            for img_path in dis_dir.glob("*"):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                stats["total_images"] += 1
                if known:
                    stats["known_crop_known_disease"] += 1
                    global_joint_id = global_index[(ci, di)]
                else:
                    stats["known_crop_unknown_disease"] += 1
                    global_joint_id = -1
                items.append((str(img_path), ci, di, global_joint_id))

    return items, stats
