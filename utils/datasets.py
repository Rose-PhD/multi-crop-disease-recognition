"""
utils/datasets.py — Dataset classes and training data index builder.

Provides:
  - IMG_EXTS: accepted image file extensions
  - build_index: walks the training directory tree
  - HierDataset: training dataset (crop/disease/image structure)
  - RegionDataset: test-region dataset (flat and hierarchical pipelines)
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_index(dataset_root):
    """
    Walk dataset_root/<crop>/<disease>/<images> and collect all samples.

    Returns:
        crops           : sorted list of crop names
        diseases_by_crop: dict[crop_name] -> sorted list of disease names
        items           : list of (img_path, crop_idx, disease_idx)
    """
    root = Path(dataset_root)
    crops = sorted([d.name for d in root.iterdir() if d.is_dir()])

    diseases_by_crop = {}
    items = []

    for ci, crop in enumerate(crops):
        ddir = root / crop
        dis_list = sorted([d.name for d in ddir.iterdir() if d.is_dir()])
        diseases_by_crop[crop] = dis_list

        for di, dis in enumerate(dis_list):
            for img in (ddir / dis).glob("*"):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                items.append((str(img), ci, di))

    return crops, diseases_by_crop, items


class HierDataset(Dataset):
    """
    Training dataset for both flat and hierarchical pipelines.

    Args:
        items      : list of (img_path, crop_id, dis_id)
        transform  : torchvision transform to apply
        global_map : dict[(crop_id, dis_id)] -> global_joint_id

    Returns per sample: (img_tensor, crop_id, dis_id, global_joint_id)
    Corrupted images are skipped by returning None (handled by safe_collate).
    """

    def __init__(self, items, transform, global_map):
        self.items = items
        self.t = transform
        self.global_map = global_map

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, crop_id, dis_id = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            print(f"[CORRUPTED] Skipping {path}", flush=True)
            return None
        img = self.t(img)
        global_dis_id = self.global_map[(crop_id, dis_id)]
        return img, crop_id, dis_id, global_dis_id


class RegionDataset(Dataset):
    """
    Test/region dataset shared by both flat and hierarchical test scripts.

    Args:
        items    : list of (img_path, crop_id, dis_local, global_joint_id)
                   global_joint_id = -1 for diseases not seen during training
        transform: torchvision transform to apply

    Returns per sample: (img_tensor, crop_id, dis_local, global_joint_id)
    Corrupted images are skipped by returning None (handled by safe_collate).
    """

    def __init__(self, items, transform):
        self.items = items
        self.t = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, crop_id, dis_local, global_joint_id = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            print(f"[WARNING] Skipping corrupted image: {path}")
            return None
        img = self.t(img)
        return img, crop_id, dis_local, global_joint_id
