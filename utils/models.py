"""
utils/models.py — Model architectures for flat and hierarchical pipelines.

Both models share the same ResNet-18 backbone loading logic with an offline
fallback to a local checkpoint when pretrained weights cannot be downloaded.

Provides:
  - FlatResNet18       : single joint-class linear head
  - HierResNet18Concat : crop head + concatenated per-crop disease heads
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

_LOCAL_WEIGHTS = "/home/nalwangar/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"


def _load_resnet18_backbone():
    """Load pretrained ResNet-18, falling back to a local checkpoint if offline."""
    try:
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        print("Loaded ResNet18 pretrained weights.")
    except Exception:
        print("Offline mode: loading local ResNet-18 weights.")
        backbone = models.resnet18(weights=None)
        backbone.load_state_dict(torch.load(_LOCAL_WEIGHTS, map_location="cpu"))
    return backbone


class FlatResNet18(nn.Module):
    """
    Flat baseline classifier.

    Architecture:
        ResNet-18 backbone (layer4 + head trainable, rest frozen)
        → Linear(512, num_joint_classes)

    A single head predicts over all (crop, disease) joint classes at once.
    """

    def __init__(self, num_joint_classes):
        super().__init__()

        backbone = _load_resnet18_backbone()

        for p in backbone.parameters():
            p.requires_grad = False
        for name, p in backbone.named_parameters():
            if "layer4" in name:
                p.requires_grad = True

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_joint_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class HierResNet18Concat(nn.Module):
    """
    Hierarchical classifier — Option C (concatenated heads).

    Architecture:
        ResNet-18 backbone (layer4 + both heads trainable, rest frozen)
        → crop_head  : Linear(512, num_crops)          — crop logits
        → heads[0..C]: Linear(512, n_diseases_for_crop) — per-crop disease logits
        → cat(heads)  : (B, total_diseases)             — concatenated disease logits

    crop_slices[ci] = (start, end) maps each crop index to its slice in the
    concatenated disease vector, used for sliced loss during training and
    two-stage inference during evaluation.
    """

    def __init__(self, crops, diseases_by_crop):
        super().__init__()

        self.crops = crops
        self.diseases_by_crop = diseases_by_crop

        backbone = _load_resnet18_backbone()

        for name, p in backbone.named_parameters():
            p.requires_grad = ("layer4" in name)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.crop_head = nn.Linear(in_features, len(crops))

        self.crop_names = list(crops)
        self.heads = nn.ModuleList([
            nn.Linear(in_features, len(diseases_by_crop[c]))
            for c in self.crop_names
        ])

        # Build (crop_id, dis_id) -> global offset mapping and label list
        self.global_labels = []
        self.offsets = {}
        index = 0
        for ci, crop in enumerate(crops):
            for di, dis in enumerate(diseases_by_crop[crop]):
                self.offsets[(ci, di)] = index
                self.global_labels.append(f"{crop}:{dis}")
                index += 1
        self.total_diseases = len(self.global_labels)

        # Build crop_id -> (start, end) slice in the concatenated disease vector
        self.crop_slices = {}
        start = 0
        for ci, crop in enumerate(crops):
            n_dis = len(diseases_by_crop[crop])
            self.crop_slices[ci] = (start, start + n_dis)
            start += n_dis

    def forward(self, x):
        feats = self.backbone(x)
        crop_logits = self.crop_head(feats)
        concat_logits = torch.cat([head(feats) for head in self.heads], dim=1)
        return crop_logits, concat_logits
