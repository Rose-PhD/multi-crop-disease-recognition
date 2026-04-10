import torch
import torch.nn as nn

from models.flat.custom_model import ResNet18


class CustomHierResNet18(nn.Module):
    """
    Hierarchical classifier built on a scratch ResNet-18 backbone.

    Architecture (mirrors HierResNet18Concat but fully from scratch):
        ResNet-18 backbone (no pretrained weights, all parameters trainable)
        → crop_head  : Linear(512, num_crops)              — crop logits
        → heads[0..C]: Linear(512, n_diseases_for_crop_c)  — per-crop disease logits
        → cat(heads)  : (B, total_diseases)                — concatenated disease logits

    crop_slices[ci] = (start, end) maps each crop index to its slice in the
    concatenated disease vector — used for sliced loss during training and
    two-stage inference during evaluation.

    Parameters
    ----------
    crops : list[str]
        Ordered list of crop names.
    diseases_by_crop : dict[str, list[str]]
        Maps each crop name to its ordered list of disease names.
    """

    def __init__(self, crops, diseases_by_crop):
        super().__init__()

        self.crops = list(crops)
        self.diseases_by_crop = diseases_by_crop

        # Build backbone from scratch and strip the classification head
        backbone = ResNet18(num_classes=1)   # num_classes is irrelevant; fc is replaced
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone             # all parameters trainable by default

        # Crop head
        self.crop_head = nn.Linear(in_features, len(self.crops))

        # Per-crop disease heads (one per crop)
        self.heads = nn.ModuleList([
            nn.Linear(in_features, len(diseases_by_crop[c]))
            for c in self.crops
        ])

        # Build global label list and (crop_id, dis_id) -> global offset mapping
        self.global_labels = []
        self.global_index = {}
        index = 0
        for ci, crop in enumerate(self.crops):
            for di, dis in enumerate(diseases_by_crop[crop]):
                self.global_index[(ci, di)] = index
                self.global_labels.append(f"{crop}:{dis}")
                index += 1
        self.total_diseases = len(self.global_labels)

        # Build crop_id -> (start, end) slice in the concatenated disease vector
        self.crop_slices = {}
        start = 0
        for ci, crop in enumerate(self.crops):
            n_dis = len(diseases_by_crop[crop])
            self.crop_slices[ci] = (start, start + n_dis)
            start += n_dis

    def forward(self, x):
        feats = self.backbone(x)
        crop_logits = self.crop_head(feats)
        concat_logits = torch.cat([head(feats) for head in self.heads], dim=1)
        return crop_logits, concat_logits
