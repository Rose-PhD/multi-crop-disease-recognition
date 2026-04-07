"""
utils/transforms.py — Image transform pipelines for training and evaluation.

ImageNet mean/std normalisation is applied in all transforms.
make_val_transform and make_test_transform are aliases of make_eval_transform
because the pipeline is identical; they differ only in where they are used.
"""

from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def make_train_transform():
    """Augmented pipeline used during training folds."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def make_eval_transform():
    """No-augmentation pipeline used for validation, testing, and inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# Aliases — same pipeline, explicit names for each call site context
make_val_transform = make_eval_transform
make_test_transform = make_eval_transform
