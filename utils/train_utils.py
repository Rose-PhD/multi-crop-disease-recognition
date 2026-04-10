"""
utils/train_utils.py — Shared training utilities.

Provides
--------
build_global_index  : build (crop_idx, disease_idx) -> global_id mapping
run_epoch_loop      : standard training loop with early stopping
"""

from pathlib import Path

import torch
from torch.optim import Adam


def build_global_index(crops, diseases_by_crop):
    """
    Build a global (crop_idx, disease_idx) -> global_id mapping and label list.

    Parameters
    ----------
    crops            : list[str]
    diseases_by_crop : dict[str, list[str]]

    Returns
    -------
    global_index  : dict[(int, int), int]
    global_labels : list[str]  — "<crop>:<disease>" for each global id
    """
    global_index  = {}
    global_labels = []
    idx = 0
    for ci, crop in enumerate(crops):
        for di, dis in enumerate(diseases_by_crop[crop]):
            global_index[(ci, di)] = idx
            global_labels.append(f"{crop}:{dis}")
            idx += 1
    return global_index, global_labels


def run_epoch_loop(fold, model, train_loader, val_loader, device, fold_dir,
                   compute_loss, *, patience=7, max_epochs=50, lr=1e-4):
    """
    Standard Adam training loop with early stopping.

    Parameters
    ----------
    compute_loss : callable(model, batch, device) -> Tensor
        Returns the scalar loss for one batch.
        Must NOT call .backward() — that is handled here.

    Returns
    -------
    best_val_loss : float
    model_path    : Path  — location of the saved best checkpoint
    """
    optimizer  = Adam(model.parameters(), lr=lr)
    best_loss  = float("inf")
    wait       = 0
    model_path = Path(fold_dir) / "best_model.pth"

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train = 0.0
        for batch in train_loader:
            if batch is None:
                continue
            optimizer.zero_grad()
            loss = compute_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                total_val += compute_loss(model, batch, device).item()

        print(f"[Fold {fold}] Epoch {epoch:3d} | Train: {total_train:.4f} | Val: {total_val:.4f}")

        if total_val < best_loss:
            best_loss = total_val
            wait = 0
            torch.save(model.state_dict(), model_path)
            print("   Best model updated!")
        else:
            wait += 1
            if wait >= patience:
                print("   Early stopping!")
                break

    return best_loss, model_path
