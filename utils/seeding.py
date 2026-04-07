"""
utils/seeding.py — Global reproducibility seed, worker seeder, and safe collate.

Sets SEED at import time so that any script importing this module
gets a consistent random state from the start.
"""

import random

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Seeded generator passed to DataLoader for deterministic shuffle order
g = torch.Generator()
g.manual_seed(SEED)


def seed_worker(worker_id):
    """Seed each DataLoader worker independently to ensure reproducibility."""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def safe_collate(batch):
    """Drop None samples (from corrupted/missing images) before collating."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)
