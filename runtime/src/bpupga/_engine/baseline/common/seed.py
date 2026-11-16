from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed process RNGs and optionally request strict PyTorch determinism.

    The default preserves the historical benchmark behavior.  The opt-in mode
    is intended for reproducibility diagnosis and may make PyTorch reject an
    operation for which no deterministic implementation exists.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if not deterministic_torch:
        return

    # Import lazily so CPU-only methods do not acquire a PyTorch dependency.
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

