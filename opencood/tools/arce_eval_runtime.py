"""Shared runtime helpers for deterministic GRACE/ARCE evaluation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_deterministic_seed(seed: int) -> None:
    """Seed evaluation RNGs and request deterministic CUDA behavior."""
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


__all__ = ["set_deterministic_seed"]
