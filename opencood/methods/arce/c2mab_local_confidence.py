"""Local CAV confidence utilities for C2MAB-ARCE.

C_i is computed from each CAV's own pre-fusion detection head output.
It is action-before / fusion-before information, so it is legal context.
"""

from __future__ import annotations

from typing import Any

import torch


def local_cav_confidences_from_psm(psm_single: Any, topk: int = 50):
    """Compute one local confidence C_i for each CAV.

    Parameters
    ----------
    psm_single:
        Pre-fusion dense classification logits, usually [sum_cav, A, H, W].
    topk:
        Number of strongest dense cells used to summarize local confidence.

    Returns
    -------
    torch.Tensor or None
        Shape [sum_cav], values clipped into [0, 1].
    """
    if psm_single is None or not torch.is_tensor(psm_single):
        return None

    with torch.no_grad():
        conf = torch.sigmoid(psm_single.detach().float())

        if conf.dim() >= 4:
            conf_map = conf.max(dim=1)[0]
        elif conf.dim() == 3:
            conf_map = conf
        else:
            return None

        flat = conf_map.reshape(conf_map.shape[0], -1)
        if flat.numel() <= 0:
            return None

        k = min(int(topk), int(flat.shape[1]))
        if k <= 0:
            return None

        vals = torch.topk(flat, k=k, dim=1).values.mean(dim=1)
        return torch.clamp(vals, 0.0, 1.0)


def get_cav_confidence(local_cav_confidences: Any, cav_idx: int, default: Any = 0.0) -> Any:
    """Return one CAV confidence value.

    If default=None, missing or invalid confidence returns None. This lets callers
    distinguish "not available" from a valid numeric confidence.
    """
    if default is None:
        fallback = None
    else:
        try:
            fallback = float(default)
        except Exception:
            fallback = 0.0

    try:
        if local_cav_confidences is None:
            return fallback

        vals = local_cav_confidences
        if hasattr(vals, "detach"):
            vals = vals.detach().cpu().flatten().tolist()
        elif hasattr(vals, "flatten"):
            vals = vals.flatten().tolist()

        idx = int(cav_idx)
        if idx < 0 or idx >= len(vals):
            return fallback

        v = vals[idx]
        if v is None:
            return fallback
        return float(v)
    except Exception:
        return fallback

def local_cav_confidence_maps_from_psm(psm_single: Any) -> Any:
    """Return per-CAV local detection confidence maps [N, H, W].

    This uses detection-head classification logits, not communication masks.
    """
    if psm_single is None:
        return None

    try:
        if not torch.is_tensor(psm_single):
            return None

        conf = torch.sigmoid(psm_single.detach().float())
        if conf.dim() == 4:
            # [N, A, H, W] -> [N, H, W]
            conf = conf.max(dim=1).values
        elif conf.dim() == 3:
            # Already [N, H, W]
            pass
        else:
            return None

        return torch.clamp(conf, 0.0, 1.0)
    except Exception:
        return None

__all__ = [
    "local_cav_confidences_from_psm",
    "local_cav_confidence_maps_from_psm",
    "get_cav_confidence",
]
