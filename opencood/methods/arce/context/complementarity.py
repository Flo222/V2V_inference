"""Mask complementarity and diversity utilities for Where2comm-ARCE.

The functions are intentionally shape-tolerant. They accept masks with shapes
such as [N,H,W], [N,1,H,W], [B,L,1,H,W], or [H,W]. Values are binarized by a
threshold before set operations.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import torch


def _to_mask(x: Any, threshold: float = 0.5, device: Optional[torch.device] = None) -> torch.Tensor:
    if x is None:
        raise ValueError('mask is None')
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if device is not None:
        x = x.to(device)
    x = x.detach()
    # remove singleton channel dimensions conservatively
    while x.dim() >= 3 and x.shape[-3] == 1:
        x = x.squeeze(-3)
    return (x.float() > threshold)


def flatten_mask(mask: Any, threshold: float = 0.5) -> torch.Tensor:
    m = _to_mask(mask, threshold=threshold)
    return m.reshape(-1).bool()


def mask_area(mask: Any, threshold: float = 0.5) -> float:
    m = flatten_mask(mask, threshold=threshold)
    return float(m.sum().item())


def mask_iou(a: Any, b: Any, threshold: float = 0.5, eps: float = 1e-6) -> float:
    ma = flatten_mask(a, threshold=threshold)
    mb = flatten_mask(b, threshold=threshold).to(ma.device)
    n = min(ma.numel(), mb.numel())
    ma, mb = ma[:n], mb[:n]
    inter = (ma & mb).sum().float()
    union = (ma | mb).sum().float()
    return float((inter / (union + eps)).item())


def ego_complementarity(cav_mask: Any, ego_mask: Any, threshold: float = 0.5, eps: float = 1e-6) -> float:
    mi = flatten_mask(cav_mask, threshold=threshold)
    me = flatten_mask(ego_mask, threshold=threshold).to(mi.device)
    n = min(mi.numel(), me.numel())
    mi, me = mi[:n], me[:n]
    novel = (mi & (~me)).sum().float()
    denom = mi.sum().float()
    return float((novel / (denom + eps)).item())


def overlap_with_selected(cav_mask: Any, selected_union_mask: Optional[Any], threshold: float = 0.5, eps: float = 1e-6) -> float:
    if selected_union_mask is None:
        return 0.0
    mi = flatten_mask(cav_mask, threshold=threshold)
    ms = flatten_mask(selected_union_mask, threshold=threshold).to(mi.device)
    n = min(mi.numel(), ms.numel())
    mi, ms = mi[:n], ms[:n]
    overlap = (mi & ms).sum().float()
    denom = mi.sum().float()
    return float((overlap / (denom + eps)).item())


def union_masks(masks: Sequence[Any], threshold: float = 0.5) -> Optional[torch.Tensor]:
    if not masks:
        return None
    outs = [flatten_mask(m, threshold=threshold) for m in masks if m is not None]
    if not outs:
        return None
    n = min(m.numel() for m in outs)
    u = outs[0][:n].clone()
    for m in outs[1:]:
        u |= m[:n].to(u.device)
    return u


def split_agent_masks(raw_masks: Any, record_len: Optional[Sequence[int]] = None, batch_index: int = 0, threshold: float = 0.5) -> List[torch.Tensor]:
    """Return masks for one batch item as a list [ego, cav1, ...].

    Common inputs:
    - [sum_cav, 1, H, W]
    - [B, L, 1, H, W]
    - list of tensors
    """
    if raw_masks is None:
        return []
    if isinstance(raw_masks, (list, tuple)):
        return [_to_mask(m, threshold=threshold) for m in raw_masks]
    x = raw_masks if torch.is_tensor(raw_masks) else torch.as_tensor(raw_masks)
    if x.dim() >= 5:  # [B,L,1,H,W]
        x = x[batch_index]
    if record_len is not None and x.dim() >= 4 and x.shape[0] == sum(int(v) for v in record_len):
        start = sum(int(record_len[i]) for i in range(batch_index))
        end = start + int(record_len[batch_index])
        x = x[start:end]
    if x.dim() == 2:
        return [_to_mask(x, threshold=threshold)]
    return [_to_mask(x[i], threshold=threshold) for i in range(int(x.shape[0]))]


def compute_pairwise_complementarity(raw_masks: Any, record_len: Optional[Sequence[int]] = None, batch_index: int = 0, threshold: float = 0.5) -> Dict[int, Dict[str, float]]:
    masks = split_agent_masks(raw_masks, record_len=record_len, batch_index=batch_index, threshold=threshold)
    if not masks:
        return {}
    ego = masks[0]
    out: Dict[int, Dict[str, float]] = {}
    for idx, m in enumerate(masks):
        out[idx] = {
            'mask_area': mask_area(m, threshold=threshold),
            'comp_to_ego': 0.0 if idx == 0 else ego_complementarity(m, ego, threshold=threshold),
            'iou_to_ego': 1.0 if idx == 0 else mask_iou(m, ego, threshold=threshold),
        }
    return out
