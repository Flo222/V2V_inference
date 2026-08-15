# -*- coding: utf-8 -*-
"""
Utility functions for RoCooper in OpenCOOD.

Save as:
    opencood/models/baselines/rocooper/components/rocooper_utils.py

This file contains reusable helpers for:
    - OpenCOOD record_len handling
    - scenario / ego / neighbor feature splitting
    - feature alignment to ego frame
    - fallback feature fusion
    - BEV feature partition / reverse partition
    - top-k block gather / scatter
    - temporal history update
    - safe shape checks

Main users:
    opencood/models/baselines/rocooper/rocooper_fuse.py
    opencood/models/baselines/rocooper/components/rocooper_aggregator.py
    opencood/models/baselines/rocooper/components/rocooper_augmentor.py
    opencood/models/baselines/rocooper/components/rocooper_block_prioritizer.py
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


# OpenCOOD defines the light-weight affine helpers used by several
# intermediate-fusion modules in coalign_fuse.py.  The previous import from
# torch_transformation_utils silently failed in this repository, which made
# RoCooper skip feature alignment and fuse geometrically misaligned CAV
# features.  Keep the fallback to avoid breaking minimal environments, but
# prefer the actual implementation.
try:
    from opencood.models.common.fuse_modules.coalign_fuse import (
        warp_affine_simple,
        normalize_pairwise_tfm,
    )
except Exception:
    warp_affine_simple = None
    normalize_pairwise_tfm = None


RecordLenType = Union[torch.Tensor, List[int], Tuple[int, ...]]
TensorOrNone = Optional[torch.Tensor]


# ----------------------------------------------------------------------
# Basic record_len helpers
# ----------------------------------------------------------------------


def record_len_to_list(record_len: RecordLenType) -> List[int]:
    """
    Convert OpenCOOD record_len to a Python list.

    Args:
        record_len:
            Tensor/List/Tuple. Example: tensor([3, 5]) means the first
            scenario has 3 CAVs and the second scenario has 5 CAVs.

    Returns:
        List[int]
    """
    if isinstance(record_len, torch.Tensor):
        return [int(v) for v in record_len.detach().cpu().tolist()]

    if isinstance(record_len, (list, tuple)):
        return [int(v) for v in record_len]

    raise TypeError(
        "record_len must be torch.Tensor, list, or tuple, "
        f"but got {type(record_len).__name__}."
    )


def validate_record_len(record_len_list: Sequence[int]) -> None:
    """
    Validate that each scenario has at least one CAV.
    """
    if len(record_len_list) == 0:
        raise ValueError("record_len is empty.")

    for i, num in enumerate(record_len_list):
        if int(num) <= 0:
            raise ValueError(
                f"Invalid record_len at index {i}: {num}. "
                "Each scenario must contain at least the ego vehicle."
            )


def validate_feature_record_len(
    features: torch.Tensor,
    record_len_list: Sequence[int],
) -> None:
    """
    Validate feature tensor and record_len compatibility.

    Args:
        features:
            Tensor with shape [sum(record_len), C, H, W].

        record_len_list:
            Python list converted from record_len.
    """
    if not isinstance(features, torch.Tensor):
        raise TypeError(
            f"features must be a torch.Tensor, got {type(features).__name__}."
        )

    if features.dim() != 4:
        raise ValueError(
            "features must have shape [sum(record_len), C, H, W], "
            f"but got {tuple(features.shape)}."
        )

    validate_record_len(record_len_list)

    total = int(sum(record_len_list))
    if total != int(features.shape[0]):
        raise ValueError(
            "sum(record_len) must equal features.shape[0]. "
            f"sum(record_len)={total}, features.shape[0]={features.shape[0]}."
        )


def cumulative_record_len(record_len_list: Sequence[int]) -> List[Tuple[int, int]]:
    """
    Return start/end indices for each scenario.

    Example:
        record_len_list = [3, 2]
        return [(0, 3), (3, 5)]
    """
    validate_record_len(record_len_list)

    ranges: List[Tuple[int, int]] = []
    start = 0

    for num in record_len_list:
        end = start + int(num)
        ranges.append((start, end))
        start = end

    return ranges


def make_ego_mask(
    record_len: RecordLenType,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a boolean mask where the first CAV in each scenario is ego.

    Args:
        record_len:
            Tensor/List/Tuple.

        device:
            Optional device.

    Returns:
        ego_mask:
            Boolean tensor with shape [sum(record_len)].
    """
    record_len_list = record_len_to_list(record_len)

    masks = []
    for num in record_len_list:
        local = torch.zeros(int(num), dtype=torch.bool, device=device)
        local[0] = True
        masks.append(local)

    return torch.cat(masks, dim=0)


def make_non_ego_mask(
    record_len: RecordLenType,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a boolean mask for all non-ego CAVs.
    """
    return ~make_ego_mask(record_len, device=device)


# ----------------------------------------------------------------------
# Scenario splitting helpers
# ----------------------------------------------------------------------


def split_by_record_len(
    tensor: torch.Tensor,
    record_len: RecordLenType,
) -> List[torch.Tensor]:
    """
    Split a tensor according to record_len along the first dimension.

    Args:
        tensor:
            Tensor with first dimension sum(record_len).

        record_len:
            Tensor/List/Tuple.

    Returns:
        List[Tensor], each item is one scenario group.
    """
    record_len_list = record_len_to_list(record_len)

    if tensor.shape[0] != sum(record_len_list):
        raise ValueError(
            "tensor.shape[0] must equal sum(record_len). "
            f"Got tensor.shape[0]={tensor.shape[0]}, "
            f"sum(record_len)={sum(record_len_list)}."
        )

    groups: List[torch.Tensor] = []
    start = 0

    for num in record_len_list:
        end = start + int(num)
        groups.append(tensor[start:end])
        start = end

    return groups


def split_ego_others(group: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split one scenario group into ego feature and other CAV features.

    Args:
        group:
            Tensor with shape [num_cav, C, H, W].

    Returns:
        ego:
            Tensor with shape [C, H, W].

        others:
            Tensor with shape [num_cav - 1, C, H, W].
    """
    if group.dim() != 4:
        raise ValueError(
            "group must have shape [num_cav, C, H, W], "
            f"but got {tuple(group.shape)}."
        )

    if group.shape[0] <= 0:
        raise ValueError("group must contain at least one CAV.")

    ego = group[0]
    others = group[1:]

    return ego, others


def merge_ego_others(
    ego: torch.Tensor,
    others: torch.Tensor,
) -> torch.Tensor:
    """
    Merge ego and other CAV features back into one group.

    Args:
        ego:
            [C, H, W] or [1, C, H, W]

        others:
            [L, C, H, W]

    Returns:
        group:
            [1 + L, C, H, W]
    """
    if ego.dim() == 3:
        ego = ego.unsqueeze(0)

    if ego.dim() != 4:
        raise ValueError(
            "ego must have shape [C, H, W] or [1, C, H, W], "
            f"but got {tuple(ego.shape)}."
        )

    if others.dim() != 4:
        raise ValueError(
            "others must have shape [L, C, H, W], "
            f"but got {tuple(others.shape)}."
        )

    return torch.cat([ego, others], dim=0)


# ----------------------------------------------------------------------
# Feature alignment helpers
# ----------------------------------------------------------------------


def align_group_to_ego(
    group: torch.Tensor,
    batch_idx: int,
    cav_num: int,
    pairwise_t_matrix: TensorOrNone,
    discrete_ratio: float = 0.4,
    downsample_rate: int = 4,
    enabled: bool = True,
    transform_direction: str = "source_to_ego",
) -> torch.Tensor:
    """
    Align all CAV feature maps in one scenario to the ego coordinate frame.

    This is a safe wrapper around OpenCOOD's:
        normalize_pairwise_tfm
        warp_affine_simple

    If the utility functions are unavailable or the transform format is
    incompatible, this function returns the original group unchanged.

    Args:
        group:
            Tensor with shape [num_cav, C, H, W].

        batch_idx:
            Scenario index in current batch.

        cav_num:
            Number of CAVs in this scenario.

        pairwise_t_matrix:
            OpenCOOD pairwise transform matrix, usually
            [B, max_cav, max_cav, 4, 4].

        discrete_ratio:
            Spatial resolution ratio used by OpenCOOD normalization.

        downsample_rate:
            Feature downsample rate.

        enabled:
            Whether to perform alignment.

        transform_direction:
            Which pairwise transform slice maps CAV features into ego frame.
            In OpenCOOD IntermediateFusionDataset, pairwise_t_matrix[i, j]
            is the transform from CAV i to CAV j.  Therefore the default
            "source_to_ego" uses pairwise_t_matrix[:, 0].  The legacy
            "ego_to_source" option uses pairwise_t_matrix[0, :] for
            compatibility/debugging.

    Returns:
        aligned_group:
            Tensor with the same shape as group.
    """
    if not enabled:
        return group

    if group.dim() != 4:
        return group

    if pairwise_t_matrix is None:
        return group

    if warp_affine_simple is None or normalize_pairwise_tfm is None:
        return group

    if not isinstance(pairwise_t_matrix, torch.Tensor):
        return group

    if pairwise_t_matrix.dim() != 5:
        return group

    if cav_num <= 1:
        return group

    try:
        _, _, height, width = group.shape

        tfm = pairwise_t_matrix.to(
            device=group.device,
            dtype=group.dtype,
        )

        normalized_tfm = normalize_pairwise_tfm(
            tfm,
            height,
            width,
            discrete_ratio,
            downsample_rate,
        )

        if batch_idx >= normalized_tfm.shape[0]:
            return group

        if cav_num > normalized_tfm.shape[1]:
            cav_num = int(normalized_tfm.shape[1])

        # OpenCOOD IntermediateFusionDataset builds pairwise_t_matrix[i, j]
        # as the transform from source CAV i to target CAV j:
        #     T_j^{-1} T_i P_i = P_j
        # Thus, to warp all CAV features into ego frame (target index 0),
        # use normalized_tfm[batch_idx, :cav_num, 0].  The previous
        # normalized_tfm[batch_idx, 0, :cav_num] is the inverse direction and
        # causes localization-damaging misalignment.
        direction = str(transform_direction).lower()
        if direction in ["source_to_ego", "src_to_ego", "cav_to_ego", "to_ego"]:
            ego_matrices = normalized_tfm[batch_idx, :cav_num, 0, :, :]
        elif direction in ["ego_to_source", "ego_to_src", "ego_to_cav", "legacy"]:
            ego_matrices = normalized_tfm[batch_idx, 0, :cav_num, :, :]
        else:
            # Safe default for OpenCOOD.
            ego_matrices = normalized_tfm[batch_idx, :cav_num, 0, :, :]

        aligned = warp_affine_simple(
            group[:cav_num],
            ego_matrices,
            (height, width),
        )

        if aligned.shape != group[:cav_num].shape:
            return group

        if cav_num == group.shape[0]:
            return aligned

        # If pairwise matrix contains fewer CAVs than group, keep the rest.
        return torch.cat([aligned, group[cav_num:]], dim=0)

    except Exception:
        return group


def estimate_ego_distances(
    record_len: RecordLenType,
    pairwise_t_matrix: TensorOrNone,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Estimate ego-CAV distances from pairwise_t_matrix.

    This helper is useful for communication impairment modules.

    Args:
        record_len:
            Tensor/List/Tuple.

        pairwise_t_matrix:
            Usually [B, max_cav, max_cav, 4, 4].

        device:
            Output device.

        dtype:
            Output dtype.

    Returns:
        distances:
            Tensor with shape [sum(record_len)].
            Ego distance is set to 1.0.
    """
    record_len_list = record_len_to_list(record_len)
    total_cav = int(sum(record_len_list))

    if device is None:
        if isinstance(pairwise_t_matrix, torch.Tensor):
            device = pairwise_t_matrix.device
        else:
            device = torch.device("cpu")

    if dtype is None:
        if isinstance(pairwise_t_matrix, torch.Tensor):
            dtype = pairwise_t_matrix.dtype
        else:
            dtype = torch.float32

    distances = torch.ones(total_cav, device=device, dtype=dtype)

    if pairwise_t_matrix is None:
        return distances

    if not isinstance(pairwise_t_matrix, torch.Tensor):
        return distances

    if pairwise_t_matrix.dim() != 5:
        return distances

    start = 0

    for b, cav_num in enumerate(record_len_list):
        if b >= pairwise_t_matrix.shape[0]:
            break

        max_cav = pairwise_t_matrix.shape[1]

        for local_idx in range(int(cav_num)):
            global_idx = start + local_idx

            if local_idx == 0:
                distances[global_idx] = 1.0
                continue

            if local_idx >= max_cav:
                distances[global_idx] = 1.0
                continue

            candidate_norms = []

            try:
                trans_1 = pairwise_t_matrix[b, 0, local_idx, :3, 3]
                candidate_norms.append(torch.norm(trans_1))
            except Exception:
                pass

            try:
                trans_2 = pairwise_t_matrix[b, local_idx, 0, :3, 3]
                candidate_norms.append(torch.norm(trans_2))
            except Exception:
                pass

            valid = [
                n for n in candidate_norms
                if torch.isfinite(n).item() and n.item() > 1e-6
            ]

            if len(valid) > 0:
                distances[global_idx] = valid[0].to(device=device, dtype=dtype)

        start += int(cav_num)

    return distances.clamp(min=1.0)


# ----------------------------------------------------------------------
# Fallback fusion
# ----------------------------------------------------------------------


def fallback_fusion_single(
    ego: torch.Tensor,
    others: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Simple fallback feature fusion for one scenario.

    Args:
        ego:
            Tensor with shape [C, H, W].

        others:
            Tensor with shape [L, C, H, W].

        mode:
            "mean", "max", "sum", or "ego".

    Returns:
        fused:
            Tensor with shape [C, H, W].
    """
    mode = str(mode).lower()

    if ego.dim() != 3:
        raise ValueError(
            f"ego must have shape [C, H, W], got {tuple(ego.shape)}."
        )

    if others.numel() == 0 or others.shape[0] == 0:
        return ego

    if others.dim() != 4:
        raise ValueError(
            f"others must have shape [L, C, H, W], got {tuple(others.shape)}."
        )

    if mode == "ego":
        return ego

    stack = torch.cat([ego.unsqueeze(0), others], dim=0)

    if mode == "max":
        return torch.max(stack, dim=0).values

    if mode == "sum":
        return torch.sum(stack, dim=0)

    # Default: mean fusion.
    return torch.mean(stack, dim=0)


def fallback_fusion_batch(
    features: torch.Tensor,
    record_len: RecordLenType,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Apply fallback fusion to a whole OpenCOOD batch.

    Args:
        features:
            [sum(record_len), C, H, W]

        record_len:
            Tensor/List/Tuple.

        mode:
            "mean", "max", "sum", or "ego".

    Returns:
        fused_features:
            [B, C, H, W]
    """
    record_len_list = record_len_to_list(record_len)
    validate_feature_record_len(features, record_len_list)

    fused = []
    start = 0

    for cav_num in record_len_list:
        end = start + int(cav_num)
        group = features[start:end]
        start = end

        ego, others = split_ego_others(group)
        fused.append(fallback_fusion_single(ego, others, mode=mode))

    return torch.stack(fused, dim=0)


# ----------------------------------------------------------------------
# Tensor / shape helpers
# ----------------------------------------------------------------------


def valid_num_heads(channels: int, requested_heads: int) -> int:
    """
    Return a valid attention head number that divides channels.

    Args:
        channels:
            Feature channel number.

        requested_heads:
            Requested number of attention heads.

    Returns:
        int
    """
    channels = int(channels)
    requested_heads = max(1, int(requested_heads))
    requested_heads = min(requested_heads, channels)

    while requested_heads > 1 and channels % requested_heads != 0:
        requested_heads -= 1

    return max(1, requested_heads)


def flatten_hw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert [N, C, H, W] to [N, H*W, C].
    """
    if x.dim() != 4:
        raise ValueError(f"flatten_hw expects [N, C, H, W], got {tuple(x.shape)}.")

    return x.flatten(2).transpose(1, 2).contiguous()


def unflatten_hw(
    x: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Convert [N, H*W, C] to [N, C, H, W].
    """
    if x.dim() != 3:
        raise ValueError(f"unflatten_hw expects [N, H*W, C], got {tuple(x.shape)}.")

    n, tokens, c = x.shape
    if tokens != int(height) * int(width):
        raise ValueError(
            "tokens must equal height * width. "
            f"tokens={tokens}, height={height}, width={width}."
        )

    return x.transpose(1, 2).contiguous().view(n, c, int(height), int(width))


def resize_like(
    x: torch.Tensor,
    ref: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resize x spatially to match ref.

    Args:
        x:
            [N, C, H, W]

        ref:
            [N, C, H_ref, W_ref]

        mode:
            Interpolation mode.

    Returns:
        resized x
    """
    if x.shape[-2:] == ref.shape[-2:]:
        return x

    if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
        return F.interpolate(
            x,
            size=ref.shape[-2:],
            mode=mode,
            align_corners=False,
        )

    return F.interpolate(
        x,
        size=ref.shape[-2:],
        mode=mode,
    )


# ----------------------------------------------------------------------
# BEV block partition utilities
# ----------------------------------------------------------------------


def get_padding_for_window(
    height: int,
    width: int,
    window_size: int,
) -> Tuple[int, int]:
    """
    Compute right/bottom padding for non-overlapping window partition.
    """
    s = int(window_size)
    if s <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")

    pad_h = (s - int(height) % s) % s
    pad_w = (s - int(width) % s) % s

    return pad_h, pad_w


def pad_feature_to_window(
    x: torch.Tensor,
    window_size: int,
    value: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Pad [N, C, H, W] so H and W are divisible by window_size.

    Returns:
        x_pad:
            [N, C, H_pad, W_pad]

        meta:
            Dict containing original and padded shape info.
    """
    if x.dim() != 4:
        raise ValueError(
            f"pad_feature_to_window expects [N, C, H, W], got {tuple(x.shape)}."
        )

    n, c, h, w = x.shape
    s = int(window_size)

    pad_h, pad_w = get_padding_for_window(h, w, s)

    x_pad = F.pad(x, (0, pad_w, 0, pad_h), value=float(value))

    meta = {
        "n": int(n),
        "c": int(c),
        "h": int(h),
        "w": int(w),
        "hp": int(h + pad_h),
        "wp": int(w + pad_w),
        "pad_h": int(pad_h),
        "pad_w": int(pad_w),
        "window_size": int(s),
        "num_h": int((h + pad_h) // s),
        "num_w": int((w + pad_w) // s),
    }

    return x_pad, meta


def partition_feature_map(
    x: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Partition feature map into non-overlapping BEV blocks.

    Args:
        x:
            Tensor with shape [N, C, H, W].

        window_size:
            Window size S.

    Returns:
        blocks:
            Tensor with shape [N, num_blocks, S*S, C].

        meta:
            Metadata for reverse_partition_feature_map.
    """
    x_pad, meta = pad_feature_to_window(x, window_size)

    n, c, hp, wp = x_pad.shape
    s = int(window_size)
    num_h = hp // s
    num_w = wp // s

    blocks = (
        x_pad.unfold(2, s, s)
        .unfold(3, s, s)
        .permute(0, 2, 3, 4, 5, 1)
        .contiguous()
    )  # [N, num_h, num_w, S, S, C]

    blocks = blocks.view(n, num_h * num_w, s * s, c)

    return blocks, meta


def reverse_partition_feature_map(
    blocks: torch.Tensor,
    meta: Dict[str, int],
) -> torch.Tensor:
    """
    Reverse non-overlapping BEV block partition.

    Args:
        blocks:
            Tensor with shape [N, num_blocks, S*S, C].

        meta:
            Metadata returned by partition_feature_map.

    Returns:
        x:
            Tensor with shape [N, C, H, W].
    """
    if blocks.dim() != 4:
        raise ValueError(
            "blocks must have shape [N, num_blocks, S*S, C], "
            f"got {tuple(blocks.shape)}."
        )

    n = int(meta["n"])
    c = int(meta["c"])
    h = int(meta["h"])
    w = int(meta["w"])
    s = int(meta["window_size"])
    num_h = int(meta["num_h"])
    num_w = int(meta["num_w"])

    expected_blocks = num_h * num_w
    expected_tokens = s * s

    if blocks.shape[0] != n:
        raise ValueError(f"blocks.shape[0]={blocks.shape[0]} but meta n={n}.")

    if blocks.shape[1] != expected_blocks:
        raise ValueError(
            f"blocks.shape[1]={blocks.shape[1]} but expected {expected_blocks}."
        )

    if blocks.shape[2] != expected_tokens:
        raise ValueError(
            f"blocks.shape[2]={blocks.shape[2]} but expected {expected_tokens}."
        )

    if blocks.shape[3] != c:
        raise ValueError(f"blocks.shape[3]={blocks.shape[3]} but meta c={c}.")

    x = blocks.view(n, num_h, num_w, s, s, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(n, c, num_h * s, num_w * s)

    return x[:, :, :h, :w].contiguous()


def partition_single_feature(
    x: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Partition one feature map [C, H, W] into blocks.

    Returns:
        blocks:
            [num_blocks, S*S, C]

        meta:
            Metadata.
    """
    if x.dim() != 3:
        raise ValueError(
            f"partition_single_feature expects [C, H, W], got {tuple(x.shape)}."
        )

    blocks, meta = partition_feature_map(x.unsqueeze(0), window_size)
    return blocks[0], meta


def reverse_single_partition(
    blocks: torch.Tensor,
    meta: Dict[str, int],
) -> torch.Tensor:
    """
    Reverse partition for one feature map.

    Args:
        blocks:
            [num_blocks, S*S, C]

    Returns:
        x:
            [C, H, W]
    """
    if blocks.dim() != 3:
        raise ValueError(
            "reverse_single_partition expects [num_blocks, S*S, C], "
            f"got {tuple(blocks.shape)}."
        )

    return reverse_partition_feature_map(blocks.unsqueeze(0), meta)[0]


# ----------------------------------------------------------------------
# Top-k block selection / gather / scatter
# ----------------------------------------------------------------------


def topk_indices_from_scores(
    scores: torch.Tensor,
    ratio: float,
    min_k: int = 1,
) -> torch.Tensor:
    """
    Select top-k indices according to a ratio.

    Args:
        scores:
            Tensor with shape [num_blocks].

        ratio:
            Selected block ratio.

        min_k:
            Minimum selected number.

    Returns:
        indices:
            Long tensor with shape [k].
    """
    if scores.dim() != 1:
        raise ValueError(
            f"scores must have shape [num_blocks], got {tuple(scores.shape)}."
        )

    num_blocks = int(scores.shape[0])
    if num_blocks <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    ratio = float(ratio)
    ratio = max(0.0, min(1.0, ratio))

    k = int(math.ceil(ratio * num_blocks))
    k = max(int(min_k), k)
    k = min(num_blocks, k)

    return torch.topk(scores, k=k, dim=0).indices


def gather_topk_blocks(
    blocks: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Gather selected blocks.

    Args:
        blocks:
            [num_blocks, tokens, C] or [N, num_blocks, tokens, C]

        indices:
            [k]

    Returns:
        selected:
            [k, tokens, C] or [N, k, tokens, C]
    """
    if indices.dtype != torch.long:
        indices = indices.long()

    if blocks.dim() == 3:
        return blocks.index_select(dim=0, index=indices)

    if blocks.dim() == 4:
        return blocks.index_select(dim=1, index=indices)

    raise ValueError(
        "blocks must have shape [num_blocks, tokens, C] or "
        f"[N, num_blocks, tokens, C], got {tuple(blocks.shape)}."
    )


def scatter_topk_blocks(
    original_blocks: torch.Tensor,
    updated_blocks: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Scatter updated top-k blocks back to original block tensor.

    Args:
        original_blocks:
            [num_blocks, tokens, C] or [N, num_blocks, tokens, C]

        updated_blocks:
            [k, tokens, C] or [N, k, tokens, C]

        indices:
            [k]

    Returns:
        new_blocks:
            Same shape as original_blocks.
    """
    if indices.dtype != torch.long:
        indices = indices.long()

    new_blocks = original_blocks.clone()

    if original_blocks.dim() == 3:
        if updated_blocks.dim() != 3:
            raise ValueError(
                "For 3D original_blocks, updated_blocks must also be 3D."
            )
        new_blocks.index_copy_(0, indices, updated_blocks)
        return new_blocks

    if original_blocks.dim() == 4:
        if updated_blocks.dim() != 4:
            raise ValueError(
                "For 4D original_blocks, updated_blocks must also be 4D."
            )
        new_blocks.index_copy_(1, indices, updated_blocks)
        return new_blocks

    raise ValueError(
        "original_blocks must have shape [num_blocks, tokens, C] or "
        f"[N, num_blocks, tokens, C], got {tuple(original_blocks.shape)}."
    )


def make_topk_mask(
    num_blocks: int,
    indices: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Make boolean top-k mask.

    Args:
        num_blocks:
            Total number of blocks.

        indices:
            Selected top-k indices.

        device:
            Output device.

    Returns:
        mask:
            Boolean tensor [num_blocks].
    """
    if device is None:
        device = indices.device

    mask = torch.zeros(int(num_blocks), dtype=torch.bool, device=device)

    if indices.numel() > 0:
        mask[indices.long()] = True

    return mask


# ----------------------------------------------------------------------
# History helpers
# ----------------------------------------------------------------------


def update_history_feature(
    history: Optional[torch.Tensor],
    current: torch.Tensor,
    tau: float = 0.8,
    init_with_current: bool = True,
) -> torch.Tensor:
    """
    Update temporal history feature.

    Formula:
        F_h = tau * F_current + (1 - tau) * F_h

    Args:
        history:
            Previous history tensor or None.

        current:
            Current feature tensor.

        tau:
            Update ratio.

        init_with_current:
            If True, initialize history with current when history is None
            or shape mismatched.

    Returns:
        updated history tensor.
    """
    tau = float(tau)
    tau = max(0.0, min(1.0, tau))

    if history is None or history.shape != current.shape:
        if init_with_current:
            return current.detach()
        return torch.zeros_like(current).detach()

    return (tau * current.detach() + (1.0 - tau) * history.detach()).detach()


def detach_dict_tensors(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detach tensor values in a dictionary.

    Useful for returning debugging info without keeping computation graphs.
    """
    out: Dict[str, Any] = {}

    for key, value in info.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach()
        elif isinstance(value, dict):
            out[key] = detach_dict_tensors(value)
        else:
            out[key] = value

    return out


# ----------------------------------------------------------------------
# Misc safe numeric helpers
# ----------------------------------------------------------------------


def safe_float(value: Any, default: float) -> float:
    """
    Convert value to float with fallback.
    """
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int) -> int:
    """
    Convert value to int with fallback.
    """
    try:
        return int(value)
    except Exception:
        return int(default)


def normal_clamped(
    mean: float,
    std: float,
    shape: Tuple[int, ...],
    device: torch.device,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> torch.Tensor:
    """
    Sample from normal distribution and clamp to [min_value, max_value].
    """
    mean = float(mean)
    std = float(std)

    if std <= 0:
        out = torch.full(shape, mean, device=device, dtype=torch.float32)
    else:
        out = torch.normal(
            mean=mean,
            std=std,
            size=shape,
            device=device,
        )

    return out.clamp(min=float(min_value), max=float(max_value))


def tensor_summary(x: torch.Tensor) -> Dict[str, Any]:
    """
    Return lightweight tensor summary for debugging.
    """
    if not isinstance(x, torch.Tensor):
        return {"type": type(x).__name__}

    if x.numel() == 0:
        return {
            "shape": list(x.shape),
            "numel": 0,
        }

    x_float = x.detach().float()

    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "mean": float(x_float.mean().item()),
        "std": float(x_float.std().item()),
        "min": float(x_float.min().item()),
        "max": float(x_float.max().item()),
    }


__all__ = [
    "record_len_to_list",
    "validate_record_len",
    "validate_feature_record_len",
    "cumulative_record_len",
    "make_ego_mask",
    "make_non_ego_mask",
    "split_by_record_len",
    "split_ego_others",
    "merge_ego_others",
    "align_group_to_ego",
    "estimate_ego_distances",
    "fallback_fusion_single",
    "fallback_fusion_batch",
    "valid_num_heads",
    "flatten_hw",
    "unflatten_hw",
    "resize_like",
    "get_padding_for_window",
    "pad_feature_to_window",
    "partition_feature_map",
    "reverse_partition_feature_map",
    "partition_single_feature",
    "reverse_single_partition",
    "topk_indices_from_scores",
    "gather_topk_blocks",
    "scatter_topk_blocks",
    "make_topk_mask",
    "update_history_feature",
    "detach_dict_tensors",
    "safe_float",
    "safe_int",
    "normal_clamped",
    "tensor_summary",
]