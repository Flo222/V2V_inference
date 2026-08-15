"""
Spatial interpolation recovery for ARCE communication simulation.

This module fills missing source packets using spatially neighboring packets.

Typical position in ARCE recovery chain:

    FEC recovered packets
    -> temporal cache
    -> spatial interpolation
    -> zero-fill

Input convention:
    packets:
        [K, C, packet_h, packet_w]

    packet_metas:
        list of FeaturePacketMeta from packetizer.py.
        Each meta describes where a packet lies in the BEV grid.

    missing_mask[i] == True:
        source packet i is still missing.

    available_mask[i] == True:
        source packet i is available.

Output:
    recovered packets with some missing packets filled by spatial interpolation.

Important:
    Spatial interpolation is a heuristic recovery method.
    It does not restore exact transmitted information.
    It only uses nearby received / recovered BEV feature patches to fill holes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from opencood.communication.transport.recovery import (
    RECOVERY_METHOD_SPATIAL_INTERPOLATION,
    build_recovery_count_dict,
    available_mask_to_missing_mask,
    missing_mask_to_available_mask,
)


def _require_tensor(x: Any, name: str = "tensor") -> torch.Tensor:
    """
    Validate torch.Tensor input.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} should be a torch.Tensor, got {type(x)}.")
    return x


def _as_bool_tensor(
    mask: Any,
    expected_len: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    name: str = "mask",
) -> torch.Tensor:
    """
    Convert mask to flattened torch.BoolTensor.
    """
    if mask is None:
        raise ValueError(f"{name} is None.")

    if torch.is_tensor(mask):
        out = mask.to(dtype=torch.bool)
        if device is not None:
            out = out.to(device=device)
    else:
        out = torch.as_tensor(mask, dtype=torch.bool, device=device)

    out = out.flatten()

    if expected_len is not None and out.numel() != int(expected_len):
        raise ValueError(
            f"{name} length mismatch: expected {expected_len}, got {out.numel()}."
        )

    return out


def _resolve_missing_mask(
    num_packets: int,
    missing_mask: Optional[Any] = None,
    available_mask: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Resolve missing mask from missing_mask or available_mask.

    Priority:
        1. missing_mask
        2. available_mask
        3. all packets available

    Returns
    -------
    torch.BoolTensor
        Shape [K]. True means packet is missing.
    """
    num_packets = int(num_packets)

    if num_packets < 0:
        raise ValueError(f"num_packets should be non-negative, got {num_packets}.")

    if missing_mask is not None:
        return _as_bool_tensor(
            missing_mask,
            expected_len=num_packets,
            device=device,
            name="missing_mask",
        )

    if available_mask is not None:
        available_mask = _as_bool_tensor(
            available_mask,
            expected_len=num_packets,
            device=device,
            name="available_mask",
        )
        return available_mask_to_missing_mask(available_mask)

    return torch.zeros(num_packets, dtype=torch.bool, device=device)


def _count_true(mask: torch.Tensor) -> int:
    """
    Count True values.
    """
    mask = _as_bool_tensor(mask, name="mask")
    return int(mask.sum().item())


def _safe_cast_patch(patch: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Cast interpolated patch back to target dtype.

    For integer tensors, interpolation output is rounded before casting.
    Spatial interpolation is usually used on dequantized float packets, but
    this keeps the function robust if integer tensors are passed.
    """
    if dtype in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return torch.round(patch).to(dtype)

    if dtype == torch.bool:
        return (patch > 0.5).to(dtype)

    return patch.to(dtype)


def _resize_patch(
    patch: torch.Tensor,
    target_hw: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resize one packet patch [C, h, w] to target size [C, target_h, target_w].

    Integer tensors are converted to float for interpolation.
    """
    patch = _require_tensor(patch, "patch")

    if patch.dim() != 3:
        raise ValueError(
            f"patch should have shape [C, h, w], got {tuple(patch.shape)}."
        )

    target_h, target_w = int(target_hw[0]), int(target_hw[1])

    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_hw should be positive, got {target_hw}.")

    if int(patch.shape[-2]) == target_h and int(patch.shape[-1]) == target_w:
        return patch

    dtype = patch.dtype
    x = patch.float().unsqueeze(0)

    mode = str(mode).strip().lower()

    if mode in ("nearest", "area"):
        resized = F.interpolate(
            x,
            size=(target_h, target_w),
            mode=mode,
        )
    elif mode in ("bilinear", "bicubic"):
        resized = F.interpolate(
            x,
            size=(target_h, target_w),
            mode=mode,
            align_corners=False,
        )
    else:
        raise ValueError(
            f"Unsupported resize mode: {mode}. "
            "Expected nearest / bilinear / bicubic / area."
        )

    return _safe_cast_patch(resized.squeeze(0), dtype)


def _get_meta_valid_hw(meta: Any) -> Tuple[int, int]:
    """
    Get valid height and width from a FeaturePacketMeta-like object.
    """
    if not hasattr(meta, "valid_h") or not hasattr(meta, "valid_w"):
        raise AttributeError("packet meta should have valid_h and valid_w.")

    return int(meta.valid_h), int(meta.valid_w)


def _get_meta_packet_id(meta: Any) -> int:
    """
    Get packet id from a FeaturePacketMeta-like object.
    """
    if not hasattr(meta, "packet_id"):
        raise AttributeError("packet meta should have packet_id.")

    return int(meta.packet_id)


def _get_meta_row_col(meta: Any) -> Tuple[int, int]:
    """
    Get row / col index from a FeaturePacketMeta-like object.
    """
    if not hasattr(meta, "row_idx") or not hasattr(meta, "col_idx"):
        raise AttributeError("packet meta should have row_idx and col_idx.")

    return int(meta.row_idx), int(meta.col_idx)


def _get_meta_spatial_slice(meta: Any) -> Tuple[slice, slice]:
    """
    Get spatial slice from FeaturePacketMeta-like object.
    """
    if hasattr(meta, "spatial_slice"):
        return meta.spatial_slice

    required = ("h_start", "h_end", "w_start", "w_end")
    for attr in required:
        if not hasattr(meta, attr):
            raise AttributeError(
                f"packet meta should have spatial_slice or attributes {required}."
            )

    return slice(int(meta.h_start), int(meta.h_end)), slice(int(meta.w_start), int(meta.w_end))


def _build_meta_maps(packet_metas: Sequence[Any]) -> Tuple[Dict[int, Any], Dict[Tuple[int, int], Any]]:
    """
    Build packet_id -> meta and (row, col) -> meta maps.
    """
    packet_id_to_meta: Dict[int, Any] = {}
    row_col_to_meta: Dict[Tuple[int, int], Any] = {}

    for meta in packet_metas:
        packet_id = _get_meta_packet_id(meta)
        row_col = _get_meta_row_col(meta)

        if packet_id in packet_id_to_meta:
            raise ValueError(f"Duplicate packet_id in packet_metas: {packet_id}.")

        if row_col in row_col_to_meta:
            raise ValueError(f"Duplicate row/col in packet_metas: {row_col}.")

        packet_id_to_meta[packet_id] = meta
        row_col_to_meta[row_col] = meta

    return packet_id_to_meta, row_col_to_meta


def _neighbor_offsets(neighbor_mode: str = "4") -> List[Tuple[int, int]]:
    """
    Return neighbor offsets.

    neighbor_mode:
        "4":
            up / down / left / right

        "8":
            8-neighborhood

        "diag":
            diagonal only
    """
    neighbor_mode = str(neighbor_mode).strip().lower()

    offsets_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    offsets_diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    if neighbor_mode in ("4", "cross"):
        return offsets_4

    if neighbor_mode in ("8", "all"):
        return offsets_4 + offsets_diag

    if neighbor_mode in ("diag", "diagonal"):
        return offsets_diag

    raise ValueError(
        f"Unsupported neighbor_mode: {neighbor_mode}. "
        "Expected 4 / 8 / diag."
    )


def _find_neighbor_packet_ids(
    meta: Any,
    row_col_to_meta: Dict[Tuple[int, int], Any],
    neighbor_mode: str = "4",
) -> List[int]:
    """
    Find neighbor packet ids for a target meta.
    """
    row, col = _get_meta_row_col(meta)
    ids = []

    for dr, dc in _neighbor_offsets(neighbor_mode):
        neighbor_meta = row_col_to_meta.get((row + dr, col + dc), None)

        if neighbor_meta is None:
            continue

        ids.append(_get_meta_packet_id(neighbor_meta))

    return ids


def _extract_valid_packet_area(packet: torch.Tensor, meta: Any) -> torch.Tensor:
    """
    Extract valid area from a padded packet tensor.

    Input packet:
        [C, packet_h, packet_w]

    Output:
        [C, valid_h, valid_w]
    """
    packet = _require_tensor(packet, "packet")

    if packet.dim() != 3:
        raise ValueError(
            f"packet should have shape [C, h, w], got {tuple(packet.shape)}."
        )

    valid_h, valid_w = _get_meta_valid_hw(meta)

    return packet[:, :valid_h, :valid_w]


def _average_neighbor_patches(
    packets: torch.Tensor,
    packet_metas: Sequence[Any],
    neighbor_ids: Sequence[int],
    target_hw: Tuple[int, int],
    resize_mode: str = "bilinear",
) -> Optional[torch.Tensor]:
    """
    Average neighbor packet valid areas after resizing them to target size.

    Returns None if no valid neighbor is provided.
    """
    if len(neighbor_ids) == 0:
        return None

    packet_id_to_meta, _ = _build_meta_maps(packet_metas)

    resized_neighbors = []

    for neighbor_id in neighbor_ids:
        neighbor_id = int(neighbor_id)

        if neighbor_id not in packet_id_to_meta:
            raise KeyError(f"neighbor_id={neighbor_id} not found in packet_metas.")

        neighbor_meta = packet_id_to_meta[neighbor_id]
        neighbor_patch = _extract_valid_packet_area(
            packets[neighbor_id],
            neighbor_meta,
        )

        neighbor_patch = _resize_patch(
            neighbor_patch,
            target_hw=target_hw,
            mode=resize_mode,
        )

        resized_neighbors.append(neighbor_patch.float())

    if not resized_neighbors:
        return None

    stacked = torch.stack(resized_neighbors, dim=0)
    return stacked.mean(dim=0)


@dataclass
class SpatialInterpolationResult:
    """
    Result of spatial interpolation recovery.

    Attributes
    ----------
    recovered : torch.Tensor
        Recovered tensor. Usually packet tensor [K, C, h, w] or feature [C, H, W].

    filled_mask : torch.BoolTensor
        Shape [K]. True means this packet was filled by spatial interpolation.

    available_mask : torch.BoolTensor
        Shape [K]. True means this packet is available after interpolation.

    missing_mask_before : torch.BoolTensor
        Shape [K]. True means this packet was missing before interpolation.

    still_missing_mask : torch.BoolTensor
        Shape [K]. True means this packet remains missing after interpolation.

    info : dict
        JSON-friendly statistics.
    """

    recovered: torch.Tensor
    filled_mask: torch.Tensor
    available_mask: torch.Tensor
    missing_mask_before: torch.Tensor
    still_missing_mask: torch.Tensor
    info: Dict[str, Any]

    @property
    def num_packets(self) -> int:
        return int(self.filled_mask.numel())

    @property
    def num_spatial_filled_packets(self) -> int:
        return _count_true(self.filled_mask)

    @property
    def recovery_ratio(self) -> float:
        if self.num_packets <= 0:
            return 1.0
        return float(_count_true(self.available_mask) / self.num_packets)

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-friendly summary.
        """
        result = copy.deepcopy(self.info)
        result.update(
            {
                "num_packets": int(self.num_packets),
                "num_spatial_filled_packets": int(self.num_spatial_filled_packets),
                "recovery_ratio": float(self.recovery_ratio),
                "recovered_shape": tuple(int(x) for x in self.recovered.shape),
                "recovered_dtype": str(self.recovered.dtype),
            }
        )
        return result


def spatial_interpolate_packets(
    packets: torch.Tensor,
    packet_metas: Sequence[Any],
    missing_mask: Optional[Any] = None,
    available_mask: Optional[Any] = None,
    neighbor_mode: str = "4",
    fallback_neighbor_mode: str = "8",
    min_neighbors: int = 1,
    resize_mode: str = "bilinear",
    iterative: bool = False,
    max_iters: int = 2,
    clone: bool = True,
) -> SpatialInterpolationResult:
    """
    Fill missing packet tensors by averaging spatial neighbors.

    Parameters
    ----------
    packets : torch.Tensor
        Packet tensor [K, C, packet_h, packet_w].

    packet_metas : sequence
        FeaturePacketMeta-like metadata list.

    missing_mask : tensor-like, optional
        Shape [K]. True means packet is missing.

    available_mask : tensor-like, optional
        Shape [K]. True means packet is available.

    neighbor_mode : str
        Primary neighborhood. Usually "4".

    fallback_neighbor_mode : str
        Fallback neighborhood when primary neighbors are insufficient.
        Usually "8".

    min_neighbors : int
        Minimum available neighbors required to fill a missing packet.

    resize_mode : str
        nearest / bilinear / bicubic / area.

    iterative : bool
        If True, packets filled in earlier iterations become available for
        later missing packets.

    max_iters : int
        Maximum interpolation passes when iterative=True.

    clone : bool
        If True, operate on a cloned tensor.

    Returns
    -------
    SpatialInterpolationResult
    """
    packets = _require_tensor(packets, "packets")

    if packets.dim() < 4:
        raise ValueError(
            "spatial_interpolate_packets expects packet tensor shape "
            f"[K, C, h, w], got {tuple(packets.shape)}."
        )

    packet_metas = list(packet_metas)
    num_packets = int(packets.shape[0])

    if len(packet_metas) != num_packets:
        raise ValueError(
            f"packet_metas length mismatch: expected {num_packets}, "
            f"got {len(packet_metas)}."
        )

    min_neighbors = int(min_neighbors)
    if min_neighbors <= 0:
        raise ValueError(f"min_neighbors should be positive, got {min_neighbors}.")

    max_iters = int(max_iters)
    if max_iters <= 0:
        raise ValueError(f"max_iters should be positive, got {max_iters}.")

    missing_before = _resolve_missing_mask(
        num_packets=num_packets,
        missing_mask=missing_mask,
        available_mask=available_mask,
        device=packets.device,
    )

    recovered = packets.clone() if clone else packets

    filled_mask = torch.zeros_like(missing_before)
    available = missing_mask_to_available_mask(missing_before)
    still_missing = missing_before.clone()

    packet_id_to_meta, row_col_to_meta = _build_meta_maps(packet_metas)

    packet_infos: List[Dict[str, Any]] = []

    num_passes = max_iters if iterative else 1

    for iter_idx in range(num_passes):
        changed = False

        missing_ids = [
            int(i)
            for i in torch.nonzero(still_missing, as_tuple=False).flatten().tolist()
        ]

        if not missing_ids:
            break

        for packet_id in missing_ids:
            target_meta = packet_id_to_meta[packet_id]
            target_hw = _get_meta_valid_hw(target_meta)

            primary_neighbors = _find_neighbor_packet_ids(
                target_meta,
                row_col_to_meta=row_col_to_meta,
                neighbor_mode=neighbor_mode,
            )
            primary_neighbors = [
                nid for nid in primary_neighbors
                if bool(available[nid].item())
            ]

            used_neighbors = primary_neighbors
            used_mode = neighbor_mode

            if len(used_neighbors) < min_neighbors and fallback_neighbor_mode:
                fallback_neighbors = _find_neighbor_packet_ids(
                    target_meta,
                    row_col_to_meta=row_col_to_meta,
                    neighbor_mode=fallback_neighbor_mode,
                )
                fallback_neighbors = [
                    nid for nid in fallback_neighbors
                    if bool(available[nid].item())
                ]

                # Merge while preserving order.
                merged = list(used_neighbors)
                for nid in fallback_neighbors:
                    if nid not in merged:
                        merged.append(nid)

                used_neighbors = merged
                used_mode = fallback_neighbor_mode

            packet_info = {
                "packet_id": int(packet_id),
                "iteration": int(iter_idx),
                "target_hw": tuple(int(x) for x in target_hw),
                "neighbor_mode": used_mode,
                "neighbor_ids": [int(x) for x in used_neighbors],
                "num_neighbors": int(len(used_neighbors)),
                "filled": False,
                "reason": "",
            }

            if len(used_neighbors) < min_neighbors:
                packet_info["reason"] = "not enough available spatial neighbors"
                packet_infos.append(packet_info)
                continue

            interpolated_valid = _average_neighbor_patches(
                packets=recovered,
                packet_metas=packet_metas,
                neighbor_ids=used_neighbors,
                target_hw=target_hw,
                resize_mode=resize_mode,
            )

            if interpolated_valid is None:
                packet_info["reason"] = "neighbor interpolation returned None"
                packet_infos.append(packet_info)
                continue

            valid_h, valid_w = target_hw
            interpolated_valid = _safe_cast_patch(
                interpolated_valid,
                recovered.dtype,
            )

            recovered[packet_id, :, :valid_h, :valid_w] = interpolated_valid

            filled_mask[packet_id] = True
            available[packet_id] = True
            still_missing[packet_id] = False
            changed = True

            packet_info["filled"] = True
            packet_info["reason"] = "filled by spatial neighbor average"
            packet_infos.append(packet_info)

        if not iterative or not changed:
            break

    num_spatial_filled = int(filled_mask.sum().item())
    num_still_missing = int(still_missing.sum().item())

    counts = build_recovery_count_dict(
        num_source_packets=num_packets,
        num_fec_recovered=0,
        num_temporal_filled=0,
        num_spatial_filled=num_spatial_filled,
        num_zero_filled=0,
        num_still_missing=num_still_missing,
    )

    info = {
        "method": RECOVERY_METHOD_SPATIAL_INTERPOLATION,
        "target": "packets",
        "input_shape": tuple(int(x) for x in packets.shape),
        "input_dtype": str(packets.dtype),
        "num_packets": int(num_packets),
        "num_missing_before": int(missing_before.sum().item()),
        "num_spatial_filled_packets": int(num_spatial_filled),
        "num_still_missing_packets": int(num_still_missing),
        "recovery_ratio": float(counts["recovery_ratio"]),
        "neighbor_mode": neighbor_mode,
        "fallback_neighbor_mode": fallback_neighbor_mode,
        "min_neighbors": int(min_neighbors),
        "resize_mode": resize_mode,
        "iterative": bool(iterative),
        "max_iters": int(max_iters),
        "packet_infos": packet_infos,
        "counts": counts,
        "note": (
            "Missing packets are filled by averaging available spatial "
            "neighbor packets. This is approximate semantic recovery, not "
            "exact packet reconstruction."
        ),
    }

    return SpatialInterpolationResult(
        recovered=recovered,
        filled_mask=filled_mask,
        available_mask=available,
        missing_mask_before=missing_before,
        still_missing_mask=still_missing,
        info=info,
    )


def spatial_interpolate_feature_by_metas(
    feature: torch.Tensor,
    packet_metas: Sequence[Any],
    missing_mask: Optional[Any] = None,
    available_mask: Optional[Any] = None,
    neighbor_mode: str = "4",
    fallback_neighbor_mode: str = "8",
    min_neighbors: int = 1,
    resize_mode: str = "bilinear",
    iterative: bool = False,
    max_iters: int = 2,
    clone: bool = True,
) -> SpatialInterpolationResult:
    """
    Fill missing spatial regions in a full feature map using neighboring regions.

    Parameters
    ----------
    feature : torch.Tensor
        Feature tensor [C, H, W].

    packet_metas : sequence
        FeaturePacketMeta-like metadata list.

    missing_mask / available_mask:
        Packet-level masks with shape [K].

    Returns
    -------
    SpatialInterpolationResult
        Recovered full feature map and packet-level statistics.
    """
    feature = _require_tensor(feature, "feature")

    if feature.dim() != 3:
        raise ValueError(
            "spatial_interpolate_feature_by_metas expects feature shape [C,H,W], "
            f"got {tuple(feature.shape)}."
        )

    packet_metas = list(packet_metas)
    num_packets = len(packet_metas)

    missing_before = _resolve_missing_mask(
        num_packets=num_packets,
        missing_mask=missing_mask,
        available_mask=available_mask,
        device=feature.device,
    )

    recovered = feature.clone() if clone else feature

    filled_mask = torch.zeros_like(missing_before)
    available = missing_mask_to_available_mask(missing_before)
    still_missing = missing_before.clone()

    packet_id_to_meta, row_col_to_meta = _build_meta_maps(packet_metas)

    packet_infos: List[Dict[str, Any]] = []

    num_passes = max_iters if iterative else 1

    for iter_idx in range(num_passes):
        changed = False

        missing_ids = [
            int(i)
            for i in torch.nonzero(still_missing, as_tuple=False).flatten().tolist()
        ]

        if not missing_ids:
            break

        for packet_id in missing_ids:
            target_meta = packet_id_to_meta[packet_id]
            target_hw = _get_meta_valid_hw(target_meta)
            target_h_slice, target_w_slice = _get_meta_spatial_slice(target_meta)

            primary_neighbors = _find_neighbor_packet_ids(
                target_meta,
                row_col_to_meta=row_col_to_meta,
                neighbor_mode=neighbor_mode,
            )
            primary_neighbors = [
                nid for nid in primary_neighbors
                if bool(available[nid].item())
            ]

            used_neighbors = primary_neighbors
            used_mode = neighbor_mode

            if len(used_neighbors) < min_neighbors and fallback_neighbor_mode:
                fallback_neighbors = _find_neighbor_packet_ids(
                    target_meta,
                    row_col_to_meta=row_col_to_meta,
                    neighbor_mode=fallback_neighbor_mode,
                )
                fallback_neighbors = [
                    nid for nid in fallback_neighbors
                    if bool(available[nid].item())
                ]

                merged = list(used_neighbors)
                for nid in fallback_neighbors:
                    if nid not in merged:
                        merged.append(nid)

                used_neighbors = merged
                used_mode = fallback_neighbor_mode

            packet_info = {
                "packet_id": int(packet_id),
                "iteration": int(iter_idx),
                "target_hw": tuple(int(x) for x in target_hw),
                "neighbor_mode": used_mode,
                "neighbor_ids": [int(x) for x in used_neighbors],
                "num_neighbors": int(len(used_neighbors)),
                "filled": False,
                "reason": "",
            }

            if len(used_neighbors) < min_neighbors:
                packet_info["reason"] = "not enough available spatial neighbors"
                packet_infos.append(packet_info)
                continue

            neighbor_patches = []

            for nid in used_neighbors:
                neighbor_meta = packet_id_to_meta[int(nid)]
                nh_slice, nw_slice = _get_meta_spatial_slice(neighbor_meta)

                patch = recovered[:, nh_slice, nw_slice]
                patch = _resize_patch(
                    patch,
                    target_hw=target_hw,
                    mode=resize_mode,
                )
                neighbor_patches.append(patch.float())

            if not neighbor_patches:
                packet_info["reason"] = "no valid neighbor patches"
                packet_infos.append(packet_info)
                continue

            interpolated = torch.stack(neighbor_patches, dim=0).mean(dim=0)
            interpolated = _safe_cast_patch(interpolated, recovered.dtype)

            recovered[:, target_h_slice, target_w_slice] = interpolated

            filled_mask[packet_id] = True
            available[packet_id] = True
            still_missing[packet_id] = False
            changed = True

            packet_info["filled"] = True
            packet_info["reason"] = "filled by spatial neighbor average"
            packet_infos.append(packet_info)

        if not iterative or not changed:
            break

    num_spatial_filled = int(filled_mask.sum().item())
    num_still_missing = int(still_missing.sum().item())

    counts = build_recovery_count_dict(
        num_source_packets=num_packets,
        num_fec_recovered=0,
        num_temporal_filled=0,
        num_spatial_filled=num_spatial_filled,
        num_zero_filled=0,
        num_still_missing=num_still_missing,
    )

    info = {
        "method": RECOVERY_METHOD_SPATIAL_INTERPOLATION,
        "target": "feature_by_metas",
        "input_shape": tuple(int(x) for x in feature.shape),
        "input_dtype": str(feature.dtype),
        "num_packets": int(num_packets),
        "num_missing_before": int(missing_before.sum().item()),
        "num_spatial_filled_packets": int(num_spatial_filled),
        "num_still_missing_packets": int(num_still_missing),
        "recovery_ratio": float(counts["recovery_ratio"]),
        "neighbor_mode": neighbor_mode,
        "fallback_neighbor_mode": fallback_neighbor_mode,
        "min_neighbors": int(min_neighbors),
        "resize_mode": resize_mode,
        "iterative": bool(iterative),
        "max_iters": int(max_iters),
        "packet_infos": packet_infos,
        "counts": counts,
        "note": (
            "Missing spatial regions in the full feature map are filled by "
            "averaging neighboring BEV feature regions."
        ),
    }

    return SpatialInterpolationResult(
        recovered=recovered,
        filled_mask=filled_mask,
        available_mask=available,
        missing_mask_before=missing_before,
        still_missing_mask=still_missing,
        info=info,
    )


def spatial_interpolate_from_fec_decode(
    decode_result: Any,
    packet_metas: Sequence[Any],
    **kwargs,
) -> SpatialInterpolationResult:
    """
    Spatially interpolate missing packets from a FECDecodeResult-like object.

    Expected fields:
        decode_result.recovered_packets
        decode_result.missing_source_mask
    """
    if not hasattr(decode_result, "recovered_packets"):
        raise AttributeError("decode_result should have recovered_packets.")

    if not hasattr(decode_result, "missing_source_mask"):
        raise AttributeError("decode_result should have missing_source_mask.")

    return spatial_interpolate_packets(
        packets=decode_result.recovered_packets,
        packet_metas=packet_metas,
        missing_mask=decode_result.missing_source_mask,
        **kwargs,
    )


class SpatialInterpolationRecovery:
    """
    Object-oriented wrapper for spatial interpolation recovery.

    YAML style:

        arce:
          recovery:
            spatial_interpolation: true
            spatial_neighbor_mode: 4
            spatial_fallback_neighbor_mode: 8
            spatial_min_neighbors: 1
            spatial_resize_mode: bilinear
            spatial_iterative: false
            spatial_max_iters: 2
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = self._extract_recovery_cfg(cfg or {})

        self.enabled = bool(
            cfg.get(
                "spatial_interpolation",
                cfg.get("spatial", True),
            )
        )

        self.neighbor_mode = str(
            cfg.get("spatial_neighbor_mode", cfg.get("neighbor_mode", "4"))
        )

        self.fallback_neighbor_mode = str(
            cfg.get(
                "spatial_fallback_neighbor_mode",
                cfg.get("fallback_neighbor_mode", "8"),
            )
        )

        self.min_neighbors = int(
            cfg.get("spatial_min_neighbors", cfg.get("min_neighbors", 1))
        )

        self.resize_mode = str(
            cfg.get("spatial_resize_mode", cfg.get("resize_mode", "bilinear"))
        )

        self.iterative = bool(
            cfg.get("spatial_iterative", cfg.get("iterative", False))
        )

        self.max_iters = int(
            cfg.get("spatial_max_iters", cfg.get("max_iters", 2))
        )

    @staticmethod
    def _extract_recovery_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept full ARCE config or direct recovery config.
        """
        if "recovery" in cfg and isinstance(cfg["recovery"], dict):
            return cfg["recovery"]

        return cfg

    def recover_packets(
        self,
        packets: torch.Tensor,
        packet_metas: Sequence[Any],
        missing_mask: Optional[Any] = None,
        available_mask: Optional[Any] = None,
        clone: bool = True,
    ) -> SpatialInterpolationResult:
        """
        Recover packet tensor by spatial interpolation.

        If disabled, return input unchanged and keep missing packets missing.
        """
        packets = _require_tensor(packets, "packets")
        packet_metas = list(packet_metas)
        num_packets = int(packets.shape[0])

        missing = _resolve_missing_mask(
            num_packets=num_packets,
            missing_mask=missing_mask,
            available_mask=available_mask,
            device=packets.device,
        )

        if not self.enabled:
            available = missing_mask_to_available_mask(missing)

            info = {
                "method": RECOVERY_METHOD_SPATIAL_INTERPOLATION,
                "enabled": False,
                "target": "packets",
                "num_packets": int(num_packets),
                "num_missing_before": int(missing.sum().item()),
                "num_spatial_filled_packets": 0,
                "num_still_missing_packets": int(missing.sum().item()),
                "recovery_ratio": (
                    float(available.sum().item() / num_packets)
                    if num_packets > 0
                    else 1.0
                ),
                "note": "Spatial interpolation recovery is disabled.",
            }

            return SpatialInterpolationResult(
                recovered=packets.clone() if clone else packets,
                filled_mask=torch.zeros_like(missing),
                available_mask=available,
                missing_mask_before=missing,
                still_missing_mask=missing,
                info=info,
            )

        return spatial_interpolate_packets(
            packets=packets,
            packet_metas=packet_metas,
            missing_mask=missing,
            neighbor_mode=self.neighbor_mode,
            fallback_neighbor_mode=self.fallback_neighbor_mode,
            min_neighbors=self.min_neighbors,
            resize_mode=self.resize_mode,
            iterative=self.iterative,
            max_iters=self.max_iters,
            clone=clone,
        )

    def recover_feature_by_metas(
        self,
        feature: torch.Tensor,
        packet_metas: Sequence[Any],
        missing_mask: Optional[Any] = None,
        available_mask: Optional[Any] = None,
        clone: bool = True,
    ) -> SpatialInterpolationResult:
        """
        Recover full feature map by spatial interpolation.
        """
        feature = _require_tensor(feature, "feature")
        packet_metas = list(packet_metas)
        num_packets = len(packet_metas)

        missing = _resolve_missing_mask(
            num_packets=num_packets,
            missing_mask=missing_mask,
            available_mask=available_mask,
            device=feature.device,
        )

        if not self.enabled:
            available = missing_mask_to_available_mask(missing)

            info = {
                "method": RECOVERY_METHOD_SPATIAL_INTERPOLATION,
                "enabled": False,
                "target": "feature_by_metas",
                "num_packets": int(num_packets),
                "num_missing_before": int(missing.sum().item()),
                "num_spatial_filled_packets": 0,
                "num_still_missing_packets": int(missing.sum().item()),
                "recovery_ratio": (
                    float(available.sum().item() / num_packets)
                    if num_packets > 0
                    else 1.0
                ),
                "note": "Spatial interpolation recovery is disabled.",
            }

            return SpatialInterpolationResult(
                recovered=feature.clone() if clone else feature,
                filled_mask=torch.zeros_like(missing),
                available_mask=available,
                missing_mask_before=missing,
                still_missing_mask=missing,
                info=info,
            )

        return spatial_interpolate_feature_by_metas(
            feature=feature,
            packet_metas=packet_metas,
            missing_mask=missing,
            neighbor_mode=self.neighbor_mode,
            fallback_neighbor_mode=self.fallback_neighbor_mode,
            min_neighbors=self.min_neighbors,
            resize_mode=self.resize_mode,
            iterative=self.iterative,
            max_iters=self.max_iters,
            clone=clone,
        )

    def recover_from_fec_decode(
        self,
        decode_result: Any,
        packet_metas: Sequence[Any],
        clone: bool = True,
    ) -> SpatialInterpolationResult:
        """
        Recover packets from a FECDecodeResult-like object.
        """
        if not hasattr(decode_result, "recovered_packets"):
            raise AttributeError("decode_result should have recovered_packets.")

        if not hasattr(decode_result, "missing_source_mask"):
            raise AttributeError("decode_result should have missing_source_mask.")

        return self.recover_packets(
            packets=decode_result.recovered_packets,
            packet_metas=packet_metas,
            missing_mask=decode_result.missing_source_mask,
            clone=clone,
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Return JSON-friendly config.
        """
        return {
            "enabled": bool(self.enabled),
            "method": RECOVERY_METHOD_SPATIAL_INTERPOLATION,
            "neighbor_mode": self.neighbor_mode,
            "fallback_neighbor_mode": self.fallback_neighbor_mode,
            "min_neighbors": int(self.min_neighbors),
            "resize_mode": self.resize_mode,
            "iterative": bool(self.iterative),
            "max_iters": int(self.max_iters),
        }

    def __repr__(self) -> str:
        return (
            "SpatialInterpolationRecovery("
            f"enabled={self.enabled}, "
            f"neighbor_mode={self.neighbor_mode}, "
            f"fallback_neighbor_mode={self.fallback_neighbor_mode}, "
            f"min_neighbors={self.min_neighbors}, "
            f"resize_mode={self.resize_mode}, "
            f"iterative={self.iterative})"
        )


SpatialInterpolation = SpatialInterpolationRecovery
SpatialInterpolator = SpatialInterpolationRecovery


__all__ = [
    "SpatialInterpolationResult",
    "spatial_interpolate_packets",
    "spatial_interpolate_feature_by_metas",
    "spatial_interpolate_from_fec_decode",
    "SpatialInterpolationRecovery",
    "SpatialInterpolation",
    "SpatialInterpolator",
]