"""
Feature packetizer for ARCE communication simulation.

This module splits an intermediate BEV feature tensor [C, H, W] into
spatial packets. It supports two packetization modes:

1. grid mode
   Split the feature map into a fixed grid, e.g. 10 x 10 = 100 packets.

2. block mode
   Split the feature map by fixed spatial block size, e.g. 8 x 8 cells.

Mask convention used across ARCE:
    loss_mask[i] == True  means packet i is lost.
    loss_mask[i] == False means packet i is received.

This module does NOT:
    - sample packet loss;
    - perform quantization;
    - perform FEC;
    - perform temporal cache recovery.

Those are handled by other ARCE modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from opencood.communication.transport.packetization import (
    PACKET_MODE_GRID,
    PACKET_MODE_BLOCK,
    PACKET_MODE_PATCH_MAX_SIZE,
    DEFAULT_PACKET_MODE,
    DEFAULT_GRID_SIZE,
    normalize_packet_mode,
    normalize_grid_size,
    normalize_block_size,
    normalize_patch_max_size,
)


TensorLike = Union[torch.Tensor, Sequence[bool]]


def _extract_packetizer_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Accept either full ARCE config or direct packetizer config.

    Supported input:
        cfg = arce_cfg
        cfg = arce_cfg["packetizer"]
    """
    cfg = cfg or {}

    if "packetizer" in cfg and isinstance(cfg["packetizer"], dict):
        return cfg["packetizer"]

    return cfg


def _validate_feature_3d(feature: torch.Tensor) -> Tuple[int, int, int]:
    """
    Validate feature tensor and return C, H, W.
    """
    if not torch.is_tensor(feature):
        raise TypeError(
            f"feature should be a torch.Tensor, got {type(feature)}."
        )

    if feature.dim() != 3:
        raise ValueError(
            "FeaturePacketizer expects a single feature tensor with shape "
            f"[C, H, W], got {tuple(feature.shape)}."
        )

    c, h, w = feature.shape

    if c <= 0 or h <= 0 or w <= 0:
        raise ValueError(
            f"feature shape should be positive, got {tuple(feature.shape)}."
        )

    return int(c), int(h), int(w)


def _normalize_feature_shape(
    feature_or_shape: Union[torch.Tensor, Sequence[int]]
) -> Tuple[int, int, int]:
    """
    Normalize tensor or shape into [C, H, W].

    Supports:
        torch.Tensor with shape [C, H, W]
        shape tuple/list [C, H, W]
        shape tuple/list [N, C, H, W], in which case last three dims are used
    """
    if torch.is_tensor(feature_or_shape):
        return _validate_feature_3d(feature_or_shape)

    if not isinstance(feature_or_shape, (list, tuple)):
        raise TypeError(
            "feature_or_shape should be a torch.Tensor, list, or tuple, "
            f"got {type(feature_or_shape)}."
        )

    if len(feature_or_shape) == 3:
        c, h, w = feature_or_shape
    elif len(feature_or_shape) == 4:
        _, c, h, w = feature_or_shape
    else:
        raise ValueError(
            "feature shape should be [C, H, W] or [N, C, H, W], "
            f"got {feature_or_shape}."
        )

    c, h, w = int(c), int(h), int(w)

    if c <= 0 or h <= 0 or w <= 0:
        raise ValueError(
            f"feature shape values should be positive, got {(c, h, w)}."
        )

    return c, h, w


def _to_bool_mask(
    mask: TensorLike,
    expected_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert tensor-like mask to torch.BoolTensor.
    """
    if torch.is_tensor(mask):
        bool_mask = mask.to(dtype=torch.bool)
        if device is not None:
            bool_mask = bool_mask.to(device=device)
    else:
        bool_mask = torch.as_tensor(mask, dtype=torch.bool, device=device)

    bool_mask = bool_mask.flatten()

    if bool_mask.numel() != expected_len:
        raise ValueError(
            f"Mask length mismatch: expected {expected_len}, "
            f"got {bool_mask.numel()}."
        )

    return bool_mask


def _make_grid_boundaries(length: int, num_chunks: int) -> List[Tuple[int, int]]:
    """
    Split a spatial dimension into num_chunks non-empty intervals.

    Example:
        length = 10, num_chunks = 3
        -> [(0, 3), (3, 6), (6, 10)]
    """
    length = int(length)
    num_chunks = int(num_chunks)

    if length <= 0:
        raise ValueError(f"length should be positive, got {length}.")

    if num_chunks <= 0:
        raise ValueError(f"num_chunks should be positive, got {num_chunks}.")

    if num_chunks > length:
        raise ValueError(
            f"num_chunks={num_chunks} cannot exceed length={length}, "
            "otherwise empty packets will be created."
        )

    intervals = []

    for idx in range(num_chunks):
        start = int(math.floor(idx * length / num_chunks))
        end = int(math.floor((idx + 1) * length / num_chunks))

        if end <= start:
            raise RuntimeError(
                "Internal error: empty interval generated. "
                f"length={length}, num_chunks={num_chunks}, idx={idx}, "
                f"start={start}, end={end}."
            )

        intervals.append((start, end))

    return intervals


@dataclass(frozen=True)
class FeaturePacketMeta:
    """
    Metadata of one spatial feature packet.

    Attributes
    ----------
    packet_id : int
        Packet index in raster-scan order.

    row_idx : int
        Packet row index in packet grid.

    col_idx : int
        Packet column index in packet grid.

    h_start, h_end, w_start, w_end : int
        Spatial slice of this packet in the original feature map.

    valid_h, valid_w : int
        Actual spatial size of this packet before padding.

    padded_h, padded_w : int
        Common padded packet size used by packetize().
    """

    packet_id: int
    row_idx: int
    col_idx: int
    h_start: int
    h_end: int
    w_start: int
    w_end: int
    valid_h: int
    valid_w: int
    padded_h: int
    padded_w: int

    @property
    def spatial_slice(self) -> Tuple[slice, slice]:
        """
        Return spatial slice in original feature map.
        """
        return slice(self.h_start, self.h_end), slice(self.w_start, self.w_end)

    @property
    def valid_slice(self) -> Tuple[slice, slice]:
        """
        Return valid slice in padded packet tensor.
        """
        return slice(0, self.valid_h), slice(0, self.valid_w)

    @property
    def spatial_area(self) -> int:
        """
        Number of valid spatial cells in this packet.
        """
        return int(self.valid_h * self.valid_w)

    def num_elements(self, channels: int) -> int:
        """
        Number of feature elements in this packet before padding.
        """
        return int(channels * self.valid_h * self.valid_w)

    def as_dict(self) -> Dict[str, int]:
        """
        Export metadata as JSON-serializable dict.
        """
        return {
            "packet_id": int(self.packet_id),
            "row_idx": int(self.row_idx),
            "col_idx": int(self.col_idx),
            "h_start": int(self.h_start),
            "h_end": int(self.h_end),
            "w_start": int(self.w_start),
            "w_end": int(self.w_end),
            "valid_h": int(self.valid_h),
            "valid_w": int(self.valid_w),
            "padded_h": int(self.padded_h),
            "padded_w": int(self.padded_w),
            "spatial_area": int(self.spatial_area),
        }


@dataclass
class PacketizationResult:
    """
    Result of feature packetization.

    packets : torch.Tensor
        Shape [M, C, packet_h, packet_w].
        Each packet is padded to a common spatial size.

    valid_mask : torch.BoolTensor
        Shape [M, 1, packet_h, packet_w].
        True means the position is valid original content.
        False means padded area.

    metas : list of FeaturePacketMeta
        Metadata for each packet.

    original_shape : tuple
        Original feature shape [C, H, W].

    mode : str
        Packetization mode.

    packet_grid_shape : tuple
        Packet grid shape [num_rows, num_cols].
    """

    packets: torch.Tensor
    valid_mask: torch.Tensor
    metas: List[FeaturePacketMeta]
    original_shape: Tuple[int, int, int]
    mode: str
    packet_grid_shape: Tuple[int, int]

    @property
    def num_packets(self) -> int:
        return int(self.packets.shape[0])

    @property
    def packet_shape(self) -> Tuple[int, int, int]:
        return tuple(int(x) for x in self.packets.shape[1:])

    def to_meta_dict(self) -> Dict[str, Any]:
        """
        Export metadata summary as dict.
        """
        return {
            "num_packets": self.num_packets,
            "original_shape": tuple(int(x) for x in self.original_shape),
            "packet_shape": self.packet_shape,
            "mode": self.mode,
            "packet_grid_shape": tuple(int(x) for x in self.packet_grid_shape),
            "metas": [m.as_dict() for m in self.metas],
        }


class FeaturePacketizer:
    """
    Packetizer for a single BEV feature tensor [C, H, W].

    Recommended YAML config:

        arce:
          packetizer:
            mode: grid
            grid_size: [10, 10]

    Or:

        arce:
          packetizer:
            mode: block
            block_size: [8, 8]
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = _extract_packetizer_cfg(cfg)

        self.cfg = cfg
        self.mode = normalize_packet_mode(cfg.get("mode", DEFAULT_PACKET_MODE))

        if self.mode == PACKET_MODE_GRID:
            self.grid_size = normalize_grid_size(
                cfg.get("grid_size", DEFAULT_GRID_SIZE)
            )
            self.block_size = None
            self.patch_max_size = None

        elif self.mode == PACKET_MODE_BLOCK:
            self.block_size = normalize_block_size(cfg.get("block_size"))
            self.grid_size = None
            self.patch_max_size = None

        elif self.mode == PACKET_MODE_PATCH_MAX_SIZE:
            self.patch_max_size = normalize_patch_max_size(
                cfg.get("patch_max_size", cfg.get("max_patch_size", 64))
            )
            self.grid_size = None
            self.block_size = None

        else:
            raise ValueError(f"Unsupported packetizer mode: {self.mode}")

        self.pad_value = float(cfg.get("pad_value", 0.0))

    def _build_grid_slices(self, c: int, h: int, w: int) -> Tuple[
        List[FeaturePacketMeta],
        Tuple[int, int],
        Tuple[int, int],
    ]:
        """
        Build packet metadata for grid packetization.
        """
        grid_h, grid_w = self.grid_size

        h_intervals = _make_grid_boundaries(h, grid_h)
        w_intervals = _make_grid_boundaries(w, grid_w)

        max_packet_h = max(end - start for start, end in h_intervals)
        max_packet_w = max(end - start for start, end in w_intervals)

        metas: List[FeaturePacketMeta] = []
        packet_id = 0

        for row_idx, (h_start, h_end) in enumerate(h_intervals):
            for col_idx, (w_start, w_end) in enumerate(w_intervals):
                valid_h = h_end - h_start
                valid_w = w_end - w_start

                metas.append(
                    FeaturePacketMeta(
                        packet_id=packet_id,
                        row_idx=row_idx,
                        col_idx=col_idx,
                        h_start=h_start,
                        h_end=h_end,
                        w_start=w_start,
                        w_end=w_end,
                        valid_h=valid_h,
                        valid_w=valid_w,
                        padded_h=max_packet_h,
                        padded_w=max_packet_w,
                    )
                )
                packet_id += 1

        return metas, (grid_h, grid_w), (max_packet_h, max_packet_w)

    def _build_block_slices(self, c: int, h: int, w: int) -> Tuple[
        List[FeaturePacketMeta],
        Tuple[int, int],
        Tuple[int, int],
    ]:
        """
        Build packet metadata for block packetization.
        """
        block_h, block_w = self.block_size

        if block_h <= 0 or block_w <= 0:
            raise ValueError(f"Invalid block_size: {self.block_size}")

        num_rows = int(math.ceil(h / block_h))
        num_cols = int(math.ceil(w / block_w))

        metas: List[FeaturePacketMeta] = []
        packet_id = 0

        for row_idx in range(num_rows):
            h_start = row_idx * block_h
            h_end = min((row_idx + 1) * block_h, h)

            for col_idx in range(num_cols):
                w_start = col_idx * block_w
                w_end = min((col_idx + 1) * block_w, w)

                valid_h = h_end - h_start
                valid_w = w_end - w_start

                metas.append(
                    FeaturePacketMeta(
                        packet_id=packet_id,
                        row_idx=row_idx,
                        col_idx=col_idx,
                        h_start=h_start,
                        h_end=h_end,
                        w_start=w_start,
                        w_end=w_end,
                        valid_h=valid_h,
                        valid_w=valid_w,
                        padded_h=block_h,
                        padded_w=block_w,
                    )
                )
                packet_id += 1

        return metas, (num_rows, num_cols), (block_h, block_w)

    def _build_patch_max_size_slices(
        self,
        c: int,
        h: int,
        w: int,
    ) -> Tuple[
        List[FeaturePacketMeta],
        Tuple[int, int],
        Tuple[int, int],
    ]:
        """
        Build packet metadata by maximum spatial patch size.

        Each patch satisfies:
            valid_h * valid_w <= self.patch_max_size

        Internally we convert patch_max_size to a rectangular block_h/block_w,
        then reuse block-style raster scan.
        """
        block_h, block_w = self._infer_block_size_from_patch_max_size(
            h=h,
            w=w,
            patch_max_size=self.patch_max_size,
        )

        num_rows = int(math.ceil(h / float(block_h)))
        num_cols = int(math.ceil(w / float(block_w)))

        metas: List[FeaturePacketMeta] = []
        packet_id = 0

        for row_idx in range(num_rows):
            h_start = row_idx * block_h
            h_end = min((row_idx + 1) * block_h, h)

            for col_idx in range(num_cols):
                w_start = col_idx * block_w
                w_end = min((col_idx + 1) * block_w, w)

                valid_h = h_end - h_start
                valid_w = w_end - w_start

                if valid_h * valid_w > self.patch_max_size:
                    raise RuntimeError(
                        "patch_max_size violation: "
                        f"valid_h={valid_h}, valid_w={valid_w}, "
                        f"area={valid_h * valid_w}, "
                        f"patch_max_size={self.patch_max_size}"
                    )

                metas.append(
                    FeaturePacketMeta(
                        packet_id=packet_id,
                        row_idx=row_idx,
                        col_idx=col_idx,
                        h_start=h_start,
                        h_end=h_end,
                        w_start=w_start,
                        w_end=w_end,
                        valid_h=valid_h,
                        valid_w=valid_w,
                        padded_h=block_h,
                        padded_w=block_w,
                    )
                )
                packet_id += 1

        return metas, (num_rows, num_cols), (block_h, block_w)

    def _infer_block_size_from_patch_max_size(
        self,
        h: int,
        w: int,
        patch_max_size: int,
    ) -> Tuple[int, int]:
        """
        Infer a rectangular block size from patch_max_size.

        patch_max_size means:
            max valid spatial cells per patch, i.e. valid_h * valid_w <= patch_max_size.

        The returned block_h/block_w follows the BEV aspect ratio roughly,
        so patches are not overly thin when H and W are very different.
        """
        h = int(h)
        w = int(w)
        patch_max_size = int(patch_max_size)

        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid feature spatial shape: H={h}, W={w}")
        if patch_max_size <= 0:
            raise ValueError(f"patch_max_size should be positive, got {patch_max_size}")

        # If the whole feature is smaller than the cap, one packet is enough.
        if h * w <= patch_max_size:
            return h, w

        # Aspect-aware rectangle:
        # block_h / block_w roughly follows h / w,
        # while block_h * block_w <= patch_max_size.
        block_h = int(math.floor(math.sqrt(patch_max_size * h / float(w))))
        block_h = max(1, min(block_h, h))

        block_w = int(patch_max_size // block_h)
        block_w = max(1, min(block_w, w))

        # Safety: ensure area <= patch_max_size.
        while block_h * block_w > patch_max_size:
            if block_w >= block_h and block_w > 1:
                block_w -= 1
            elif block_h > 1:
                block_h -= 1
            else:
                break

        return int(block_h), int(block_w)

    def build_packet_metas(
        self,
        feature_or_shape: Union[torch.Tensor, Sequence[int]],
    ) -> Tuple[List[FeaturePacketMeta], Tuple[int, int], Tuple[int, int]]:
        """
        Build packet metadata without extracting packet tensors.

        Returns
        -------
        metas : list
            Packet metadata.

        packet_grid_shape : tuple
            [num_rows, num_cols].

        padded_packet_hw : tuple
            [packet_h, packet_w].
        """
        c, h, w = _normalize_feature_shape(feature_or_shape)

        if self.mode == PACKET_MODE_GRID:
            return self._build_grid_slices(c, h, w)

        if self.mode == PACKET_MODE_BLOCK:
            return self._build_block_slices(c, h, w)

        if self.mode == PACKET_MODE_PATCH_MAX_SIZE:
            return self._build_patch_max_size_slices(c, h, w)

        raise ValueError(f"Unsupported packetizer mode: {self.mode}")

    def get_num_packets(
        self,
        feature_or_shape: Union[torch.Tensor, Sequence[int]],
    ) -> int:
        """
        Return the number of packets for a feature shape.
        """
        metas, _, _ = self.build_packet_metas(feature_or_shape)
        return int(len(metas))

    def packetize(
        self,
        feature: torch.Tensor,
        clone: bool = True,
        pad_value: Optional[float] = None,
    ) -> PacketizationResult:
        """
        Split one feature tensor [C, H, W] into padded packet tensors.

        Parameters
        ----------
        feature : torch.Tensor
            Input feature with shape [C, H, W].

        clone : bool
            If True, copy packet content into a new contiguous tensor.
            Recommended for FEC / quantized packet operations.

        pad_value : float, optional
            Value for padded area. If None, use self.pad_value.

        Returns
        -------
        PacketizationResult
            packets:
                [M, C, packet_h, packet_w]

            valid_mask:
                [M, 1, packet_h, packet_w]
        """
        c, h, w = _validate_feature_3d(feature)

        metas, packet_grid_shape, padded_hw = self.build_packet_metas(
            (c, h, w)
        )
        packet_h, packet_w = padded_hw

        if pad_value is None:
            pad_value = self.pad_value

        packets = feature.new_full(
            (len(metas), c, packet_h, packet_w),
            fill_value=float(pad_value),
        )

        valid_mask = torch.zeros(
            (len(metas), 1, packet_h, packet_w),
            dtype=torch.bool,
            device=feature.device,
        )

        for meta in metas:
            h_slice, w_slice = meta.spatial_slice
            vh_slice, vw_slice = meta.valid_slice

            patch = feature[:, h_slice, w_slice]
            if clone:
                patch = patch.clone()

            packets[
                meta.packet_id,
                :,
                vh_slice,
                vw_slice,
            ] = patch

            valid_mask[
                meta.packet_id,
                :,
                vh_slice,
                vw_slice,
            ] = True

        return PacketizationResult(
            packets=packets,
            valid_mask=valid_mask,
            metas=metas,
            original_shape=(c, h, w),
            mode=self.mode,
            packet_grid_shape=packet_grid_shape,
        )

    def unpacketize(
        self,
        packets: torch.Tensor,
        metas: List[FeaturePacketMeta],
        original_shape: Sequence[int],
        base_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reconstruct a [C, H, W] feature tensor from packet tensors.

        Parameters
        ----------
        packets : torch.Tensor
            Shape [M, C, packet_h, packet_w].

        metas : list of FeaturePacketMeta
            Packet metadata.

        original_shape : sequence
            Original [C, H, W] shape.

        base_feature : torch.Tensor, optional
            If provided, copy it first and overwrite packet regions.
            If None, create a zero tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed feature with shape [C, H, W].
        """
        if packets.dim() != 4:
            raise ValueError(
                f"packets should have shape [M, C, ph, pw], "
                f"got {tuple(packets.shape)}."
            )

        c, h, w = _normalize_feature_shape(original_shape)

        if packets.shape[0] != len(metas):
            raise ValueError(
                f"Number of packets and metas mismatch: "
                f"{packets.shape[0]} vs {len(metas)}."
            )

        if packets.shape[1] != c:
            raise ValueError(
                f"Channel mismatch: packets C={packets.shape[1]}, "
                f"original C={c}."
            )

        if base_feature is not None:
            output = base_feature.clone()
            if tuple(output.shape) != (c, h, w):
                raise ValueError(
                    f"base_feature shape should be {(c, h, w)}, "
                    f"got {tuple(output.shape)}."
                )
        else:
            output = packets.new_zeros((c, h, w))

        for meta in metas:
            h_slice, w_slice = meta.spatial_slice
            vh_slice, vw_slice = meta.valid_slice

            output[:, h_slice, w_slice] = packets[
                meta.packet_id,
                :,
                vh_slice,
                vw_slice,
            ]

        return output

    def build_spatial_packet_id_map(
        self,
        feature_or_shape: Union[torch.Tensor, Sequence[int]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Build a [H, W] int64 map indicating packet_id for each spatial cell.
        """
        c, h, w = _normalize_feature_shape(feature_or_shape)
        metas, _, _ = self.build_packet_metas((c, h, w))

        packet_id_map = torch.empty(
            (h, w),
            dtype=torch.long,
            device=device,
        )

        for meta in metas:
            h_slice, w_slice = meta.spatial_slice
            packet_id_map[h_slice, w_slice] = int(meta.packet_id)

        return packet_id_map

    def build_spatial_loss_mask(
        self,
        loss_mask: TensorLike,
        feature_or_shape: Union[torch.Tensor, Sequence[int]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convert packet-level loss mask to spatial loss mask.

        Parameters
        ----------
        loss_mask : tensor-like
            Shape [M]. True means packet is lost.

        feature_or_shape : tensor or shape
            Original feature or shape.

        Returns
        -------
        torch.BoolTensor
            Shape [H, W]. True means this spatial cell belongs to a lost packet.
        """
        c, h, w = _normalize_feature_shape(feature_or_shape)
        metas, _, _ = self.build_packet_metas((c, h, w))

        loss_mask = _to_bool_mask(
            loss_mask,
            expected_len=len(metas),
            device=device,
        )

        spatial_mask = torch.zeros(
            (h, w),
            dtype=torch.bool,
            device=device,
        )

        for meta in metas:
            if bool(loss_mask[meta.packet_id].item()):
                h_slice, w_slice = meta.spatial_slice
                spatial_mask[h_slice, w_slice] = True

        return spatial_mask

    def apply_loss_mask(
        self,
        feature: torch.Tensor,
        loss_mask: TensorLike,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Apply packet-level loss mask to feature.

        Lost packet regions are filled with fill_value.

        Parameters
        ----------
        feature : torch.Tensor
            Shape [C, H, W].

        loss_mask : tensor-like
            Shape [M]. True means packet is lost.

        fill_value : float
            Value used to fill lost regions.

        Returns
        -------
        torch.Tensor
            Feature with lost packet regions replaced by fill_value.
        """
        c, h, w = _validate_feature_3d(feature)
        metas, _, _ = self.build_packet_metas((c, h, w))

        loss_mask = _to_bool_mask(
            loss_mask,
            expected_len=len(metas),
            device=feature.device,
        )

        output = feature.clone()

        for meta in metas:
            if bool(loss_mask[meta.packet_id].item()):
                h_slice, w_slice = meta.spatial_slice
                output[:, h_slice, w_slice] = float(fill_value)

        return output

    def apply_receive_mask(
        self,
        feature: torch.Tensor,
        receive_mask: TensorLike,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Apply packet-level receive mask to feature.

        Regions of non-received packets are filled with fill_value.

        receive_mask[i] == True means packet i is received.
        """
        receive_mask = _to_bool_mask(
            receive_mask,
            expected_len=self.get_num_packets(feature),
            device=feature.device,
        )
        loss_mask = ~receive_mask

        return self.apply_loss_mask(
            feature=feature,
            loss_mask=loss_mask,
            fill_value=fill_value,
        )

    def select_packets(
        self,
        packets: torch.Tensor,
        mask: TensorLike,
        mask_means_keep: bool = True,
    ) -> torch.Tensor:
        """
        Select packets by a boolean mask.

        Parameters
        ----------
        packets : torch.Tensor
            Shape [M, C, ph, pw].

        mask : tensor-like
            Shape [M].

        mask_means_keep : bool
            If True, mask=True means keep packet.
            If False, mask=True means drop packet.

        Returns
        -------
        torch.Tensor
            Selected packet tensor.
        """
        if packets.dim() != 4:
            raise ValueError(
                f"packets should have shape [M, C, ph, pw], "
                f"got {tuple(packets.shape)}."
            )

        mask = _to_bool_mask(
            mask,
            expected_len=packets.shape[0],
            device=packets.device,
        )

        if not mask_means_keep:
            mask = ~mask

        return packets[mask]

    def get_config(self) -> Dict[str, Any]:
        """
        Export packetizer config.
        """
        return {
            "mode": self.mode,
            "grid_size": tuple(self.grid_size) if self.grid_size else None,
            "block_size": tuple(self.block_size) if self.block_size else None,
            "patch_max_size": (
                int(self.patch_max_size)
                if self.patch_max_size is not None
                else None
            ),
            "pad_value": float(self.pad_value),
        }

    def __repr__(self) -> str:
        if self.mode == PACKET_MODE_GRID:
            detail = f"grid_size={self.grid_size}"
        elif self.mode == PACKET_MODE_BLOCK:
            detail = f"block_size={self.block_size}"
        elif self.mode == PACKET_MODE_PATCH_MAX_SIZE:
            detail = f"patch_max_size={self.patch_max_size}"
        else:
            detail = "unknown"
        return f"FeaturePacketizer(mode={self.mode}, {detail})"