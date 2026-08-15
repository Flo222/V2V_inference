"""
Temporal cache recovery for ARCE communication simulation.

This module fills missing source packets using cached packets from previous
frames of the same communication link.

Typical position in ARCE recovery chain:

    FEC recovered packets
    -> temporal cache
    -> spatial interpolation
    -> zero-fill

Input convention:
    packets:
        [K, ...]
        K is the number of source packets.

    missing_mask[i] == True:
        source packet i is still missing.

    available_mask[i] == True:
        source packet i is available.

Cache convention:
    Cache is maintained per link_id, for example:
        link_id = (batch_idx, sender_idx)
        link_id = (frame_id, ego_id, sender_id)
        link_id = (ego_id, sender_id)

Important:
    Temporal cache is approximate recovery.
    It does not reconstruct the exact current-frame packet. It reuses a recent
    packet from the same link. This is useful when adjacent frames are similar,
    but it can be stale under fast motion or scene changes.

Recommended usage:
    1. After FEC decode, use temporal cache to fill missing packets.
    2. After final reconstruction, update cache using packets that are trusted.
       Usually use directly received + FEC-recovered + temporal/spatial-filled
       packets, but be careful not to pollute the cache with pure zero-fill
       packets unless you explicitly want that behavior.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from opencood.communication.transport.recovery import (
    RECOVERY_METHOD_TEMPORAL_CACHE,
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
            f"{name} length mismatch: expected {expected_len}, "
            f"got {out.numel()}."
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


def _normalize_link_key(link_id: Any = None) -> str:
    """
    Normalize link id to a stable dictionary key.

    repr(link_id) is used so tuples like (ego_id, sender_id) become
    JSON-friendly and hash-stable enough for cache indexing.
    """
    if link_id is None:
        return "__global_temporal_cache_link__"

    return repr(link_id)


def _count_true(mask: torch.Tensor) -> int:
    """
    Count True values.
    """
    mask = _as_bool_tensor(mask, name="mask")
    return int(mask.sum().item())


def _safe_cast_packet(packet: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Cast cached packet to target dtype.

    For integer tensors, round before casting.
    Temporal cache is usually applied to dequantized float packets, but this
    helper keeps integer-packet paths robust.
    """
    if dtype in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return torch.round(packet).to(dtype)

    if dtype == torch.bool:
        return (packet > 0.5).to(dtype)

    return packet.to(dtype)


def _compute_age_frames(
    cached_frame_id: Optional[int],
    current_frame_id: Optional[int],
) -> Optional[int]:
    """
    Compute cache age in frames.
    """
    if cached_frame_id is None or current_frame_id is None:
        return None

    return max(0, int(current_frame_id) - int(cached_frame_id))


def _compute_temporal_weight(
    age_frames: Optional[int],
    cache_tau: float,
    decay_mode: str = "none",
) -> float:
    """
    Compute cache confidence / decay weight.

    decay_mode:
        none:
            weight = 1.0

        exp:
            weight = exp(-age / cache_tau)

    Notes:
        By default this module does not scale cached features. It only logs
        temporal weight. Scaling feature values can change the feature
        distribution, so keep decay_mode='none' unless you intentionally want
        feature attenuation.
    """
    decay_mode = str(decay_mode).strip().lower()

    if decay_mode in ("none", "off", "disabled"):
        return 1.0

    if decay_mode in ("exp", "exponential"):
        if age_frames is None:
            return 1.0

        cache_tau = float(cache_tau)
        if cache_tau <= 0.0:
            return 1.0

        return float(math.exp(-float(age_frames) / cache_tau))

    raise ValueError(
        f"Unsupported temporal decay_mode: {decay_mode}. "
        "Expected none or exp."
    )


@dataclass
class TemporalCacheEntry:
    """
    One temporal cache entry for a communication link.

    Attributes
    ----------
    packets : torch.Tensor
        Cached packet tensor [K, ...].

    available_mask : torch.BoolTensor
        Shape [K]. True means cached packet i is available / trusted.

    frame_id : int or None
        Frame id when this cache entry was last updated.

    link_key : str
        Normalized link id key.

    num_updates : int
        Number of times this entry has been updated.

    info : dict
        Extra metadata.
    """

    packets: torch.Tensor
    available_mask: torch.Tensor
    frame_id: Optional[int]
    link_key: str
    num_updates: int = 1
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_packets(self) -> int:
        return int(self.packets.shape[0])

    @property
    def num_available_packets(self) -> int:
        return _count_true(self.available_mask)

    @property
    def packet_shape(self) -> Tuple[int, ...]:
        return tuple(int(x) for x in self.packets.shape)

    def is_shape_compatible(self, packets: torch.Tensor) -> bool:
        """
        Check whether cached packets have the same shape as input packets.
        """
        return tuple(self.packets.shape) == tuple(packets.shape)

    def to_device(
        self,
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
    ) -> "TemporalCacheEntry":
        """
        Return a shallow copy with tensors moved to target device / dtype.
        """
        packets = self.packets.to(device=device)

        if dtype is not None:
            packets = packets.to(dtype=dtype)

        available_mask = self.available_mask.to(device=device, dtype=torch.bool)

        return TemporalCacheEntry(
            packets=packets,
            available_mask=available_mask,
            frame_id=self.frame_id,
            link_key=self.link_key,
            num_updates=self.num_updates,
            info=copy.deepcopy(self.info),
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-friendly summary without serializing packet tensors.
        """
        return {
            "link_key": self.link_key,
            "frame_id": self.frame_id,
            "num_updates": int(self.num_updates),
            "packet_shape": tuple(int(x) for x in self.packets.shape),
            "packet_dtype": str(self.packets.dtype),
            "packet_device": str(self.packets.device),
            "num_packets": int(self.num_packets),
            "num_available_packets": int(self.num_available_packets),
            "info": copy.deepcopy(self.info),
        }


@dataclass
class TemporalCacheResult:
    """
    Result of temporal cache recovery.

    Attributes
    ----------
    recovered : torch.Tensor
        Recovered packet tensor [K, ...].

    filled_mask : torch.BoolTensor
        Shape [K]. True means packet i was filled from temporal cache.

    available_mask : torch.BoolTensor
        Shape [K]. True means packet i is available after temporal recovery.

    missing_mask_before : torch.BoolTensor
        Shape [K]. True means packet i was missing before temporal recovery.

    still_missing_mask : torch.BoolTensor
        Shape [K]. True means packet i remains missing after temporal recovery.

    info : dict
        JSON-friendly recovery statistics.
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
    def num_temporal_filled_packets(self) -> int:
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
                "num_temporal_filled_packets": int(self.num_temporal_filled_packets),
                "recovery_ratio": float(self.recovery_ratio),
                "recovered_shape": tuple(int(x) for x in self.recovered.shape),
                "recovered_dtype": str(self.recovered.dtype),
            }
        )
        return result


def temporal_fill_packets(
    packets: torch.Tensor,
    cache_entry: Optional[TemporalCacheEntry],
    missing_mask: Optional[Any] = None,
    available_mask: Optional[Any] = None,
    current_frame_id: Optional[int] = None,
    max_cache_age: int = 5,
    cache_tau: float = 5.0,
    decay_mode: str = "none",
    clone: bool = True,
) -> TemporalCacheResult:
    """
    Fill missing packets from a given temporal cache entry.

    Parameters
    ----------
    packets : torch.Tensor
        Current packet tensor [K, ...].

    cache_entry : TemporalCacheEntry or None
        Cached packets for the same link.

    missing_mask : tensor-like, optional
        Shape [K]. True means packet is missing.

    available_mask : tensor-like, optional
        Shape [K]. True means packet is available.

    current_frame_id : int, optional
        Current frame id, used to check cache age.

    max_cache_age : int
        Maximum allowed age in frames.

    cache_tau : float
        Temporal decay parameter. Only used when decay_mode="exp".

    decay_mode : str
        none or exp.

    clone : bool
        If True, operate on a cloned tensor.

    Returns
    -------
    TemporalCacheResult
    """
    packets = _require_tensor(packets, "packets")

    if packets.dim() < 1:
        raise ValueError(
            f"packets should have at least 1 dimension, got {tuple(packets.shape)}."
        )

    num_packets = int(packets.shape[0])

    missing_before = _resolve_missing_mask(
        num_packets=num_packets,
        missing_mask=missing_mask,
        available_mask=available_mask,
        device=packets.device,
    )

    recovered = packets.clone() if clone else packets
    filled_mask = torch.zeros_like(missing_before)
    still_missing = missing_before.clone()
    available_after = missing_mask_to_available_mask(missing_before)

    cache_hit = cache_entry is not None
    cache_valid = False
    cache_age = None
    temporal_weight = 1.0
    reason = ""

    if cache_entry is None:
        reason = "no cache entry"
    else:
        cache_age = _compute_age_frames(cache_entry.frame_id, current_frame_id)

        if cache_age is not None and cache_age > int(max_cache_age):
            reason = f"cache expired: age={cache_age}, max_cache_age={max_cache_age}"
        elif not cache_entry.is_shape_compatible(packets):
            reason = (
                "cache shape mismatch: "
                f"cache={tuple(cache_entry.packets.shape)}, "
                f"current={tuple(packets.shape)}"
            )
        else:
            cache_valid = True
            reason = "cache valid"

    if cache_valid:
        cache_on_device = cache_entry.to_device(
            device=packets.device,
            dtype=packets.dtype,
        )

        temporal_weight = _compute_temporal_weight(
            age_frames=cache_age,
            cache_tau=cache_tau,
            decay_mode=decay_mode,
        )

        cached_available = cache_on_device.available_mask
        fill_candidates = missing_before & cached_available

        if bool(fill_candidates.any().item()):
            cached_packets = cache_on_device.packets

            if decay_mode in ("exp", "exponential") and packets.dtype not in (
                torch.int8,
                torch.uint8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.bool,
            ):
                fill_values = cached_packets[fill_candidates].float() * float(temporal_weight)
                recovered[fill_candidates] = _safe_cast_packet(fill_values, packets.dtype)
            else:
                recovered[fill_candidates] = _safe_cast_packet(
                    cached_packets[fill_candidates],
                    packets.dtype,
                )

            filled_mask[fill_candidates] = True
            available_after[fill_candidates] = True
            still_missing[fill_candidates] = False

    num_temporal_filled = int(filled_mask.sum().item())
    num_still_missing = int(still_missing.sum().item())

    counts = build_recovery_count_dict(
        num_source_packets=num_packets,
        num_fec_recovered=0,
        num_temporal_filled=num_temporal_filled,
        num_spatial_filled=0,
        num_zero_filled=0,
        num_still_missing=num_still_missing,
    )

    info = {
        "method": RECOVERY_METHOD_TEMPORAL_CACHE,
        "target": "packets",
        "input_shape": tuple(int(x) for x in packets.shape),
        "input_dtype": str(packets.dtype),
        "num_packets": int(num_packets),
        "num_missing_before": int(missing_before.sum().item()),
        "num_temporal_filled_packets": int(num_temporal_filled),
        "num_still_missing_packets": int(num_still_missing),
        "recovery_ratio": float(counts["recovery_ratio"]),
        "cache_hit": bool(cache_hit),
        "cache_valid": bool(cache_valid),
        "cache_frame_id": None if cache_entry is None else cache_entry.frame_id,
        "current_frame_id": current_frame_id,
        "cache_age": cache_age,
        "max_cache_age": int(max_cache_age),
        "cache_tau": float(cache_tau),
        "decay_mode": str(decay_mode),
        "temporal_weight": float(temporal_weight),
        "reason": reason,
        "counts": counts,
        "note": (
            "Missing packets are filled from a previous cached feature packet "
            "of the same link. This is approximate temporal recovery."
        ),
    }

    return TemporalCacheResult(
        recovered=recovered,
        filled_mask=filled_mask,
        available_mask=available_after,
        missing_mask_before=missing_before,
        still_missing_mask=still_missing,
        info=info,
    )


class TemporalFeatureCache:
    """
    Per-link temporal feature packet cache.

    YAML style:

        arce:
          recovery:
            temporal_cache: true
            cache_tau: 5.0
            max_cache_age: 5
            temporal_decay_mode: none
            store_on_cpu: false
            detach_cache: true
            max_links: 1024
            merge_update: true

    Main methods:
        recover_packets(...)
            Fill current missing packets from cache.

        update(...)
            Update cache for a link with trusted current packets.

        update_from_fec_decode(...)
            Update cache from FECDecodeResult-like object.

        recover_from_fec_decode(...)
            Fill missing packets from FECDecodeResult-like object.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = self._extract_recovery_cfg(cfg or {})

        self.enabled = bool(
            cfg.get(
                "temporal_cache",
                cfg.get("temporal", True),
            )
        )

        self.cache_tau = float(cfg.get("cache_tau", 5.0))
        if self.cache_tau < 0.0:
            raise ValueError(f"cache_tau should be non-negative, got {self.cache_tau}.")

        self.max_cache_age = int(cfg.get("max_cache_age", 5))
        if self.max_cache_age < 0:
            raise ValueError(
                f"max_cache_age should be non-negative, got {self.max_cache_age}."
            )

        self.temporal_decay_mode = str(
            cfg.get("temporal_decay_mode", cfg.get("decay_mode", "none"))
        ).strip().lower()

        self.store_on_cpu = bool(cfg.get("store_on_cpu", False))
        self.detach_cache = bool(cfg.get("detach_cache", True))
        self.merge_update = bool(cfg.get("merge_update", True))
        self.max_links = int(cfg.get("max_links", 1024))

        if self.max_links <= 0:
            raise ValueError(f"max_links should be positive, got {self.max_links}.")

        self.cache: Dict[str, TemporalCacheEntry] = {}

    @staticmethod
    def _extract_recovery_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept full ARCE config or direct recovery config.
        """
        if "recovery" in cfg and isinstance(cfg["recovery"], dict):
            return cfg["recovery"]

        return cfg

    def _prepare_packets_for_store(self, packets: torch.Tensor) -> torch.Tensor:
        """
        Prepare packet tensor before storing in cache.
        """
        packets = _require_tensor(packets, "packets")

        if self.detach_cache:
            packets = packets.detach()

        packets = packets.clone()

        if self.store_on_cpu:
            packets = packets.cpu()

        return packets

    def _prepare_mask_for_store(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Prepare bool mask before storing in cache.
        """
        mask = _as_bool_tensor(mask, name="available_mask")

        if self.detach_cache:
            mask = mask.detach()

        mask = mask.clone()

        if self.store_on_cpu:
            mask = mask.cpu()

        return mask

    def _evict_if_needed(self) -> None:
        """
        Evict oldest cache entries if number of links exceeds max_links.
        """
        if len(self.cache) <= self.max_links:
            return

        def sort_key(item):
            _, entry = item
            if entry.frame_id is None:
                return -1
            return int(entry.frame_id)

        while len(self.cache) > self.max_links:
            oldest_key, _ = min(self.cache.items(), key=sort_key)
            self.cache.pop(oldest_key, None)

    def get_entry(self, link_id: Any = None) -> Optional[TemporalCacheEntry]:
        """
        Get cache entry for a link.
        """
        key = _normalize_link_key(link_id)
        return self.cache.get(key, None)

    def has_entry(self, link_id: Any = None) -> bool:
        """
        Return whether a link has a cache entry.
        """
        return self.get_entry(link_id) is not None

    def is_entry_valid(
        self,
        link_id: Any = None,
        current_frame_id: Optional[int] = None,
        packets: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Check whether a cache entry exists, is not expired, and optionally
        has the same packet shape as the current packets.
        """
        entry = self.get_entry(link_id)

        if entry is None:
            return False

        age = _compute_age_frames(entry.frame_id, current_frame_id)
        if age is not None and age > self.max_cache_age:
            return False

        if packets is not None and not entry.is_shape_compatible(packets):
            return False

        return True

    def clear(self, link_id: Any = None) -> None:
        """
        Clear all cache entries or one link cache entry.
        """
        if link_id is None:
            self.cache.clear()
            return

        key = _normalize_link_key(link_id)
        self.cache.pop(key, None)

    def remove_expired(self, current_frame_id: Optional[int]) -> int:
        """
        Remove entries older than max_cache_age.

        Returns
        -------
        int
            Number of removed entries.
        """
        if current_frame_id is None:
            return 0

        keys_to_remove = []

        for key, entry in self.cache.items():
            age = _compute_age_frames(entry.frame_id, current_frame_id)
            if age is not None and age > self.max_cache_age:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self.cache.pop(key, None)

        return int(len(keys_to_remove))

    def update(
        self,
        link_id: Any,
        packets: torch.Tensor,
        frame_id: Optional[int] = None,
        available_mask: Optional[Any] = None,
        missing_mask: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> TemporalCacheEntry:
        """
        Update temporal cache for one link.

        Parameters
        ----------
        link_id : any
            Link identifier.

        packets : torch.Tensor
            Packet tensor [K, ...].

        frame_id : int, optional
            Current frame id.

        available_mask : tensor-like, optional
            Shape [K]. True means packet is trusted and should be cached.

        missing_mask : tensor-like, optional
            Shape [K]. True means packet should not be updated.

        info : dict, optional
            Extra metadata.

        Returns
        -------
        TemporalCacheEntry
            Updated cache entry.
        """
        packets = _require_tensor(packets, "packets")

        if packets.dim() < 1:
            raise ValueError(
                f"packets should have at least 1 dimension, got {tuple(packets.shape)}."
            )

        key = _normalize_link_key(link_id)
        num_packets = int(packets.shape[0])

        missing = _resolve_missing_mask(
            num_packets=num_packets,
            missing_mask=missing_mask,
            available_mask=available_mask,
            device=packets.device,
        )
        current_available = missing_mask_to_available_mask(missing)

        old_entry = self.cache.get(key, None)

        if (
            self.merge_update
            and old_entry is not None
            and old_entry.is_shape_compatible(packets)
        ):
            old_on_device = old_entry.to_device(
                device=packets.device,
                dtype=packets.dtype,
            )

            merged_packets = old_on_device.packets.clone()
            merged_available = old_on_device.available_mask.clone()

            if bool(current_available.any().item()):
                merged_packets[current_available] = packets[current_available]
                merged_available[current_available] = True

            num_updates = int(old_entry.num_updates + 1)
        else:
            merged_packets = torch.zeros_like(packets)
            merged_available = torch.zeros(
                num_packets,
                dtype=torch.bool,
                device=packets.device,
            )

            if bool(current_available.any().item()):
                merged_packets[current_available] = packets[current_available]
                merged_available[current_available] = True

            num_updates = 1

        store_packets = self._prepare_packets_for_store(merged_packets)
        store_available = self._prepare_mask_for_store(merged_available)

        entry_info = {
            "last_update_frame_id": frame_id,
            "num_packets": int(num_packets),
            "num_updated_packets": int(current_available.sum().item()),
            "num_cached_available_packets": int(merged_available.sum().item()),
            "packet_shape": tuple(int(x) for x in packets.shape),
            "packet_dtype": str(packets.dtype),
        }

        if info:
            entry_info.update(copy.deepcopy(info))

        entry = TemporalCacheEntry(
            packets=store_packets,
            available_mask=store_available,
            frame_id=None if frame_id is None else int(frame_id),
            link_key=key,
            num_updates=num_updates,
            info=entry_info,
        )

        self.cache[key] = entry
        self._evict_if_needed()

        return entry

    def put(
        self,
        link_id: Any,
        packets: torch.Tensor,
        frame_id: Optional[int] = None,
        available_mask: Optional[Any] = None,
        missing_mask: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> TemporalCacheEntry:
        """
        Overwrite cache for one link.

        This is similar to update(), but it does not merge with the previous
        cache entry.
        """
        old_merge_update = self.merge_update
        self.merge_update = False

        try:
            entry = self.update(
                link_id=link_id,
                packets=packets,
                frame_id=frame_id,
                available_mask=available_mask,
                missing_mask=missing_mask,
                info=info,
            )
        finally:
            self.merge_update = old_merge_update

        return entry

    def recover_packets(
        self,
        link_id: Any,
        packets: torch.Tensor,
        missing_mask: Optional[Any] = None,
        available_mask: Optional[Any] = None,
        current_frame_id: Optional[int] = None,
        clone: bool = True,
    ) -> TemporalCacheResult:
        """
        Recover missing packets from temporal cache.

        If cache is disabled, missing packets remain missing.
        """
        packets = _require_tensor(packets, "packets")
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
                "method": RECOVERY_METHOD_TEMPORAL_CACHE,
                "enabled": False,
                "target": "packets",
                "link_id": repr(link_id),
                "num_packets": int(num_packets),
                "num_missing_before": int(missing.sum().item()),
                "num_temporal_filled_packets": 0,
                "num_still_missing_packets": int(missing.sum().item()),
                "recovery_ratio": (
                    float(available.sum().item() / num_packets)
                    if num_packets > 0
                    else 1.0
                ),
                "reason": "temporal cache disabled",
            }

            return TemporalCacheResult(
                recovered=packets.clone() if clone else packets,
                filled_mask=torch.zeros_like(missing),
                available_mask=available,
                missing_mask_before=missing,
                still_missing_mask=missing,
                info=info,
            )

        entry = self.get_entry(link_id)

        result = temporal_fill_packets(
            packets=packets,
            cache_entry=entry,
            missing_mask=missing,
            current_frame_id=current_frame_id,
            max_cache_age=self.max_cache_age,
            cache_tau=self.cache_tau,
            decay_mode=self.temporal_decay_mode,
            clone=clone,
        )

        result.info["enabled"] = True
        result.info["link_id"] = repr(link_id)
        result.info["link_key"] = _normalize_link_key(link_id)

        return result

    def recover_from_fec_decode(
        self,
        link_id: Any,
        decode_result: Any,
        current_frame_id: Optional[int] = None,
        clone: bool = True,
    ) -> TemporalCacheResult:
        """
        Recover missing packets from a FECDecodeResult-like object.

        Expected fields:
            decode_result.recovered_packets
            decode_result.missing_source_mask
        """
        if not hasattr(decode_result, "recovered_packets"):
            raise AttributeError("decode_result should have recovered_packets.")

        if not hasattr(decode_result, "missing_source_mask"):
            raise AttributeError("decode_result should have missing_source_mask.")

        return self.recover_packets(
            link_id=link_id,
            packets=decode_result.recovered_packets,
            missing_mask=decode_result.missing_source_mask,
            current_frame_id=current_frame_id,
            clone=clone,
        )

    def update_from_fec_decode(
        self,
        link_id: Any,
        decode_result: Any,
        frame_id: Optional[int] = None,
        include_fec_recovered: bool = True,
    ) -> TemporalCacheEntry:
        """
        Update cache from a FECDecodeResult-like object.

        By default, cache packets that are directly received or FEC-recovered:

            available_mask = decode_result.recovered_source_mask

        If include_fec_recovered=False:
            only cache directly received source packets.
        """
        if not hasattr(decode_result, "recovered_packets"):
            raise AttributeError("decode_result should have recovered_packets.")

        if include_fec_recovered:
            if not hasattr(decode_result, "recovered_source_mask"):
                raise AttributeError("decode_result should have recovered_source_mask.")

            available_mask = decode_result.recovered_source_mask
        else:
            if not hasattr(decode_result, "direct_received_source_mask"):
                raise AttributeError(
                    "decode_result should have direct_received_source_mask."
                )

            available_mask = decode_result.direct_received_source_mask

        return self.update(
            link_id=link_id,
            packets=decode_result.recovered_packets,
            frame_id=frame_id,
            available_mask=available_mask,
            info={
                "source": "fec_decode_result",
                "include_fec_recovered": bool(include_fec_recovered),
            },
        )

    def update_from_recovery_result(
        self,
        link_id: Any,
        recovery_result: Any,
        frame_id: Optional[int] = None,
        exclude_zero_filled: bool = True,
    ) -> TemporalCacheEntry:
        """
        Update cache from a recovery result.

        Expected fields:
            recovery_result.recovered
            recovery_result.available_mask

        If exclude_zero_filled=True and recovery_result has filled_mask from
        zero-fill, you should pass a trusted available_mask externally instead.
        This method cannot always infer whether filled_mask came from zero-fill
        or another recovery method, so use carefully.
        """
        if not hasattr(recovery_result, "recovered"):
            raise AttributeError("recovery_result should have recovered.")

        if not hasattr(recovery_result, "available_mask"):
            raise AttributeError("recovery_result should have available_mask.")

        available_mask = recovery_result.available_mask

        if exclude_zero_filled and hasattr(recovery_result, "info"):
            method = recovery_result.info.get("method", "")
            if str(method).lower() in ("zero", "zero_fill"):
                if hasattr(recovery_result, "filled_mask"):
                    available_mask = recovery_result.available_mask & (~recovery_result.filled_mask)

        return self.update(
            link_id=link_id,
            packets=recovery_result.recovered,
            frame_id=frame_id,
            available_mask=available_mask,
            info={
                "source": "recovery_result",
                "exclude_zero_filled": bool(exclude_zero_filled),
            },
        )

    def get_cache_summary(self) -> Dict[str, Any]:
        """
        Return JSON-friendly summary of all cache entries.
        """
        return {
            "enabled": bool(self.enabled),
            "num_links": int(len(self.cache)),
            "max_links": int(self.max_links),
            "cache_tau": float(self.cache_tau),
            "max_cache_age": int(self.max_cache_age),
            "temporal_decay_mode": self.temporal_decay_mode,
            "store_on_cpu": bool(self.store_on_cpu),
            "detach_cache": bool(self.detach_cache),
            "merge_update": bool(self.merge_update),
            "entries": {
                key: entry.as_dict()
                for key, entry in self.cache.items()
            },
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return JSON-friendly temporal cache config.
        """
        return {
            "enabled": bool(self.enabled),
            "method": RECOVERY_METHOD_TEMPORAL_CACHE,
            "cache_tau": float(self.cache_tau),
            "max_cache_age": int(self.max_cache_age),
            "temporal_decay_mode": self.temporal_decay_mode,
            "store_on_cpu": bool(self.store_on_cpu),
            "detach_cache": bool(self.detach_cache),
            "merge_update": bool(self.merge_update),
            "max_links": int(self.max_links),
        }

    def __repr__(self) -> str:
        return (
            "TemporalFeatureCache("
            f"enabled={self.enabled}, "
            f"num_links={len(self.cache)}, "
            f"cache_tau={self.cache_tau}, "
            f"max_cache_age={self.max_cache_age}, "
            f"decay_mode={self.temporal_decay_mode})"
        )


TemporalCache = TemporalFeatureCache
TemporalCacheRecovery = TemporalFeatureCache


__all__ = [
    "TemporalCacheEntry",
    "TemporalCacheResult",
    "temporal_fill_packets",
    "TemporalFeatureCache",
    "TemporalCache",
    "TemporalCacheRecovery",
]