"""
Unified partial reconstruction controller for ARCE communication simulation.

This module combines packet recovery methods after channel loss and FEC decode.

Recommended recovery chain:

    FEC decode result
    -> temporal cache
    -> spatial interpolation
    -> zero-fill
    -> recovered source packets [K, ...]
    -> unpacketize back to feature [C, H, W]

Important:
    FEC is handled before this module.

    This module only handles source packets that are still missing after:
        1. direct packet reception;
        2. FEC recovery.

Input convention:
    packets:
        [K, ...]
        K is the number of original source packets.

    missing_mask[i] == True:
        source packet i is still missing.

    available_mask[i] == True:
        source packet i is available.

Typical usage after FEC decode:

    partial = PartialReconstructor(arce_cfg)

    result = partial.recover_from_fec_decode(
        decode_result=decode_result,
        packet_metas=packet_result.metas,
        link_id=(ego_id, sender_id),
        frame_id=frame_id,
    )

    recovered_packets = result.recovered_packets

    recovered_feature = packetizer.unpacketize(
        recovered_packets,
        packet_result.metas,
        packet_result.original_shape,
    )

Cache update policy:
    By default, temporal cache is updated with trusted packets:
        direct received packets
        FEC recovered packets
        temporal filled packets
        spatial filled packets

    Zero-filled packets are excluded from cache by default to avoid cache
    pollution.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from opencood.communication.transport.recovery import (
    RECOVERY_METHOD_ZERO_FILL,
    RECOVERY_METHOD_SPATIAL_INTERPOLATION,
    RECOVERY_METHOD_TEMPORAL_CACHE,
    DEFAULT_RECOVERY_PRIORITY,
    normalize_recovery_config,
    normalize_recovery_priority,
    missing_mask_to_available_mask,
    available_mask_to_missing_mask,
    build_recovery_count_dict,
)

from opencood.communication.transport.recovery.zero_fill import (
    ZeroFillRecovery,
)

from opencood.communication.transport.recovery.spatial_interpolation import (
    SpatialInterpolationRecovery,
)

from opencood.communication.transport.recovery.temporal_cache import (
    TemporalFeatureCache,
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


def _optional_bool_tensor(
    mask: Optional[Any],
    expected_len: int,
    device: Union[str, torch.device],
    name: str,
    default_value: bool = False,
) -> torch.Tensor:
    """
    Convert optional mask to bool tensor.

    If mask is None, return all default_value.
    """
    if mask is None:
        return torch.full(
            (int(expected_len),),
            bool(default_value),
            dtype=torch.bool,
            device=device,
        )

    return _as_bool_tensor(
        mask,
        expected_len=expected_len,
        device=device,
        name=name,
    )


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
        available = _as_bool_tensor(
            available_mask,
            expected_len=num_packets,
            device=device,
            name="available_mask",
        )
        return available_mask_to_missing_mask(available)

    return torch.zeros(num_packets, dtype=torch.bool, device=device)


def _count_true(mask: torch.Tensor) -> int:
    """
    Count True values in a bool tensor.
    """
    mask = _as_bool_tensor(mask, name="mask")
    return int(mask.sum().item())


def _extract_recovery_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Accept full ARCE config or direct recovery config.
    """
    cfg = cfg or {}

    if "recovery" in cfg and isinstance(cfg["recovery"], dict):
        return cfg["recovery"]

    return cfg


def _as_bool(value: Any) -> bool:
    """
    Convert common config values to bool.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "y", "on"):
            return True
        if text in ("false", "0", "no", "n", "off"):
            return False

    return bool(value)


@dataclass
class PartialReconstructionResult:
    """
    Result of partial reconstruction.

    Attributes
    ----------
    recovered_packets : torch.Tensor
        Final recovered source packets, shape [K, ...].

    recovered_feature : torch.Tensor or None
        Optional unpacketized feature [C, H, W].
        This is filled only when recover_and_unpacketize() is used.

    available_mask : torch.BoolTensor
        Shape [K]. True means source packet is available after all recovery.

    missing_mask_before : torch.BoolTensor
        Shape [K]. True means source packet was missing before partial recovery.

    still_missing_mask : torch.BoolTensor
        Shape [K]. True means source packet remains missing after partial recovery.
        If zero-fill is enabled, this is usually all False.

    direct_received_mask : torch.BoolTensor
        Shape [K]. True means packet was directly received from channel.

    fec_recovered_mask : torch.BoolTensor
        Shape [K]. True means packet was recovered by FEC.

    temporal_filled_mask : torch.BoolTensor
        Shape [K]. True means packet was filled from temporal cache.

    spatial_filled_mask : torch.BoolTensor
        Shape [K]. True means packet was filled by spatial interpolation.

    zero_filled_mask : torch.BoolTensor
        Shape [K]. True means packet was filled by zero-fill.

    cache_update_mask : torch.BoolTensor
        Shape [K]. True means this packet was used to update temporal cache.

    method_infos : dict
        Per-method logs.

    info : dict
        JSON-friendly summary.
    """

    recovered_packets: torch.Tensor
    recovered_feature: Optional[torch.Tensor]

    available_mask: torch.Tensor
    missing_mask_before: torch.Tensor
    still_missing_mask: torch.Tensor

    direct_received_mask: torch.Tensor
    fec_recovered_mask: torch.Tensor
    temporal_filled_mask: torch.Tensor
    spatial_filled_mask: torch.Tensor
    zero_filled_mask: torch.Tensor

    cache_update_mask: torch.Tensor

    method_infos: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_packets(self) -> int:
        return int(self.available_mask.numel())

    @property
    def num_available_packets(self) -> int:
        return _count_true(self.available_mask)

    @property
    def num_still_missing_packets(self) -> int:
        return _count_true(self.still_missing_mask)

    @property
    def recovery_ratio(self) -> float:
        if self.num_packets <= 0:
            return 1.0
        return float(self.num_available_packets / self.num_packets)

    def as_dict(self) -> Dict[str, Any]:
        """
        Return JSON-friendly summary.

        Masks are summarized by counts instead of serializing full tensors.
        """
        result = copy.deepcopy(self.info)

        result.update(
            {
                "num_packets": int(self.num_packets),
                "num_available_packets": int(self.num_available_packets),
                "num_still_missing_packets": int(self.num_still_missing_packets),
                "recovery_ratio": float(self.recovery_ratio),
                "num_direct_received_packets": int(_count_true(self.direct_received_mask)),
                "num_fec_recovered_packets": int(_count_true(self.fec_recovered_mask)),
                "num_temporal_filled_packets": int(_count_true(self.temporal_filled_mask)),
                "num_spatial_filled_packets": int(_count_true(self.spatial_filled_mask)),
                "num_zero_filled_packets": int(_count_true(self.zero_filled_mask)),
                "num_cache_update_packets": int(_count_true(self.cache_update_mask)),
                "recovered_packets_shape": tuple(
                    int(x) for x in self.recovered_packets.shape
                ),
                "recovered_packets_dtype": str(self.recovered_packets.dtype),
                "has_recovered_feature": self.recovered_feature is not None,
                "recovered_feature_shape": (
                    None
                    if self.recovered_feature is None
                    else tuple(int(x) for x in self.recovered_feature.shape)
                ),
                "method_infos": copy.deepcopy(self.method_infos),
            }
        )

        return result


class PartialReconstructor:
    """
    Unified partial reconstruction controller.

    YAML style:

        arce:
          recovery:
            temporal_cache: true
            spatial_interpolation: true
            zero_fill: true

            priority:
              - temporal_cache
              - spatial_interpolation
              - zero_fill

            cache_tau: 5.0
            max_cache_age: 5

            update_cache: true
            cache_update_include_direct: true
            cache_update_include_fec: true
            cache_update_include_temporal: true
            cache_update_include_spatial: true
            cache_update_include_zero: false

    Main APIs:
        recover_packets(...)
            Run temporal/spatial/zero recovery on packet tensor.

        recover_from_fec_decode(...)
            Run recovery directly from FECDecodeResult.

        recover_and_unpacketize(...)
            Run recovery and then reconstruct [C, H, W] feature.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.full_cfg = cfg or {}
        self.recovery_cfg_raw = _extract_recovery_cfg(cfg)

        self.recovery_cfg = normalize_recovery_config(cfg or {})

        self.enabled = _as_bool(
            self.recovery_cfg_raw.get("enabled", True)
        )

        self.priority = normalize_recovery_priority(
            self.recovery_cfg_raw.get(
                "priority",
                self.recovery_cfg.get("priority", DEFAULT_RECOVERY_PRIORITY),
            )
        )

        self.temporal_cache = TemporalFeatureCache(cfg or {})
        self.spatial_recovery = SpatialInterpolationRecovery(cfg or {})
        self.zero_recovery = ZeroFillRecovery(cfg or {})

        self.update_cache_enabled = _as_bool(
            self.recovery_cfg_raw.get("update_cache", True)
        )

        self.cache_update_include_direct = _as_bool(
            self.recovery_cfg_raw.get("cache_update_include_direct", True)
        )
        self.cache_update_include_fec = _as_bool(
            self.recovery_cfg_raw.get("cache_update_include_fec", True)
        )
        self.cache_update_include_temporal = _as_bool(
            self.recovery_cfg_raw.get("cache_update_include_temporal", True)
        )
        self.cache_update_include_spatial = _as_bool(
            self.recovery_cfg_raw.get("cache_update_include_spatial", True)
        )
        self.cache_update_include_zero = _as_bool(
            self.recovery_cfg_raw.get("cache_update_include_zero", False)
        )

        self.update_cache_when_temporal_disabled = _as_bool(
            self.recovery_cfg_raw.get("update_cache_when_temporal_disabled", False)
        )

        temporal_fusion_cfg = {}
        if isinstance(self.full_cfg, dict):
            temporal_fusion_cfg = self.full_cfg.get("temporal_fusion", {}) or {}
            if not temporal_fusion_cfg and isinstance(self.full_cfg.get("arce", None), dict):
                temporal_fusion_cfg = self.full_cfg.get("arce", {}).get("temporal_fusion", {}) or {}

        self.temporal_fusion_enabled = _as_bool(
            temporal_fusion_cfg.get("enabled", True)
        )
        self.temporal_fusion_beta = float(
            temporal_fusion_cfg.get("beta", 5.0)
        )
        self.temporal_fusion_tau_stale_ms = float(
            temporal_fusion_cfg.get("tau_stale_ms", 300.0)
        )
        self.temporal_fusion_use_delay_penalty = _as_bool(
            temporal_fusion_cfg.get("use_delay_penalty", True)
        )
        self.update_if_recv_ge_cache = (
            str(temporal_fusion_cfg.get("update_rule", "recv_ge_cache")).strip().lower()
            == "recv_ge_cache"
        )

    def _build_cache_update_mask(
        self,
        direct_received_mask: torch.Tensor,
        fec_recovered_mask: torch.Tensor,
        temporal_filled_mask: torch.Tensor,
        spatial_filled_mask: torch.Tensor,
        zero_filled_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build trusted packet mask for temporal cache update.
        """
        cache_mask = torch.zeros_like(direct_received_mask)

        if self.cache_update_include_direct:
            cache_mask = cache_mask | direct_received_mask

        if self.cache_update_include_fec:
            cache_mask = cache_mask | fec_recovered_mask

        if self.cache_update_include_temporal:
            cache_mask = cache_mask | temporal_filled_mask

        if self.cache_update_include_spatial:
            cache_mask = cache_mask | spatial_filled_mask

        if self.cache_update_include_zero:
            cache_mask = cache_mask | zero_filled_mask

        return cache_mask

    def _maybe_update_cache(
        self,
        link_id: Any,
        packets: torch.Tensor,
        frame_id: Optional[int],
        cache_update_mask: torch.Tensor,
        update_cache: bool,
        method_infos: Dict[str, Any],
        q_recv: Optional[float] = None,
        q_cache: Optional[float] = None,
        q_eff: Optional[float] = None,
        temporal_alpha: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Update temporal cache if enabled and requested.
        """
        if not update_cache:
            method_infos["cache_update"] = {
                "updated": False,
                "reason": "update_cache argument is False",
            }
            return None

        if not self.update_cache_enabled:
            method_infos["cache_update"] = {
                "updated": False,
                "reason": "update_cache disabled by config",
            }
            return None

        if not self.temporal_cache.enabled and not self.update_cache_when_temporal_disabled:
            method_infos["cache_update"] = {
                "updated": False,
                "reason": "temporal cache disabled",
            }
            return None

        if cache_update_mask.numel() == 0 or not bool(cache_update_mask.any().item()):
            method_infos["cache_update"] = {
                "updated": False,
                "reason": "no trusted packets for cache update",
            }
            return None

        entry = self.temporal_cache.update(
            link_id=link_id,
            packets=packets,
            frame_id=frame_id,
            available_mask=cache_update_mask,
            info={
                "source": "partial_reconstruction",
                "num_cache_update_packets": int(cache_update_mask.sum().item()),
                "quality": (
                    None
                    if (q_eff is None and q_recv is None)
                    else float(q_eff if q_eff is not None else q_recv)
                ),
                "q_recv": None if q_recv is None else float(q_recv),
                "q_eff": None if q_eff is None else float(q_eff),
                "q_cache": None if q_cache is None else float(q_cache),
                "temporal_alpha": None if temporal_alpha is None else float(temporal_alpha),
            },
        )

        method_infos["cache_update"] = {
            "updated": True,
            "entry": entry.as_dict(),
        }

        return entry

    def recover_packets(
        self,
        packets: torch.Tensor,
        packet_metas: Optional[Sequence[Any]] = None,
        missing_mask: Optional[Any] = None,
        available_mask: Optional[Any] = None,
        direct_received_mask: Optional[Any] = None,
        fec_recovered_mask: Optional[Any] = None,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        update_cache: bool = True,
        clone: bool = True,
        delay_ms: float = 0.0,
    ) -> PartialReconstructionResult:
        """
        Run partial reconstruction on source packets.

        Parameters
        ----------
        packets : torch.Tensor
            Source packet tensor [K, ...].
            Usually this is FECDecodeResult.recovered_packets.

        packet_metas : sequence, optional
            FeaturePacketMeta list from packetizer.py.
            Required for spatial interpolation.

        missing_mask : tensor-like, optional
            Shape [K]. True means packet is missing before partial recovery.

        available_mask : tensor-like, optional
            Shape [K]. True means packet is available before partial recovery.

        direct_received_mask : tensor-like, optional
            Shape [K]. True means packet was directly received.

        fec_recovered_mask : tensor-like, optional
            Shape [K]. True means packet was recovered by FEC.

        link_id : any
            Communication link id, for temporal cache.

        frame_id : int, optional
            Current frame id, for temporal cache age and update.

        update_cache : bool
            Whether to update temporal cache after reconstruction.

        clone : bool
            If True, operate on cloned packet tensor.

        Returns
        -------
        PartialReconstructionResult
        """
        packets = _require_tensor(packets, "packets")

        if packets.dim() < 1:
            raise ValueError(
                f"packets should have at least 1 dimension, got {tuple(packets.shape)}."
            )

        num_packets = int(packets.shape[0])
        device = packets.device

        missing_before = _resolve_missing_mask(
            num_packets=num_packets,
            missing_mask=missing_mask,
            available_mask=available_mask,
            device=device,
        )

        current_packets = packets.clone() if clone else packets
        current_available = missing_mask_to_available_mask(missing_before)
        current_missing = missing_before.clone()

        fec_recovered_mask = _optional_bool_tensor(
            fec_recovered_mask,
            expected_len=num_packets,
            device=device,
            name="fec_recovered_mask",
            default_value=False,
        )

        if direct_received_mask is None:
            direct_received_mask = current_available & (~fec_recovered_mask)
        else:
            direct_received_mask = _as_bool_tensor(
                direct_received_mask,
                expected_len=num_packets,
                device=device,
                name="direct_received_mask",
            )

        current_available = current_available | direct_received_mask | fec_recovered_mask
        current_missing = available_mask_to_missing_mask(current_available)

        temporal_filled_mask = torch.zeros(
            num_packets,
            dtype=torch.bool,
            device=device,
        )
        spatial_filled_mask = torch.zeros_like(temporal_filled_mask)
        zero_filled_mask = torch.zeros_like(temporal_filled_mask)

        method_infos: Dict[str, Any] = {}

        if not self.enabled:
            cache_update_mask = self._build_cache_update_mask(
                direct_received_mask=direct_received_mask,
                fec_recovered_mask=fec_recovered_mask,
                temporal_filled_mask=temporal_filled_mask,
                spatial_filled_mask=spatial_filled_mask,
                zero_filled_mask=zero_filled_mask,
            )

            self._maybe_update_cache(
                link_id=link_id,
                packets=current_packets,
                frame_id=frame_id,
                cache_update_mask=cache_update_mask,
                update_cache=update_cache,
                method_infos=method_infos,
            )

            info = {
                "enabled": False,
                "priority": tuple(self.priority),
                "link_id": repr(link_id),
                "frame_id": frame_id,
                "num_packets": int(num_packets),
                "num_missing_before": int(missing_before.sum().item()),
                "num_still_missing_packets": int(current_missing.sum().item()),
                "note": "Partial reconstruction is disabled.",
            }

            return PartialReconstructionResult(
                recovered_packets=current_packets,
                recovered_feature=None,
                available_mask=current_available,
                missing_mask_before=missing_before,
                still_missing_mask=current_missing,
                direct_received_mask=direct_received_mask,
                fec_recovered_mask=fec_recovered_mask,
                temporal_filled_mask=temporal_filled_mask,
                spatial_filled_mask=spatial_filled_mask,
                zero_filled_mask=zero_filled_mask,
                cache_update_mask=cache_update_mask,
                method_infos=method_infos,
                info=info,
            )

        for method in self.priority:
            if not bool(current_missing.any().item()):
                break

            if method == RECOVERY_METHOD_TEMPORAL_CACHE:
                temporal_result = self.temporal_cache.recover_packets(
                    link_id=link_id,
                    packets=current_packets,
                    missing_mask=current_missing,
                    current_frame_id=frame_id,
                    clone=True,
                )

                current_packets = temporal_result.recovered
                temporal_filled_mask = temporal_filled_mask | temporal_result.filled_mask
                current_available = temporal_result.available_mask
                current_missing = temporal_result.still_missing_mask

                method_infos[RECOVERY_METHOD_TEMPORAL_CACHE] = temporal_result.as_dict()

            elif method == RECOVERY_METHOD_SPATIAL_INTERPOLATION:
                if packet_metas is None:
                    method_infos[RECOVERY_METHOD_SPATIAL_INTERPOLATION] = {
                        "enabled": False,
                        "skipped": True,
                        "reason": "packet_metas is None, spatial interpolation cannot run",
                    }
                    continue

                spatial_result = self.spatial_recovery.recover_packets(
                    packets=current_packets,
                    packet_metas=packet_metas,
                    missing_mask=current_missing,
                    clone=True,
                )

                current_packets = spatial_result.recovered
                spatial_filled_mask = spatial_filled_mask | spatial_result.filled_mask
                current_available = spatial_result.available_mask
                current_missing = spatial_result.still_missing_mask

                method_infos[RECOVERY_METHOD_SPATIAL_INTERPOLATION] = spatial_result.as_dict()

            elif method == RECOVERY_METHOD_ZERO_FILL:
                zero_result = self.zero_recovery.recover_packets(
                    packets=current_packets,
                    missing_mask=current_missing,
                    clone=True,
                )

                current_packets = zero_result.recovered
                zero_filled_mask = zero_filled_mask | zero_result.filled_mask
                current_available = zero_result.available_mask
                current_missing = zero_result.still_missing_mask

                method_infos[RECOVERY_METHOD_ZERO_FILL] = zero_result.as_dict()

            else:
                method_infos[str(method)] = {
                    "skipped": True,
                    "reason": f"unsupported recovery method in priority: {method}",
                }

        # PDF Step 4: temporal quality-aware fusion.
        # q_recv measures current recovered packet completeness.
        # q_cache is read from the previous cache entry for the same link.
        q_recv = (
            1.0
            if num_packets <= 0
            else max(
                0.0,
                min(
                    1.0,
                    1.0 - float(current_missing.sum().item()) / float(num_packets),
                ),
            )
        )

        cache_entry_for_fusion = None
        q_cache = 0.0
        if hasattr(self, "temporal_cache") and self.temporal_cache is not None:
            try:
                cache_entry_for_fusion = self.temporal_cache.get_entry(link_id)
            except Exception:
                cache_entry_for_fusion = None

        if cache_entry_for_fusion is not None:
            try:
                cache_info = getattr(cache_entry_for_fusion, "info", {}) or {}
                q_cache = float(
                    cache_info.get(
                        "quality",
                        cache_info.get(
                            "q_recv",
                            cache_info.get("recovery_ratio", 0.0),
                        ),
                    )
                    or 0.0
                )
            except Exception:
                q_cache = 0.0

            if q_cache <= 0.0:
                try:
                    q_cache = float(
                        cache_entry_for_fusion.num_available_packets
                        / max(1, cache_entry_for_fusion.num_packets)
                    )
                except Exception:
                    q_cache = 0.0

        delay_ms_f = max(float(delay_ms), 0.0)
        tau_stale_ms = max(float(self.temporal_fusion_tau_stale_ms), 1e-6)
        if bool(self.temporal_fusion_use_delay_penalty):
            q_delay = math.exp(-delay_ms_f / tau_stale_ms)
        else:
            q_delay = 1.0
        q_eff = float(q_recv) * float(q_delay)

        temporal_alpha = 1.0
        fusion_info = {
            "enabled": bool(self.temporal_fusion_enabled),
            "applied": False,
            "q_recv": float(q_recv),
            "q_delay": float(q_delay),
            "q_eff": float(q_eff),
            "q_cache": float(q_cache),
            "delay_ms": float(delay_ms_f),
            "tau_stale_ms": float(tau_stale_ms),
            "use_delay_penalty": bool(self.temporal_fusion_use_delay_penalty),
            "alpha": float(temporal_alpha),
        }

        if (
            bool(self.temporal_fusion_enabled)
            and cache_entry_for_fusion is not None
            and hasattr(cache_entry_for_fusion, "is_shape_compatible")
            and cache_entry_for_fusion.is_shape_compatible(current_packets)
        ):
            cache_entry_device = cache_entry_for_fusion.to_device(
                device=current_packets.device,
                dtype=current_packets.dtype,
            )
            alpha_tensor = torch.sigmoid(
                torch.tensor(
                    float(self.temporal_fusion_beta) * (float(q_eff) - float(q_cache)),
                    dtype=current_packets.dtype,
                    device=current_packets.device,
                )
            )
            current_packets = (
                alpha_tensor * current_packets
                + (1.0 - alpha_tensor) * cache_entry_device.packets
            )
            temporal_alpha = float(alpha_tensor.detach().cpu().item())
            fusion_info.update(
                {
                    "applied": True,
                    "alpha": float(temporal_alpha),
                    "beta": float(self.temporal_fusion_beta),
                    "q_delay": float(q_delay),
                    "q_eff": float(q_eff),
                    "delay_ms": float(delay_ms_f),
                    "tau_stale_ms": float(tau_stale_ms),
                    "cache_frame_id": getattr(cache_entry_for_fusion, "frame_id", None),
                }
            )
        else:
            if cache_entry_for_fusion is None:
                fusion_info["reason"] = "no cache entry"
            elif not bool(self.temporal_fusion_enabled):
                fusion_info["reason"] = "temporal fusion disabled"
            else:
                fusion_info["reason"] = "cache shape incompatible"

        method_infos["temporal_quality_fusion"] = fusion_info

        cache_update_mask = self._build_cache_update_mask(
            direct_received_mask=direct_received_mask,
            fec_recovered_mask=fec_recovered_mask,
            temporal_filled_mask=temporal_filled_mask,
            spatial_filled_mask=spatial_filled_mask,
            zero_filled_mask=zero_filled_mask,
        )

        update_cache_after_quality_gate = bool(update_cache)
        if self.update_if_recv_ge_cache:
            update_cache_after_quality_gate = (
                update_cache_after_quality_gate
                and float(q_eff) >= float(q_cache)
            )
            method_infos["cache_update_quality_gate"] = {
                "enabled": True,
                "passed": bool(float(q_eff) >= float(q_cache)),
                "q_recv": float(q_recv),
                "q_cache": float(q_cache),
                "rule": "q_eff >= q_cache",
            }

        self._maybe_update_cache(
            link_id=link_id,
            packets=current_packets,
            frame_id=frame_id,
            cache_update_mask=cache_update_mask,
            update_cache=update_cache_after_quality_gate,
            method_infos=method_infos,
            q_recv=float(q_recv),
            q_cache=float(q_cache),
            q_eff=float(q_eff),
            temporal_alpha=float(temporal_alpha),
        )

        counts = build_recovery_count_dict(
            num_source_packets=num_packets,
            num_fec_recovered=int(fec_recovered_mask.sum().item()),
            num_temporal_filled=int(temporal_filled_mask.sum().item()),
            num_spatial_filled=int(spatial_filled_mask.sum().item()),
            num_zero_filled=int(zero_filled_mask.sum().item()),
            num_still_missing=int(current_missing.sum().item()),
        )

        info = {
            "enabled": True,
            "priority": tuple(self.priority),
            "link_id": repr(link_id),
            "frame_id": frame_id,
            "num_packets": int(num_packets),
            "num_missing_before": int(missing_before.sum().item()),
            "num_available_after": int(current_available.sum().item()),
            "num_still_missing_packets": int(current_missing.sum().item()),
            "num_direct_received_packets": int(direct_received_mask.sum().item()),
            "num_fec_recovered_packets": int(fec_recovered_mask.sum().item()),
            "num_temporal_filled_packets": int(temporal_filled_mask.sum().item()),
            "num_spatial_filled_packets": int(spatial_filled_mask.sum().item()),
            "num_zero_filled_packets": int(zero_filled_mask.sum().item()),
            "num_cache_update_packets": int(cache_update_mask.sum().item()),
            "recovery_ratio": float(counts["recovery_ratio"]),
            "q_recv": float(q_recv),
            "q_delay": float(q_delay),
            "q_eff": float(q_eff),
            "q_cache": float(q_cache),
            "delay_ms": float(delay_ms_f),
            "tau_stale_ms": float(tau_stale_ms),
            "temporal_alpha": float(temporal_alpha),
            "temporal_fusion_applied": bool(method_infos.get("temporal_quality_fusion", {}).get("applied", False)),
            "counts": counts,
            "note": (
                "Partial reconstruction applies temporal cache, spatial "
                "interpolation, and zero-fill according to configured priority. "
                "FEC recovery is assumed to have already happened before this step."
            ),
        }

        return PartialReconstructionResult(
            recovered_packets=current_packets,
            recovered_feature=None,
            available_mask=current_available,
            missing_mask_before=missing_before,
            still_missing_mask=current_missing,
            direct_received_mask=direct_received_mask,
            fec_recovered_mask=fec_recovered_mask,
            temporal_filled_mask=temporal_filled_mask,
            spatial_filled_mask=spatial_filled_mask,
            zero_filled_mask=zero_filled_mask,
            cache_update_mask=cache_update_mask,
            method_infos=method_infos,
            info=info,
        )

    def recover_from_fec_decode(
        self,
        decode_result: Any,
        packet_metas: Optional[Sequence[Any]] = None,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        update_cache: bool = True,
        clone: bool = True,
    ) -> PartialReconstructionResult:
        """
        Run partial reconstruction directly from FECDecodeResult-like object.

        Expected fields:
            decode_result.recovered_packets
            decode_result.missing_source_mask
            decode_result.direct_received_source_mask
            decode_result.fec_recovered_source_mask
        """
        if not hasattr(decode_result, "recovered_packets"):
            raise AttributeError("decode_result should have recovered_packets.")

        if not hasattr(decode_result, "missing_source_mask"):
            raise AttributeError("decode_result should have missing_source_mask.")

        direct_mask = getattr(
            decode_result,
            "direct_received_source_mask",
            None,
        )
        fec_mask = getattr(
            decode_result,
            "fec_recovered_source_mask",
            None,
        )

        return self.recover_packets(
            packets=decode_result.recovered_packets,
            packet_metas=packet_metas,
            missing_mask=decode_result.missing_source_mask,
            direct_received_mask=direct_mask,
            fec_recovered_mask=fec_mask,
            link_id=link_id,
            frame_id=frame_id,
            update_cache=update_cache,
            clone=clone,
        )

    def recover_and_unpacketize(
        self,
        decode_result: Any,
        packetizer: Any,
        packet_metas: Sequence[Any],
        original_shape: Sequence[int],
        link_id: Any = None,
        frame_id: Optional[int] = None,
        update_cache: bool = True,
        clone: bool = True,
    ) -> PartialReconstructionResult:
        """
        Run partial reconstruction and then unpacketize to [C, H, W].

        Parameters
        ----------
        decode_result : object
            FECDecodeResult-like object.

        packetizer : object
            FeaturePacketizer-like object with unpacketize() method.

        packet_metas : sequence
            Packet metadata list.

        original_shape : sequence
            Original feature shape [C, H, W].

        Returns
        -------
        PartialReconstructionResult
            Same result as recover_from_fec_decode(), but with
            result.recovered_feature filled.
        """
        if not hasattr(packetizer, "unpacketize"):
            raise AttributeError("packetizer should have unpacketize() method.")

        result = self.recover_from_fec_decode(
            decode_result=decode_result,
            packet_metas=packet_metas,
            link_id=link_id,
            frame_id=frame_id,
            update_cache=update_cache,
            clone=clone,
        )

        recovered_feature = packetizer.unpacketize(
            result.recovered_packets,
            list(packet_metas),
            original_shape,
        )

        result.recovered_feature = recovered_feature
        result.info["recovered_feature_shape"] = tuple(
            int(x) for x in recovered_feature.shape
        )
        result.info["recovered_feature_dtype"] = str(recovered_feature.dtype)

        return result

    def clear_cache(self, link_id: Any = None) -> None:
        """
        Clear temporal cache.

        If link_id is None, clear all links.
        """
        self.temporal_cache.clear(link_id=link_id)

    def get_cache_summary(self) -> Dict[str, Any]:
        """
        Return temporal cache summary.
        """
        return self.temporal_cache.get_cache_summary()

    def get_config(self) -> Dict[str, Any]:
        """
        Return JSON-friendly partial reconstruction config.
        """
        return {
            "enabled": bool(self.enabled),
            "priority": tuple(self.priority),
            "temporal_cache": self.temporal_cache.get_config(),
            "spatial_interpolation": self.spatial_recovery.get_config(),
            "zero_fill": self.zero_recovery.get_config(),
            "update_cache_enabled": bool(self.update_cache_enabled),
            "cache_update_include_direct": bool(self.cache_update_include_direct),
            "cache_update_include_fec": bool(self.cache_update_include_fec),
            "cache_update_include_temporal": bool(self.cache_update_include_temporal),
            "cache_update_include_spatial": bool(self.cache_update_include_spatial),
            "cache_update_include_zero": bool(self.cache_update_include_zero),
            "update_cache_when_temporal_disabled": bool(
                self.update_cache_when_temporal_disabled
            ),
        }

    def __repr__(self) -> str:
        return (
            "PartialReconstructor("
            f"enabled={self.enabled}, "
            f"priority={self.priority}, "
            f"update_cache={self.update_cache_enabled})"
        )


def partial_reconstruct_from_fec_decode(
    decode_result: Any,
    packet_metas: Optional[Sequence[Any]],
    cfg: Optional[Dict[str, Any]] = None,
    link_id: Any = None,
    frame_id: Optional[int] = None,
    update_cache: bool = True,
) -> PartialReconstructionResult:
    """
    Convenience function for one-shot partial reconstruction.

    Note:
        This creates a new PartialReconstructor, so temporal cache will not
        persist across calls. For real ARCE inference, instantiate
        PartialReconstructor once and reuse it across frames.
    """
    reconstructor = PartialReconstructor(cfg or {})

    return reconstructor.recover_from_fec_decode(
        decode_result=decode_result,
        packet_metas=packet_metas,
        link_id=link_id,
        frame_id=frame_id,
        update_cache=update_cache,
    )


# Backward-compatible aliases.
PartialRecovery = PartialReconstructor
ARCEPartialReconstructor = PartialReconstructor


__all__ = [
    "PartialReconstructionResult",
    "PartialReconstructor",
    "PartialRecovery",
    "ARCEPartialReconstructor",
    "partial_reconstruct_from_fec_decode",
]