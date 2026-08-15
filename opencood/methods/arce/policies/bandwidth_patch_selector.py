"""Bandwidth-aware patch selection for final ARCE/C2MAB alignment.

Final experimental semantics:
  - Where2comm first produces a communication mask / sparse message.
  - The masked message is split into spatial packets/patches.
  - For every feasible CAV-action candidate, only top-ranked source patches
    under the per-link byte budget are allowed to enter quantization and FEC.
  - FEC is applied to selected patches only; unselected patches are missing by
    budget and are later recovered by temporal cache / spatial interpolation /
    zero-fill.

This module does not quantize, FEC encode, sample packet loss, or reconstruct
packets. It only selects source patches and estimates their encoded size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from opencood.methods.arce.policies.action_adapter import get_action_field
from opencood.communication.transport.packetization.size_estimator import (
    quant_mode_to_bits,
    estimate_redundancy_packets,
)


@dataclass
class PatchSelectionResult:
    selected_mask: torch.Tensor
    missing_mask: torch.Tensor
    valid_patch_mask: torch.Tensor
    missing_by_budget_mask: torch.Tensor
    selected_indices: List[int]
    ordered_indices: List[int]

    budget_bytes: float
    estimated_transmitted_bytes: float
    source_bytes: float
    parity_bytes: float
    metadata_bytes: float

    num_total_patches: int
    num_valid_patches: int
    num_selected_patches: int
    num_missing_by_budget: int
    num_parity_packets: int
    selected_patch_ratio: float
    effective_patch_ratio: float
    feasible: bool
    reason: str

    quant_mode: str
    quant_bits: int
    fec_type: str
    redundancy_ratio: float
    group_size: Optional[int]

    min_patch_ratio: float
    min_patch_count: int
    metadata_bytes_per_packet: float
    mask_threshold: float
    lambda_mask: float
    lambda_activation: float
    lambda_complementarity: float
    complementarity: float

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Masks can be long; keep them for debugging and exact replay.
        for key in (
            "selected_mask",
            "missing_mask",
            "valid_patch_mask",
            "missing_by_budget_mask",
        ):
            value = getattr(self, key)
            d[key] = value.detach().cpu().to(torch.bool).tolist()
        return d


class BandwidthAwarePatchSelector:
    """Select important source patches under a per-frame/per-link byte budget."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        root = cfg or {}
        scheduler_cfg = root.get("scheduler", {}) if isinstance(root.get("scheduler", {}), dict) else {}
        patch_cfg = root.get("patch_selection", {}) if isinstance(root.get("patch_selection", {}), dict) else {}

        # Backward-compatible flat / scheduler keys, then final patch_selection keys.
        merged: Dict[str, Any] = {}
        merged.update(scheduler_cfg)
        merged.update({k: v for k, v in root.items() if k in (
            "patch_selector", "min_patch_ratio", "min_patch_count",
            "metadata_bytes_per_packet", "strict_min_patch",
        )})
        merged.update(patch_cfg)

        self.enabled = bool(merged.get("enabled", True))
        self.selector = str(merged.get("patch_selector", merged.get("ranking", "score_per_byte"))).strip().lower()
        self.source = str(merged.get("source", "where2comm_mask")).strip().lower()
        self.min_patch_ratio = float(merged.get("min_patch_ratio", merged.get("min_effective_patch_ratio", 0.2)))
        self.min_patch_count = int(merged.get("min_patch_count", 1))
        self.metadata_bytes_per_packet = float(merged.get("metadata_bytes_per_packet", 8.0))
        self.strict_min_patch = bool(merged.get("strict_min_patch", True))
        self.mask_threshold = float(merged.get("mask_threshold", 0.05))

        score_cfg = merged.get("score", {}) if isinstance(merged.get("score", {}), dict) else {}
        self.lambda_mask = float(merged.get("lambda_mask", score_cfg.get("lambda_mask", 1.0)))
        self.lambda_activation = float(merged.get("lambda_activation", score_cfg.get("lambda_activation", 0.2)))
        self.lambda_complementarity = float(merged.get("lambda_complementarity", score_cfg.get("lambda_complementarity", 0.3)))

        if self.selector not in ("score_per_byte", "activation_topk", "mask_topk"):
            raise ValueError(
                f"Unsupported patch_selector/ranking={self.selector}. "
                "Supported: score_per_byte, activation_topk, mask_topk."
            )

    @staticmethod
    def _action_get(action: Any, name: str, default: Any = None) -> Any:
        return get_action_field(action, name, default)

    @staticmethod
    def _normalize_message_mask(
        message_mask: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if message_mask is None:
            return None
        mask = message_mask
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.to(device=device, dtype=dtype)
        # Accept [H,W], [1,H,W], [C,H,W], [N,1,H,W] with N=1.
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif mask.dim() == 3:
            if mask.shape[0] != 1:
                mask = mask.mean(dim=0, keepdim=True)
            mask = mask.unsqueeze(0)  # [1,1,H,W]
        elif mask.dim() == 4:
            if mask.shape[0] != 1:
                mask = mask[:1]
            if mask.shape[1] != 1:
                mask = mask.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported message_mask shape {tuple(mask.shape)}")

        if tuple(mask.shape[-2:]) != tuple(target_hw):
            mask = F.interpolate(mask, size=target_hw, mode="bilinear", align_corners=False)
        return mask.clamp(0.0, 1.0)[0, 0]  # [H,W]

    @staticmethod
    def _patch_mask_scores(
        metas: Sequence[Any],
        message_mask_hw: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        vals: List[float] = []
        for m in metas:
            if message_mask_hw is None:
                vals.append(1.0)
                continue
            patch = message_mask_hw[int(m.h_start): int(m.h_end), int(m.w_start): int(m.w_end)]
            if patch.numel() == 0:
                vals.append(0.0)
            else:
                vals.append(float(patch.mean().detach().cpu().item()))
        return torch.tensor(vals, device=device, dtype=dtype)

    def compute_activation_scores(self, packets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute mean(abs(packet)) over valid spatial cells."""
        if packets.dim() != 4:
            raise ValueError(f"packets should be [M,C,ph,pw], got {tuple(packets.shape)}")
        if valid_mask.dim() != 4:
            raise ValueError(f"valid_mask should be [M,1,ph,pw], got {tuple(valid_mask.shape)}")
        if int(valid_mask.shape[0]) != int(packets.shape[0]):
            raise ValueError("valid_mask and packets have different packet counts")
        mask = valid_mask.to(device=packets.device, dtype=packets.dtype)
        masked_abs = packets.abs() * mask
        denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * float(packets.shape[1])
        return masked_abs.sum(dim=(1, 2, 3)) / denom

    @staticmethod
    def estimate_source_packet_bytes(metas: Sequence[Any], channels: int, quant_bits: int) -> List[float]:
        out = []
        for m in metas:
            valid_h = int(getattr(m, "valid_h"))
            valid_w = int(getattr(m, "valid_w"))
            bits = int(channels * valid_h * valid_w * quant_bits)
            out.append(float(int(math.ceil(bits / 8.0))))
        return out

    def _estimate_total_for_indices(
        self,
        indices: Sequence[int],
        source_packet_bytes: Sequence[float],
        fec_type: str,
        redundancy_ratio: float,
        group_size: Optional[int],
    ) -> Dict[str, float]:
        k = int(len(indices))
        if k <= 0:
            return {"source_bytes": 0.0, "parity_bytes": 0.0, "metadata_bytes": 0.0, "total_bytes": 0.0, "num_parity_packets": 0}
        source_bytes = float(sum(float(source_packet_bytes[i]) for i in indices))
        avg_source_packet_bytes = source_bytes / float(k)
        parity_packets = int(estimate_redundancy_packets(
            num_source_packets=k,
            fec_type=fec_type,
            redundancy_ratio=redundancy_ratio,
            group_size=group_size,
        ))
        parity_bytes = float(parity_packets * avg_source_packet_bytes)
        metadata_bytes = float((k + parity_packets) * self.metadata_bytes_per_packet)
        total_bytes = float(source_bytes + parity_bytes + metadata_bytes)
        return {
            "source_bytes": source_bytes,
            "parity_bytes": parity_bytes,
            "metadata_bytes": metadata_bytes,
            "total_bytes": total_bytes,
            "num_parity_packets": parity_packets,
        }

    def select(
        self,
        packetization_result: Any,
        action: Any,
        budget_bytes: float,
        message_mask: Optional[torch.Tensor] = None,
        complementarity: float = 0.0,
    ) -> PatchSelectionResult:
        packets = packetization_result.packets
        valid_mask = packetization_result.valid_mask
        metas = list(packetization_result.metas)

        num_total = int(packetization_result.num_packets)
        device = packets.device
        dtype = packets.dtype

        quant_mode = str(self._action_get(action, "quant_mode", "fp16")).strip().lower()
        quant_bits = int(quant_mode_to_bits(quant_mode))
        fec_type = str(self._action_get(action, "fec_type", "none")).strip().lower()
        redundancy_ratio = float(self._action_get(action, "redundancy_ratio", 0.0))
        group_size = self._action_get(action, "xor_group_size", self._action_get(action, "group_size", None))
        if group_size is not None:
            group_size = int(group_size)

        budget_bytes = float(max(0.0, budget_bytes))
        original_shape = tuple(int(x) for x in packetization_result.original_shape)
        if len(original_shape) == 4:
            _, _, H, W = original_shape
        elif len(original_shape) == 3:
            _, H, W = original_shape
        else:
            raise ValueError(
                "packetization_result.original_shape should be [C,H,W] or [B,C,H,W], "
                f"got {original_shape}."
            )
        message_mask_hw = self._normalize_message_mask(message_mask, (H, W), device, dtype)

        activation_scores = self.compute_activation_scores(packets, valid_mask)
        mask_scores = self._patch_mask_scores(metas, message_mask_hw, device, dtype)
        valid_patch_mask = mask_scores >= float(self.mask_threshold)
        # If no Where2comm mask is given, all patches are valid.
        if message_mask_hw is None:
            valid_patch_mask = torch.ones(num_total, device=device, dtype=torch.bool)

        comp = float(complementarity)
        raw_scores = (
            float(self.lambda_mask) * mask_scores
            + float(self.lambda_activation) * activation_scores
            + float(self.lambda_complementarity) * comp
        )
        raw_scores = torch.where(valid_patch_mask, raw_scores, torch.full_like(raw_scores, -1e30))

        channels = int(packets.shape[1])
        source_packet_bytes = self.estimate_source_packet_bytes(metas, channels, quant_bits)
        per_packet_cost = torch.tensor(source_packet_bytes, device=device, dtype=dtype).clamp_min(1.0)

        if self.selector == "activation_topk":
            rank_scores = torch.where(valid_patch_mask, activation_scores, torch.full_like(activation_scores, -1e30))
        elif self.selector == "mask_topk":
            rank_scores = torch.where(valid_patch_mask, mask_scores, torch.full_like(mask_scores, -1e30))
        else:
            rank_scores = raw_scores / per_packet_cost

        ordered = torch.argsort(rank_scores, descending=True).detach().cpu().tolist()
        ordered = [int(i) for i in ordered if bool(valid_patch_mask[int(i)].detach().cpu().item())]

        selected: List[int] = []
        last_est = {"source_bytes": 0.0, "parity_bytes": 0.0, "metadata_bytes": 0.0, "total_bytes": 0.0, "num_parity_packets": 0}
        for idx in ordered:
            candidate = selected + [int(idx)]
            est = self._estimate_total_for_indices(candidate, source_packet_bytes, fec_type, redundancy_ratio, group_size)
            if float(est["total_bytes"]) <= budget_bytes:
                selected = candidate
                last_est = est

        num_valid = int(valid_patch_mask.sum().detach().cpu().item())
        selected_ratio = float(len(selected) / max(1, num_valid))
        effective_ratio = selected_ratio
        min_count_ok = int(len(selected)) >= int(self.min_patch_count)
        min_ratio_ok = effective_ratio >= float(self.min_patch_ratio)
        feasible = bool(num_valid > 0 and min_count_ok and min_ratio_ok)

        reason = "ok"
        if not feasible:
            if num_valid <= 0:
                reason = "no_valid_patch"
            elif not min_count_ok:
                reason = f"selected_patches<{self.min_patch_count}"
            elif not min_ratio_ok:
                reason = f"selected_ratio<{self.min_patch_ratio}"
            if self.strict_min_patch:
                selected = []
                last_est = {"source_bytes": 0.0, "parity_bytes": 0.0, "metadata_bytes": 0.0, "total_bytes": 0.0, "num_parity_packets": 0}
                selected_ratio = 0.0
                effective_ratio = 0.0

        selected_mask = torch.zeros(num_total, dtype=torch.bool, device=device)
        if selected:
            selected_mask[torch.as_tensor(selected, dtype=torch.long, device=device)] = True
        missing_mask = ~selected_mask
        missing_by_budget_mask = valid_patch_mask & (~selected_mask)

        return PatchSelectionResult(
            selected_mask=selected_mask,
            missing_mask=missing_mask,
            valid_patch_mask=valid_patch_mask.to(torch.bool),
            missing_by_budget_mask=missing_by_budget_mask.to(torch.bool),
            selected_indices=[int(x) for x in selected],
            ordered_indices=[int(x) for x in ordered],
            budget_bytes=float(budget_bytes),
            estimated_transmitted_bytes=float(last_est["total_bytes"]),
            source_bytes=float(last_est["source_bytes"]),
            parity_bytes=float(last_est["parity_bytes"]),
            metadata_bytes=float(last_est["metadata_bytes"]),
            num_total_patches=int(num_total),
            num_valid_patches=int(num_valid),
            num_selected_patches=int(len(selected)),
            num_missing_by_budget=int(missing_by_budget_mask.sum().detach().cpu().item()),
            num_parity_packets=int(last_est["num_parity_packets"]),
            selected_patch_ratio=float(selected_ratio),
            effective_patch_ratio=float(effective_ratio),
            feasible=bool(feasible and (len(selected) > 0)),
            reason=reason if len(selected) == 0 else "ok",
            quant_mode=quant_mode,
            quant_bits=int(quant_bits),
            fec_type=fec_type,
            redundancy_ratio=float(redundancy_ratio),
            group_size=group_size,
            min_patch_ratio=float(self.min_patch_ratio),
            min_patch_count=int(self.min_patch_count),
            metadata_bytes_per_packet=float(self.metadata_bytes_per_packet),
            mask_threshold=float(self.mask_threshold),
            lambda_mask=float(self.lambda_mask),
            lambda_activation=float(self.lambda_activation),
            lambda_complementarity=float(self.lambda_complementarity),
            complementarity=float(comp),
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "patch_selector": self.selector,
            "source": self.source,
            "min_patch_ratio": float(self.min_patch_ratio),
            "min_patch_count": int(self.min_patch_count),
            "metadata_bytes_per_packet": float(self.metadata_bytes_per_packet),
            "strict_min_patch": bool(self.strict_min_patch),
            "mask_threshold": float(self.mask_threshold),
            "lambda_mask": float(self.lambda_mask),
            "lambda_activation": float(self.lambda_activation),
            "lambda_complementarity": float(self.lambda_complementarity),
        }
