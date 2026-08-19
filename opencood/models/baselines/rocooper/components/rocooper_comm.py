# -*- coding: utf-8 -*-
"""
RoCooper communication impairment module for OpenCOOD.

Save as:
    opencood/models/baselines/rocooper/components/rocooper_comm.py

This module simulates the lossy V2V transmission stage in RoCooper:
    1. Wireless channel fading:
        - Rician fading
        - Additive white Gaussian noise

    2. Network performance degradation:
        - Bandwidth limitation / resolution reduction
        - Packet loss by zeroing transmitted feature blocks
        - Transmission delay
        - Communication disruption / frame drop

Important design choice:
    The ego vehicle feature is kept lossless by default. Only non-ego CAV
    features are impaired, because RoCooper uses lossless ego feature as the
    anchoring foundation for feature recovery and fusion.

Expected input:
    features: torch.Tensor
        Shape [sum(record_len), C, H, W]

    record_len: torch.Tensor or list
        Number of CAVs in each scenario within the batch.
        The first CAV in each scenario is treated as ego.

    pairwise_t_matrix: torch.Tensor, optional
        Usually shape [B, max_cav, max_cav, 4, 4] in OpenCOOD.
        Used to estimate ego-CAV distance for distance-aware channel fading.

Return:
    impaired_features, comm_info
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.communication.channel.channel_manager import ChannelManager


TensorOrNone = Optional[torch.Tensor]


class RoCooperComm(nn.Module):
    """
    Communication impairment simulator for RoCooper.

    Recommended yaml block:

        rocooper_comm:
          enabled: true
          impair_ego: false
          communication_range: 70

          train_with_impairment: true
          test_with_impairment: true

          channel_fading:
            enabled: true
            type: "rician"
            snr_db: 15.0
            p0: 1.0
            path_loss_exponent: 2.0
            mu_h_real: 1.0
            mu_h_imag: 0.0
            sigma_h: 0.1
            sigma_w: 0.01
            distance_aware: true

          network_loss:
            enabled: true

            bandwidth_limit:
              enabled: true
              mean: 0.10
              std: 0.05
              compression_ratio: 1
              mode: "resolution_reduction"

            packet_loss:
              enabled: true
              mean: 0.15
              std: 0.05
              granularity: "block"
              zero_fraction: 0.15
              block_size: 4

            delay:
              enabled: true
              mean_ms: 60
              std_ms: 10
              frame_interval_ms: 100
              max_delay_frames: 3

            frame_drop:
              enabled: true
              mean: 0.05
              std: 0.05
              drop_whole_cav_feature: true
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super(RoCooperComm, self).__init__()

        self.cfg = cfg or {}

        self.enabled = bool(self.cfg.get("enabled", True))
        self.impair_ego = bool(self.cfg.get("impair_ego", False))

        self.train_with_impairment = bool(
            self.cfg.get("train_with_impairment", True)
        )
        self.test_with_impairment = bool(
            self.cfg.get("test_with_impairment", True)
        )

        self.communication_range = float(
            self.cfg.get("communication_range", 70.0)
        )
        self.drop_beyond_range = bool(
            self.cfg.get("drop_beyond_range", False)
        )

        self.return_masks = bool(self.cfg.get("return_masks", False))

        # Measurement-only wire accounting. This does not alter impairment.
        wire_cfg = self.cfg.get("wire_audit", {}) or {}
        self.wire_packet_size_bytes = max(1, int(wire_cfg.get("packet_size_bytes", 1024)))
        self.wire_bytes_per_value = int(wire_cfg.get("bytes_per_value", 0))

        # Channel fading config
        self.channel_cfg = self.cfg.get("channel_fading", {}) or {}
        self.channel_enabled = bool(self.channel_cfg.get("enabled", False))

        # Network loss config
        self.network_cfg = self.cfg.get("network_loss", {}) or {}
        self.network_enabled = bool(self.network_cfg.get("enabled", False))

        self.bandwidth_cfg = self.network_cfg.get("bandwidth_limit", {}) or {}
        self.packet_cfg = self.network_cfg.get("packet_loss", {}) or {}
        self.delay_cfg = self.network_cfg.get("delay", {}) or {}
        self.frame_drop_cfg = self.network_cfg.get("frame_drop", {}) or {}

        # VCQI config. This module can report an approximate VCQI computed
        # from configured mean impairment values. The actual feature impairment
        # below is controlled by the explicit channel/network configs.
        self.use_vcqi = bool(self.cfg.get("use_vcqi", False))
        self.vcqi_cfg = self.cfg.get("vcqi", {}) or {}

        # Delay history queue.
        # Each item is a detached feature tensor from a previous forward pass.
        self._history_queue: List[torch.Tensor] = []

        # Set only by the experiment-level adapter.  Keeping this optional
        # preserves legacy checkpoints, while new experiments take all
        # physical parameters from ChannelManager rather than this baseline.
        self.channel_manager: Optional[ChannelManager] = None
        self._public_channel_profile: Optional[Dict[str, Any]] = None
        self._public_channel_state: Optional[str] = None
        self._public_channel_frame_id: Optional[int] = None
        # Append-only audit stream.  The normal ``comm_info`` remains the
        # frame-local baseline interface; this export is read-only metadata.
        self.records: List[Dict[str, Any]] = []

    def set_channel_manager(self, channel_manager: ChannelManager) -> None:
        """Receive the experiment-owned physical channel (no reset here)."""
        if not isinstance(channel_manager, ChannelManager):
            raise TypeError("channel_manager must be a ChannelManager instance")
        self.channel_manager = channel_manager
        # Fading and frame-drop are RoCooper-specific disturbance models, not
        # part of the shared physical channel contract.  Disable them once an
        # experiment manager is injected so every varying condition comes
        # from the public bandwidth/loss/delay profile.
        self.channel_enabled = False
        self.frame_drop_cfg["enabled"] = False

    def set_channel_context(
        self,
        channel_manager: ChannelManager,
        profile: Dict[str, Any],
        state_name: str,
        frame_id: int,
    ) -> None:
        """Set the public profile selected by the thin RoCooper adapter."""
        self.set_channel_manager(channel_manager)
        self._public_channel_profile = dict(profile)
        self._public_channel_state = str(state_name)
        self._public_channel_frame_id = int(frame_id)

    def get_records(self) -> List[Dict[str, Any]]:
        return list(self.records)

    def reset_records(self) -> None:
        self.records = []

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features: torch.Tensor,
        record_len: Union[torch.Tensor, List[int], Tuple[int, ...]],
        pairwise_t_matrix: TensorOrNone = None,
        data_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply communication impairments to non-ego CAV features.

        Args:
            features:
                Tensor with shape [sum(record_len), C, H, W].

            record_len:
                Number of CAVs in each batch scenario.

            pairwise_t_matrix:
                Optional pairwise transform matrix used to estimate distance.

            data_dict:
                Optional OpenCOOD batch dict. Currently unused but kept for
                compatibility with the top-level model.

        Returns:
            impaired_features:
                Tensor with the same shape as input features.

            comm_info:
                Dictionary containing impairment statistics.
        """
        del data_dict

        if not self._should_apply_impairment():
            return features, {
                "enabled": False,
                "reason": "impairment_disabled_or_inactive_for_mode",
            }

        if features.dim() != 4:
            raise ValueError(
                "RoCooperComm expects features with shape [N, C, H, W], "
                f"but got shape {tuple(features.shape)}."
            )

        device = features.device
        dtype = features.dtype

        record_len_list = self._record_len_to_list(record_len)
        num_cav = features.shape[0]

        if sum(record_len_list) != num_cav:
            raise ValueError(
                "sum(record_len) must equal features.shape[0]. "
                f"sum(record_len)={sum(record_len_list)}, "
                f"features.shape[0]={num_cav}."
            )

        ego_mask = self._make_ego_mask(record_len_list, device=device)
        impair_mask = ~ego_mask if not self.impair_ego else torch.ones_like(ego_mask)

        distances = self._estimate_distances(
            record_len_list=record_len_list,
            pairwise_t_matrix=pairwise_t_matrix,
            device=device,
            dtype=dtype,
        )

        if self.drop_beyond_range:
            beyond_range_mask = distances > self.communication_range
            impair_mask = impair_mask | beyond_range_mask
        else:
            beyond_range_mask = torch.zeros_like(impair_mask)

        x = features.clone()

        comm_info: Dict[str, Any] = {
            "enabled": True,
            "num_total_cav": int(num_cav),
            "num_ego": int(ego_mask.sum().item()),
            "num_non_ego": int((~ego_mask).sum().item()),
            "num_impaired_candidates": int(impair_mask.sum().item()),
            "communication_range": self.communication_range,
        }

        if self.use_vcqi:
            comm_info["approx_vcqi"] = self.compute_config_vcqi()

        if self.return_masks:
            comm_info["ego_mask"] = ego_mask.detach()
            comm_info["impair_mask"] = impair_mask.detach()
            comm_info["distance"] = distances.detach()
            comm_info["beyond_range_mask"] = beyond_range_mask.detach()

        # --------------------------------------------------------------
        # 1. Bandwidth limitation / resolution reduction
        # --------------------------------------------------------------
        x, bandwidth_info = self._apply_bandwidth_limit(x, impair_mask)
        comm_info.update(bandwidth_info)

        wire_records = self._build_wire_records(
            features=features,
            record_len_list=record_len_list,
            bandwidth_info=bandwidth_info,
            beyond_range_mask=beyond_range_mask,
        )
        comm_info["wire_records"] = wire_records
        comm_info["source_payload_bytes"] = int(sum(
            int(r.get("source_payload_bytes", 0)) for r in wire_records
        ))
        comm_info["transmitted_wire_bytes"] = int(sum(
            int(r.get("transmitted_wire_bytes", 0)) for r in wire_records
        ))
        comm_info["received_wire_bytes"] = None
        for wire_record in wire_records:
            audit_record = dict(wire_record)
            audit_record["state"] = (
                self._public_channel_state
                or audit_record.get("state", "rocooper_configured_channel")
            )
            audit_record["frame_id"] = self._public_channel_frame_id
            self.records.append(audit_record)

        # --------------------------------------------------------------
        # 2. Wireless channel fading + AWGN
        # --------------------------------------------------------------
        x, channel_info = self._apply_channel_fading(
            x=x,
            impair_mask=impair_mask,
            distances=distances,
        )
        comm_info.update(channel_info)

        # --------------------------------------------------------------
        # 3. Packet loss: zero transmitted feature blocks
        # --------------------------------------------------------------
        x, packet_info = self._apply_packet_loss(x, impair_mask)
        comm_info.update(packet_info)

        # --------------------------------------------------------------
        # 4. Delay: replace current feature with buffered previous feature
        # --------------------------------------------------------------
        x, delay_info = self._apply_delay(x, impair_mask)
        comm_info.update(delay_info)

        # --------------------------------------------------------------
        # 5. Frame drop / communication disruption
        # --------------------------------------------------------------
        x, frame_drop_info = self._apply_frame_drop(x, impair_mask)
        comm_info.update(frame_drop_info)

        # --------------------------------------------------------------
        # 6. Drop beyond communication range, optional
        # --------------------------------------------------------------
        if self.drop_beyond_range and beyond_range_mask.any():
            x[beyond_range_mask] = 0
            comm_info["num_beyond_range_dropped"] = int(
                beyond_range_mask.sum().item()
            )
        else:
            comm_info["num_beyond_range_dropped"] = 0

        return x, comm_info

    def _build_wire_records(
        self,
        features: torch.Tensor,
        record_len_list: List[int],
        bandwidth_info: Dict[str, Any],
        beyond_range_mask: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """Build per-link pre-loss wire records for the current RoCooper model.

        Resolution-reduction events transmit the low-resolution tensor; fading,
        block loss, delay, and frame drop happen after bytes enter the link and
        therefore do not reduce transmitted_wire_bytes.
        """
        if features.dim() != 4:
            return []
        _, channels, height, width = features.shape
        bytes_per_value = (
            int(self.wire_bytes_per_value)
            if int(self.wire_bytes_per_value) > 0
            else int(features.element_size())
        )
        event_indices = set(int(v) for v in bandwidth_info.get(
            "bandwidth_event_indices", []
        ))
        low_h = int(bandwidth_info.get("bandwidth_low_height", height))
        low_w = int(bandwidth_info.get("bandwidth_low_width", width))

        records: List[Dict[str, Any]] = []
        global_idx = 0
        for batch_idx, cav_num in enumerate(record_len_list):
            for local_idx in range(int(cav_num)):
                idx = global_idx + local_idx
                if local_idx == 0 and not self.impair_ego:
                    continue
                source_payload = int(channels * height * width * bytes_per_value)
                dropped_by_range = bool(
                    torch.is_tensor(beyond_range_mask)
                    and idx < beyond_range_mask.numel()
                    and bool(beyond_range_mask[idx].item())
                    and self.drop_beyond_range
                )
                if dropped_by_range:
                    tx_payload = 0
                elif idx in event_indices:
                    tx_payload = int(channels * low_h * low_w * bytes_per_value)
                else:
                    tx_payload = source_payload
                tx_packets = (
                    int(math.ceil(tx_payload / float(self.wire_packet_size_bytes)))
                    if tx_payload > 0 else 0
                )
                records.append({
                    "batch": int(batch_idx),
                    "cav": int(local_idx),
                    "sender_id": int(local_idx),
                    "global_idx": int(idx),
                    "link_key": "b{}_cav{}".format(batch_idx, local_idx),
                    "state": "rocooper_configured_channel",
                    "kind": "rocooper_wire_audit",
                    "source_payload_bytes": source_payload,
                    "tx_payload_bytes": tx_payload,
                    "packet_size_bytes": int(self.wire_packet_size_bytes),
                    "num_source_packets": (
                        int(math.ceil(source_payload / float(self.wire_packet_size_bytes)))
                        if source_payload > 0 else 0
                    ),
                    "num_transmitted_packets": tx_packets,
                    "transmitted_wire_bytes": int(
                        tx_packets * self.wire_packet_size_bytes
                    ),
                    "received_wire_bytes": None,
                    "bandwidth_resolution_reduced": bool(idx in event_indices),
                    "dropped_before_send_beyond_range": dropped_by_range,
                    "channels": int(channels),
                    "height": int(height),
                    "width": int(width),
                    "transmitted_height": int(low_h if idx in event_indices else height),
                    "transmitted_width": int(low_w if idx in event_indices else width),
                })
            global_idx += int(cav_num)
        return records

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    def _should_apply_impairment(self) -> bool:
        if not self.enabled:
            return False

        if self.training and not self.train_with_impairment:
            return False

        if (not self.training) and not self.test_with_impairment:
            return False

        return True

    # ------------------------------------------------------------------
    # Record length / ego mask / distance
    # ------------------------------------------------------------------

    @staticmethod
    def _record_len_to_list(
        record_len: Union[torch.Tensor, List[int], Tuple[int, ...]]
    ) -> List[int]:
        if isinstance(record_len, torch.Tensor):
            return [int(v) for v in record_len.detach().cpu().tolist()]
        return [int(v) for v in record_len]

    @staticmethod
    def _make_ego_mask(
        record_len_list: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        masks = []
        for num in record_len_list:
            if num <= 0:
                raise ValueError(f"Invalid record_len value: {num}")
            local = torch.zeros(num, dtype=torch.bool, device=device)
            local[0] = True
            masks.append(local)
        return torch.cat(masks, dim=0)

    def _estimate_distances(
        self,
        record_len_list: List[int],
        pairwise_t_matrix: TensorOrNone,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Estimate ego-CAV distances.

        OpenCOOD pairwise_t_matrix usually has shape:
            [B, max_cav, max_cav, 4, 4]

        The exact direction of the transform may vary across OpenCOOD versions.
        For distance estimation, we try both [ego, cav] and [cav, ego], then
        choose a valid non-zero norm.
        """
        total_cav = sum(record_len_list)
        distances = torch.ones(total_cav, device=device, dtype=dtype)

        if pairwise_t_matrix is None:
            return distances

        if not isinstance(pairwise_t_matrix, torch.Tensor):
            return distances

        if pairwise_t_matrix.dim() != 5:
            return distances

        start = 0
        batch_size = len(record_len_list)

        for b in range(batch_size):
            num = record_len_list[b]

            if b >= pairwise_t_matrix.shape[0]:
                break

            max_cav_in_matrix = pairwise_t_matrix.shape[1]

            for local_idx in range(num):
                global_idx = start + local_idx

                if local_idx == 0:
                    distances[global_idx] = 1.0
                    continue

                if local_idx >= max_cav_in_matrix:
                    distances[global_idx] = 1.0
                    continue

                cand_norms: List[torch.Tensor] = []

                try:
                    t_ego_to_cav = pairwise_t_matrix[b, 0, local_idx, :3, 3]
                    norm_1 = torch.norm(t_ego_to_cav)
                    cand_norms.append(norm_1)
                except Exception:
                    pass

                try:
                    t_cav_to_ego = pairwise_t_matrix[b, local_idx, 0, :3, 3]
                    norm_2 = torch.norm(t_cav_to_ego)
                    cand_norms.append(norm_2)
                except Exception:
                    pass

                valid_norms = [
                    n for n in cand_norms
                    if torch.isfinite(n).item() and n.item() > 1e-6
                ]

                if len(valid_norms) == 0:
                    distances[global_idx] = 1.0
                else:
                    distances[global_idx] = valid_norms[0].to(device=device, dtype=dtype)

            start += num

        return distances.clamp(min=1.0)

    # ------------------------------------------------------------------
    # Random helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_normal_clamped(
        mean: float,
        std: float,
        shape: Tuple[int, ...],
        device: torch.device,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> torch.Tensor:
        if std <= 0:
            out = torch.full(shape, float(mean), device=device, dtype=torch.float32)
        else:
            out = torch.normal(
                mean=float(mean),
                std=float(std),
                size=shape,
                device=device,
            )

        return out.clamp(min=min_value, max=max_value)

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    # ------------------------------------------------------------------
    # Bandwidth limitation
    # ------------------------------------------------------------------

    def _apply_bandwidth_limit(
        self,
        x: torch.Tensor,
        impair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cfg = self.bandwidth_cfg
        public_profile = self._public_channel_profile
        enabled = self.network_enabled and bool(cfg.get("enabled", False))

        info: Dict[str, Any] = {
            "bandwidth_enabled": enabled,
            "num_bandwidth_limited": 0,
            "bandwidth_event_indices": [],
            "bandwidth_low_height": int(x.shape[-2]),
            "bandwidth_low_width": int(x.shape[-1]),
        }

        if not enabled:
            return x, info

        mean = self._as_float(cfg.get("mean", 0.0), 0.0)
        std = self._as_float(cfg.get("std", 0.0), 0.0)
        compression_ratio = max(
            1.0,
            self._as_float(cfg.get("compression_ratio", 1.0), 1.0),
        )
        mode = str(cfg.get("mode", "resolution_reduction"))

        # Under the public environment the per-frame byte budget is derived
        # solely from its bandwidth profile.  The original RoCooper operation
        # (spatial down/up-sampling) remains the payload codec.
        if public_profile is not None:
            bandwidth_mbps = max(0.0, float(public_profile["bandwidth_mbps"]))
            frame_interval_ms = max(
                1e-6,
                float(self.channel_manager.channel_cfg.get("frame_interval_ms", 100.0)),
            )
            budget_bytes = bandwidth_mbps * 1_000_000.0 / 8.0 * frame_interval_ms / 1000.0
            payload_bytes = float(x[0].numel() * x.element_size()) if x.shape[0] else 0.0
            compression_ratio = max(1.0, payload_bytes / max(budget_bytes, 1.0))
            mean, std = 1.0, 0.0
            info["bandwidth_mbps"] = bandwidth_mbps
            info["frame_budget_bytes"] = budget_bytes

        if compression_ratio <= 1.0:
            info["bandwidth_compression_ratio"] = compression_ratio
            return x, info

        num_cav, _, height, width = x.shape
        probs = self._sample_normal_clamped(
            mean=mean,
            std=std,
            shape=(num_cav,),
            device=x.device,
            min_value=0.0,
            max_value=1.0,
        )

        event_mask = torch.rand(num_cav, device=x.device) < probs
        event_mask = event_mask & impair_mask

        if not event_mask.any():
            info["bandwidth_compression_ratio"] = compression_ratio
            return x, info

        if mode != "resolution_reduction":
            # Fallback to resolution reduction for unknown modes.
            mode = "resolution_reduction"

        # Approximate compression ratio by reducing spatial resolution by
        # sqrt(compression_ratio) along H and W, then upsampling back.
        scale = math.sqrt(compression_ratio)
        new_h = max(1, int(round(height / scale)))
        new_w = max(1, int(round(width / scale)))

        selected = x[event_mask]

        # Downsample and upsample. This keeps shape unchanged but removes
        # high-frequency spatial details, approximating bandwidth pressure.
        selected_low = F.interpolate(
            selected,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        selected_restored = F.interpolate(
            selected_low,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        x = x.clone()
        x[event_mask] = selected_restored

        info["num_bandwidth_limited"] = int(event_mask.sum().item())
        info["bandwidth_event_indices"] = [
            int(v) for v in torch.nonzero(event_mask, as_tuple=False).view(-1).detach().cpu().tolist()
        ]
        info["bandwidth_low_height"] = int(new_h)
        info["bandwidth_low_width"] = int(new_w)
        info["bandwidth_prob_mean"] = mean
        info["bandwidth_prob_std"] = std
        info["bandwidth_compression_ratio"] = compression_ratio
        info["bandwidth_mode"] = mode

        if self.return_masks:
            info["bandwidth_event_mask"] = event_mask.detach()

        return x, info

    # ------------------------------------------------------------------
    # Channel fading
    # ------------------------------------------------------------------

    def _apply_channel_fading(
        self,
        x: torch.Tensor,
        impair_mask: torch.Tensor,
        distances: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cfg = self.channel_cfg
        enabled = self.channel_enabled

        info: Dict[str, Any] = {
            "channel_fading_enabled": enabled,
        }

        if not enabled:
            return x, info

        channel_type = str(cfg.get("type", "rician")).lower()
        if channel_type not in ["rician", "rayleigh", "none"]:
            channel_type = "rician"

        if channel_type == "none":
            return x, info

        num_cav = x.shape[0]
        device = x.device
        dtype = x.dtype

        selected_mask = impair_mask
        if not selected_mask.any():
            info["num_channel_impaired"] = 0
            return x, info

        snr_db = self._as_float(cfg.get("snr_db", 15.0), 15.0)
        snr_linear = 10.0 ** (snr_db / 10.0)

        p0 = self._as_float(cfg.get("p0", 1.0), 1.0)
        path_loss_exponent = self._as_float(
            cfg.get("path_loss_exponent", 2.0),
            2.0,
        )
        distance_aware = bool(cfg.get("distance_aware", True))

        mu_h_real = self._as_float(cfg.get("mu_h_real", 1.0), 1.0)
        mu_h_imag = self._as_float(cfg.get("mu_h_imag", 0.0), 0.0)
        sigma_h = self._as_float(cfg.get("sigma_h", 0.1), 0.1)
        sigma_w_cfg = self._as_float(cfg.get("sigma_w", 0.01), 0.01)

        # Prevent the path-loss term from numerically destroying all feature
        # magnitudes in early debugging. Users can override these clamps.
        min_gain = self._as_float(cfg.get("min_gain", 0.05), 0.05)
        max_gain = self._as_float(cfg.get("max_gain", 10.0), 10.0)

        # Rician channel coefficient h ~ CN(mu_h, sigma_h^2).
        # We apply the magnitude of h as a real-valued multiplicative factor
        # because BEV features are real tensors.
        if channel_type == "rician":
            real = torch.normal(
                mean=mu_h_real,
                std=sigma_h,
                size=(num_cav, 1, 1, 1),
                device=device,
            )
            imag = torch.normal(
                mean=mu_h_imag,
                std=sigma_h,
                size=(num_cav, 1, 1, 1),
                device=device,
            )
            h_abs = torch.sqrt(real * real + imag * imag).to(dtype=dtype)
        else:
            # Rayleigh fallback: zero-mean complex Gaussian magnitude.
            real = torch.normal(
                mean=0.0,
                std=sigma_h,
                size=(num_cav, 1, 1, 1),
                device=device,
            )
            imag = torch.normal(
                mean=0.0,
                std=sigma_h,
                size=(num_cav, 1, 1, 1),
                device=device,
            )
            h_abs = torch.sqrt(real * real + imag * imag).to(dtype=dtype)

        if distance_aware:
            d = distances.view(num_cav, 1, 1, 1).to(device=device, dtype=dtype)
            path_gain = torch.sqrt(
                torch.tensor(p0, device=device, dtype=dtype)
                / torch.pow(d.clamp(min=1.0), path_loss_exponent)
            )
            path_gain = path_gain.clamp(min=min_gain, max=max_gain)
        else:
            path_gain = torch.ones(
                (num_cav, 1, 1, 1),
                device=device,
                dtype=dtype,
            )

        coefficient = path_gain * h_abs

        x = x.clone()

        selected_x = x[selected_mask]
        selected_coef = coefficient[selected_mask]

        # Feature-power-based AWGN. The configured sigma_w acts as a floor.
        with torch.no_grad():
            power = selected_x.float().pow(2).mean(dim=(1, 2, 3), keepdim=True)
            noise_std_from_snr = torch.sqrt(power / max(snr_linear, 1e-6))
            sigma_w_tensor = torch.full_like(noise_std_from_snr, sigma_w_cfg)
            noise_std = torch.maximum(noise_std_from_snr, sigma_w_tensor)

        noise = torch.randn_like(selected_x) * noise_std.to(dtype=selected_x.dtype)

        x[selected_mask] = selected_coef * selected_x + noise

        info["num_channel_impaired"] = int(selected_mask.sum().item())
        info["channel_type"] = channel_type
        info["snr_db"] = snr_db
        info["distance_aware_channel"] = distance_aware
        info["channel_gain_mean"] = float(
            coefficient[selected_mask].detach().float().mean().item()
        )

        if self.return_masks:
            info["channel_event_mask"] = selected_mask.detach()
            info["channel_coefficient"] = coefficient.detach()

        return x, info

    # ------------------------------------------------------------------
    # Packet / block loss
    # ------------------------------------------------------------------

    def _apply_packet_loss(
        self,
        x: torch.Tensor,
        impair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cfg = self.packet_cfg
        public_profile = self._public_channel_profile
        enabled = self.network_enabled and bool(cfg.get("enabled", False))

        info: Dict[str, Any] = {
            "packet_loss_enabled": enabled,
            "num_packet_loss_cav": 0,
            "packet_pixel_zero_ratio": 0.0,
        }

        if not enabled:
            return x, info

        mean = self._as_float(cfg.get("mean", 0.0), 0.0)
        std = self._as_float(cfg.get("std", 0.0), 0.0)
        zero_fraction = self._as_float(cfg.get("zero_fraction", mean), mean)
        zero_fraction = max(0.0, min(1.0, zero_fraction))

        granularity = str(cfg.get("granularity", "block")).lower()
        block_size = max(1, self._as_int(cfg.get("block_size", 4), 4))

        num_cav, _, height, width = x.shape
        device = x.device

        if public_profile is None:
            probs = self._sample_normal_clamped(
                mean=mean,
                std=std,
                shape=(num_cav,),
                device=device,
                min_value=0.0,
                max_value=1.0,
            )
            event_mask = (torch.rand(num_cav, device=device) < probs) & impair_mask
        else:
            # A public loss sample is taken below for every transmitted
            # feature block.  Do not make a second baseline-private event.
            mean = float(public_profile.get("packet_loss_rate", 0.0))
            std = 0.0
            zero_fraction = 1.0
            event_mask = impair_mask.clone()

        if not event_mask.any() or zero_fraction <= 0:
            return x, info

        x = x.clone()
        pixel_masks = torch.ones(
            (num_cav, 1, height, width),
            device=device,
            dtype=x.dtype,
        )

        selected_indices = torch.nonzero(event_mask, as_tuple=False).view(-1)

        if granularity == "element":
            random_mask = (
                torch.rand(
                    (len(selected_indices), 1, height, width),
                    device=device,
                    dtype=x.dtype,
                )
                >= zero_fraction
            ).to(dtype=x.dtype)

            pixel_masks[selected_indices] = random_mask

        elif granularity == "channel":
            channel_masks = torch.ones_like(x)
            for idx in selected_indices.tolist():
                num_channels = x.shape[1]
                num_drop = int(math.ceil(zero_fraction * num_channels))
                num_drop = max(0, min(num_channels, num_drop))

                if num_drop > 0:
                    perm = torch.randperm(num_channels, device=device)
                    drop_ch = perm[:num_drop]
                    channel_masks[idx, drop_ch, :, :] = 0

            x = x * channel_masks
            info["num_packet_loss_cav"] = int(event_mask.sum().item())
            info["packet_granularity"] = granularity
            info["packet_loss_prob_mean"] = mean
            info["packet_loss_prob_std"] = std
            info["packet_zero_fraction"] = zero_fraction

            if self.return_masks:
                info["packet_event_mask"] = event_mask.detach()
                info["packet_channel_mask"] = channel_masks.detach()

            zero_ratio = float((channel_masks == 0).float().mean().item())
            info["packet_pixel_zero_ratio"] = zero_ratio
            return x, info

        else:
            # Default: block-level loss, matching the RoCooper paper wording:
            # randomly zero out portions of transmitted feature blocks.
            grid_h = int(math.ceil(height / block_size))
            grid_w = int(math.ceil(width / block_size))
            num_blocks = grid_h * grid_w
            num_drop = int(math.ceil(zero_fraction * num_blocks))
            num_drop = max(0, min(num_blocks, num_drop))

            for idx in selected_indices.tolist():
                if num_drop <= 0:
                    continue

                block_mask = torch.ones(
                    (grid_h, grid_w),
                    device=device,
                    dtype=x.dtype,
                )

                if public_profile is None:
                    perm = torch.randperm(num_blocks, device=device)
                    drop_blocks = perm[:num_drop]
                    block_mask.view(-1)[drop_blocks] = 0
                else:
                    loss_mask, _ = self.channel_manager.sample_packet_loss(
                        num_packets=num_blocks,
                        link_id=("rocooper", int(idx)),
                        frame_id=self._public_channel_frame_id,
                        state=self._public_channel_state,
                        device=device,
                    )
                    block_mask.view(-1)[loss_mask] = 0

                pixel_mask = block_mask.repeat_interleave(
                    block_size,
                    dim=0,
                ).repeat_interleave(
                    block_size,
                    dim=1,
                )

                pixel_mask = pixel_mask[:height, :width]
                pixel_masks[idx, 0] = pixel_mask

        x = x * pixel_masks

        info["num_packet_loss_cav"] = int(event_mask.sum().item())
        info["packet_granularity"] = granularity
        info["packet_loss_prob_mean"] = mean
        info["packet_loss_prob_std"] = std
        info["packet_zero_fraction"] = zero_fraction
        info["packet_block_size"] = block_size
        info["packet_pixel_zero_ratio"] = float(
            (pixel_masks == 0).float().mean().item()
        )

        if self.return_masks:
            info["packet_event_mask"] = event_mask.detach()
            info["packet_pixel_mask"] = pixel_masks.detach()

        return x, info

    # ------------------------------------------------------------------
    # Delay
    # ------------------------------------------------------------------

    def _apply_delay(
        self,
        x: torch.Tensor,
        impair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cfg = self.delay_cfg
        public_profile = self._public_channel_profile
        enabled = self.network_enabled and bool(cfg.get("enabled", False))

        info: Dict[str, Any] = {
            "delay_enabled": enabled,
            "num_delayed_cav": 0,
            "delay_frames_mean": 0.0,
        }

        if not enabled:
            self._push_history(x)
            return x, info

        mean_ms = self._as_float(cfg.get("mean_ms", 0.0), 0.0)
        std_ms = self._as_float(cfg.get("std_ms", 0.0), 0.0)
        frame_interval_ms = max(
            1e-6,
            self._as_float(cfg.get("frame_interval_ms", 100.0), 100.0),
        )
        max_delay_frames = max(
            0,
            self._as_int(cfg.get("max_delay_frames", 3), 3),
        )
        if public_profile is not None:
            mean_ms = max(0.0, float(public_profile.get("delay_ms", 0.0)))
            std_ms = 0.0
            max_delay_frames = max(0, int(math.ceil(mean_ms / frame_interval_ms)))

        # Push current impaired feature first.
        # For delay=1, we will fetch the previous element in the queue.
        self._push_history(x)

        if max_delay_frames <= 0:
            return x, info

        num_cav = x.shape[0]
        device = x.device

        delay_ms = self._sample_delay_ms(
            mean_ms=mean_ms,
            std_ms=std_ms,
            shape=(num_cav,),
            device=device,
        )

        delay_frames = torch.round(delay_ms / frame_interval_ms).long()
        delay_frames = delay_frames.clamp(min=0, max=max_delay_frames)

        delay_event_mask = (delay_frames > 0) & impair_mask

        if not delay_event_mask.any():
            return x, info

        delayed_x = x.clone()

        for idx in torch.nonzero(delay_event_mask, as_tuple=False).view(-1).tolist():
            d = int(delay_frames[idx].item())

            history_index = len(self._history_queue) - 1 - d

            if history_index < 0:
                continue

            hist = self._history_queue[history_index]

            # Batch composition can vary during shuffled training. We only
            # use a delayed feature when shape is exactly compatible.
            if hist.shape != x.shape:
                continue

            delayed_x[idx] = hist[idx].to(device=x.device, dtype=x.dtype)

        used_mask = delay_event_mask

        info["num_delayed_cav"] = int(used_mask.sum().item())
        info["delay_ms_mean"] = mean_ms
        info["delay_ms_std"] = std_ms
        info["frame_interval_ms"] = frame_interval_ms
        info["max_delay_frames"] = max_delay_frames
        info["delay_frames_mean"] = float(
            delay_frames[used_mask].float().mean().item()
            if used_mask.any()
            else 0.0
        )

        if self.return_masks:
            info["delay_event_mask"] = delay_event_mask.detach()
            info["delay_frames"] = delay_frames.detach()

        return delayed_x, info

    def _push_history(self, x: torch.Tensor) -> None:
        max_delay_frames = max(
            0,
            self._as_int(self.delay_cfg.get("max_delay_frames", 3), 3),
        )
        if self._public_channel_profile is not None:
            frame_interval_ms = max(
                1e-6,
                float(self.channel_manager.channel_cfg.get("frame_interval_ms", 100.0)),
            )
            max_delay_frames = max(
                0,
                int(math.ceil(float(self._public_channel_profile.get("delay_ms", 0.0)) / frame_interval_ms)),
            )
        max_len = max_delay_frames + 1

        self._history_queue.append(x.detach())

        if len(self._history_queue) > max_len:
            self._history_queue = self._history_queue[-max_len:]

    @staticmethod
    def _sample_delay_ms(
        mean_ms: float,
        std_ms: float,
        shape: Tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        if std_ms <= 0:
            delay_ms = torch.full(
                shape,
                float(mean_ms),
                device=device,
                dtype=torch.float32,
            )
        else:
            delay_ms = torch.normal(
                mean=float(mean_ms),
                std=float(std_ms),
                size=shape,
                device=device,
            )

        return delay_ms.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Frame drop / communication disruption
    # ------------------------------------------------------------------

    def _apply_frame_drop(
        self,
        x: torch.Tensor,
        impair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cfg = self.frame_drop_cfg
        enabled = self.network_enabled and bool(cfg.get("enabled", False))

        info: Dict[str, Any] = {
            "frame_drop_enabled": enabled,
            "num_frame_dropped_cav": 0,
        }

        if not enabled:
            return x, info

        mean = self._as_float(cfg.get("mean", 0.0), 0.0)
        std = self._as_float(cfg.get("std", 0.0), 0.0)
        drop_whole_cav_feature = bool(
            cfg.get("drop_whole_cav_feature", True)
        )

        if not drop_whole_cav_feature:
            return x, info

        num_cav = x.shape[0]
        device = x.device

        probs = self._sample_normal_clamped(
            mean=mean,
            std=std,
            shape=(num_cav,),
            device=device,
            min_value=0.0,
            max_value=1.0,
        )

        event_mask = torch.rand(num_cav, device=device) < probs
        event_mask = event_mask & impair_mask

        if not event_mask.any():
            return x, info

        x = x.clone()
        x[event_mask] = 0

        info["num_frame_dropped_cav"] = int(event_mask.sum().item())
        info["frame_drop_prob_mean"] = mean
        info["frame_drop_prob_std"] = std

        if self.return_masks:
            info["frame_drop_event_mask"] = event_mask.detach()

        return x, info

    # ------------------------------------------------------------------
    # VCQI reporting
    # ------------------------------------------------------------------

    def compute_config_vcqi(self) -> float:
        """
        Compute an approximate VCQI from configured mean impairment values.

        This is a reporting helper, not the actual simulator. The simulator
        above uses explicit feature-level corruption.

        Formula mirrors the paper's decomposition:
            VCQI = VCQI_channel^beta * VCQI_network^gamma

        Unknown constants are read from yaml; defaults are intentionally simple.
        """
        vcqi_cfg = self.vcqi_cfg

        alpha0 = self._as_float(vcqi_cfg.get("alpha0", 1.0), 1.0)
        alpha1 = self._as_float(vcqi_cfg.get("alpha1", 1.0), 1.0)
        alpha2 = self._as_float(vcqi_cfg.get("alpha2", 1.0), 1.0)
        alpha3 = self._as_float(vcqi_cfg.get("alpha3", 1.0), 1.0)
        alpha4 = self._as_float(vcqi_cfg.get("alpha4", 1.0), 1.0)
        alpha_c = self._as_float(vcqi_cfg.get("alpha_c", 1.0), 1.0)
        beta = self._as_float(vcqi_cfg.get("beta", 1.0), 1.0)
        gamma = self._as_float(vcqi_cfg.get("gamma", 1.0), 1.0)

        channel_cfg = self.channel_cfg
        p0 = self._as_float(channel_cfg.get("p0", 1.0), 1.0)
        sigma_h = self._as_float(channel_cfg.get("sigma_h", 0.1), 0.1)
        sigma_w = self._as_float(channel_cfg.get("sigma_w", 0.01), 0.01)

        # Use communication_range as a simple d0 estimate.
        d0 = max(1.0, self.communication_range)

        vcqi_channel = (
            math.sqrt(max(p0, 1e-12)) / d0
        ) * math.sqrt(
            alpha_c / max(sigma_h ** 2 + sigma_w ** 2, 1e-12)
        )

        bandwidth_cfg = self.bandwidth_cfg
        packet_cfg = self.packet_cfg
        delay_cfg = self.delay_cfg
        frame_drop_cfg = self.frame_drop_cfg

        pb = self._as_float(bandwidth_cfg.get("mean", 0.0), 0.0)
        cb = self._as_float(bandwidth_cfg.get("compression_ratio", 1.0), 1.0)

        pp = self._as_float(packet_cfg.get("mean", 0.0), 0.0)
        cp = self._as_float(
            packet_cfg.get("zero_fraction", packet_cfg.get("mean", 0.0)),
            0.0,
        )

        delay_ms = self._as_float(delay_cfg.get("mean_ms", 0.0), 0.0)
        frame_interval_ms = max(
            1e-6,
            self._as_float(delay_cfg.get("frame_interval_ms", 100.0), 100.0),
        )
        i3 = delay_ms / frame_interval_ms

        i4 = self._as_float(frame_drop_cfg.get("mean", 0.0), 0.0)

        i1 = pb * cb
        i2 = pp * cp

        vcqi_network = 1.0 / max(
            alpha0 + alpha1 * i1 + alpha2 * i2 + alpha3 * i3 + alpha4 * i4,
            1e-12,
        )

        vcqi = (vcqi_channel ** beta) * (vcqi_network ** gamma)

        min_value = self._as_float(vcqi_cfg.get("min_value", 0.0), 0.0)
        max_value = self._as_float(vcqi_cfg.get("max_value", 20.0), 20.0)

        return float(max(min_value, min(max_value, vcqi)))


# Optional aliases for tolerant imports
CommunicationImpairment = RoCooperComm
RocooperComm = RoCooperComm
