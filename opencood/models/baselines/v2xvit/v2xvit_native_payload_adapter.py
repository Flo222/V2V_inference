from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import torch

from opencood.communication.interface import NativePayload


try:
    from opencood.methods.arce.arce_fixed_comm import ARCEFixedComm
    _ARCE_FIXED_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    ARCEFixedComm = None
    _ARCE_FIXED_IMPORT_ERROR = exc

try:
    from opencood.methods.arce.arce_c2mab_comm import ARCEC2MABComm
    _ARCE_C2MAB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    ARCEC2MABComm = None
    _ARCE_C2MAB_IMPORT_ERROR = exc


class V2XViTNativePayloadAdapter:
    """Bridge V2X-ViT's native post-compressor feature to ARCE.

    Correct placement:
        backbone -> shrink -> native compressor
        -> NativePayload/ARCE
        -> regroup/padding -> prior repeat -> V2XTransformer

    Only the actual flattened CAV feature tensor is communicated. ``max_cav``
    padding and HxW-repeated priors are created after communication and hence
    cannot inflate the physical payload.
    """

    def __init__(
        self,
        arce_cfg: Optional[Dict[str, Any]],
        dataset_name: str,
        prior_bytes_per_link: int = 12,
        pose_bytes_per_link: int = 64,
    ):
        cfg = copy.deepcopy(arce_cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.dataset_name = str(dataset_name)

        # V2X-ViT's method-native payload is already the post-compressor dense
        # feature. ARCE must not apply a Where2Comm-specific compact mask.
        cfg["transport_mode"] = "payload_native"
        compact_cfg = copy.deepcopy(cfg.get("compact_sparse", {}) or {})
        compact_cfg.update(
            {
                "enabled": False,
                "source": "none",
                "budget_aware_topk": False,
            }
        )
        cfg["compact_sparse"] = compact_cfg

        payload_cfg = copy.deepcopy(cfg.get("payload", {}) or {})
        payload_cfg.setdefault("interface", "native_payload_v1")
        payload_cfg.setdefault("stage", "post_native_compressor")
        payload_cfg.setdefault("prior_bytes_per_link", int(prior_bytes_per_link))
        payload_cfg.setdefault("pose_bytes_per_link", int(pose_bytes_per_link))
        cfg["payload"] = payload_cfg

        self.cfg = cfg
        self.prior_bytes_per_link = int(payload_cfg["prior_bytes_per_link"])
        self.pose_bytes_per_link = int(payload_cfg["pose_bytes_per_link"])
        self.executor = None
        self.executor_type = "disabled"

        if not self.enabled:
            return

        mode = str(cfg.get("mode", cfg.get("policy", "fixed"))).strip().lower()
        policy = str(cfg.get("policy", mode)).strip().lower()
        use_c2mab = (
            mode in ("dc2mab", "c2mab")
            or policy in ("dc2mab_sender_ego", "c2mab_sender_ego")
        )

        if use_c2mab:
            if ARCEC2MABComm is None:
                raise ImportError(
                    "Cannot import ARCEC2MABComm: {}".format(
                        _ARCE_C2MAB_IMPORT_ERROR
                    )
                )
            self.executor = ARCEC2MABComm(cfg)
            self.executor_type = "c2mab"
        else:
            if ARCEFixedComm is None:
                raise ImportError(
                    "Cannot import ARCEFixedComm: {}".format(
                        _ARCE_FIXED_IMPORT_ERROR
                    )
                )
            self.executor = ARCEFixedComm(cfg)
            self.executor_type = "fixed_or_random"

    @property
    def per_link_aux_bytes(self) -> int:
        return int(self.prior_bytes_per_link + self.pose_bytes_per_link)

    def build_payload(
        self,
        features: torch.Tensor,
        record_len: Any,
    ) -> NativePayload:
        return NativePayload(
            values=features,
            record_len=record_len,
            payload_type="v2xvit_dense_feature",
            stage="post_shrink_and_native_compressor",
            layout="NCHW",
            metadata={
                "dataset": self.dataset_name,
                "baseline": "V2X-ViT",
                "ego_transmitted": False,
                "max_cav_padding_transmitted": False,
                "prior_repeated_over_hw": False,
                "prior_bytes_per_link": int(self.prior_bytes_per_link),
                "pose_bytes_per_link": int(self.pose_bytes_per_link),
                "per_link_aux_bytes": int(self.per_link_aux_bytes),
            },
        ).validate()

    def communicate(
        self,
        features: torch.Tensor,
        record_len: Any,
        data_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        payload = self.build_payload(features, record_len)
        payload_summary = payload.summary()

        if not self.enabled or self.executor is None:
            return payload.values, {
                "enabled": False,
                "mode": "disabled",
                "native_payload": payload_summary,
            }

        kwargs = {
            "features": payload.values,
            "record_len": payload.record_len,
            "data_dict": data_dict,
            "ego_index": 0,
            "update_cache": True,
            "return_records": True,
            "message_masks": None,
        }

        result = self.executor.communicate_flattened_features(**kwargs)
        if isinstance(result, tuple):
            recovered, comm_info = result
        else:
            recovered = result
            comm_info = None

        recovered_payload = payload.with_values(recovered)
        info = comm_info if isinstance(comm_info, dict) else {
            "records": comm_info if isinstance(comm_info, list) else [],
        }
        info = copy.deepcopy(info)
        info["enabled"] = True
        info["executor_type"] = self.executor_type
        info["native_payload"] = payload_summary
        info["recovered_payload"] = recovered_payload.summary()
        return recovered_payload.values, info


__all__ = ["V2XViTNativePayloadAdapter"]
