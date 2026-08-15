"""
Fixed-policy ARCE communication pipeline.

Final byte-stream version with real FEC / redundancy.

Pipeline:
    Where2comm masked feature F
        -> temporal policy:
             good / medium: current frame
             bad: previous frame
        -> quantize_feature(F)
        -> reinterpret Q(F) as byte stream
        -> split into fixed 1024-Byte source packets
        -> FEC encode:
             encoded_packets = source_packets + parity / repair packets
        -> system / link budget selects encoded packets to transmit
        -> Bernoulli packet loss:
             receive_i ~ Bernoulli(1 - PLR_t)
        -> FEC decode to recover source packets
        -> still-missing source packets are zero-filled
        -> rebuild byte stream
        -> dequantize back to feature
        -> return to Where2comm attention fusion
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from opencood.methods.arce import (
    ARCE_POLICY_RANDOM,
    ARCE_MODE_DISABLED,
    ARCE_MODE_BYPASS,
    normalize_arce_config,
    extract_arce_cfg,
    should_apply_to_agent,
)
from opencood.methods.arce.fixed_policy import ARCEAction, FixedARCEPolicy
from opencood.methods.arce.random_policy import RandomARCEPolicy
from opencood.communication.transport.packetization.byte_stream_packetizer import ByteStreamPacketizer
from opencood.methods.arce.policies.payload_transport import (
    apply_payload_native_transport_to_arce_cfg,
    is_payload_native_transport,
    normalize_transport_mode,
)
from opencood.methods.arce.policies.action_adapter import (
    get_action_field,
    normalize_runtime_action,
    runtime_action_as_dict,
)
from opencood.methods.arce.policies.spatial_importance import (
    ARCESpatialImportance,
)
from opencood.methods.arce.priority_block_fec_transport import (
    PriorityBlockFECTransport,
    SCHEDULING_MODE as RAPTORQ_SCHEDULING_MODE,
)
from opencood.communication.transport.quantization.feature_quantizer import FeatureQuantizer
from opencood.methods.arce.audit import CompressionAuditor, FECRecoveryAuditor

from opencood.communication.transport.fec import (
    FEC_TYPE_NONE,
    FEC_TYPE_XOR,
    FEC_TYPE_RAPTOR_SIM,
    FEC_TYPE_RAPTORQ,
    normalize_fec_type,
)
from opencood.communication.transport.fec.fec_base import (
    FECEncodeResult,
    FECDecodeResult,
    EncodedPacketMeta,
)
from opencood.communication.transport.fec.fec_xor import XORFEC
from opencood.communication.transport.fec.fec_raptor_sim import RaptorSimFEC


CHANNEL_STATE_ID_TO_NAME = {
    0: "good",
    1: "medium",
    2: "bad",
}

VALID_CHANNEL_STATE_NAMES = ("good", "medium", "bad")

LATE_POLICY_ALLOW = "allow"
LATE_POLICY_DROP = "drop"
LATE_POLICY_CACHE_ONLY = "cache_only"

VALID_LATE_POLICIES = (
    LATE_POLICY_ALLOW,
    LATE_POLICY_DROP,
    LATE_POLICY_CACHE_ONLY,
)


def _require_tensor(x: Any, name: str = "tensor") -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} should be a torch.Tensor, got {type(x)}.")
    return x


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "y", "on"):
            return True
        if text in ("false", "0", "no", "n", "off"):
            return False
    return bool(value)


def _as_positive_int(value: Any, name: str) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to int, got {value}.")
    if value <= 0:
        raise ValueError(f"{name} should be positive, got {value}.")
    return value


def _stable_int_seed(base_seed: int, *items: Any) -> int:
    text = "|".join(repr(item) for item in items).encode("utf-8")
    digest = hashlib.md5(text).hexdigest()
    offset = int(digest[:8], 16)
    return int((int(base_seed) + offset) % (2**32 - 1))


def _normalize_late_policy(policy: Optional[str]) -> str:
    if policy is None:
        return LATE_POLICY_CACHE_ONLY
    policy = str(policy).strip().lower()
    if policy not in VALID_LATE_POLICIES:
        raise ValueError(
            f"Unsupported late_policy: {policy}. "
            f"Expected one of {VALID_LATE_POLICIES}."
        )
    return policy


def _merge_dict(
    base: Optional[Dict[str, Any]],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    result = copy.deepcopy(base or {})
    result.update(copy.deepcopy(override or {}))
    return result


def _safe_get_action_field(action: Any, name: str, default: Any = None) -> Any:
    try:
        return get_action_field(action, name, default)
    except Exception:
        if isinstance(action, dict):
            return action.get(name, default)
        return getattr(action, name, default)


def _mask_summary(mask: torch.Tensor, true_name: str = "true") -> Dict[str, Any]:
    mask = mask.to(dtype=torch.bool).flatten()
    n = int(mask.numel())
    num_true = int(mask.sum().item())
    return {
        "length": n,
        f"num_{true_name}": num_true,
        f"ratio_{true_name}": float(num_true / n) if n > 0 else 0.0,
    }


def _state_profile_defaults(state_name: str) -> Dict[str, Any]:
    state_name = str(state_name).lower()
    if state_name == "good":
        return {
            "state_name": "good",
            "bandwidth_mbps": 27.0,
            "plr": 0.05,
            "loss_rate": 0.05,
            "delay_ms": 10.0,
            "fixed_delay_ms": 10.0,
            "temporal_source": "current",
        }
    if state_name == "bad":
        return {
            "state_name": "bad",
            "bandwidth_mbps": 1.0,
            "plr": 0.35,
            "loss_rate": 0.35,
            "delay_ms": 100.0,
            "fixed_delay_ms": 100.0,
            "temporal_source": "previous_frame",
        }
    return {
        "state_name": "medium",
        "bandwidth_mbps": 5.0,
        "plr": 0.20,
        "loss_rate": 0.20,
        "delay_ms": 50.0,
        "fixed_delay_ms": 50.0,
        "temporal_source": "current",
    }


@dataclass
class ARCECommResult:
    recovered_feature: torch.Tensor
    record: Dict[str, Any]
    packetization_result: Optional[Any] = None
    quantization_result: Optional[Any] = None
    encode_result: Optional[Any] = None
    decode_result: Optional[Any] = None
    partial_result: Optional[Any] = None

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.record)


class ARCEFixedComm:
    """
    Fixed / random ARCE communication executor.

    This version implements actual byte-stream FEC / redundancy.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.full_cfg = cfg or {}
        self.arce_cfg_raw = extract_arce_cfg(cfg or {})
        self.arce_cfg = normalize_arce_config(cfg or {})

        self.enabled = bool(self.arce_cfg["enabled"])
        self.mode = self.arce_cfg["mode"]
        self.seed = int(self.arce_cfg["seed"])
        self.link_scope = self.arce_cfg["link_scope"]
        self.record_per_frame = bool(self.arce_cfg["record_per_frame"])
        self.record_per_link = bool(self.arce_cfg["record_per_link"])
        self.log_interval = int(self.arce_cfg["log_interval"])
        self.verbose = bool(self.arce_cfg["verbose"])
        self.debug = bool(self.arce_cfg["debug"])

        self.max_records = _as_positive_int(
            self.arce_cfg_raw.get("max_records", 100000),
            "arce.max_records",
        )
        self.keep_tensor_results = _as_bool(
            self.arce_cfg_raw.get("keep_tensor_results", False)
        )

        self.late_policy = _normalize_late_policy(
            self.arce_cfg_raw.get("late_policy", None)
        )
        self.enable_deadline_drop = _as_bool(
            self.arce_cfg_raw.get("enable_deadline_drop", False)
        )

        self.default_ego_index = int(self.arce_cfg_raw.get("ego_index", 0))

        self.policy_name = str(self.arce_cfg.get("policy", "fixed")).strip().lower()
        if self.policy_name == ARCE_POLICY_RANDOM:
            self.action_policy = RandomARCEPolicy(self.arce_cfg_raw)
        else:
            self.action_policy = FixedARCEPolicy(self.arce_cfg_raw)

        self.fixed_policy = self.action_policy
        self.byte_packetizer = ByteStreamPacketizer(self.arce_cfg_raw)

        scheduler_cfg = self.arce_cfg_raw.get("scheduler", {}) or {}
        self.tx_window_ms = float(
            scheduler_cfg.get(
                "tx_window_ms",
                self.arce_cfg_raw.get("deadline_ms", 100.0),
            )
        )
        self.budget_source = str(
            scheduler_cfg.get("budget_source", getattr(self, "arce_cfg_raw", {}).get("budget_source", "system_budget"))
        )
        self.budget_scope = str(
            scheduler_cfg.get("budget_scope", "system_equal_split")
        ).strip().lower()
        self.system_budget_mbps = float(
            scheduler_cfg.get(
                "system_budget_mbps",
                scheduler_cfg.get("total_budget_mbps", 5.0),
            )
        )

        channel_cfg = self.arce_cfg_raw.get("channel", {}) or {}

        self.loss_model = str(channel_cfg.get("loss_model", "bernoulli")).strip().lower()
        self.bernoulli_loss_rates = {
            "good": 0.05,
            "medium": 0.20,
            "bad": 0.35,
        }
        self.bernoulli_loss_rates.update(
            channel_cfg.get("bernoulli_loss_rates", {}) or {}
        )

        self.latency_model_type = str(
            channel_cfg.get("latency_model", "fixed_state_delay")
        ).strip().lower()
        self.fixed_delay_ms = {
            "good": 10.0,
            "medium": 50.0,
            "bad": 100.0,
        }
        self.fixed_delay_ms.update(channel_cfg.get("fixed_delay_ms", {}) or {})

        profiles_cfg = channel_cfg.get("profiles", {}) or {}
        self.channel_profiles = {
            "good": _merge_dict(_state_profile_defaults("good"), profiles_cfg.get("good", {})),
            "medium": _merge_dict(_state_profile_defaults("medium"), profiles_cfg.get("medium", {})),
            "bad": _merge_dict(_state_profile_defaults("bad"), profiles_cfg.get("bad", {})),
        }

        delay_cfg = self.arce_cfg_raw.get("delay", {}) or {}
        self.delay_policy_by_state = delay_cfg.get(
            "policy_by_state",
            {
                "good": "current",
                "medium": "current",
                "bad": "previous_frame",
            },
        )

        self.fec_cfg = self.arce_cfg_raw.get("fec", {}) or {}
        self.redundancy_cfg = self.arce_cfg_raw.get("redundancy", {}) or {}
        raptorq_cfg = self.fec_cfg.get("raptorq", {}) or {}
        self.raptorq_block_source_packets = int(
            raptorq_cfg.get("source_packets_per_block", 20)
        )
        self.raptorq_transport = PriorityBlockFECTransport(
            source_packet_bytes=int(self.byte_packetizer.packet_size_bytes),
            block_source_packets=self.raptorq_block_source_packets,
        )
        self.transport_mode = normalize_transport_mode(self.arce_cfg_raw)
        self.arce_cfg_raw = apply_payload_native_transport_to_arce_cfg(self.arce_cfg_raw)
        self.spatial_importance = ARCESpatialImportance(
            self.arce_cfg_raw.get("spatial_importance", {}) or {}
        )
        self.uses_arce_spatial_importance = bool(
            self.spatial_importance.enabled
        )

        self.markov_cfg = self._extract_markov_cfg()
        self.markov_enabled = _as_bool(self.markov_cfg.get("enabled", False))
        self.markov_states = [
            str(s).lower() for s in self.markov_cfg.get("states", ["good", "medium", "bad"])
        ]
        self.markov_init_state = str(
            self.markov_cfg.get("init_state", self.markov_cfg.get("initial_state", "medium"))
        ).lower()
        self.markov_transition_matrix = self.markov_cfg.get(
            "transition_matrix",
            [
                [0.85, 0.13, 0.02],
                [0.10, 0.80, 0.10],
                [0.03, 0.17, 0.80],
            ],
        )
        if isinstance(self.markov_transition_matrix, dict):
            self.markov_transition_matrix = [
                [
                    self.markov_transition_matrix["good"]["good"],
                    self.markov_transition_matrix["good"]["medium"],
                    self.markov_transition_matrix["good"]["bad"],
                ],
                [
                    self.markov_transition_matrix["medium"]["good"],
                    self.markov_transition_matrix["medium"]["medium"],
                    self.markov_transition_matrix["medium"]["bad"],
                ],
                [
                    self.markov_transition_matrix["bad"]["good"],
                    self.markov_transition_matrix["bad"]["medium"],
                    self.markov_transition_matrix["bad"]["bad"],
                ],
            ]
        self._markov_state_by_link: Dict[Any, str] = {}

        self.prev_feature_cache: Dict[Any, torch.Tensor] = {}
        # Receiver-side cache is separate from the sender-side delay cache.
        # It stores only feature units that were actually recovered.
        self.receiver_feature_cache: Dict[Any, Dict[str, Any]] = {}
        temporal_cfg = self.arce_cfg_raw.get("temporal_fusion", {}) or {}
        self.receiver_cache_max_age_frames = int(
            temporal_cfg.get("max_age_frames", 1)
        )
        if self.receiver_cache_max_age_frames < 1:
            raise ValueError(
                "temporal_fusion.max_age_frames must be >= 1, got {}.".format(
                    self.receiver_cache_max_age_frames
                )
            )

        self.records: List[Dict[str, Any]] = []
        self.frame_records: Dict[Any, List[Dict[str, Any]]] = {}

        self.num_processed_links = 0
        self.num_bypassed_links = 0
        self.num_late_links = 0
        self.num_dropped_by_late = 0
        self._loss_call_index = 0
        self._markov_call_index = 0

        # Read-only diagnostics for Experiment 1. Disabled by default.
        self.compression_auditor = CompressionAuditor(
            self.arce_cfg_raw.get("compression_audit", {}) or {}
        )
        # Read-only diagnostics for Experiment 3. Disabled by default.
        self.fec_recovery_auditor = FECRecoveryAuditor(
            self.arce_cfg_raw.get("fec_recovery_audit", {}) or {}
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _extract_markov_cfg(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        if isinstance(self.full_cfg, dict):
            top = self.full_cfg.get("channel_state_markov", None)
            if isinstance(top, dict):
                result.update(copy.deepcopy(top))

        raw = self.arce_cfg_raw
        top_raw = raw.get("channel_state_markov", None)
        if isinstance(top_raw, dict):
            result.update(copy.deepcopy(top_raw))

        channel_cfg = raw.get("channel", {}) or {}
        if str(channel_cfg.get("mode", "")).strip().lower() == "markov":
            result["enabled"] = True

        if "states" in channel_cfg:
            result["states"] = copy.deepcopy(channel_cfg["states"])
        if "init_state" in channel_cfg:
            result["init_state"] = channel_cfg["init_state"]
        if "initial_state" in channel_cfg:
            result["initial_state"] = channel_cfg["initial_state"]
        if "transition_matrix" in channel_cfg:
            result["transition_matrix"] = copy.deepcopy(channel_cfg["transition_matrix"])

        return result

    def _get_base_quant_cfg(self) -> Dict[str, Any]:
        return copy.deepcopy(self.arce_cfg_raw.get("quantization", {}))

    def _build_quantizer(self, action: Any) -> FeatureQuantizer:
        quant_cfg = _merge_dict(
            self._get_base_quant_cfg(),
            action.to_quant_config() if hasattr(action, "to_quant_config") else {},
        )

        # IMPORTANT:
        # INT4 communication should be byte-packed to match the intended
        # 4-bit communication semantics. Without this, FeatureQuantizer keeps
        # q_tensor in int8 storage, and ByteStreamPacketizer sends 1 byte/value.
        try:
            action_quant_mode = self._get_action_quant_mode(action)
        except Exception:
            action_quant_mode = str(quant_cfg.get("mode", "fp32")).strip().lower()

        if str(action_quant_mode).strip().lower() in ("int4", "4bit", "int4_uniform"):
            quant_cfg["mode"] = "int4"
            quant_cfg["enabled"] = True
            quant_cfg["pack_int4"] = True

        return FeatureQuantizer({"quantization": quant_cfg})

    def _get_action_quant_mode(self, action: Any) -> str:
        quant_mode = _safe_get_action_field(action, "quant_mode", None)
        if quant_mode is None:
            quant_mode = _safe_get_action_field(action, "quant", None)
        if quant_mode is None:
            quant_mode = self._get_base_quant_cfg().get("mode", "fp16")
        return str(quant_mode).strip().lower()

    def _get_action_fec_type(self, action: Any) -> str:
        fec_type = _safe_get_action_field(action, "fec_type", None)
        if fec_type is None:
            fec_type = self.fec_cfg.get("type", self.fec_cfg.get("default_type", "none"))
        if str(fec_type).lower() == "action":
            fec_type = self.fec_cfg.get("default_type", "raptor_sim")
        return normalize_fec_type(fec_type)

    def _get_action_redundancy_ratio(self, action: Any) -> float:
        rho = _safe_get_action_field(action, "redundancy_ratio", None)
        if rho is None:
            rho = _safe_get_action_field(action, "rho", None)
        if rho is None:
            rho = self.fec_cfg.get(
                "redundancy_ratio",
                self.redundancy_cfg.get("default_ratio", 0.0),
            )
        return float(max(0.0, float(rho)))

    def _get_action_xor_group_size(self, action: Any) -> int:
        group_size = _safe_get_action_field(action, "xor_group_size", None)
        if group_size is None:
            group_size = _safe_get_action_field(action, "group_size", None)
        if group_size is None:
            group_size = self.fec_cfg.get("xor_group_size", self.fec_cfg.get("group_size", 4))
        return int(max(1, int(group_size)))

    def _get_action_decode_overhead(self, action: Any) -> float:
        value = _safe_get_action_field(action, "decode_overhead", None)
        if value is None:
            value = self.fec_cfg.get("decode_overhead", 0.0)
        return float(max(0.0, float(value)))

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------

    def _append_record(self, record: Dict[str, Any]) -> None:
        if not self.record_per_link:
            return

        self.records.append(copy.deepcopy(record))
        if len(self.records) > self.max_records:
            overflow = len(self.records) - self.max_records
            self.records = self.records[overflow:]

        frame_id = record.get("frame_id", None)
        if self.record_per_frame:
            self.frame_records.setdefault(frame_id, []).append(copy.deepcopy(record))

    def clear_records(self) -> None:
        self.records.clear()
        self.frame_records.clear()

    def reset(self, clear_cache: bool = True, clear_records: bool = True) -> None:
        if clear_cache:
            self.prev_feature_cache.clear()
            self.receiver_feature_cache.clear()
            self._markov_state_by_link.clear()

        if clear_records:
            self.clear_records()

        self.num_processed_links = 0
        self.num_bypassed_links = 0
        self.num_late_links = 0
        self.num_dropped_by_late = 0
        self._loss_call_index = 0
        self._markov_call_index = 0
        if hasattr(self, "compression_auditor"):
            self.compression_auditor.reset()
        if hasattr(self, "fec_recovery_auditor"):
            self.fec_recovery_auditor.reset()

    def set_channel_state(self, state: str) -> None:
        self._markov_state_by_link["__global__"] = self._normalize_state_name(state)

    # ------------------------------------------------------------------
    # Channel / loss / delay
    # ------------------------------------------------------------------

    def _normalize_state_name(self, state: Optional[str]) -> str:
        if state is None:
            return "medium"

        state = str(state).strip().lower()
        if state == "mid":
            state = "medium"
        if state in VALID_CHANNEL_STATE_NAMES:
            return state

        raise ValueError(
            f"Unsupported channel state: {state}. "
            f"Expected one of {VALID_CHANNEL_STATE_NAMES}."
        )

    def _profile_for_state(self, state_name: str) -> Dict[str, Any]:
        state_name = self._normalize_state_name(state_name)
        profile = copy.deepcopy(self.channel_profiles.get(state_name, _state_profile_defaults(state_name)))
        profile["state_name"] = state_name
        profile["plr"] = float(self.bernoulli_loss_rates.get(state_name, profile.get("plr", 0.2)))
        profile["loss_rate"] = float(profile["plr"])
        profile["delay_ms"] = float(self.fixed_delay_ms.get(state_name, profile.get("delay_ms", 50.0)))
        profile["fixed_delay_ms"] = float(profile["delay_ms"])
        return profile

    def _sample_markov_state(self, link_id: Any, frame_id: Optional[int]) -> str:
        if not self.markov_enabled:
            global_state = self._markov_state_by_link.get("__global__", None)
            return global_state or "medium"

        key = repr(link_id)
        prev_state = self._markov_state_by_link.get(key, None)

        if prev_state is None:
            current_state = self._normalize_state_name(self.markov_init_state)
            self._markov_state_by_link[key] = current_state
            return current_state

        prev_state = self._normalize_state_name(prev_state)
        try:
            row_idx = self.markov_states.index(prev_state)
        except ValueError:
            row_idx = self.markov_states.index("medium")

        probs = torch.tensor(
            self.markov_transition_matrix[row_idx],
            dtype=torch.float32,
        )
        probs = probs / probs.sum().clamp_min(1e-12)

        self._markov_call_index += 1
        seed = _stable_int_seed(
            self.seed,
            "markov",
            key,
            frame_id,
            self._markov_call_index,
        )
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        next_idx = int(torch.multinomial(probs, 1, generator=g).item())
        current_state = self._normalize_state_name(self.markov_states[next_idx])
        self._markov_state_by_link[key] = current_state
        return current_state

    def _resolve_active_channel_state(
        self,
        requested_channel_state: Optional[str],
        link_id: Any,
        frame_id: Optional[int],
    ) -> Tuple[str, str]:
        if requested_channel_state is not None:
            state = self._normalize_state_name(requested_channel_state)
            self._markov_state_by_link[repr(link_id)] = state
            return state, "dataset_link_markov"

        if self.markov_enabled:
            return self._sample_markov_state(link_id=link_id, frame_id=frame_id), "internal_markov"

        global_state = self._markov_state_by_link.get("__global__", None)
        if global_state is not None:
            return self._normalize_state_name(global_state), "fixed_runtime"

        return "medium", "default_medium"

    def _sample_bernoulli_loss(
        self,
        num_packets: int,
        state_name: str,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = device or torch.device("cpu")
        state_name = self._normalize_state_name(state_name)
        plr = float(self.bernoulli_loss_rates[state_name])

        self._loss_call_index += 1
        seed = _stable_int_seed(
            self.seed,
            "bernoulli",
            repr(link_id),
            frame_id,
            self._loss_call_index,
            state_name,
        )
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        receive_mask_cpu = torch.rand((int(num_packets),), generator=g) < (1.0 - plr)
        receive_mask = receive_mask_cpu.to(device=device)
        loss_mask = ~receive_mask

        info = {
            "model": "bernoulli",
            "formula": "receive_i ~ Bernoulli(1 - PLR_t)",
            "frame_id": frame_id,
            "link_id": repr(link_id),
            "channel_state": state_name,
            "plr": float(plr),
            "receive_prob": float(1.0 - plr),
            "num_packets": int(num_packets),
            "num_received": int(receive_mask.sum().item()),
            "num_lost": int(loss_mask.sum().item()),
            "empirical_loss": float(loss_mask.float().mean().item())
            if int(num_packets) > 0
            else 0.0,
        }
        return loss_mask, info

    def _estimate_fixed_latency(
        self,
        transmitted_bytes: float,
        state_name: str,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        bandwidth_mbps: Optional[float] = None,
    ) -> Dict[str, Any]:
        state_name = self._normalize_state_name(state_name)
        delay_ms = float(self.fixed_delay_ms[state_name])

        return {
            "model": "fixed_state_delay",
            "frame_id": frame_id,
            "link_id": repr(link_id),
            "channel_state": state_name,
            "bandwidth_mbps": float(bandwidth_mbps)
            if bandwidth_mbps is not None
            else None,
            "transmitted_bytes": float(transmitted_bytes),
            "transmission_delay_ms": 0.0,
            "processing_delay_ms": 0.0,
            "jitter_ms": 0.0,
            "total_delay_ms": float(delay_ms),
            "late": False,
            "deadline_ms": None,
        }

    def _get_temporal_tx_feature(
        self,
        feature: torch.Tensor,
        state_name: str,
        link_id: Any,
        agent_index: Optional[int],
        ego_index: Optional[int],
    ) -> Tuple[torch.Tensor, str]:
        state_name = self._normalize_state_name(state_name)
        policy = str(
            self.delay_policy_by_state.get(state_name, "current")
        ).strip().lower()

        cache_key = (
            repr(link_id),
            int(agent_index if agent_index is not None else -1),
            int(ego_index if ego_index is not None else self.default_ego_index),
        )

        if policy in ("previous_frame", "prev", "t-1", "previous"):
            prev = self.prev_feature_cache.get(cache_key, None)
            if prev is not None:
                return prev.to(device=feature.device, dtype=feature.dtype), "previous_frame"
            return feature, "current_no_history"

        return feature, "current"

    def _update_prev_feature_cache(
        self,
        feature: torch.Tensor,
        link_id: Any,
        agent_index: Optional[int],
        ego_index: Optional[int],
    ) -> None:
        cache_key = (
            repr(link_id),
            int(agent_index if agent_index is not None else -1),
            int(ego_index if ego_index is not None else self.default_ego_index),
        )
        self.prev_feature_cache[cache_key] = feature.detach().clone()

    @staticmethod
    def _receiver_cache_key(
        link_id: Any,
        agent_index: Optional[int],
        ego_index: Optional[int],
        default_ego_index: int,
    ) -> Tuple[str, int, int]:
        return (
            repr(link_id),
            int(agent_index if agent_index is not None else -1),
            int(ego_index if ego_index is not None else default_ego_index),
        )

    @staticmethod
    def _frame_distance(current_frame_id: Any, cached_frame_id: Any) -> Optional[int]:
        try:
            return int(current_frame_id) - int(cached_frame_id)
        except (TypeError, ValueError):
            return None

    def _get_receiver_cache_entry(
        self,
        link_id: Any,
        agent_index: Optional[int],
        ego_index: Optional[int],
        frame_id: Any,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        cache_key = self._receiver_cache_key(
            link_id,
            agent_index,
            ego_index,
            self.default_ego_index,
        )
        entry = self.receiver_feature_cache.get(cache_key)
        if not isinstance(entry, dict):
            return None, "empty"

        age = self._frame_distance(frame_id, entry.get("frame_id"))
        if age is not None:
            if age <= 0:
                return None, "non_forward_frame"
            if age > int(self.receiver_cache_max_age_frames):
                return None, "expired"

        return entry, "available"

    def receiver_cache_context(
        self,
        link_id: Any,
        agent_index: Optional[int],
        ego_index: Optional[int],
        frame_id: Any,
    ) -> Dict[str, Any]:
        """Return Cache state available before current action execution."""
        cache_key = self._receiver_cache_key(
            link_id,
            agent_index,
            ego_index,
            self.default_ego_index,
        )
        raw_entry = self.receiver_feature_cache.get(cache_key)
        entry, status = self._get_receiver_cache_entry(
            link_id,
            agent_index,
            ego_index,
            frame_id,
        )

        age_frames = None
        if isinstance(raw_entry, dict):
            age_frames = self._frame_distance(
                frame_id,
                raw_entry.get("frame_id"),
            )

        valid_units = 0
        total_units = 0
        valid_ratio = 0.0

        if entry is not None:
            valid_flat = entry.get("valid_flat")
            if torch.is_tensor(valid_flat):
                valid_flat = valid_flat.detach().to(
                    dtype=torch.bool
                ).flatten()
                total_units = int(valid_flat.numel())
                valid_units = int(valid_flat.sum().item())
                if total_units > 0:
                    valid_ratio = (
                        float(valid_units) / float(total_units)
                    )
            else:
                status = "invalid_valid_mask"

        max_age = max(int(self.receiver_cache_max_age_frames), 1)
        if age_frames is None:
            age_norm = 0.0
        else:
            age_norm = max(
                0.0,
                min(
                    1.0,
                    float(age_frames) / float(max_age),
                ),
            )

        available = bool(
            status == "available"
            and total_units > 0
            and valid_units > 0
        )

        return {
            "cache_available": available,
            "cache_status": str(status),
            "cache_valid_unit_ratio": float(valid_ratio),
            "cache_num_valid_units": int(valid_units),
            "cache_num_total_units": int(total_units),
            "cache_age_frames": age_frames,
            "cache_age_norm": float(age_norm),
            "cache_max_age_frames": int(max_age),
            "cache_context_source":
                "receiver_feature_cache_valid_unit_ratio",
        }

    def _update_receiver_feature_cache(
        self,
        recovered_feature: torch.Tensor,
        compact_meta: Optional[Dict[str, Any]],
        current_unit_valid_mask: Optional[torch.Tensor],
        link_id: Any,
        agent_index: Optional[int],
        ego_index: Optional[int],
        frame_id: Any,
    ) -> None:
        if (
            not isinstance(compact_meta, dict)
            or compact_meta.get("layout") != "KC"
            or not torch.is_tensor(compact_meta.get("indices"))
            or not torch.is_tensor(current_unit_valid_mask)
        ):
            return

        C, H, W = recovered_feature.shape
        unit_ids = compact_meta["indices"].to(
            device=recovered_feature.device,
            dtype=torch.long,
        ).flatten()
        current_unit_valid_mask = current_unit_valid_mask.to(
            device=recovered_feature.device,
            dtype=torch.bool,
        ).flatten()
        count = min(int(unit_ids.numel()), int(current_unit_valid_mask.numel()))
        unit_ids = unit_ids[:count]
        current_unit_valid_mask = current_unit_valid_mask[:count]

        valid_flat = torch.zeros(
            H * W,
            dtype=torch.bool,
            device=recovered_feature.device,
        )
        if count > 0:
            valid_flat[unit_ids[current_unit_valid_mask]] = True

        cache_key = self._receiver_cache_key(
            link_id,
            agent_index,
            ego_index,
            self.default_ego_index,
        )
        self.receiver_feature_cache[cache_key] = {
            "feature": recovered_feature.detach().clone(),
            "valid_flat": valid_flat.detach().clone(),
            "frame_id": frame_id,
            "original_shape": (int(C), int(H), int(W)),
        }

    @staticmethod
    def _compact_unit_packet_coverage(
        compact_meta: Optional[Dict[str, Any]],
        packet_result: Any,
        recovered_source_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        info = {
            "supported": False,
            "reason": "unsupported_layout",
            "unit_bytes": None,
            "units_per_packet": None,
        }
        if (
            not isinstance(compact_meta, dict)
            or compact_meta.get("layout") != "KC"
        ):
            return None, info

        num_units = int(compact_meta.get("num_tokens", 0))
        original_num_bytes = int(packet_result.original_num_bytes)
        packet_size_bytes = int(packet_result.packet_size_bytes)
        if num_units <= 0:
            info["reason"] = "empty_payload"
            return torch.zeros(
                0,
                dtype=torch.bool,
                device=recovered_source_mask.device,
            ), info
        if original_num_bytes <= 0 or original_num_bytes % num_units != 0:
            info["reason"] = "non_integral_unit_bytes"
            return None, info

        unit_bytes = int(original_num_bytes // num_units)
        if unit_bytes <= 0:
            info["reason"] = "invalid_unit_bytes"
            return None, info

        source_mask = recovered_source_mask.to(dtype=torch.bool).flatten()
        unit_index = torch.arange(
            num_units,
            device=source_mask.device,
            dtype=torch.long,
        )
        first_packet = torch.div(unit_index * unit_bytes, packet_size_bytes, rounding_mode="floor")
        last_packet = (
            torch.div((unit_index + 1) * unit_bytes - 1, packet_size_bytes, rounding_mode="floor")
        )

        missing = (~source_mask).to(dtype=torch.long)
        prefix = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=source_mask.device),
                torch.cumsum(missing, dim=0),
            ],
            dim=0,
        )
        missing_count = prefix[last_packet + 1] - prefix[first_packet]
        valid_units = missing_count == 0

        info.update({
            "supported": True,
            "reason": "ok",
            "unit_bytes": int(unit_bytes),
            "units_per_packet": (
                int(packet_size_bytes // unit_bytes)
                if packet_size_bytes % unit_bytes == 0
                else None
            ),
        })
        return valid_units, info

    def _apply_receiver_temporal_cache(
        self,
        recovered_feature: torch.Tensor,
        compact_meta: Optional[Dict[str, Any]],
        current_unit_valid_mask: Optional[torch.Tensor],
        recovered_source_mask: torch.Tensor,
        packet_result: Any,
        cache_enabled: int,
        link_id: Any,
        agent_index: Optional[int],
        ego_index: Optional[int],
        frame_id: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        stats = {
            "enabled": bool(cache_enabled),
            "cache_status": "disabled",
            "cache_hit": False,
            "num_total_units": 0,
            "num_current_recovered_units": 0,
            "num_temporal_filled_units": 0,
            "num_effective_recovered_units": 0,
            "num_temporal_filled_packets": 0,
            # Legacy aliases are retained for compatibility:
            # q_cache is packet-level; q_eff is unit-level.
            "q_cache": 0.0,
            "q_eff": 0.0,
            "q_recv_unit": 0.0,
            "q_cache_unit": 0.0,
            "q_eff_unit": 0.0,
            "q_recv_packet": 0.0,
            "q_cache_packet": 0.0,
            "q_eff_packet": 0.0,
        }
        if not torch.is_tensor(current_unit_valid_mask):
            stats["cache_status"] = "unsupported_payload"
            return recovered_feature, stats

        current_valid = current_unit_valid_mask.to(
            device=recovered_feature.device,
            dtype=torch.bool,
        ).flatten()
        num_units = int(current_valid.numel())
        num_current_units = int(current_valid.sum().item())
        q_recv_unit = float(
            current_valid.float().mean().item()
            if num_units > 0 else 1.0
        )

        source_mask_for_quality = recovered_source_mask.to(
            device=recovered_feature.device,
            dtype=torch.bool,
        ).flatten()
        num_source_for_quality = int(source_mask_for_quality.numel())
        q_recv_packet = float(
            source_mask_for_quality.float().mean().item()
            if num_source_for_quality > 0 else 1.0
        )

        stats["num_total_units"] = int(num_units)
        stats["num_current_recovered_units"] = int(num_current_units)
        stats["num_effective_recovered_units"] = int(num_current_units)
        stats["q_recv_unit"] = float(q_recv_unit)
        stats["q_eff_unit"] = float(q_recv_unit)
        stats["q_recv_packet"] = float(q_recv_packet)
        stats["q_eff_packet"] = float(q_recv_packet)
        stats["q_eff"] = float(q_recv_unit)

        if not int(cache_enabled):
            return recovered_feature, stats
        if (
            not isinstance(compact_meta, dict)
            or compact_meta.get("layout") != "KC"
            or not torch.is_tensor(compact_meta.get("indices"))
        ):
            stats["cache_status"] = "unsupported_payload"
            return recovered_feature, stats

        entry, cache_status = self._get_receiver_cache_entry(
            link_id,
            agent_index,
            ego_index,
            frame_id,
        )
        stats["cache_status"] = cache_status
        if entry is None:
            return recovered_feature, stats

        cached_feature = entry.get("feature")
        cached_valid = entry.get("valid_flat")
        if (
            not torch.is_tensor(cached_feature)
            or not torch.is_tensor(cached_valid)
            or tuple(cached_feature.shape) != tuple(recovered_feature.shape)
        ):
            stats["cache_status"] = "shape_mismatch"
            return recovered_feature, stats

        C, H, W = recovered_feature.shape
        unit_ids = compact_meta["indices"].to(
            device=recovered_feature.device,
            dtype=torch.long,
        ).flatten()
        count = min(int(unit_ids.numel()), num_units)
        unit_ids = unit_ids[:count]
        current_valid = current_valid[:count]
        cached_valid = cached_valid.to(
            device=recovered_feature.device,
            dtype=torch.bool,
        ).flatten()

        temporal_unit_mask = (
            (~current_valid)
            & cached_valid[unit_ids]
        )
        num_temporal_units = int(temporal_unit_mask.sum().item())
        if num_temporal_units > 0:
            out_flat = recovered_feature.reshape(C, H * W)
            cached_flat = cached_feature.to(
                device=recovered_feature.device,
                dtype=recovered_feature.dtype,
            ).reshape(C, H * W)
            fill_ids = unit_ids[temporal_unit_mask]
            out_flat[:, fill_ids] = cached_flat[:, fill_ids]

        effective_valid = current_valid | temporal_unit_mask
        source_mask = recovered_source_mask.to(
            device=recovered_feature.device,
            dtype=torch.bool,
        ).flatten()
        num_source_packets = int(source_mask.numel())
        temporal_packets = 0
        if count > 0 and num_source_packets > 0:
            original_num_bytes = int(packet_result.original_num_bytes)
            packet_size_bytes = int(packet_result.packet_size_bytes)
            if original_num_bytes % count == 0:
                unit_bytes = int(original_num_bytes // count)
                packet_index = torch.arange(
                    num_source_packets,
                    device=recovered_feature.device,
                    dtype=torch.long,
                )
                first_unit = torch.div(packet_index * packet_size_bytes, unit_bytes, rounding_mode="floor")
                last_byte = torch.minimum(
                    (packet_index + 1) * packet_size_bytes,
                    torch.full_like(
                        packet_index,
                        original_num_bytes,
                    ),
                ) - 1
                last_unit = torch.div(last_byte, unit_bytes, rounding_mode="floor")
                first_unit = torch.clamp(first_unit, 0, count - 1)
                last_unit = torch.clamp(last_unit, 0, count - 1)

                invalid = (~effective_valid).to(dtype=torch.long)
                prefix = torch.cat(
                    [
                        torch.zeros(
                            1,
                            dtype=torch.long,
                            device=recovered_feature.device,
                        ),
                        torch.cumsum(invalid, dim=0),
                    ],
                    dim=0,
                )
                packet_complete = (
                    prefix[last_unit + 1] - prefix[first_unit]
                ) == 0
                temporal_packets = int(
                    ((~source_mask) & packet_complete).sum().item()
                )

        q_recv_unit = float(
            current_valid.float().mean().item()
            if count > 0 else 1.0
        )
        q_cache_unit = float(
            num_temporal_units / max(count, 1)
        )
        q_eff_unit = float(
            effective_valid.float().mean().item()
            if count > 0 else 1.0
        )

        q_recv_packet = float(
            source_mask.float().mean().item()
            if num_source_packets > 0 else 1.0
        )
        q_cache_packet = float(
            temporal_packets / max(num_source_packets, 1)
        )
        q_eff_packet = float(
            (
                int(source_mask.sum().item())
                + int(temporal_packets)
            ) / max(num_source_packets, 1)
        )

        stats.update({
            "cache_hit": bool(num_temporal_units > 0),
            "num_total_units": int(count),
            "num_temporal_filled_units": int(num_temporal_units),
            "num_effective_recovered_units": int(effective_valid.sum().item()),
            "num_temporal_filled_packets": int(temporal_packets),
            "q_recv_unit": float(q_recv_unit),
            "q_cache_unit": float(q_cache_unit),
            "q_eff_unit": float(q_eff_unit),
            "q_recv_packet": float(q_recv_packet),
            "q_cache_packet": float(q_cache_packet),
            "q_eff_packet": float(q_eff_packet),
            "q_cache": float(q_cache_packet),
            "q_eff": float(q_eff_unit),
        })
        return recovered_feature, stats

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def _system_budget_bytes(self) -> float:
        return float(
            self.system_budget_mbps
            * 1_000_000.0
            * (self.tx_window_ms / 1000.0)
            / 8.0
        )

    def _per_link_budget_bytes(self, num_collaborators: int) -> float:
        if num_collaborators <= 0:
            return 0.0
        return float(self._system_budget_bytes() / float(num_collaborators))


    def _record_budget_scope(self) -> str:
        """Return the frame-budget source used for audit records."""
        raw_cfg = getattr(self, "arce_cfg_raw", {}) or {}
        scheduler_cfg = raw_cfg.get("scheduler", {}) or {}
        budget_source = str(
            scheduler_cfg.get(
                "budget_source",
                raw_cfg.get("budget_source", getattr(self, "budget_source", "system_budget")),
            )
        ).lower()
        budget_scope = str(
            scheduler_cfg.get(
                "budget_scope",
                raw_cfg.get("budget_scope", getattr(self, "budget_scope", "system_equal_split")),
            )
        ).lower()

        if (
            budget_source in ("channel_profiles", "channel_profile", "markov_channel")
            or budget_scope in ("global_sum_link", "channel_profiles", "channel_profile")
        ):
            return "channel_profile_frame_budget"
        return str(getattr(self, "budget_scope", "system_equal_split"))

    def _record_frame_budget_bytes(self, channel_state=None) -> float:
        """Return the frame-level budget used before fixed equal-split allocation."""
        profile = None
        if channel_state is not None and hasattr(self, "_profile_for_state"):
            try:
                profile = self._profile_for_state(channel_state)
            except Exception:
                profile = None
        return float(
            self._frame_budget_bytes_from_channel_profile(
                profile,
                budget_bytes=None,
                channel_state=channel_state,
            )
        )


    def _frame_budget_bytes_from_channel_profile(
        self,
        channel_profile: Dict[str, Any],
        budget_bytes: Optional[float] = None,
        channel_state: Optional[str] = None,
    ) -> float:

        # C2MAB explicit allocated budget has highest priority.
        # When C2MAB / oracle passes an explicit allocated budget for the
        # selected sender-action proposal, it must override channel-state
        # default budgets. Otherwise INT4/INT8 low-rate actions are selected
        # by the oracle but the executor still transmits using the full
        # medium/bad/good channel budget.
        if budget_bytes is not None:
            return float(max(0.0, budget_bytes))
        """
        Return frame-level byte budget.

        If scheduler.budget_source == channel_profiles, derive budget from
        current Markov channel state:
            good   -> profiles.good.bandwidth_mbps
            medium -> profiles.medium.bandwidth_mbps
            bad    -> profiles.bad.bandwidth_mbps

        Otherwise keep legacy fixed system_budget_mbps behavior.
        """
        raw_cfg = getattr(self, "arce_cfg_raw", {}) or {}
        scheduler_cfg = raw_cfg.get("scheduler", {}) or {}
        if not isinstance(scheduler_cfg, dict):
            scheduler_cfg = {}

        budget_source = str(
            scheduler_cfg.get(
                "budget_source",
                raw_cfg.get("budget_source", getattr(self, "budget_source", "system_budget")),
            )
        ).strip().lower()

        budget_scope = str(
            scheduler_cfg.get(
                "budget_scope",
                raw_cfg.get("budget_scope", getattr(self, "budget_scope", "system_equal_split")),
            )
        ).strip().lower()

        use_channel_profiles = (
            budget_source in ("channel_profiles", "channel_profile", "profiles")
            or budget_scope in ("global_sum_link", "channel_profiles", "channel_profile")
        )

        if not use_channel_profiles:
            if budget_bytes is not None:
                return float(max(0.0, budget_bytes))
            return float(self._system_budget_bytes())

        profiles = raw_cfg.get("profiles", {}) or {}
        if not profiles and isinstance(raw_cfg.get("channel", None), dict):
            profiles = raw_cfg.get("channel", {}).get("profiles", {}) or {}

        if not isinstance(profiles, dict):
            profiles = {}

        state = None
        if channel_state is not None:
            state = str(channel_state).strip().lower()

        if state is None and isinstance(channel_profile, dict):
            for key in ("state", "name", "channel_state"):
                if channel_profile.get(key) is not None:
                    state = str(channel_profile.get(key)).strip().lower()
                    break

        if state in profiles:
            channel_profile = profiles[state]

        if channel_profile is None:
            channel_profile = {}

        bandwidth_mbps = float(channel_profile.get("bandwidth_mbps", 0.0) or 0.0)

        if bandwidth_mbps <= 0:
            return float(self._system_budget_bytes())

        tx_window_ms = float(
            scheduler_cfg.get(
                "tx_window_ms",
                scheduler_cfg.get(
                    "frame_interval_ms",
                    raw_cfg.get("tx_window_ms", raw_cfg.get("frame_interval_ms", 100.0)),
                ),
            )
        )

        return float(bandwidth_mbps * 1e6 / 8.0 * (tx_window_ms / 1000.0))

    def _link_budget_bytes_for_state(
        self,
        channel_state: Optional[str],
        num_collaborators: int,
    ) -> float:
        """Resolve the fixed-baseline per-link budget with the same channel config.

        Fixed baseline sends every non-ego collaborator. Under global_sum_link /
        channel_profiles, the frame-level channel budget is shared equally across
        all fixed-send collaborators. This keeps the total frame budget comparable
        with ARCE-C2MAB while preserving the fixed action policy.
        """
        if int(num_collaborators) <= 0:
            return 0.0

        raw_cfg = getattr(self, "arce_cfg_raw", {}) or {}
        scheduler_cfg = raw_cfg.get("scheduler", {}) or {}
        if not isinstance(scheduler_cfg, dict):
            scheduler_cfg = {}

        budget_source = str(
            scheduler_cfg.get(
                "budget_source",
                raw_cfg.get("budget_source", getattr(self, "budget_source", "system_budget")),
            )
        ).strip().lower()

        budget_scope = str(
            scheduler_cfg.get(
                "budget_scope",
                raw_cfg.get("budget_scope", getattr(self, "budget_scope", "system_equal_split")),
            )
        ).strip().lower()

        use_channel_profiles = (
            budget_source in ("channel_profiles", "channel_profile", "profiles")
            or budget_scope in ("global_sum_link", "channel_profiles", "channel_profile")
        )

        if use_channel_profiles:
            state = self._normalize_state_name(channel_state)
            profile = self._profile_for_state(state)
            frame_budget = self._frame_budget_bytes_from_channel_profile(
                channel_profile=profile,
                budget_bytes=None,
                channel_state=state,
            )
        else:
            frame_budget = self._system_budget_bytes()

        # Fixed baseline sends all collaborators, so split the frame budget across
        # all non-ego links. ARCE-C2MAB can concentrate this same frame budget on
        # a selected super-arm through its oracle.
        return float(frame_budget / float(max(1, int(num_collaborators))))

    def _select_encoded_packets_by_budget(
        self,
        encoded_packets: torch.Tensor,
        budget_bytes: float,
        packet_size_bytes: int,
    ) -> torch.Tensor:
        num_packets = int(encoded_packets.shape[0])
        if num_packets == 0:
            return torch.zeros((0,), dtype=torch.bool, device=encoded_packets.device)

        if math.isinf(float(budget_bytes)):
            return torch.ones((num_packets,), dtype=torch.bool, device=encoded_packets.device)

        budget_bytes = float(max(0.0, budget_bytes))
        max_packets = int(math.floor(budget_bytes / float(packet_size_bytes)))
        max_packets = max(0, min(num_packets, max_packets))

        mask = torch.zeros((num_packets,), dtype=torch.bool, device=encoded_packets.device)
        if max_packets > 0:
            mask[:max_packets] = True
        return mask

    # ------------------------------------------------------------------
    # FEC helpers
    # ------------------------------------------------------------------

    def _build_fec_codec(self, action: Any):
        fec_type = self._get_action_fec_type(action)
        rho = self._get_action_redundancy_ratio(action)
        group_size = self._get_action_xor_group_size(action)
        decode_overhead = self._get_action_decode_overhead(action)

        if fec_type == FEC_TYPE_NONE or rho <= 0.0:
            return None, {
                "enabled": False,
                "type": "none",
                "fec_type": "none",
                "redundancy_ratio": 0.0,
                "group_size": group_size,
                "decode_overhead": decode_overhead,
            }

        fec_cfg = {
            "enabled": True,
            "type": fec_type,
            "redundancy_ratio": float(rho),
            "group_size": int(group_size),
            "xor_group_size": int(group_size),
            "decode_overhead": float(decode_overhead),
            "seed": int(self.seed),
        }

        extra = getattr(action, "extra", None)
        if isinstance(extra, dict):
            for key in (
                "degree_distribution",
                "robust_soliton_c",
                "robust_soliton_delta",
                "fixed_degree",
                "max_degree",
                "num_repair_packets",
                "repair_packets",
                "max_decode_iters",
                "seed",
            ):
                if key in extra:
                    fec_cfg[key] = extra[key]

        if fec_type == FEC_TYPE_XOR:
            return XORFEC({"fec": fec_cfg}), fec_cfg

        if fec_type == FEC_TYPE_RAPTORQ:
            fec_cfg.update({
                "type": FEC_TYPE_RAPTORQ,
                "codec": "raptorq",
                "standard": "RFC6330",
                "source_packets_per_block": int(
                    self.raptorq_block_source_packets
                ),
                "source_packet_bytes": int(
                    self.raptorq_transport.source_packet_bytes
                ),
                "wire_packet_bytes": int(
                    self.raptorq_transport.wire_packet_bytes
                ),
            })
            return self.raptorq_transport, fec_cfg

        if fec_type == FEC_TYPE_RAPTOR_SIM:
            fec_cfg["type"] = FEC_TYPE_RAPTOR_SIM
            return RaptorSimFEC({"fec": fec_cfg}), fec_cfg

        raise ValueError(f"Unsupported FEC type for byte-stream ARCE: {fec_type}")

    def _make_no_fec_encode_result(
        self,
        source_packets: torch.Tensor,
    ) -> FECEncodeResult:
        metas = [
            EncodedPacketMeta(
                encoded_id=i,
                kind="source",
                source_id=i,
                group_id=None,
                source_ids=(i,),
                note="direct source packet without FEC",
            )
            for i in range(int(source_packets.shape[0]))
        ]

        result = FECEncodeResult(
            source_packets=source_packets,
            encoded_packets=source_packets.clone(),
            encoded_metas=metas,
            fec_type=FEC_TYPE_NONE,
            redundancy_ratio_config=0.0,
            group_size=None,
            decode_overhead=0.0,
            info={
                "fec_type": FEC_TYPE_NONE,
                "enabled": False,
                "num_source_packets": int(source_packets.shape[0]),
                "num_parity_packets": 0,
                "num_encoded_packets": int(source_packets.shape[0]),
            },
        )
        result.validate()
        return result

    def _decode_no_fec(
        self,
        encode_result: FECEncodeResult,
        receive_mask: torch.Tensor,
    ) -> FECDecodeResult:
        source_packets = encode_result.source_packets
        k = int(source_packets.shape[0])
        receive_mask = receive_mask.to(dtype=torch.bool, device=source_packets.device).flatten()

        recovered_packets = torch.zeros_like(source_packets)
        direct_received_source_mask = torch.zeros(k, dtype=torch.bool, device=source_packets.device)

        if k > 0:
            source_receive = receive_mask[:k]
            recovered_packets[source_receive] = source_packets[source_receive]
            direct_received_source_mask = source_receive.clone()

        fec_recovered_source_mask = torch.zeros(k, dtype=torch.bool, device=source_packets.device)
        recovered_source_mask = direct_received_source_mask | fec_recovered_source_mask
        missing_source_mask = ~recovered_source_mask
        loss_mask = ~receive_mask

        num_recovered = int(recovered_source_mask.sum().item())
        recovery_ratio = float(num_recovered / k) if k > 0 else 1.0

        result = FECDecodeResult(
            recovered_packets=recovered_packets,
            recovered_source_mask=recovered_source_mask,
            direct_received_source_mask=direct_received_source_mask,
            fec_recovered_source_mask=fec_recovered_source_mask,
            missing_source_mask=missing_source_mask,
            receive_mask=receive_mask,
            loss_mask=loss_mask,
            fec_type=FEC_TYPE_NONE,
            full_recovery=bool(num_recovered == k),
            recovery_ratio=float(recovery_ratio),
            info={
                "fec_type": FEC_TYPE_NONE,
                "enabled": False,
                "num_source_packets": int(k),
                "num_encoded_packets": int(receive_mask.numel()),
                "num_direct_received_source_packets": int(direct_received_source_mask.sum().item()),
                "num_fec_recovered_source_packets": 0,
                "num_missing_source_packets_after_no_fec": int(missing_source_mask.sum().item()),
            },
        )
        result.validate()
        return result

    # ------------------------------------------------------------------
    # Main one-link communication
    # ------------------------------------------------------------------



    @staticmethod
    def _to_spatial_map(value, feature, interpolation_mode):
        if value is None or not torch.is_tensor(value):
            return None

        C, H, W = feature.shape
        spatial = value.to(device=feature.device)
        if spatial.dim() == 3:
            spatial = spatial[0] if spatial.shape[0] == 1 else spatial.float().mean(dim=0)
        elif spatial.dim() != 2:
            return None

        if spatial.shape[-2:] != (H, W):
            kwargs = {
                "input": spatial.view(1, 1, spatial.shape[-2], spatial.shape[-1]).float(),
                "size": (H, W),
                "mode": interpolation_mode,
            }
            if interpolation_mode != "nearest":
                kwargs["align_corners"] = False
            spatial = torch.nn.functional.interpolate(**kwargs)[0, 0]
        return spatial

    @staticmethod
    def _descending_order(values):
        try:
            return torch.argsort(values, descending=True, stable=True)
        except TypeError:
            return torch.argsort(values, descending=True)

    @staticmethod
    def _compact_meta_for_record(compact_meta):
        if not isinstance(compact_meta, dict):
            return None

        record_meta = {
            key: copy.deepcopy(value)
            for key, value in compact_meta.items()
            if key not in ("indices", "priority")
        }
        indices = compact_meta.get("indices")
        if torch.is_tensor(indices):
            ids = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
            record_meta["unit_id_checksum"] = int(ids.sum().item()) if ids.numel() else 0
            record_meta["unit_ids_preview"] = ids[:16].tolist()
        return record_meta

    def _compact_feature_by_message_mask(
        self,
        feature,
        message_mask,
        priority_map=None,
        action=None,
        budget_bytes=None,
        channel_profile=None,
    ):
        """
        Budget-aware compact packing for Where2comm messages.

        The binary Where2Comm mask defines the candidate set. The continuous
        sender-side confidence map only orders candidates inside that set.
        """
        try:
            cfg = (getattr(self, "arce_cfg_raw", {}) or {}).get("compact_sparse", {}) or {}
        except Exception:
            cfg = {}

        if is_payload_native_transport(getattr(self, "transport_mode", "")):
            return feature, None

        if not bool(cfg.get("enabled", False)):
            return feature, None

        if feature.dim() != 3:
            return feature, None

        C, H, W = feature.shape
        importance_result = None
        priority_layout_enabled = bool(
            cfg.get(
                "priority_layout_enabled",
                (getattr(self, "arce_cfg_raw", {}) or {}).get(
                    "priority_layout_enabled", False
                ),
            )
        )
        if self.spatial_importance.enabled:
            if not priority_layout_enabled:
                raise ValueError(
                    "ARCE spatial importance requires KC priority layout."
                )
            importance_result = self.spatial_importance.compute(feature)
            candidate2d = importance_result.candidate_mask[0]
            priority2d = importance_result.priority_map[0]
            candidate_threshold = 0.5
            flat_candidate = candidate2d.reshape(-1).float()
            flat_priority = priority2d.reshape(-1).float()
            cand = torch.nonzero(
                flat_candidate > candidate_threshold,
                as_tuple=False,
            ).flatten()
        elif priority_layout_enabled:
            if message_mask is None or not torch.is_tensor(message_mask):
                return feature, None
            candidate2d = self._to_spatial_map(message_mask, feature, "nearest")
            if candidate2d is None:
                return feature, None

            require_native_priority = bool(cfg.get("require_native_priority", True))
            priority2d = self._to_spatial_map(priority_map, feature, "bilinear")
            if priority2d is None:
                if require_native_priority:
                    raise ValueError(
                        "compact_sparse requires a sender-side priority_map separate "
                        "from the binary candidate mask."
                    )
                priority2d = candidate2d.float()

            candidate_threshold = float(cfg.get("candidate_threshold", 0.5))
            flat_candidate = candidate2d.reshape(-1).float()
            flat_priority = priority2d.reshape(-1).float()
            cand = torch.nonzero(
                flat_candidate > candidate_threshold,
                as_tuple=False,
            ).flatten()

        else:
            if message_mask is None or not torch.is_tensor(message_mask):
                return feature, None
            score2d = self._to_spatial_map(
                priority_map if torch.is_tensor(priority_map) else message_mask,
                feature,
                "bilinear",
            )
            if score2d is None:
                return feature, None
            candidate_threshold = float(cfg.get("threshold", 0.0))
            flat_candidate = score2d.reshape(-1).float()
            flat_priority = flat_candidate
            cand = torch.nonzero(
                flat_candidate > candidate_threshold,
                as_tuple=False,
            ).flatten()
            if cand.numel() == 0:
                cand = torch.arange(H * W, device=feature.device)
        scores = flat_priority[cand]
        if bool(cfg.get("sort_by_score", True)):
            cand = cand[self._descending_order(scores)]
            scores = flat_priority[cand]

        # Compute budget-aware top-K.
        # budget_bytes can be passed by executor; if absent, use full candidate set.
        dynamic_topk = bool(cfg.get("budget_aware_topk", False))
        max_tokens_cfg = int(cfg.get("max_tokens", -1))

        K_budget = cand.numel()
        used_budget_bytes = None

        if dynamic_topk and budget_bytes is not None:
            b = float(max(0.0, budget_bytes))

            # Quantization byte width.
            q = "fp16"
            try:
                q = str(self._get_action_quant_mode(action)).lower()
            except Exception:
                try:
                    q = str(_safe_get_action_field(action, "quant_mode", "fp16")).lower()
                except Exception:
                    q = "fp16"

            if q in ("fp32", "float32", "none", "raw"):
                bytes_per_value = 4.0
            elif q in ("fp16", "float16", "half"):
                bytes_per_value = 2.0
            elif q in ("int8", "8bit", "int8_uniform"):
                bytes_per_value = 1.0
            elif q in ("int4", "4bit", "int4_uniform"):
                bytes_per_value = 0.5
            else:
                bytes_per_value = 2.0

            # Redundancy ratio.
            try:
                rho = float(_safe_get_action_field(action, "redundancy_ratio", _safe_get_action_field(action, "rho", 0.0)))
            except Exception:
                rho = 0.0

            token_cost = float(C) * bytes_per_value * float(1.0 + max(0.0, rho))
            if token_cost <= 0:
                token_cost = float(C) * 2.0

            K_budget = int(b // token_cost)
            K_budget = max(1, min(int(cand.numel()), K_budget))
            used_budget_bytes = float(K_budget * token_cost)

        if max_tokens_cfg > 0:
            K_budget = min(K_budget, max_tokens_cfg)

        selected = cand[:K_budget]

        selected_scores = (
            flat_priority[selected]
            if selected.numel() > 0
            else flat_priority.new_empty((0,))
        )
        selected_inside = int(
            (flat_candidate[selected] > candidate_threshold).sum().item()
        )
        selected_total = int(selected.numel())
        selected_outside = int(selected_total - selected_inside)

        flat_feature = feature.reshape(C, H * W)
        if priority_layout_enabled:
            compact_feature = (
                flat_feature[:, selected].transpose(0, 1).contiguous()
            )
            layout, channel_dim = "KC", 1
            if importance_result is not None:
                candidate_source = importance_result.candidate_source
                priority_source = importance_result.priority_source
            else:
                candidate_source = "where2comm_binary_mask"
                priority_source = "where2comm_sender_confidence"
        else:
            compact_feature = (
                flat_feature[:, selected]
                .contiguous()
                .view(C, int(selected.numel()), 1)
            )
            layout, channel_dim = "CK1", 0
            candidate_source = "legacy_masked_confidence"
            priority_source = "legacy_masked_confidence"

        compact_meta = {
            "enabled": True,
            "empty_candidate": bool(selected.numel() == 0),
            "indices": selected,
            "priority": selected_scores,
            "layout": layout,
            "channel_dim": int(channel_dim),
            "priority_layout_enabled": bool(priority_layout_enabled),
            "candidate_source": candidate_source,
            "priority_source": priority_source,
            "spatial_importance": (
                copy.deepcopy(importance_result.stats)
                if importance_result is not None else None
            ),
            "original_shape": (int(C), int(H), int(W)),
            "num_tokens": int(selected.numel()),
            "num_total_tokens": int(H * W),
            "mask_ratio": float(selected.numel() / max(H * W, 1)),
            "candidate_threshold": float(candidate_threshold),
            "budget_aware_topk": bool(dynamic_topk),
            "budget_bytes": float(budget_bytes) if budget_bytes is not None else None,
            "estimated_payload_budget_bytes": used_budget_bytes,
            "mask_alignment": {
                "selected_tokens_total": int(selected_total),
                "selected_tokens_inside_mask": int(selected_inside),
                "selected_tokens_outside_mask": int(selected_outside),
                "outside_mask_ratio": float(selected_outside / max(selected_total, 1)),
                "selected_score_min": float(selected_scores.min().detach().cpu()) if selected_total > 0 else None,
                "selected_score_max": float(selected_scores.max().detach().cpu()) if selected_total > 0 else None,
                "candidate_tokens_total": int(cand.numel()),
                "candidate_ratio": float(cand.numel() / max(H * W, 1)),
            },
        }

        return compact_feature, compact_meta


    def _scatter_compact_feature(self, compact_feature, compact_meta, reference_feature):
        """
        Scatter recovered compact message [K,C] back to dense [C,H,W].
        Missing/unreceived tokens are already zero-filled in compact_feature.
        """
        if compact_meta is None or not compact_meta.get("enabled", False):
            return compact_feature

        indices = compact_meta.get("indices", None)
        if indices is None or not torch.is_tensor(indices):
            return compact_feature

        C, H, W = reference_feature.shape
        out = torch.zeros_like(reference_feature)

        out_flat = out.reshape(C, H * W)
        if compact_meta.get("layout") == "KC":
            vals = compact_feature.reshape(-1, C)
            K = min(vals.shape[0], indices.numel())
            if K <= 0:
                return out
            out_flat[:, indices[:K].to(out_flat.device)] = (
                vals[:K].transpose(0, 1).to(out_flat.dtype)
            )
        else:
            vals = compact_feature.reshape(C, -1)
            K = min(vals.shape[1], indices.numel())
            if K <= 0:
                return out
            out_flat[:, indices[:K].to(out_flat.device)] = (
                vals[:, :K].to(out_flat.dtype)
            )
        return out

    def communicate_feature(
        self,
        feature: torch.Tensor,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        agent_index: Optional[int] = None,
        ego_index: Optional[int] = None,
        channel_state: Optional[str] = None,
        action_override: Optional[ARCEAction] = None,
        budget_bytes: Optional[float] = None,
        message_mask: Optional[torch.Tensor] = None,
        priority_map: Optional[torch.Tensor] = None,
        complementarity: float = 0.0,
        update_cache: bool = True,
        return_result: bool = False,
    ):
        feature = _require_tensor(feature, "feature")
        if feature.dim() != 3:
            raise ValueError(
                "communicate_feature expects one feature with shape [C, H, W], "
                f"got {tuple(feature.shape)}."
            )

        if ego_index is None:
            ego_index = self.default_ego_index
        if agent_index is None:
            agent_index = -1

        requested_channel_state = (
            None if channel_state is None else self._normalize_state_name(channel_state)
        )

        active_channel_state, channel_state_source = self._resolve_active_channel_state(
            requested_channel_state=requested_channel_state,
            link_id=link_id,
            frame_id=frame_id,
        )

        apply_to_this_agent = should_apply_to_agent(
            agent_index=agent_index,
            ego_index=ego_index,
            link_scope=self.link_scope,
        )

        base_record = {
            "frame_id": frame_id,
            "link_id": repr(link_id),
            "agent_index": int(agent_index),
            "ego_index": int(ego_index),
            "input_shape": tuple(int(x) for x in feature.shape),
            "input_dtype": str(feature.dtype),
            "device": str(feature.device),
            "arce_enabled": bool(self.enabled),
            "arce_mode": self.mode,
            "applied": bool(apply_to_this_agent),
            "channel_state": active_channel_state,
            "requested_channel_state": requested_channel_state,
            "channel_state_source": channel_state_source,
        }

        if (not self.enabled) or self.mode == ARCE_MODE_DISABLED:
            record = copy.deepcopy(base_record)
            record.update(
                {
                    "bypassed": True,
                    "bypass_reason": "ARCE disabled",
                    "output_shape": tuple(int(x) for x in feature.shape),
                }
            )
            self.num_bypassed_links += 1
            self._append_record(record)
            result = ARCECommResult(recovered_feature=feature, record=record)
            return result if return_result else (feature, record)

        if self.mode == ARCE_MODE_BYPASS or not apply_to_this_agent:
            reason = (
                "ARCE bypass mode"
                if self.mode == ARCE_MODE_BYPASS
                else "agent not in ARCE link scope"
            )
            record = copy.deepcopy(base_record)
            record.update(
                {
                    "bypassed": True,
                    "bypass_reason": reason,
                    "output_shape": tuple(int(x) for x in feature.shape),
                }
            )
            self.num_bypassed_links += 1
            self._append_record(record)
            result = ARCECommResult(recovered_feature=feature, record=record)
            return result if return_result else (feature, record)

        self.num_processed_links += 1

        channel_profile = self._profile_for_state(active_channel_state)
        bandwidth_mbps = float(channel_profile.get("bandwidth_mbps", 0.0))

        if action_override is not None:
            action = action_override
            action_source = "override"
        else:
            # For a fair fixed baseline, YAML fixed_action must override
            # the default FixedARCEPolicy state-action table.
            fixed_action_cfg = None
            if str(self.policy_name).strip().lower() == "fixed":
                fixed_action_cfg = self.arce_cfg_raw.get("fixed_action", None)

            if isinstance(fixed_action_cfg, dict) and int(fixed_action_cfg.get("send", 1)) == 1:
                action = fixed_action_cfg
                action_source = "fixed_action"
            else:
                action = self.action_policy.select(channel_profile=channel_profile)
                action_source = str(self.policy_name)

        action = normalize_runtime_action(action)
        action_dict = runtime_action_as_dict(action)

        cache_enabled = int(_safe_get_action_field(action, "cache_enabled", 0))
        send_flag = int(_safe_get_action_field(action, "send", 1))
        if send_flag == 0:
            recovered_feature = torch.zeros_like(feature)

            if update_cache:
                self._update_prev_feature_cache(
                    feature=feature,
                    link_id=link_id,
                    agent_index=agent_index,
                    ego_index=ego_index,
                )

            record = copy.deepcopy(base_record)
            record.update(
                {
                    "bypassed": False,
                    "bypass_reason": None,
                    "action_source": action_source,
                    "action": action_dict,
                    "no_send": True,
                    "output_shape": tuple(int(x) for x in recovered_feature.shape),
                    "output_dtype": str(recovered_feature.dtype),
                    "tx_bytes": 0.0,
                    "rx_bytes": 0.0,
                    "raw_bytes": int(feature.numel() * feature.element_size()),
                    "compressed_bytes": 0.0,
                    "encoded_bytes": 0.0,
                    "received_bytes": 0.0,
                    "effective_received_bytes": 0.0,
                    "packetization": {
                        "mode": "byte_stream",
                        "num_packets": 0,
                    },
                    "fec_encode": {
                        "enabled": False,
                        "fec_type": "none",
                        "num_source_packets": 0,
                        "num_parity_packets": 0,
                        "num_encoded_packets": 0,
                    },
                    "fec_decode": {
                        "enabled": False,
                        "fec_type": "none",
                        "num_fec_recovered_source_packets": 0,
                    },
                    "packet": {
                        "num_source_packets": 0,
                        "num_parity_packets": 0,
                        "num_encoded_packets": 0,
                        "num_received_packets": 0,
                    },
                    "size": {
                        "raw_numel": int(feature.numel()),
                        "raw_bytes_fp32_reference": float(feature.numel() * 4),
                        "compressed_bytes": 0.0,
                        "actual_transmitted_bytes": 0.0,
                        "actual_received_bytes": 0.0,
                        "actual_num_source_packets": 0,
                        "actual_num_parity_packets": 0,
                        "actual_num_encoded_packets": 0,
                        "actual_num_lost_encoded_packets": 0,
                        "bandwidth_budget_bytes": float(budget_bytes)
                        if budget_bytes is not None
                        else None,
                    },
                    "quality": {
                        "q_recv": 0.0,
                        "q_cache": 0.0,
                        "num_source_packets": 0,
                        "num_still_missing": 0,
                    },
                    "late": False,
                    "dropped_by_late": False,
                }
            )

            self._append_record(record)
            result = ARCECommResult(recovered_feature=recovered_feature, record=record)
            return result if return_result else (recovered_feature, record)

        # 1. State-aware temporal source.
        # cache_enabled=0 strictly uses current feature and forbids previous_frame.
        # cache_enabled=1 allows state-aware temporal source such as previous_frame.
        if int(cache_enabled):
            feature_tx, temporal_source = self._get_temporal_tx_feature(
                feature=feature,
                state_name=active_channel_state,
                link_id=link_id,
                agent_index=agent_index,
                ego_index=ego_index,
            )
        else:
            feature_tx = feature
            temporal_source = "current_cache_disabled"


        # 1.5. Where2comm mask-guided compact message packing.
        # GRACE/ARCE transmits only selected Where2comm tokens instead of the full dense feature map.
        compact_meta = None
        feature_tx, compact_meta = self._compact_feature_by_message_mask(
            feature_tx,
            message_mask,
            priority_map=priority_map,
            action=action,
            budget_bytes=self._frame_budget_bytes_from_channel_profile(
                channel_profile=channel_profile,
                budget_bytes=budget_bytes,
                channel_state=active_channel_state,
            ),
            channel_profile=channel_profile,
        )

        if (
            isinstance(compact_meta, dict)
            and compact_meta.get("priority_layout_enabled")
            and compact_meta.get("empty_candidate")
        ):
            recovered_feature = torch.zeros_like(feature)
            frame_budget_bytes = (
                self._frame_budget_bytes_from_channel_profile(
                    channel_profile,
                    budget_bytes,
                    active_channel_state,
                )
            )
            if update_cache:
                self._update_prev_feature_cache(
                    feature, link_id, agent_index, ego_index
                )

            record = copy.deepcopy(base_record)
            record.update({
                "bypassed": False,
                "action_source": action_source,
                "action": action_dict,
                "no_send": False,
                "no_effective_send": True,
                "empty_candidate": True,
                "empty_candidate_reason":
                    "where2comm_candidate_mask_empty",
                "temporal_source": temporal_source,
                "output_shape": tuple(recovered_feature.shape),
                "output_dtype": str(recovered_feature.dtype),
                "compact_sparse":
                    self._compact_meta_for_record(compact_meta),
                "quantization": {
                    "mode": str(self._get_action_quant_mode(action)),
                    "skipped": True,
                    "reason": "empty_candidate",
                },
                "packetization": {
                    "mode": "byte_stream",
                    "num_packets": 0,
                },
                "packet": {
                    "num_source_packets": 0,
                    "num_encoded_packets": 0,
                    "num_transmitted_packets": 0,
                    "num_received_packets": 0,
                },
                "bandwidth_selection": {
                    "mode": "empty_candidate",
                    "budget_bytes": float(frame_budget_bytes),
                    "source_symbol_bytes": int(
                        self.byte_packetizer.packet_size_bytes
                    ),
                    "wire_packet_bytes": None,
                    "packet_metadata_bytes": 0,
                    "scheduling": "not_executed_empty_candidate",
                    "num_fec_blocks": 0,
                    "num_tx_packets": 0,
                    "num_missing_by_budget": 0,
                },
                "size": {
                    "raw_numel": int(feature.numel()),
                    "raw_bytes_fp32_reference":
                        float(feature.numel() * 4),
                    "compressed_bytes": 0.0,
                    "actual_num_source_packets": 0,
                    "actual_num_encoded_packets": 0,
                    "actual_transmitted_bytes": 0.0,
                    "actual_received_bytes": 0.0,
                    "bandwidth_budget_bytes":
                        float(frame_budget_bytes),
                },
                "quality": {"q_recv": 0.0, "q_cache": 0.0},
                "tx_bytes": 0.0,
                "rx_bytes": 0.0,
                "transmitted_bytes": 0.0,
                "received_bytes": 0.0,
                "actual_transmitted_bytes": 0.0,
                "actual_received_bytes": 0.0,
                "raw_bytes":
                    int(feature.numel() * feature.element_size()),
                "compressed_bytes": 0.0,
                "encoded_bytes": 0.0,
                "effective_received_bytes": 0.0,
                "late": False,
                "dropped_by_late": False,
            })
            self._append_record(record)
            result = ARCECommResult(recovered_feature, record)
            return (
                result
                if return_result
                else (recovered_feature, record)
            )

        # 2. Quantize first.
        quantizer = self._build_quantizer(action)
        quant_mode = self._get_action_quant_mode(action)
        if compact_meta is not None and compact_meta.get("layout") == "KC":
            quant_result = quantizer.quantize(
                feature_tx,
                mode=quant_mode,
                channel_dim=1,
            )
        else:
            quant_result = quantizer.quantize_feature(
                feature_tx,
                mode=quant_mode,
            )

        if quant_result.packed_tensor is not None:
            stream_tensor = quant_result.packed_tensor
            source_tensor_kind = "packed_int4"
        else:
            stream_tensor = quant_result.q_tensor
            source_tensor_kind = "q_tensor"

        # 3. Byte-stream packetization after quantization.
        packet_result = self.byte_packetizer.packetize(
            stream_tensor,
            source_tensor_kind=source_tensor_kind,
        )
        source_packets = packet_result.packets
        num_source_packets = int(packet_result.num_packets)
        packet_size_bytes = int(packet_result.packet_size_bytes)

        # 4. Resolve the physical link budget before FEC. RaptorQ block
        # admission jointly reserves source and repair packets under this
        # budget; estimated action cost is never used as an execution budget.
        frame_budget_bytes = self._frame_budget_bytes_from_channel_profile(
            channel_profile=channel_profile,
            budget_bytes=budget_bytes,
            channel_state=active_channel_state,
        )
        fec_codec, fec_runtime_cfg = self._build_fec_codec(action)
        block_fec_plan = None
        wire_packet_size_bytes = int(packet_size_bytes)

        if isinstance(fec_codec, PriorityBlockFECTransport):
            block_fec_plan = fec_codec.encode_under_budget(
                source_packets=source_packets,
                budget_bytes=frame_budget_bytes,
                redundancy_ratio=float(
                    fec_runtime_cfg.get("redundancy_ratio", 0.0)
                ),
            )
            encode_result = block_fec_plan
            fec_encode_dict = block_fec_plan.as_dict(
                include_blocks=True,
                include_metas=False,
            )
            encoded_packets = block_fec_plan.encoded_packets
            num_encoded_packets = int(block_fec_plan.num_encoded_packets)
            num_parity_packets = int(block_fec_plan.num_repair_packets)
            wire_packet_size_bytes = int(
                block_fec_plan.wire_packet_bytes
            )
            tx_mask = torch.ones(
                num_encoded_packets,
                dtype=torch.bool,
                device=encoded_packets.device,
            )
            source_tx_mask = block_fec_plan.admitted_source_mask
            parity_tx_mask = torch.ones(
                num_parity_packets,
                dtype=torch.bool,
                device=encoded_packets.device,
            )
            num_tx_source_packets = int(
                block_fec_plan.num_admitted_source_packets
            )
            num_tx_parity_packets = int(num_parity_packets)
            num_source_dropped_by_budget = int(
                block_fec_plan.num_source_dropped_by_budget
            )
            num_parity_dropped_by_budget = 0
            num_missing_by_budget = int(num_source_dropped_by_budget)
        else:
            if fec_codec is None:
                encode_result = self._make_no_fec_encode_result(source_packets)
                fec_encode_dict = encode_result.as_dict(include_metas=False)
                fec_encode_dict["enabled"] = False
            else:
                encode_result = fec_codec.encode(source_packets)
                fec_encode_dict = encode_result.as_dict(include_metas=False)
                fec_encode_dict["enabled"] = True

            encoded_packets = encode_result.encoded_packets
            num_encoded_packets = int(encode_result.num_encoded_packets)
            num_parity_packets = int(encode_result.num_parity_packets)
            tx_mask = self._select_encoded_packets_by_budget(
                encoded_packets=encoded_packets,
                budget_bytes=frame_budget_bytes,
                packet_size_bytes=wire_packet_size_bytes,
            )
            source_tx_mask = tx_mask[:num_source_packets]
            parity_tx_mask = tx_mask[num_source_packets:num_encoded_packets]
            num_tx_source_packets = int(source_tx_mask.sum().item())
            num_tx_parity_packets = int(parity_tx_mask.sum().item())
            num_source_dropped_by_budget = int(
                num_source_packets - num_tx_source_packets
            )
            num_parity_dropped_by_budget = int(
                num_parity_packets - num_tx_parity_packets
            )
            num_missing_by_budget = int(
                num_source_dropped_by_budget + num_parity_dropped_by_budget
            )

        num_tx_packets = int(tx_mask.sum().item())
        num_protected_source_packets = int(
            block_fec_plan.num_protected_source_packets
            if block_fec_plan is not None
            else 0
        )
        num_tail_source_packets = int(
            block_fec_plan.num_tail_source_packets
            if block_fec_plan is not None
            else num_tx_source_packets
        )
        protected_redundancy_ratio = float(
            block_fec_plan.protected_redundancy_ratio
            if block_fec_plan is not None
            else 0.0
        )
        overall_redundancy_ratio = float(
            block_fec_plan.overall_redundancy_ratio
            if block_fec_plan is not None
            else num_parity_packets / max(1, num_tx_source_packets)
        )

        if num_tx_source_packets + num_tx_parity_packets != num_tx_packets:
            raise RuntimeError(
                "Budget packet accounting mismatch: "
                "tx_source={} tx_parity={} tx_total={}.".format(
                    num_tx_source_packets,
                    num_tx_parity_packets,
                    num_tx_packets,
                )
            )

        # 6. Bernoulli loss only on actually transmitted encoded packets.
        full_loss_mask = torch.ones(
            (num_encoded_packets,),
            dtype=torch.bool,
            device=feature.device,
        )

        if num_tx_packets > 0:
            tx_packets = encoded_packets[tx_mask]
            raw_loss_mask_tx, channel_loss_info = self._sample_bernoulli_loss(
                num_packets=int(tx_packets.shape[0]),
                state_name=active_channel_state,
                link_id=link_id,
                frame_id=frame_id,
                device=feature.device,
            )
            full_loss_mask[tx_mask] = raw_loss_mask_tx
        else:
            raw_loss_mask_tx = torch.empty(
                (0,),
                dtype=torch.bool,
                device=feature.device,
            )
            channel_loss_info = {
                "model": "bernoulli",
                "formula": "receive_i ~ Bernoulli(1 - PLR_t)",
                "frame_id": frame_id,
                "link_id": repr(link_id),
                "channel_state": active_channel_state,
                "plr": float(self.bernoulli_loss_rates[active_channel_state]),
                "receive_prob": float(1.0 - self.bernoulli_loss_rates[active_channel_state]),
                "num_packets": 0,
                "num_received": 0,
                "num_lost": 0,
                "empirical_loss": 0.0,
                "reason": "zero_budget",
            }

        receive_mask = ~full_loss_mask
        received_packet_mask = receive_mask

        transmitted_bytes = float(num_tx_packets * wire_packet_size_bytes)
        received_bytes = float(
            int(received_packet_mask.sum().item()) * wire_packet_size_bytes
        )

        # 7. FEC decode.
        if block_fec_plan is not None:
            decode_result = fec_codec.decode(
                plan=block_fec_plan,
                receive_mask=receive_mask,
                fill_value=0,
            )
        elif fec_codec is None:
            decode_result = self._decode_no_fec(
                encode_result=encode_result,
                receive_mask=receive_mask,
            )
        else:
            decode_result = fec_codec.decode(
                encode_result=encode_result,
                receive_mask=receive_mask,
                fill_value=0,
            )

        decoded_source_packets = decode_result.recovered_packets

        # 8. Fixed delay by state.
        latency_info = self._estimate_fixed_latency(
            transmitted_bytes=transmitted_bytes,
            state_name=active_channel_state,
            link_id=link_id,
            frame_id=frame_id,
            bandwidth_mbps=bandwidth_mbps,
        )

        # 9. Rebuild quantized byte stream and dequantize.
        recovered_stream_tensor = self.byte_packetizer.unpacketize(
            packets=decoded_source_packets,
            meta=packet_result,
        )

        if source_tensor_kind == "packed_int4":
            recovered_feature = quantizer.unpack_and_dequantize_int4(
                packed_tensor=recovered_stream_tensor,
                meta=quant_result.meta,
                original_numel=int(quant_result.q_tensor.numel()),
                shape=tuple(int(x) for x in quant_result.q_tensor.shape),
                output_dtype=feature.dtype,
            )
        else:
            recovered_feature = quantizer.dequantize(
                q_tensor=recovered_stream_tensor,
                meta=quant_result.meta,
                output_dtype=feature.dtype,
            )

        # Experiment-3 read-only counterfactual: reconstruct the payload using
        # only directly received systematic source packets. FEC output used by
        # normal inference remains unchanged.
        direct_recovered_feature_compact = None
        if getattr(self, "fec_recovery_auditor", None) is not None and self.fec_recovery_auditor.enabled:
            direct_source_receive_mask = (
                decode_result.direct_received_source_mask
            )
            direct_source_packets = source_packets.clone()
            direct_source_packets[~direct_source_receive_mask] = 0
            direct_stream_tensor = self.byte_packetizer.unpacketize(
                packets=direct_source_packets,
                meta=packet_result,
            )
            if source_tensor_kind == "packed_int4":
                direct_recovered_feature_compact = quantizer.unpack_and_dequantize_int4(
                    packed_tensor=direct_stream_tensor,
                    meta=quant_result.meta,
                    original_numel=int(quant_result.q_tensor.numel()),
                    shape=tuple(int(x) for x in quant_result.q_tensor.shape),
                    output_dtype=feature.dtype,
                )
            else:
                direct_recovered_feature_compact = quantizer.dequantize(
                    q_tensor=direct_stream_tensor,
                    meta=quant_result.meta,
                    output_dtype=feature.dtype,
                )

        if update_cache:
            self._update_prev_feature_cache(
                feature=feature,
                link_id=link_id,
                agent_index=agent_index,
                ego_index=ego_index,
            )

        fec_decode_dict = decode_result.as_dict()
        num_direct_received_source_packets = int(
            decode_result.num_direct_received_source_packets
        )

        # Keep the compact receive tensor for read-only compression audit.
        # The formal output remains the dense/scattered feature below.
        recovered_feature_compact = recovered_feature
        current_unit_valid_mask, unit_coverage_info = (
            self._compact_unit_packet_coverage(
                compact_meta=compact_meta,
                packet_result=packet_result,
                recovered_source_mask=decode_result.recovered_source_mask,
            )
        )

        # 7.5. Scatter compact recovered payload back to dense BEV feature map.
        if compact_meta is not None:
            recovered_feature = self._scatter_compact_feature(
                recovered_feature,
                compact_meta,
                reference_feature=feature,
            )

        recovered_feature, temporal_cache_info = (
            self._apply_receiver_temporal_cache(
                recovered_feature=recovered_feature,
                compact_meta=compact_meta,
                current_unit_valid_mask=current_unit_valid_mask,
                recovered_source_mask=decode_result.recovered_source_mask,
                packet_result=packet_result,
                cache_enabled=cache_enabled,
                link_id=link_id,
                agent_index=agent_index,
                ego_index=ego_index,
                frame_id=frame_id,
            )
        )

        if update_cache:
            self._update_receiver_feature_cache(
                recovered_feature=recovered_feature,
                compact_meta=compact_meta,
                current_unit_valid_mask=current_unit_valid_mask,
                link_id=link_id,
                agent_index=agent_index,
                ego_index=ego_index,
                frame_id=frame_id,
            )

        num_fec_recovered_source_packets = int(
            decode_result.num_fec_recovered_source_packets
        )
        num_missing_source_packets = int(decode_result.num_missing_source_packets)
        num_recovered_source_packets = int(decode_result.num_recovered_source_packets)
        recovery_ratio = float(decode_result.recovery_ratio)
        num_temporal_filled_packets = int(
            temporal_cache_info["num_temporal_filled_packets"]
        )
        num_still_missing_packets = max(
            0,
            num_missing_source_packets - num_temporal_filled_packets,
        )

        num_lost_by_bernoulli = int(raw_loss_mask_tx.sum().item())
        num_received_encoded_packets = int(received_packet_mask.sum().item())

        q_recv = float(recovery_ratio)
        q_cache = float(temporal_cache_info["q_cache"])
        q_eff = float(temporal_cache_info["q_eff"])

        size_info = {
            "raw_numel": int(feature.numel()),
            "raw_bytes_fp32_reference": float(feature.numel() * 4),
            "quantized_num_bytes": float(packet_result.original_num_bytes),
            "compressed_bytes": float(packet_result.original_num_bytes),
            "actual_num_source_packets": int(num_source_packets),
            "actual_num_admitted_source_packets": int(num_tx_source_packets),
            "actual_num_protected_source_packets": int(
                num_protected_source_packets
            ),
            "actual_num_tail_source_packets": int(num_tail_source_packets),
            "actual_num_parity_packets": int(num_parity_packets),
            "actual_num_encoded_packets": int(num_encoded_packets),
            "actual_effective_redundancy_ratio": float(
                overall_redundancy_ratio
            ),
            "actual_protected_redundancy_ratio": float(
                protected_redundancy_ratio
            ),
            "actual_overall_redundancy_ratio": float(
                overall_redundancy_ratio
            ),
            "actual_avg_source_packet_bytes": float(
                packet_result.original_num_bytes / max(1, num_source_packets)
            ),
            "actual_parity_bytes": float(
                num_parity_packets * wire_packet_size_bytes
            ),
            "actual_metadata_bytes": float(
                num_tx_packets
                * max(0, wire_packet_size_bytes - packet_size_bytes)
            ),
            "actual_transmitted_bytes": float(transmitted_bytes),
            "actual_received_bytes": float(received_bytes),
            "actual_transmitted_mb": float(transmitted_bytes / 1_000_000.0),
            "actual_received_mb": float(received_bytes / 1_000_000.0),
            "actual_num_received_encoded_packets": int(num_received_encoded_packets),
            "actual_num_lost_encoded_packets": int(num_encoded_packets - num_received_encoded_packets),
            "actual_num_transmitted_source_packets": int(num_tx_source_packets),
            "actual_num_transmitted_parity_packets": int(num_tx_parity_packets),
            "actual_transmitted_source_bytes": float(
                num_tx_source_packets * wire_packet_size_bytes
            ),
            "actual_transmitted_parity_bytes": float(
                num_tx_parity_packets * wire_packet_size_bytes
            ),
            "num_source_dropped_by_budget": int(num_source_dropped_by_budget),
            "num_parity_dropped_by_budget": int(num_parity_dropped_by_budget),
            "num_missing_by_budget": int(num_missing_by_budget),
            "num_lost_by_bernoulli": int(num_lost_by_bernoulli),
            "num_direct_received_source_packets": int(num_direct_received_source_packets),
            "num_fec_recovered_source_packets": int(num_fec_recovered_source_packets),
            "num_missing_source_packets": int(num_missing_source_packets),
            "num_recovered_source_packets": int(num_recovered_source_packets),
            "bandwidth_budget_bytes": float(frame_budget_bytes),
            "system_budget_mbps": float(self.system_budget_mbps),
            "tx_window_ms": float(self.tx_window_ms),
        }

        record = copy.deepcopy(base_record)
        record.update(
            {
                "bypassed": False,
                "bypass_reason": None,
                "output_shape": tuple(int(x) for x in recovered_feature.shape),
                "output_dtype": str(recovered_feature.dtype),
                "action": action_dict,
                "action_source": action_source,
                "no_send": False,
                "temporal_source": temporal_source,
                "delay_policy": self.delay_policy_by_state.get(active_channel_state, "current"),
                "channel": {
                    "profile": copy.deepcopy(channel_profile),
                    "loss": copy.deepcopy(channel_loss_info),
                    "latency": copy.deepcopy(latency_info),
                    "late_policy": {
                        "late": False,
                        "late_policy": "disabled",
                        "overridden": False,
                        "reason": "Fixed state delay is used; bad state directly uses previous frame.",
                    },
                },
                "packetization": packet_result.to_meta_dict(),

                "compact_sparse": self._compact_meta_for_record(compact_meta),
                "transport": {
                    "transport_mode": str(getattr(self, "transport_mode", "compact_sparse")),
                    "mask_used_by_arce": bool(compact_meta is not None),
                    "budget_aware_topk": bool(
                        compact_meta.get("budget_aware_topk", False)
                    ) if isinstance(compact_meta, dict) else False,
                },
                "byte_stream_packetization": packet_result.to_meta_dict(),
                "quantization": quant_result.as_dict(),
                "fec_encode": fec_encode_dict,
                "fec_decode": fec_decode_dict,
                "partial_reconstruction": {
                    "enabled": True,
                    "method": (
                        "fec_then_temporal_cache_then_zero_fill"
                        if int(cache_enabled)
                        else "fec_then_zero_fill_missing_source_packets"
                    ),
                    "num_direct_received_packets": int(num_direct_received_source_packets),
                    "num_fec_recovered_packets": int(num_fec_recovered_source_packets),
                    "num_temporal_filled_packets": int(
                        num_temporal_filled_packets
                    ),
                    "num_spatial_filled_packets": 0,
                    "num_zero_filled_packets": int(num_still_missing_packets),
                    "num_still_missing": int(num_still_missing_packets),
                    "recovery_ratio": float(recovery_ratio),
                    "effective_recovery_ratio": float(q_eff),
                    "num_current_recovered_units": int(
                        temporal_cache_info["num_current_recovered_units"]
                    ),
                    "num_temporal_filled_units": int(
                        temporal_cache_info["num_temporal_filled_units"]
                    ),
                    "num_effective_recovered_units": int(
                        temporal_cache_info["num_effective_recovered_units"]
                    ),
                    "unit_packet_coverage": copy.deepcopy(
                        unit_coverage_info
                    ),
                    "temporal_cache": copy.deepcopy(
                        temporal_cache_info
                    ),
                },
                "bandwidth_selection": {
                    "mode": "system_equal_split"
                    if budget_bytes is not None
                    else self.budget_scope,
                    "budget_bytes": float(frame_budget_bytes),
                    "source_symbol_bytes": int(packet_size_bytes),
                    "wire_packet_bytes": int(wire_packet_size_bytes),
                    "packet_metadata_bytes": int(
                        max(0, wire_packet_size_bytes - packet_size_bytes)
                    ),
                    "scheduling": (
                        RAPTORQ_SCHEDULING_MODE
                        if block_fec_plan is not None
                        else "encoded_stream_prefix"
                    ),
                    "num_fec_blocks": int(
                        block_fec_plan.num_protected_blocks
                        if block_fec_plan is not None
                        else 0
                    ),
                    "num_tail_blocks": int(
                        block_fec_plan.num_tail_blocks
                        if block_fec_plan is not None
                        else 0
                    ),
                    "num_source_packets": int(num_source_packets),
                    "num_admitted_source_packets": int(num_tx_source_packets),
                    "num_protected_source_packets": int(
                        num_protected_source_packets
                    ),
                    "num_tail_source_packets": int(num_tail_source_packets),
                    "num_parity_packets": int(num_parity_packets),
                    "num_encoded_packets": int(num_encoded_packets),
                    "num_tx_packets": int(num_tx_packets),
                    "num_tx_source_packets": int(num_tx_source_packets),
                    "num_tx_parity_packets": int(num_tx_parity_packets),
                    "num_source_dropped_by_budget": int(
                        num_source_dropped_by_budget
                    ),
                    "num_parity_dropped_by_budget": int(
                        num_parity_dropped_by_budget
                    ),
                    "num_missing_by_budget": int(num_missing_by_budget),
                    "selected_encoded_packet_ratio": float(
                        num_tx_packets / max(1, num_encoded_packets)
                    ),
                    "admitted_source_packet_ratio": float(
                        num_tx_source_packets / max(1, num_source_packets)
                    ),
                    "protected_redundancy_ratio": float(
                        protected_redundancy_ratio
                    ),
                    "overall_redundancy_ratio": float(
                        overall_redundancy_ratio
                    ),
                },
                "patch_summary": {
                    "packetization": "byte_stream_not_spatial_patch",
                    "num_total_patches": int(num_source_packets),
                    "num_valid_patches": int(num_source_packets),
                    "num_selected_source_patches": int(num_tx_source_packets),
                    "num_encoded_patches": int(num_encoded_packets),
                    "num_received_patches": int(num_received_encoded_packets),
                    "num_fec_recovered_patches": int(num_fec_recovered_source_packets),
                    "num_source_dropped_by_budget": int(
                        num_source_dropped_by_budget
                    ),
                    "num_parity_dropped_by_budget": int(
                        num_parity_dropped_by_budget
                    ),
                    "num_missing_by_budget": int(num_missing_by_budget),
                    "num_missing_by_loss": int(num_lost_by_bernoulli),
                    "selected_patch_ratio": float(
                        num_tx_source_packets / max(1, num_source_packets)
                    ),
                    "effective_patch_ratio": float(q_recv),
                },
                "packet": {
                    "num_source_packets": int(num_source_packets),
                    "num_admitted_source_packets": int(num_tx_source_packets),
                    "num_protected_source_packets": int(
                        num_protected_source_packets
                    ),
                    "num_tail_source_packets": int(num_tail_source_packets),
                    "num_parity_packets": int(num_parity_packets),
                    "num_encoded_packets": int(num_encoded_packets),
                    "num_transmitted_packets": int(num_tx_packets),
                    "num_transmitted_source_packets": int(num_tx_source_packets),
                    "num_transmitted_parity_packets": int(num_tx_parity_packets),
                    "num_source_dropped_by_budget": int(
                        num_source_dropped_by_budget
                    ),
                    "num_parity_dropped_by_budget": int(
                        num_parity_dropped_by_budget
                    ),
                    "num_received_packets": int(num_received_encoded_packets),
                    "num_direct_received_source_packets": int(num_direct_received_source_packets),
                    "num_fec_recovered_source_packets": int(num_fec_recovered_source_packets),
                    "num_missing_source_packets": int(num_missing_source_packets),
                    "packet_size_bytes": int(wire_packet_size_bytes),
                    "source_symbol_bytes": int(packet_size_bytes),
                    "protected_redundancy_ratio": float(
                        protected_redundancy_ratio
                    ),
                    "overall_redundancy_ratio": float(
                        overall_redundancy_ratio
                    ),
                },
                "size": size_info,
                "quality": {
                    "q_recv": float(q_recv),
                    "q_cache": float(q_cache),
                    "q_eff": float(q_eff),
                    "q_recv_unit": float(
                        temporal_cache_info.get("q_recv_unit", 0.0)
                    ),
                    "q_cache_unit": float(
                        temporal_cache_info.get("q_cache_unit", 0.0)
                    ),
                    "q_eff_unit": float(
                        temporal_cache_info.get("q_eff_unit", 0.0)
                    ),
                    "q_recv_packet": float(
                        temporal_cache_info.get("q_recv_packet", q_recv)
                    ),
                    "q_cache_packet": float(
                        temporal_cache_info.get("q_cache_packet", q_cache)
                    ),
                    "q_eff_packet": float(
                        temporal_cache_info.get("q_eff_packet", q_recv)
                    ),
                    "num_source_packets": int(num_source_packets),
                    "num_still_missing": int(num_still_missing_packets),
                    "num_fec_recovered_packets": int(num_fec_recovered_source_packets),
                    "num_temporal_filled_packets": int(
                        num_temporal_filled_packets
                    ),
                },
                "raw_loss_mask_summary": _mask_summary(raw_loss_mask_tx, true_name="lost"),
                "final_loss_mask_summary": _mask_summary(full_loss_mask, true_name="lost"),
                "receive_mask_summary": _mask_summary(receive_mask, true_name="received"),
                "tx_bytes": float(transmitted_bytes),
                "rx_bytes": float(received_bytes),
                "raw_bytes": int(feature.numel() * feature.element_size()),
                "compressed_bytes": float(packet_result.original_num_bytes),
                "encoded_bytes": float(
                    num_encoded_packets * wire_packet_size_bytes
                ),
                "received_bytes": float(received_bytes),
                "effective_received_bytes": float(received_bytes),
                "late": False,
                "dropped_by_late": False,
                "notes": {
                    "packetization": "Quantize first, then flatten Q(F) into a byte stream and split by fixed packet length.",
                    "fec": "FEC is applied to byte-stream source packets before Bernoulli packet loss.",
                    "loss": "Each transmitted encoded packet independently follows receive_i ~ Bernoulli(1 - PLR_t).",
                    "delay": "Good/Medium use current frame; Bad uses previous frame.",
                },
            }
        )

        # Experiment-1 audit is read-only and disabled unless explicitly enabled
        # in the temporary audit config. It never changes recovered_feature.
        audit_summary = self.compression_auditor.record(
            frame_id=frame_id,
            link_id=link_id,
            agent_index=int(agent_index),
            ego_index=int(ego_index),
            requested_quant_mode=quant_mode,
            actual_quant_mode=quant_result.mode,
            source_tensor_kind=source_tensor_kind,
            feature_input=feature,
            source_feature=feature_tx,
            quant_dequantized=quant_result.dequantized,
            recovered_compact=recovered_feature_compact,
            recovered_dense=recovered_feature,
            stream_tensor=stream_tensor,
            packet_result=packet_result,
            source_tx_mask=source_tx_mask,
            budget_accounting={
                "accounting_version": 2,
                "bandwidth_budget_bytes": float(frame_budget_bytes),
                "packet_size_bytes": int(wire_packet_size_bytes),
                "source_symbol_bytes": int(packet_size_bytes),
                "num_source_packets": int(num_source_packets),
                "num_admitted_source_packets": int(num_tx_source_packets),
                "num_protected_source_packets": int(
                    num_protected_source_packets
                ),
                "num_tail_source_packets": int(num_tail_source_packets),
                "num_parity_packets": int(num_parity_packets),
                "num_encoded_packets": int(num_encoded_packets),
                "num_transmitted_packets": int(num_tx_packets),
                "num_transmitted_source_packets": int(num_tx_source_packets),
                "num_transmitted_parity_packets": int(num_tx_parity_packets),
                "num_source_dropped_by_budget": int(num_source_dropped_by_budget),
                "num_parity_dropped_by_budget": int(num_parity_dropped_by_budget),
                "num_missing_by_budget": int(num_missing_by_budget),
                "actual_transmitted_bytes": float(transmitted_bytes),
                "actual_transmitted_source_bytes": float(
                    num_tx_source_packets * wire_packet_size_bytes
                ),
                "actual_transmitted_parity_bytes": float(
                    num_tx_parity_packets * wire_packet_size_bytes
                ),
                "num_lost_by_bernoulli": int(num_lost_by_bernoulli),
                "num_direct_received_source_packets": int(
                    num_direct_received_source_packets
                ),
                "num_fec_recovered_source_packets": int(
                    num_fec_recovered_source_packets
                ),
                "num_recovered_source_packets": int(num_recovered_source_packets),
                "num_missing_source_packets": int(num_missing_source_packets),
                "protected_redundancy_ratio": float(
                    protected_redundancy_ratio
                ),
                "overall_redundancy_ratio": float(
                    overall_redundancy_ratio
                ),
            },
            comm_record=record,
        )
        if audit_summary is not None:
            record["compression_audit"] = audit_summary

        fec_audit_summary = None
        if getattr(self, "fec_recovery_auditor", None) is not None and self.fec_recovery_auditor.enabled:
            if direct_recovered_feature_compact is None:
                raise RuntimeError("Experiment-3 direct-only tensor was not constructed.")
            fec_audit_summary = self.fec_recovery_auditor.record(
                frame_id=frame_id,
                link_id=link_id,
                ego_index=int(ego_index),
                agent_index=int(agent_index),
                quant_mode=str(quant_result.mode),
                fec_type=str(fec_runtime_cfg.get("fec_type", fec_runtime_cfg.get("type", "none"))),
                redundancy_ratio=float(fec_runtime_cfg.get("redundancy_ratio", 0.0)),
                plr=float(channel_loss_info.get("plr", 0.0)),
                quant_dequantized=quant_result.dequantized,
                direct_recovered_compact=direct_recovered_feature_compact,
                fec_recovered_compact=recovered_feature_compact,
                source_tx_mask=source_tx_mask,
                parity_tx_mask=parity_tx_mask,
                source_receive_mask=(
                    decode_result.direct_received_source_mask
                ),
                parity_receive_mask=(
                    block_fec_plan.repair_receive_mask(receive_mask)
                    if block_fec_plan is not None
                    else (
                        receive_mask[num_source_packets:num_encoded_packets]
                        & parity_tx_mask
                    )
                ),
                num_source_packets=int(num_source_packets),
                num_admitted_source_packets=int(num_tx_source_packets),
                num_parity_packets=int(num_parity_packets),
                num_encoded_packets=int(num_encoded_packets),
                num_tx_source_packets=int(num_tx_source_packets),
                num_tx_parity_packets=int(num_tx_parity_packets),
                num_source_dropped_by_budget=int(num_source_dropped_by_budget),
                num_parity_dropped_by_budget=int(num_parity_dropped_by_budget),
                num_direct_received_source_packets=int(num_direct_received_source_packets),
                num_fec_recovered_source_packets=int(num_fec_recovered_source_packets),
                num_missing_source_packets=int(num_missing_source_packets),
                bandwidth_budget_bytes=float(frame_budget_bytes),
                actual_transmitted_bytes=float(transmitted_bytes),
                actual_received_bytes=float(received_bytes),
                packet_size_bytes=int(wire_packet_size_bytes),
            )
            if fec_audit_summary is not None:
                record["fec_recovery_audit"] = fec_audit_summary

        self._append_record(record)

        if self.keep_tensor_results:
            result = ARCECommResult(
                recovered_feature=recovered_feature,
                record=record,
                packetization_result=packet_result,
                quantization_result=quant_result,
                encode_result=encode_result,
                decode_result=decode_result,
            )
        else:
            result = ARCECommResult(
                recovered_feature=recovered_feature,
                record=record,
            )

        return result if return_result else (recovered_feature, record)

    # ------------------------------------------------------------------
    # Batch / agent helpers
    # ------------------------------------------------------------------

    def _infer_frame_id_from_data_dict(
        self,
        data_dict: Any = None,
        fallback: Any = None,
    ):
        if fallback is not None:
            return fallback

        if not isinstance(data_dict, dict):
            return None

        for key in ("frame_id", "timestamp", "sample_idx", "sample_id"):
            if key in data_dict:
                value = data_dict[key]
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        return int(value.detach().cpu().item())
                    return tuple(value.detach().cpu().flatten().tolist())
                return value

        return None

    def _get_external_channel_state(self, data_dict, batch_idx, cav_idx):
        """
        Read per-link channel state from data_dict['channel_state_ids'].

        Expected:
            channel_state_ids: [B, max_cav]

        Mapping:
            0 -> good
            1 -> medium
            2 -> bad
            -1 -> ego / padding
        """
        if not isinstance(data_dict, dict):
            return None, "no_data_dict"

        if "channel_state_ids" not in data_dict:
            return None, "no_channel_state_ids"

        state_ids = data_dict["channel_state_ids"]

        try:
            if torch.is_tensor(state_ids):
                state_id = int(
                    state_ids[int(batch_idx), int(cav_idx)]
                    .detach()
                    .cpu()
                    .item()
                )
            else:
                state_id = int(state_ids[int(batch_idx)][int(cav_idx)])
        except Exception as e:
            return None, "failed_to_read_channel_state_ids:{}".format(repr(e))

        if state_id < 0:
            return None, "ego_or_padding"

        state_name = CHANNEL_STATE_ID_TO_NAME.get(state_id, None)
        if state_name not in VALID_CHANNEL_STATE_NAMES:
            return None, "invalid_channel_state_id:{}".format(state_id)

        return state_name, "dataset_link_markov"

    def communicate_flattened_features(
        self,
        features: torch.Tensor,
        record_len: Any,
        data_dict: Any = None,
        frame_id: Optional[int] = None,
        ego_index: Optional[int] = None,
        update_cache: bool = True,
        return_records: bool = True,
        message_masks: Optional[torch.Tensor] = None,
        priority_maps: Optional[torch.Tensor] = None,
    ):
        """
        Communicate OpenCOOD flattened CAV features.

        features: [sum(record_len), C, H, W]
        record_len: [B]
        """
        features = _require_tensor(features, "features")
        if is_payload_native_transport(getattr(self, "transport_mode", "")):
            message_masks = None
            priority_maps = None

        if features.dim() != 4:
            raise ValueError(
                "communicate_flattened_features expects features with shape "
                f"[sum(record_len), C, H, W], got {tuple(features.shape)}."
            )

        if torch.is_tensor(record_len):
            record_len_list = [
                int(x) for x in record_len.detach().cpu().flatten().tolist()
            ]
        elif isinstance(record_len, (list, tuple)):
            record_len_list = [int(x) for x in record_len]
        else:
            raise TypeError(
                "record_len should be a torch.Tensor, list, or tuple, "
                f"got {type(record_len)}."
            )

        if len(record_len_list) == 0:
            raise ValueError("record_len should not be empty.")

        total_cav = int(sum(record_len_list))
        if total_cav != int(features.shape[0]):
            raise ValueError(
                "record_len does not match flattened feature count: "
                f"sum(record_len)={total_cav}, features.shape[0]={features.shape[0]}."
            )

        if ego_index is None:
            ego_index = self.default_ego_index

        frame_id = self._infer_frame_id_from_data_dict(
            data_dict=data_dict,
            fallback=frame_id,
        )

        recovered = features.clone()
        records: List[Dict[str, Any]] = []

        offset = 0
        for batch_idx, num_cav in enumerate(record_len_list):
            collaborator_indices = [
                cav_idx for cav_idx in range(num_cav)
                if int(cav_idx) != int(ego_index)
            ]
            num_collaborators = len(collaborator_indices)
            per_link_budget_bytes = self._per_link_budget_bytes(num_collaborators)

            for cav_idx in range(num_cav):
                global_idx = offset + cav_idx

                link_id = (
                    int(batch_idx),
                    int(ego_index),
                    int(cav_idx),
                )

                channel_state, external_state_source = self._get_external_channel_state(
                    data_dict=data_dict,
                    batch_idx=batch_idx,
                    cav_idx=cav_idx,
                )

                cav_message_mask = None
                if message_masks is not None:
                    cav_message_mask = message_masks[global_idx]

                cav_priority_map = None
                if priority_maps is not None:
                    cav_priority_map = priority_maps[global_idx]

                budget_for_link = (
                    self._link_budget_bytes_for_state(channel_state, num_collaborators)
                    if int(cav_idx) != int(ego_index)
                    else 0.0
                )

                feature_hat, record = self.communicate_feature(
                    feature=features[global_idx],
                    link_id=link_id,
                    frame_id=frame_id,
                    agent_index=cav_idx,
                    ego_index=ego_index,
                    channel_state=channel_state,
                    budget_bytes=budget_for_link,
                    message_mask=cav_message_mask,
                    priority_map=cav_priority_map,
                    update_cache=update_cache,
                    return_result=False,
                )

                record["external_channel_state_source"] = external_state_source
                record["system_budget"] = {
                    "budget_scope": self._record_budget_scope(),
                    "allocation_scope": "fixed_equal_split",
                    "system_budget_mbps": float(self.system_budget_mbps),
                    "tx_window_ms": float(self.tx_window_ms),
                    "system_budget_bytes": float(self._record_frame_budget_bytes(channel_state if "channel_state" in locals() else None)),
                    "frame_budget_bytes": float(self._record_frame_budget_bytes(channel_state if "channel_state" in locals() else None)),
                    "num_collaborators": int(num_collaborators),
                    "per_link_budget_bytes": float(budget_for_link),
                }

                recovered[global_idx] = feature_hat
                records.append(record)

            offset += num_cav

        comm_info = {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "link_scope": self.link_scope,
            "frame_id": frame_id,
            "num_batches": int(len(record_len_list)),
            "record_len": tuple(int(x) for x in record_len_list),
            "num_input_features": int(features.shape[0]),
            "num_records_this_forward": int(len(records)),
            "summary": self.get_summary(),
        }

        if return_records:
            comm_info["records"] = records

        return recovered, comm_info

    def communicate_agent_features(
        self,
        features: torch.Tensor,
        frame_id: Optional[int] = None,
        ego_index: Optional[int] = None,
        batch_index: Optional[int] = None,
        update_cache: bool = True,
        return_records: bool = True,
    ):
        features = _require_tensor(features, "features")

        if ego_index is None:
            ego_index = self.default_ego_index

        if features.dim() == 4:
            num_agents = int(features.shape[0])
            recovered = features.clone()
            records = []

            collaborator_indices = [
                agent_idx for agent_idx in range(num_agents)
                if int(agent_idx) != int(ego_index)
            ]
            per_link_budget_bytes = self._per_link_budget_bytes(
                len(collaborator_indices)
            )

            for agent_idx in range(num_agents):
                link_id = (
                    batch_index,
                    int(ego_index),
                    int(agent_idx),
                )

                channel_state = None
                budget_for_link = (
                    self._link_budget_bytes_for_state(channel_state, len(collaborator_indices))
                    if int(agent_idx) != int(ego_index)
                    else 0.0
                )

                feature_hat, record = self.communicate_feature(
                    feature=features[agent_idx],
                    link_id=link_id,
                    frame_id=frame_id,
                    agent_index=agent_idx,
                    ego_index=ego_index,
                    budget_bytes=budget_for_link,
                    update_cache=update_cache,
                    return_result=False,
                )

                record["system_budget"] = {
                    "budget_scope": self._record_budget_scope(),
                    "allocation_scope": "fixed_equal_split",
                    "system_budget_mbps": float(self.system_budget_mbps),
                    "tx_window_ms": float(self.tx_window_ms),
                    "system_budget_bytes": float(self._record_frame_budget_bytes(channel_state if "channel_state" in locals() else None)),
                    "frame_budget_bytes": float(self._record_frame_budget_bytes(channel_state if "channel_state" in locals() else None)),
                    "num_collaborators": int(len(collaborator_indices)),
                    "per_link_budget_bytes": float(budget_for_link),
                }

                recovered[agent_idx] = feature_hat
                records.append(record)

            return (recovered, records) if return_records else recovered

        if features.dim() == 5:
            batch_size = int(features.shape[0])
            num_agents = int(features.shape[1])
            recovered = features.clone()
            batch_records: Dict[int, List[Dict[str, Any]]] = {}

            collaborator_indices = [
                agent_idx for agent_idx in range(num_agents)
                if int(agent_idx) != int(ego_index)
            ]
            per_link_budget_bytes = self._per_link_budget_bytes(
                len(collaborator_indices)
            )

            for b in range(batch_size):
                batch_records[b] = []

                for agent_idx in range(num_agents):
                    link_id = (
                        int(b),
                        int(ego_index),
                        int(agent_idx),
                    )

                    channel_state = None
                    budget_for_link = (
                        self._link_budget_bytes_for_state(channel_state, len(collaborator_indices))
                        if int(agent_idx) != int(ego_index)
                        else 0.0
                    )

                    feature_hat, record = self.communicate_feature(
                        feature=features[b, agent_idx],
                        link_id=link_id,
                        frame_id=frame_id,
                        agent_index=agent_idx,
                        ego_index=ego_index,
                        budget_bytes=budget_for_link,
                        update_cache=update_cache,
                        return_result=False,
                    )

                    record["system_budget"] = {
                        "budget_scope": self._record_budget_scope(),
                        "allocation_scope": "fixed_equal_split",
                        "system_budget_mbps": float(self.system_budget_mbps),
                        "tx_window_ms": float(self.tx_window_ms),
                        "system_budget_bytes": float(self._record_frame_budget_bytes(channel_state if "channel_state" in locals() else None)),
                    "frame_budget_bytes": float(self._record_frame_budget_bytes(channel_state if "channel_state" in locals() else None)),
                        "num_collaborators": int(len(collaborator_indices)),
                        "per_link_budget_bytes": float(budget_for_link),
                    }

                    recovered[b, agent_idx] = feature_hat
                    batch_records[b].append(record)

            return (recovered, batch_records) if return_records else recovered

        raise ValueError(
            "communicate_agent_features expects shape [N,C,H,W] or [B,N,C,H,W], "
            f"got {tuple(features.shape)}."
        )

    def __call__(self, features: torch.Tensor, *args, **kwargs):
        if len(args) >= 1:
            maybe_record_len = args[0]
            if torch.is_tensor(maybe_record_len) or isinstance(
                maybe_record_len, (list, tuple)
            ):
                return self.communicate_flattened_features(
                    features,
                    maybe_record_len,
                    *args[1:],
                    **kwargs,
                )

        return self.communicate_agent_features(features, *args, **kwargs)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def get_records(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.records)

    def get_frame_records(self, frame_id: Any) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.frame_records.get(frame_id, []))

    def get_summary(self) -> Dict[str, Any]:
        num_records = len(self.records)

        total_tx = 0.0
        total_rx = 0.0
        total_lost = 0
        total_encoded = 0
        total_source = 0
        total_parity = 0
        total_fec_recovered = 0

        total_missing_by_budget = 0
        total_lost_by_bernoulli = 0

        for record in self.records:
            if record.get("bypassed", False):
                continue

            size = record.get("size", {})
            total_tx += float(size.get("actual_transmitted_bytes", 0.0))
            total_rx += float(size.get("actual_received_bytes", 0.0))
            total_lost += int(size.get("actual_num_lost_encoded_packets", 0))
            total_encoded += int(size.get("actual_num_encoded_packets", 0))
            total_source += int(size.get("actual_num_source_packets", 0))
            total_parity += int(size.get("actual_num_parity_packets", 0))
            total_fec_recovered += int(size.get("num_fec_recovered_source_packets", 0))
            total_missing_by_budget += int(size.get("num_missing_by_budget", 0))
            total_lost_by_bernoulli += int(size.get("num_lost_by_bernoulli", 0))

        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "num_records": int(num_records),
            "num_processed_links": int(self.num_processed_links),
            "num_bypassed_links": int(self.num_bypassed_links),
            "num_late_links": int(self.num_late_links),
            "num_dropped_by_late": int(self.num_dropped_by_late),
            "packetization_mode": "byte_stream",
            "loss_model": "bernoulli",
            "latency_model": "fixed_state_delay",
            "fec_enabled": bool(total_parity > 0),
            "total_transmitted_bytes": float(total_tx),
            "total_received_bytes": float(total_rx),
            "total_transmitted_mb": float(total_tx / 1_000_000.0),
            "total_received_mb": float(total_rx / 1_000_000.0),
            "total_encoded_packets": int(total_encoded),
            "total_source_packets": int(total_source),
            "total_parity_packets": int(total_parity),
            "total_lost_encoded_packets": int(total_lost),
            "total_fec_recovered_source_packets": int(total_fec_recovered),
            "total_missing_by_budget": int(total_missing_by_budget),
            "total_lost_by_bernoulli": int(total_lost_by_bernoulli),
            "encoded_packet_loss_ratio": (
                float(total_lost / total_encoded) if total_encoded > 0 else 0.0
            ),
            "system_budget": {
                "budget_scope": self._record_budget_scope(),
                "allocation_scope": "fixed_equal_split",
                "system_budget_mbps": float(self.system_budget_mbps),
                "tx_window_ms": float(self.tx_window_ms),
                "system_budget_bytes": float(self._record_frame_budget_bytes(channel_state if "channel_state" in locals() else None)),
                    "frame_budget_bytes": float(self._record_frame_budget_bytes(channel_state if "channel_state" in locals() else None)),
            },
            "bernoulli_loss_rates": copy.deepcopy(self.bernoulli_loss_rates),
            "fixed_delay_ms": copy.deepcopy(self.fixed_delay_ms),
            "delay_policy_by_state": copy.deepcopy(self.delay_policy_by_state),
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "arce": copy.deepcopy(self.arce_cfg),
            "late_policy": self.late_policy,
            "max_records": int(self.max_records),
            "keep_tensor_results": bool(self.keep_tensor_results),
            "byte_packetizer": self.byte_packetizer.get_config(),
            "action_policy": self.action_policy.get_config(),
            "fixed_policy": self.fixed_policy.get_config(),
            "loss_model": "bernoulli",
            "bernoulli_loss_rates": copy.deepcopy(self.bernoulli_loss_rates),
            "latency_model": "fixed_state_delay",
            "fixed_delay_ms": copy.deepcopy(self.fixed_delay_ms),
            "delay_policy_by_state": copy.deepcopy(self.delay_policy_by_state),
            "fec": copy.deepcopy(self.fec_cfg),
            "redundancy": copy.deepcopy(self.redundancy_cfg),
            "markov": {
                "enabled": bool(self.markov_enabled),
                "states": copy.deepcopy(self.markov_states),
                "init_state": self.markov_init_state,
                "transition_matrix": copy.deepcopy(self.markov_transition_matrix),
            },
            "system_budget": {
                "budget_scope": self.budget_scope,
                "system_budget_mbps": float(self.system_budget_mbps),
                "tx_window_ms": float(self.tx_window_ms),
                "system_budget_bytes": float(self._record_frame_budget_bytes(channel_state if "channel_state" in locals() else None)),
                    "frame_budget_bytes": float(self._record_frame_budget_bytes(channel_state if "channel_state" in locals() else None)),
            },
        }

    def __repr__(self) -> str:
        return (
            "ARCEFixedComm("
            f"enabled={self.enabled}, "
            f"mode={self.mode}, "
            f"link_scope={self.link_scope}, "
            f"packetization=byte_stream, "
            f"fec=enabled, "
            f"loss=bernoulli, "
            f"latency=fixed_state_delay, "
            f"num_records={len(self.records)})"
        )


# Compatibility aliases.
FixedARCEComm = ARCEFixedComm
ARCEComm = ARCEFixedComm

__all__ = [
    "LATE_POLICY_ALLOW",
    "LATE_POLICY_DROP",
    "LATE_POLICY_CACHE_ONLY",
    "VALID_LATE_POLICIES",
    "ARCECommResult",
    "ARCEFixedComm",
    "FixedARCEComm",
    "ARCEComm",
]
