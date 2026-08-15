"""
Fixed ARCE policy for OPV2V / V2X-ViT communication simulation.

This module defines a hand-crafted fixed policy that maps channel states to
ARCE communication actions.

Typical pipeline position:

    ChannelManager
        -> current channel state: good / medium / bad

    FixedARCEPolicy
        -> ARCEAction:
            quant_mode
            fec_type
            redundancy_ratio
            xor_group_size
            raptor-like config
            recovery priority

    ARCEFixedComm
        -> executes:
            packetization
            quantization
            FEC encode
            channel loss / latency
            FEC decode
            partial reconstruction

Default state-aware policy:

    good:
        FP16 + no FEC

    medium:
        INT8 + XOR FEC, about 25% redundancy

    bad:
        INT4 + Raptor-like fountain FEC, about 50% redundancy

All default values are intentionally easy to override from YAML.

Example YAML:

    arce:
      policy: fixed

      fixed_policy:
        default_state: medium

        profiles:
          good:
            quant_mode: fp16
            fec_type: none
            redundancy_ratio: 0.0
            recovery: arce

          medium:
            quant_mode: int8
            fec_type: xor
            xor_group_size: 4
            redundancy_ratio: 0.25
            recovery: arce

          bad:
            quant_mode: int4
            fec_type: raptor_sim
            redundancy_ratio: 0.5
            decode_overhead: 0.0
            degree_distribution: robust_soliton
            recovery: arce

Important:
    This file only selects actions.
    It does NOT:
        - quantize features;
        - encode FEC packets;
        - sample packet loss;
        - reconstruct missing packets;
        - modify feature tensors.

Those are handled by:
    opencood.communication.transport.quantization.*
    opencood.communication.transport.fec.*
    opencood.communication.channel.*
    opencood.communication.transport.recovery.*
    opencood.methods.arce.arce_fixed_comm
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from opencood.communication.channel import (
    CHANNEL_STATE_GOOD,
    CHANNEL_STATE_MEDIUM,
    CHANNEL_STATE_BAD,
    VALID_CHANNEL_STATES,
    normalize_channel_state,
)

from opencood.communication.transport.quantization import (
    QUANT_MODE_FP32,
    QUANT_MODE_FP16,
    QUANT_MODE_INT8,
    QUANT_MODE_INT4,
    normalize_quant_mode,
    quant_mode_to_bits,
    compression_ratio_from_quant_mode,
)

from opencood.communication.transport.fec import (
    FEC_TYPE_NONE,
    FEC_TYPE_XOR,
    FEC_TYPE_RAPTOR_SIM,
    normalize_fec_type,
    normalize_redundancy_ratio,
    normalize_group_size,
    normalize_decode_overhead,
    get_fec_config_summary,
)

from opencood.communication.transport.recovery import (
    RECOVERY_METHOD_ARCE,
    RECOVERY_METHOD_ZERO_FILL,
    DEFAULT_RECOVERY_PRIORITY,
    normalize_recovery_method,
    normalize_recovery_priority,
)


ACTION_PROFILE_KEYS = (
    "quant_mode",
    "fec_type",
    "redundancy_ratio",
    "xor_group_size",
    "group_size",
    "decode_overhead",
    "recovery",
    "recovery_priority",
    "degree_distribution",
    "robust_soliton_c",
    "robust_soliton_delta",
    "fixed_degree",
    "max_degree",
    "num_repair_packets",
    "repair_packets",
    "max_decode_iters",
    "seed",
)


DEFAULT_STATE_ACTIONS: Dict[str, Dict[str, Any]] = {
    CHANNEL_STATE_GOOD: {
        "name": "good_fp16_no_fec",
        "channel_state": CHANNEL_STATE_GOOD,
        "quant_mode": QUANT_MODE_FP16,
        "fec_type": FEC_TYPE_NONE,
        "redundancy_ratio": 0.0,
        "xor_group_size": 4,
        "decode_overhead": 0.0,
        "recovery": RECOVERY_METHOD_ARCE,
        "recovery_priority": DEFAULT_RECOVERY_PRIORITY,
    },
    CHANNEL_STATE_MEDIUM: {
        "name": "medium_int8_xor",
        "channel_state": CHANNEL_STATE_MEDIUM,
        "quant_mode": QUANT_MODE_INT8,
        "fec_type": FEC_TYPE_XOR,
        "redundancy_ratio": 0.25,
        "xor_group_size": 4,
        "decode_overhead": 0.0,
        "recovery": RECOVERY_METHOD_ARCE,
        "recovery_priority": DEFAULT_RECOVERY_PRIORITY,
    },
    CHANNEL_STATE_BAD: {
        "name": "bad_int4_raptor_like",
        "channel_state": CHANNEL_STATE_BAD,
        "quant_mode": QUANT_MODE_INT4,
        "fec_type": FEC_TYPE_RAPTOR_SIM,
        "redundancy_ratio": 0.50,
        "xor_group_size": 4,
        "decode_overhead": 0.0,
        "recovery": RECOVERY_METHOD_ARCE,
        "recovery_priority": DEFAULT_RECOVERY_PRIORITY,
        "degree_distribution": "robust_soliton",
        "robust_soliton_c": 0.1,
        "robust_soliton_delta": 0.5,
        "max_decode_iters": 10000,
    },
}


DEFAULT_FALLBACK_ORDER: Dict[str, Tuple[str, ...]] = {
    CHANNEL_STATE_GOOD: (
        CHANNEL_STATE_GOOD,
        CHANNEL_STATE_MEDIUM,
        CHANNEL_STATE_BAD,
    ),
    CHANNEL_STATE_MEDIUM: (
        CHANNEL_STATE_MEDIUM,
        CHANNEL_STATE_BAD,
    ),
    CHANNEL_STATE_BAD: (
        CHANNEL_STATE_BAD,
    ),
}


def _extract_arce_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Accept full config or direct ARCE config.
    """
    cfg = cfg or {}

    if "arce" in cfg and isinstance(cfg["arce"], dict):
        return cfg["arce"]

    return cfg


def _extract_fixed_policy_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Accept full config, ARCE config, or direct fixed_policy config.
    """
    arce_cfg = _extract_arce_cfg(cfg)

    if "fixed_policy" in arce_cfg and isinstance(arce_cfg["fixed_policy"], dict):
        return arce_cfg["fixed_policy"]

    return arce_cfg


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


def _as_non_negative_int(value: Any, name: str) -> int:
    """
    Convert value to non-negative int.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to int, got {value}.")

    if value < 0:
        raise ValueError(f"{name} should be non-negative, got {value}.")

    return value


def _as_positive_int(value: Any, name: str) -> int:
    """
    Convert value to positive int.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to int, got {value}.")

    if value <= 0:
        raise ValueError(f"{name} should be positive, got {value}.")

    return value


def _as_non_negative_float(value: Any, name: str) -> float:
    """
    Convert value to non-negative float.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to float, got {value}.")

    if value < 0.0:
        raise ValueError(f"{name} should be non-negative, got {value}.")

    return value


def _normalize_optional_int(value: Any, name: str) -> Optional[int]:
    """
    Normalize optional int config.
    """
    if value is None:
        return None

    return _as_non_negative_int(value, name)


def _get_profile_dict(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-state action profiles.

    Supported YAML keys:
        profiles
        state_profiles
        state_actions
        actions
        by_state
    """
    for key in ("profiles", "state_profiles", "state_actions", "actions", "by_state"):
        if key in cfg and isinstance(cfg[key], dict):
            return cfg[key]

    return {}


def _extract_common_action_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract common action keys from fixed_policy config.

    These keys are applied to every state profile before per-state overrides.
    """
    overrides = {}

    for key in ACTION_PROFILE_KEYS:
        if key in cfg:
            overrides[key] = cfg[key]

    return overrides


def _normalize_extra_dict(extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a deepcopy of JSON-friendly extra fields.
    """
    result = {}

    for key, value in extra.items():
        if isinstance(value, dict):
            result[key] = copy.deepcopy(value)
        elif isinstance(value, (list, tuple)):
            result[key] = tuple(value)
        else:
            result[key] = value

    return result


@dataclass
class ARCEAction:
    """
    One fixed ARCE communication action.

    Attributes
    ----------
    name : str
        Human-readable action name.

    channel_state : str
        good / medium / bad profile that this action is designed for.

    quant_mode : str
        fp32 / fp16 / int8 / int4.

    fec_type : str
        none / xor / raptor_sim.

    redundancy_ratio : float
        Redundancy ratio rho.
        For raptor_sim:
            repair packets P = ceil(K * rho).
        For xor:
            actual redundancy is mainly determined by xor_group_size,
            but redundancy_ratio is still recorded and used for size/latency
            estimation.

    xor_group_size : int
        Source packets per XOR group.

    decode_overhead : float
        Raptor-like reference decode overhead.

    recovery : str
        arce / zero_fill / temporal_cache / spatial_interpolation / none.

    recovery_priority : tuple
        Partial reconstruction priority.

    extra : dict
        Additional FEC/policy fields, such as robust_soliton_c.
    """

    name: str
    channel_state: str
    quant_mode: str
    fec_type: str
    redundancy_ratio: float = 0.0
    xor_group_size: int = 4
    decode_overhead: float = 0.0
    recovery: str = RECOVERY_METHOD_ARCE
    recovery_priority: Tuple[str, ...] = DEFAULT_RECOVERY_PRIORITY
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.channel_state = normalize_channel_state(self.channel_state)
        self.quant_mode = normalize_quant_mode(self.quant_mode)
        self.fec_type = normalize_fec_type(self.fec_type)
        self.redundancy_ratio = normalize_redundancy_ratio(self.redundancy_ratio)
        self.xor_group_size = normalize_group_size(self.xor_group_size)
        self.decode_overhead = normalize_decode_overhead(self.decode_overhead)
        self.recovery = normalize_recovery_method(self.recovery)

        if self.recovery == RECOVERY_METHOD_ARCE:
            self.recovery_priority = normalize_recovery_priority(
                self.recovery_priority
            )
        elif self.recovery == RECOVERY_METHOD_ZERO_FILL:
            self.recovery_priority = normalize_recovery_priority(
                (RECOVERY_METHOD_ZERO_FILL,)
            )
        else:
            self.recovery_priority = normalize_recovery_priority(
                self.recovery_priority
            )

        if self.fec_type == FEC_TYPE_NONE:
            self.redundancy_ratio = 0.0

        if self.fec_type == FEC_TYPE_XOR and self.redundancy_ratio <= 0.0:
            self.redundancy_ratio = 1.0 / float(self.xor_group_size)

        self.extra = _normalize_extra_dict(self.extra)

    @property
    def quant_bits(self) -> int:
        """
        Bits per feature value under this action.
        """
        return int(quant_mode_to_bits(self.quant_mode))

    @property
    def compression_ratio(self) -> float:
        """
        Compression ratio relative to FP32.
        """
        return float(compression_ratio_from_quant_mode(self.quant_mode))

    @property
    def effective_redundancy_ratio_for_latency(self) -> float:
        """
        Redundancy ratio used for byte / latency estimation.

        For XOR, this approximates one parity packet per group.
        For exact packet count, FEC encoder / size_estimator should be used.
        """
        if self.fec_type == FEC_TYPE_NONE:
            return 0.0

        if self.fec_type == FEC_TYPE_XOR:
            if self.redundancy_ratio > 0.0:
                return float(self.redundancy_ratio)
            return float(1.0 / self.xor_group_size)

        return float(self.redundancy_ratio)

    def to_quant_config(self) -> Dict[str, Any]:
        """
        Build quantization config for FeatureQuantizer.
        """
        return {
            "enabled": self.quant_mode != QUANT_MODE_FP32,
            "mode": self.quant_mode,
            "raw_bits": 32,
        }

    def to_fec_config(self) -> Dict[str, Any]:
        """
        Build FEC config for NoFEC / XORFEC / RaptorSimFEC.
        """
        cfg = {
            "enabled": self.fec_type != FEC_TYPE_NONE,
            "type": self.fec_type,
            "redundancy_ratio": float(self.redundancy_ratio),
            "group_size": int(self.xor_group_size),
            "decode_overhead": float(self.decode_overhead),
        }

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
            if key in self.extra:
                cfg[key] = self.extra[key]

        return cfg

    def to_recovery_config(self) -> Dict[str, Any]:
        """
        Build recovery config for PartialReconstructor.
        """
        return {
            "priority": tuple(self.recovery_priority),
            "zero_fill": RECOVERY_METHOD_ZERO_FILL in self.recovery_priority,
            "spatial_interpolation": "spatial_interpolation" in self.recovery_priority,
            "temporal_cache": "temporal_cache" in self.recovery_priority,
        }

    def to_latency_action(self) -> Dict[str, Any]:
        """
        Build compact action fields for latency feasibility checks.
        """
        return {
            "compression_ratio": float(self.compression_ratio),
            "redundancy_ratio": float(self.effective_redundancy_ratio_for_latency),
            "quant_mode": self.quant_mode,
            "fec_type": self.fec_type,
        }

    def as_dict(self) -> Dict[str, Any]:
        """
        Export JSON-friendly action summary.
        """
        result = asdict(self)
        result["recovery_priority"] = tuple(self.recovery_priority)
        result["extra"] = copy.deepcopy(self.extra)
        result["quant_bits"] = int(self.quant_bits)
        result["compression_ratio"] = float(self.compression_ratio)
        result["effective_redundancy_ratio_for_latency"] = float(
            self.effective_redundancy_ratio_for_latency
        )
        result["fec_config"] = self.to_fec_config()
        result["quant_config"] = self.to_quant_config()
        result["recovery_config"] = self.to_recovery_config()
        return result

    def copy_with(self, **kwargs) -> "ARCEAction":
        """
        Return a copy of this action with selected fields replaced.
        """
        data = self.as_dict()

        for derived_key in (
            "quant_bits",
            "compression_ratio",
            "effective_redundancy_ratio_for_latency",
            "fec_config",
            "quant_config",
            "recovery_config",
        ):
            data.pop(derived_key, None)

        data.update(kwargs)

        return ARCEAction(**data)


def normalize_action_config(
    cfg: Dict[str, Any],
    channel_state: str,
    default_name: Optional[str] = None,
) -> ARCEAction:
    """
    Normalize one action config into ARCEAction.

    Parameters
    ----------
    cfg : dict
        Raw action config.

    channel_state : str
        State key that this action belongs to.

    default_name : str, optional
        Fallback action name.

    Returns
    -------
    ARCEAction
    """
    cfg = copy.deepcopy(cfg)
    channel_state = normalize_channel_state(cfg.get("channel_state", channel_state))

    name = str(
        cfg.get(
            "name",
            default_name or f"{channel_state}_fixed_action",
        )
    )

    quant_mode = cfg.get("quant_mode", DEFAULT_STATE_ACTIONS[channel_state]["quant_mode"])
    fec_type = cfg.get("fec_type", DEFAULT_STATE_ACTIONS[channel_state]["fec_type"])

    redundancy_ratio = cfg.get(
        "redundancy_ratio",
        DEFAULT_STATE_ACTIONS[channel_state].get("redundancy_ratio", 0.0),
    )

    xor_group_size = cfg.get(
        "xor_group_size",
        cfg.get(
            "group_size",
            DEFAULT_STATE_ACTIONS[channel_state].get("xor_group_size", 4),
        ),
    )

    decode_overhead = cfg.get(
        "decode_overhead",
        DEFAULT_STATE_ACTIONS[channel_state].get("decode_overhead", 0.0),
    )

    recovery = cfg.get(
        "recovery",
        DEFAULT_STATE_ACTIONS[channel_state].get("recovery", RECOVERY_METHOD_ARCE),
    )

    recovery_priority = cfg.get(
        "recovery_priority",
        cfg.get(
            "priority",
            DEFAULT_STATE_ACTIONS[channel_state].get(
                "recovery_priority",
                DEFAULT_RECOVERY_PRIORITY,
            ),
        ),
    )

    known_keys = {
        "name",
        "channel_state",
        "quant_mode",
        "fec_type",
        "redundancy_ratio",
        "xor_group_size",
        "group_size",
        "decode_overhead",
        "recovery",
        "recovery_priority",
        "priority",
    }

    extra = {
        key: value
        for key, value in cfg.items()
        if key not in known_keys
    }

    return ARCEAction(
        name=name,
        channel_state=channel_state,
        quant_mode=quant_mode,
        fec_type=fec_type,
        redundancy_ratio=redundancy_ratio,
        xor_group_size=xor_group_size,
        decode_overhead=decode_overhead,
        recovery=recovery,
        recovery_priority=tuple(recovery_priority),
        extra=extra,
    )


class FixedARCEPolicy:
    """
    Fixed state-aware ARCE policy.

    It maps channel state to ARCEAction:

        good   -> action_good
        medium -> action_medium
        bad    -> action_bad

    Main APIs:
        get_action(channel_state)
        select(channel_state / channel_profile)
        select_with_feasibility(...)
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.full_cfg = cfg or {}
        self.cfg = _extract_fixed_policy_cfg(cfg)

        self.enabled = _as_bool(self.cfg.get("enabled", True))

        self.default_state = normalize_channel_state(
            self.cfg.get(
                "default_state",
                self.cfg.get("channel_state", CHANNEL_STATE_MEDIUM),
            )
        )

        self.strict_state = _as_bool(self.cfg.get("strict_state", False))
        self.use_feasibility_fallback = _as_bool(
            self.cfg.get("use_feasibility_fallback", False)
        )

        self.actions = self._build_actions(self.cfg)
        self.fallback_order = self._build_fallback_order(self.cfg)

    def _build_actions(self, cfg: Dict[str, Any]) -> Dict[str, ARCEAction]:
        """
        Build per-state action table.
        """
        action_cfgs = copy.deepcopy(DEFAULT_STATE_ACTIONS)

        common_overrides = _extract_common_action_overrides(cfg)

        # Apply common overrides to every state.
        for state in VALID_CHANNEL_STATES:
            action_cfgs[state].update(common_overrides)

        # Apply per-state profiles.
        profile_dict = _get_profile_dict(cfg)

        for raw_state, profile in profile_dict.items():
            state = normalize_channel_state(raw_state)

            if not isinstance(profile, dict):
                raise ValueError(
                    f"fixed_policy profile for state {raw_state} should be dict, "
                    f"got {type(profile)}."
                )

            action_cfgs[state].update(profile)
            action_cfgs[state]["channel_state"] = state

        actions = {}

        for state in VALID_CHANNEL_STATES:
            actions[state] = normalize_action_config(
                action_cfgs[state],
                channel_state=state,
                default_name=f"{state}_fixed_action",
            )

        return actions

    def _build_fallback_order(self, cfg: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
        """
        Build feasibility fallback order.

        YAML style:
            fallback_order:
              good: [good, medium, bad]
              medium: [medium, bad]
              bad: [bad]
        """
        raw = cfg.get("fallback_order", None)

        if raw is None:
            return copy.deepcopy(DEFAULT_FALLBACK_ORDER)

        if not isinstance(raw, dict):
            raise ValueError(
                "fixed_policy.fallback_order should be a dict, "
                f"got {type(raw)}."
            )

        result = {}

        for state in VALID_CHANNEL_STATES:
            order = raw.get(state, DEFAULT_FALLBACK_ORDER[state])

            if isinstance(order, str):
                order = [
                    item.strip()
                    for item in order.split(",")
                    if item.strip()
                ]

            if not isinstance(order, (list, tuple)):
                raise ValueError(
                    f"fallback_order.{state} should be list/tuple/string, "
                    f"got {type(order)}."
                )

            normalized = tuple(normalize_channel_state(item) for item in order)

            if len(normalized) == 0:
                normalized = DEFAULT_FALLBACK_ORDER[state]

            result[state] = normalized

        return result

    def get_action(self, channel_state: Optional[str] = None) -> ARCEAction:
        """
        Return action for channel state.

        Parameters
        ----------
        channel_state : str, optional
            good / medium / bad. If None, use self.default_state.
        """
        if channel_state is None:
            channel_state = self.default_state

        try:
            state = normalize_channel_state(channel_state)
        except ValueError:
            if self.strict_state:
                raise
            state = self.default_state

        if state not in self.actions:
            if self.strict_state:
                raise KeyError(f"No fixed ARCE action for state: {state}.")
            state = self.default_state

        return copy.deepcopy(self.actions[state])

    def get_action_from_profile(self, channel_profile: Dict[str, Any]) -> ARCEAction:
        """
        Select action from channel profile dictionary.

        Expected profile fields:
            state_name
        """
        if not isinstance(channel_profile, dict):
            raise TypeError(
                f"channel_profile should be dict, got {type(channel_profile)}."
            )

        state = channel_profile.get(
            "state_name",
            channel_profile.get("channel_state", self.default_state),
        )

        return self.get_action(state)

    def select(
        self,
        channel_state: Optional[str] = None,
        channel_profile: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ARCEAction:
        """
        Select fixed ARCE action.

        Priority:
            1. channel_profile["state_name"]
            2. channel_state
            3. default_state
        """
        if not self.enabled:
            return ARCEAction(
                name="disabled_fp32_no_fec",
                channel_state=channel_state or self.default_state,
                quant_mode=QUANT_MODE_FP32,
                fec_type=FEC_TYPE_NONE,
                redundancy_ratio=0.0,
                xor_group_size=4,
                decode_overhead=0.0,
                recovery=RECOVERY_METHOD_ZERO_FILL,
                recovery_priority=(RECOVERY_METHOD_ZERO_FILL,),
                extra={"enabled": False},
            )

        if channel_profile is not None:
            return self.get_action_from_profile(channel_profile)

        return self.get_action(channel_state)

    def select_with_feasibility(
        self,
        raw_bytes: float,
        channel_manager: Any,
        channel_state: Optional[str] = None,
        channel_profile: Optional[Dict[str, Any]] = None,
        link_id: Any = None,
        frame_id: Optional[int] = None,
        deadline_ms: Optional[float] = None,
        use_max_jitter: bool = True,
    ) -> Tuple[ARCEAction, Dict[str, Any]]:
        """
        Select an action and optionally fall back if it violates latency deadline.

        This method requires a ChannelManager-like object that has:

            is_action_feasible(
                raw_bytes,
                compression_ratio,
                redundancy_ratio,
                link_id,
                frame_id,
                state,
                deadline_ms,
                use_max_jitter,
            )

        If use_feasibility_fallback=False, this only checks the selected action
        and returns it together with feasibility info.
        """
        if channel_profile is not None:
            state = channel_profile.get(
                "state_name",
                channel_profile.get("channel_state", self.default_state),
            )
        else:
            state = channel_state or self.default_state

        state = normalize_channel_state(state)

        selected = self.get_action(state)

        if channel_manager is None:
            return selected, {
                "checked": False,
                "reason": "channel_manager is None",
                "selected_action": selected.as_dict(),
            }

        def check(action: ARCEAction) -> Tuple[bool, Dict[str, Any]]:
            feasible, info = channel_manager.is_action_feasible(
                raw_bytes=raw_bytes,
                compression_ratio=action.compression_ratio,
                redundancy_ratio=action.effective_redundancy_ratio_for_latency,
                link_id=link_id,
                frame_id=frame_id,
                state=state,
                deadline_ms=deadline_ms,
                use_max_jitter=use_max_jitter,
            )
            info["action"] = action.as_dict()
            return bool(feasible), info

        feasible, selected_info = check(selected)

        if feasible or not self.use_feasibility_fallback:
            selected_info["checked"] = True
            selected_info["fallback_used"] = False
            selected_info["selected_action_name"] = selected.name
            return selected, selected_info

        tried = [
            {
                "state": state,
                "action_name": selected.name,
                "feasible": bool(feasible),
                "info": selected_info,
            }
        ]

        for fallback_state in self.fallback_order.get(state, (state,)):
            fallback_state = normalize_channel_state(fallback_state)
            fallback_action = self.get_action(fallback_state)

            if fallback_action.name == selected.name:
                continue

            ok, info = check(fallback_action)

            tried.append(
                {
                    "state": fallback_state,
                    "action_name": fallback_action.name,
                    "feasible": bool(ok),
                    "info": info,
                }
            )

            if ok:
                info["checked"] = True
                info["fallback_used"] = True
                info["original_state"] = state
                info["fallback_state"] = fallback_state
                info["tried"] = tried
                info["selected_action_name"] = fallback_action.name
                return fallback_action, info

        fallback_info = selected_info
        fallback_info["checked"] = True
        fallback_info["fallback_used"] = False
        fallback_info["fallback_failed"] = True
        fallback_info["tried"] = tried
        fallback_info["selected_action_name"] = selected.name
        fallback_info["reason"] = (
            "No feasible fallback action found; returning original selected action."
        )

        return selected, fallback_info

    def get_fec_config(self, channel_state: Optional[str] = None) -> Dict[str, Any]:
        """
        Return FEC config for selected channel state.
        """
        return self.get_action(channel_state).to_fec_config()

    def get_quant_config(self, channel_state: Optional[str] = None) -> Dict[str, Any]:
        """
        Return quantization config for selected channel state.
        """
        return self.get_action(channel_state).to_quant_config()

    def get_recovery_config(self, channel_state: Optional[str] = None) -> Dict[str, Any]:
        """
        Return recovery config for selected channel state.
        """
        return self.get_action(channel_state).to_recovery_config()

    def get_action_summary(self, num_source_packets: Optional[int] = None) -> Dict[str, Any]:
        """
        Return JSON-friendly summary of all state actions.

        If num_source_packets is provided, include estimated FEC packet counts.
        """
        result = {}

        for state, action in self.actions.items():
            item = action.as_dict()

            item["fec_summary"] = get_fec_config_summary(
                fec_type=action.fec_type,
                redundancy_ratio=action.redundancy_ratio,
                group_size=action.xor_group_size,
                decode_overhead=action.decode_overhead,
                num_source_packets=num_source_packets,
            )

            result[state] = item

        return result

    def get_config(self) -> Dict[str, Any]:
        """
        Return JSON-friendly fixed policy config.
        """
        return {
            "enabled": bool(self.enabled),
            "default_state": self.default_state,
            "strict_state": bool(self.strict_state),
            "use_feasibility_fallback": bool(self.use_feasibility_fallback),
            "actions": self.get_action_summary(),
            "fallback_order": {
                state: tuple(order)
                for state, order in self.fallback_order.items()
            },
        }

    def __repr__(self) -> str:
        return (
            "FixedARCEPolicy("
            f"enabled={self.enabled}, "
            f"default_state={self.default_state}, "
            f"use_feasibility_fallback={self.use_feasibility_fallback})"
        )


# Compatibility aliases.
ARCEFixedPolicy = FixedARCEPolicy
FixedPolicy = FixedARCEPolicy


__all__ = [
    "ACTION_PROFILE_KEYS",
    "DEFAULT_STATE_ACTIONS",
    "DEFAULT_FALLBACK_ORDER",
    "ARCEAction",
    "normalize_action_config",
    "FixedARCEPolicy",
    "ARCEFixedPolicy",
    "FixedPolicy",
]