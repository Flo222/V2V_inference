"""Random ARCE action policy.

This policy is used for the Random baseline. It samples one valid
ARCEAction from a predefined action space. The channel state still comes
from the current link / Markov channel; this policy only decides the
communication action: quantization + FEC + redundancy.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

from opencood.methods.arce import extract_arce_cfg
from opencood.methods.arce.policy.fixed_policy import (
    ARCEAction,
    normalize_action_config,
)
from opencood.communication.channel import (
    CHANNEL_STATE_MEDIUM,
    normalize_channel_state,
)


DEFAULT_RANDOM_ACTION_SPACE: List[Dict[str, Any]] = [
    {
        "name": "fp16_none",
        "quant_mode": "fp16",
        "fec_type": "none",
        "redundancy_ratio": 0.0,
        "recovery": "arce",
    },
    {
        "name": "int8_none",
        "quant_mode": "int8",
        "fec_type": "none",
        "redundancy_ratio": 0.0,
        "recovery": "arce",
    },
    {
        "name": "int4_none",
        "quant_mode": "int4",
        "fec_type": "none",
        "redundancy_ratio": 0.0,
        "recovery": "arce",
    },
    {
        "name": "int8_xor_r025",
        "quant_mode": "int8",
        "fec_type": "xor",
        "xor_group_size": 4,
        "redundancy_ratio": 0.25,
        "recovery": "arce",
    },
    {
        "name": "int8_xor_r050",
        "quant_mode": "int8",
        "fec_type": "xor",
        "xor_group_size": 4,
        "redundancy_ratio": 0.50,
        "recovery": "arce",
    },
    {
        "name": "int4_xor_r025",
        "quant_mode": "int4",
        "fec_type": "xor",
        "xor_group_size": 4,
        "redundancy_ratio": 0.25,
        "recovery": "arce",
    },
    {
        "name": "int4_xor_r050",
        "quant_mode": "int4",
        "fec_type": "xor",
        "xor_group_size": 4,
        "redundancy_ratio": 0.50,
        "recovery": "arce",
    },
    {
        "name": "int8_raptor_r025",
        "quant_mode": "int8",
        "fec_type": "raptor_sim",
        "redundancy_ratio": 0.25,
        "recovery": "arce",
    },
    {
        "name": "int8_raptor_r050",
        "quant_mode": "int8",
        "fec_type": "raptor_sim",
        "redundancy_ratio": 0.50,
        "recovery": "arce",
    },
    {
        "name": "int4_raptor_r025",
        "quant_mode": "int4",
        "fec_type": "raptor_sim",
        "redundancy_ratio": 0.25,
        "recovery": "arce",
    },
    {
        "name": "int4_raptor_r050",
        "quant_mode": "int4",
        "fec_type": "raptor_sim",
        "redundancy_ratio": 0.50,
        "recovery": "arce",
    },
]


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


def _extract_random_policy_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    arce_cfg = extract_arce_cfg(cfg or {})
    if "random_policy" in arce_cfg and isinstance(arce_cfg["random_policy"], dict):
        return arce_cfg["random_policy"]
    return {}


class RandomARCEPolicy:
    """Uniform random policy over a valid ARCE action space."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.full_cfg = cfg or {}
        self.arce_cfg = extract_arce_cfg(cfg or {})
        self.cfg = _extract_random_policy_cfg(cfg or {})

        self.enabled = _as_bool(self.cfg.get("enabled", True))
        self.seed = int(self.cfg.get("seed", self.arce_cfg.get("seed", 0)))
        self.sample_mode = str(self.cfg.get("sample_mode", "uniform")).strip().lower()

        if self.sample_mode != "uniform":
            raise ValueError(
                f"Unsupported random_policy.sample_mode={self.sample_mode}. "
                "Currently only 'uniform' is supported."
            )

        self.rng = random.Random(self.seed)

        raw_actions = self.cfg.get("action_space", None)
        if raw_actions is None:
            raw_actions = DEFAULT_RANDOM_ACTION_SPACE

        if not isinstance(raw_actions, (list, tuple)) or len(raw_actions) == 0:
            raise ValueError("random_policy.action_space should be a non-empty list.")

        self.actions: List[ARCEAction] = []
        for idx, action_cfg in enumerate(raw_actions):
            if not isinstance(action_cfg, dict):
                raise TypeError(
                    f"random_policy.action_space[{idx}] should be dict, "
                    f"got {type(action_cfg)}."
                )
            action_cfg = copy.deepcopy(action_cfg)
            channel_state = action_cfg.get("channel_state", CHANNEL_STATE_MEDIUM)
            action = normalize_action_config(
                action_cfg,
                channel_state=channel_state,
                default_name=action_cfg.get("name", f"random_action_{idx}"),
            )
            self.actions.append(action)

        # Disabled random policy falls back to fp32 + no fec.
        self.disabled_action = normalize_action_config(
            {
                "name": "random_disabled_fp32_none",
                "quant_mode": "fp32",
                "fec_type": "none",
                "redundancy_ratio": 0.0,
                "recovery": "zero_fill",
            },
            channel_state=CHANNEL_STATE_MEDIUM,
            default_name="random_disabled_fp32_none",
        )

    def _state_from_inputs(
        self,
        channel_state: Optional[str] = None,
        channel_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        state = channel_state
        if isinstance(channel_profile, dict):
            state = (
                channel_profile.get("state_name", None)
                or channel_profile.get("channel_state", None)
                or channel_profile.get("state", None)
                or state
            )
        try:
            return normalize_channel_state(state or CHANNEL_STATE_MEDIUM)
        except Exception:
            return CHANNEL_STATE_MEDIUM

    def select(
        self,
        channel_state: Optional[str] = None,
        channel_profile: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ARCEAction:
        if not self.enabled:
            return copy.deepcopy(self.disabled_action)

        action = copy.deepcopy(self.rng.choice(self.actions))

        # Keep the actual Markov link state in the action record, while the
        # quant/fec/redundancy values remain randomly sampled.
        current_state = self._state_from_inputs(
            channel_state=channel_state,
            channel_profile=channel_profile,
        )
        return action.copy_with(channel_state=current_state)

    def get_config(self) -> Dict[str, Any]:
        return {
            "policy": "random",
            "enabled": bool(self.enabled),
            "seed": int(self.seed),
            "sample_mode": self.sample_mode,
            "num_actions": len(self.actions),
            "actions": [action.as_dict() for action in self.actions],
        }

    def __repr__(self) -> str:
        return (
            "RandomARCEPolicy("
            f"enabled={self.enabled}, "
            f"seed={self.seed}, "
            f"num_actions={len(self.actions)})"
        )


__all__ = [
    "DEFAULT_RANDOM_ACTION_SPACE",
    "RandomARCEPolicy",
]
