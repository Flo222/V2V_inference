"""
ARCE communication package for OpenCOOD / OPV2V.

This subpackage contains the ARCE communication pipeline used in
V2X-ViT / OPV2V experiments.

Planned modules:

1. arce_fixed_comm.py
   Main fixed-policy ARCE communication pipeline:
       feature
       -> packetization
       -> quantization
       -> FEC encoding
       -> channel loss / latency
       -> FEC decoding
       -> partial reconstruction
       -> recovered feature

2. policy_fixed.py, optional later
   Fixed ARCE action selection:
       Good   channel -> weaker protection / higher quality
       Medium channel -> balanced quantization + redundancy
       Bad    channel -> stronger compression / stronger redundancy

3. policy_dynamic.py, optional later
   Dynamic policy / bandit / learned ARCE controller.

Design note:
    Keep this __init__.py lightweight.

    Do not eagerly import ARCEFixedComm here. The ARCE communication pipeline
    depends on torch, packetization, quantization, FEC, recovery, and channel
    modules. Importing it here can easily introduce circular imports.

    Concrete modules should be imported directly where they are used:

        from opencood.methods.arce.arce_fixed_comm import ARCEFixedComm

    instead of:

        from opencood.methods.arce import ARCEFixedComm
"""

from typing import Any, Dict, Optional


ARCE_MODE_DISABLED = "disabled"
ARCE_MODE_FIXED = "fixed"
ARCE_MODE_BYPASS = "bypass"

DEFAULT_ARCE_MODE = ARCE_MODE_FIXED

VALID_ARCE_MODES = (
    ARCE_MODE_DISABLED,
    ARCE_MODE_FIXED,
    ARCE_MODE_BYPASS,
)


ARCE_POLICY_FIXED = "fixed"
ARCE_POLICY_STATIC = "static"
ARCE_POLICY_RANDOM = "random"
ARCE_POLICY_NONE = "none"

DEFAULT_ARCE_POLICY = ARCE_POLICY_FIXED

VALID_ARCE_POLICIES = (
    ARCE_POLICY_FIXED,
    ARCE_POLICY_STATIC,
    ARCE_POLICY_RANDOM,
    ARCE_POLICY_NONE,
)


LINK_SCOPE_NON_EGO = "non_ego"
LINK_SCOPE_ALL = "all"
LINK_SCOPE_COOPERATIVE_ONLY = "cooperative_only"

DEFAULT_LINK_SCOPE = LINK_SCOPE_NON_EGO

VALID_LINK_SCOPES = (
    LINK_SCOPE_NON_EGO,
    LINK_SCOPE_ALL,
    LINK_SCOPE_COOPERATIVE_ONLY,
)


DEFAULT_FIXED_POLICY = {
    "channel_state": "medium",
    "quant_mode": "int8",
    "fec_type": "xor",
    "redundancy_ratio": 0.25,
    "xor_group_size": 4,
    "recovery": "arce",
}


def _as_bool(value: Any) -> bool:
    """
    Convert common config values to bool.

    Supported string values:
        true / false
        yes / no
        1 / 0
        on / off
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


def _as_int(value: Any, name: str) -> int:
    """
    Convert value to int.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} should be convertible to int, got {value}.")


def _as_positive_int(value: Any, name: str) -> int:
    """
    Convert value to positive int.
    """
    value = _as_int(value, name)

    if value <= 0:
        raise ValueError(f"{name} should be positive, got {value}.")

    return value


def _as_non_negative_int(value: Any, name: str) -> int:
    """
    Convert value to non-negative int.
    """
    value = _as_int(value, name)

    if value < 0:
        raise ValueError(f"{name} should be non-negative, got {value}.")

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


def extract_arce_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract ARCE config from a full config dictionary.

    Supported input:
        extract_arce_cfg(full_cfg)
        extract_arce_cfg(full_cfg["arce"])

    Parameters
    ----------
    cfg : dict or None
        Full YAML config or direct ARCE config.

    Returns
    -------
    dict
        ARCE config dictionary.
    """
    cfg = cfg or {}

    if "arce" in cfg and isinstance(cfg["arce"], dict):
        return cfg["arce"]

    return cfg


def normalize_arce_mode(mode: Optional[str] = None) -> str:
    """
    Normalize ARCE mode.

    Supported modes:
        disabled:
            Do not apply ARCE communication.

        fixed:
            Use fixed ARCE communication policy.

        bypass:
            Keep ARCE module constructed but bypass communication impairment.
            This is useful for debugging.

    Returns
    -------
    str
        Canonical ARCE mode.
    """
    if mode is None:
        return DEFAULT_ARCE_MODE

    mode = str(mode).strip().lower()

    if mode not in VALID_ARCE_MODES:
        raise ValueError(
            f"Unsupported ARCE mode: {mode}. "
            f"Supported modes: {VALID_ARCE_MODES}."
        )

    return mode


def is_valid_arce_mode(mode: Optional[str]) -> bool:
    """
    Check whether ARCE mode is valid.
    """
    try:
        normalize_arce_mode(mode)
        return True
    except ValueError:
        return False


def normalize_arce_policy(policy: Optional[str] = None) -> str:
    """
    Normalize ARCE policy name.

    Supported policies:
        fixed:
            Fixed hand-crafted ARCE action.

        static:
            Alias-like policy name for fixed/static settings.

        none:
            No ARCE action selection.
    """
    if policy is None:
        return DEFAULT_ARCE_POLICY

    policy = str(policy).strip().lower()

    if policy not in VALID_ARCE_POLICIES:
        raise ValueError(
            f"Unsupported ARCE policy: {policy}. "
            f"Supported policies: {VALID_ARCE_POLICIES}."
        )

    if policy == ARCE_POLICY_STATIC:
        return ARCE_POLICY_FIXED

    return policy


def normalize_link_scope(scope: Optional[str] = None) -> str:
    """
    Normalize communication link scope.

    Supported scopes:
        non_ego:
            Apply communication impairment only to non-ego collaborative
            features. Ego feature is kept local and clean.

        all:
            Apply communication module to all features, including ego.
            This is usually not recommended for OPV2V cooperative perception.

        cooperative_only:
            Alias of non_ego.
    """
    if scope is None:
        return DEFAULT_LINK_SCOPE

    scope = str(scope).strip().lower()

    if scope not in VALID_LINK_SCOPES:
        raise ValueError(
            f"Unsupported ARCE link scope: {scope}. "
            f"Supported scopes: {VALID_LINK_SCOPES}."
        )

    if scope == LINK_SCOPE_COOPERATIVE_ONLY:
        return LINK_SCOPE_NON_EGO

    return scope


def normalize_fixed_policy_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Normalize fixed ARCE policy config.

    Supported input:
        normalize_fixed_policy_config(arce_cfg)
        normalize_fixed_policy_config(arce_cfg["fixed_policy"])

    YAML style:

        arce:
          fixed_policy:
            channel_state: medium
            quant_mode: int8
            fec_type: xor
            redundancy_ratio: 0.25
            xor_group_size: 4
            recovery: arce

    Returns
    -------
    dict
        Normalized fixed policy dictionary.
    """
    cfg = cfg or {}

    if "fixed_policy" in cfg and isinstance(cfg["fixed_policy"], dict):
        cfg = cfg["fixed_policy"]

    policy = dict(DEFAULT_FIXED_POLICY)
    policy.update(cfg)

    policy["channel_state"] = str(policy.get("channel_state", "medium")).strip().lower()
    policy["quant_mode"] = str(policy.get("quant_mode", "int8")).strip().lower()
    policy["fec_type"] = str(policy.get("fec_type", "xor")).strip().lower()
    policy["recovery"] = str(policy.get("recovery", "arce")).strip().lower()

    policy["redundancy_ratio"] = _as_non_negative_float(
        policy.get("redundancy_ratio", 0.25),
        "fixed_policy.redundancy_ratio",
    )

    policy["xor_group_size"] = _as_positive_int(
        policy.get("xor_group_size", 4),
        "fixed_policy.xor_group_size",
    )

    return policy


def normalize_arce_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Normalize ARCE config.

    Supported YAML style:

        arce:
          enabled: true
          mode: fixed
          policy: fixed
          seed: 2026

          link_scope: non_ego
          record_per_frame: true
          record_per_link: true
          log_interval: 1

          fixed_policy:
            channel_state: medium
            quant_mode: int8
            fec_type: xor
            redundancy_ratio: 0.25
            xor_group_size: 4
            recovery: arce

    Returns
    -------
    dict
        Normalized ARCE config.
    """
    arce_cfg = extract_arce_cfg(cfg)

    mode = normalize_arce_mode(arce_cfg.get("mode", DEFAULT_ARCE_MODE))

    if "enabled" in arce_cfg:
        enabled = _as_bool(arce_cfg.get("enabled"))
    else:
        enabled = mode != ARCE_MODE_DISABLED

    if not enabled:
        mode = ARCE_MODE_DISABLED

    policy = normalize_arce_policy(arce_cfg.get("policy", DEFAULT_ARCE_POLICY))

    if mode == ARCE_MODE_DISABLED:
        policy = ARCE_POLICY_NONE

    seed = _as_non_negative_int(arce_cfg.get("seed", 0), "arce.seed")

    link_scope = normalize_link_scope(
        arce_cfg.get("link_scope", DEFAULT_LINK_SCOPE)
    )

    record_per_frame = _as_bool(arce_cfg.get("record_per_frame", True))
    record_per_link = _as_bool(arce_cfg.get("record_per_link", True))

    log_interval = _as_positive_int(
        arce_cfg.get("log_interval", 1),
        "arce.log_interval",
    )

    verbose = _as_bool(arce_cfg.get("verbose", False))
    debug = _as_bool(arce_cfg.get("debug", False))

    fixed_policy = normalize_fixed_policy_config(arce_cfg)

    return {
        "enabled": bool(enabled),
        "mode": mode,
        "policy": policy,
        "seed": int(seed),
        "link_scope": link_scope,
        "record_per_frame": bool(record_per_frame),
        "record_per_link": bool(record_per_link),
        "log_interval": int(log_interval),
        "verbose": bool(verbose),
        "debug": bool(debug),
        "fixed_policy": fixed_policy,
    }


def should_apply_to_agent(agent_index: int, ego_index: int = 0, link_scope: str = DEFAULT_LINK_SCOPE) -> bool:
    """
    Decide whether ARCE communication should be applied to an agent feature.

    Parameters
    ----------
    agent_index : int
        Current agent index.

    ego_index : int
        Ego agent index. In many OpenCOOD settings, ego is index 0.

    link_scope : str
        non_ego / all / cooperative_only.

    Returns
    -------
    bool
        True if this agent feature should pass through ARCE communication.
    """
    link_scope = normalize_link_scope(link_scope)

    agent_index = int(agent_index)
    ego_index = int(ego_index)

    if link_scope == LINK_SCOPE_ALL:
        return True

    if link_scope == LINK_SCOPE_NON_EGO:
        return agent_index != ego_index

    raise ValueError(f"Unexpected normalized link_scope: {link_scope}")


def get_arce_config_summary(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return JSON-friendly ARCE config summary.
    """
    return normalize_arce_config(cfg)


__all__ = [
    "ARCE_MODE_DISABLED",
    "ARCE_MODE_FIXED",
    "ARCE_MODE_BYPASS",
    "DEFAULT_ARCE_MODE",
    "VALID_ARCE_MODES",
    "ARCE_POLICY_FIXED",
    "ARCE_POLICY_STATIC",
    "ARCE_POLICY_RANDOM",
    "ARCE_POLICY_NONE",
    "DEFAULT_ARCE_POLICY",
    "VALID_ARCE_POLICIES",
    "LINK_SCOPE_NON_EGO",
    "LINK_SCOPE_ALL",
    "LINK_SCOPE_COOPERATIVE_ONLY",
    "DEFAULT_LINK_SCOPE",
    "VALID_LINK_SCOPES",
    "DEFAULT_FIXED_POLICY",
    "extract_arce_cfg",
    "normalize_arce_mode",
    "is_valid_arce_mode",
    "normalize_arce_policy",
    "normalize_link_scope",
    "normalize_fixed_policy_config",
    "normalize_arce_config",
    "should_apply_to_agent",
    "get_arce_config_summary",
]