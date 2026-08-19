"""Channel profile and budget manager for GRACE / C2MAB.

This module owns channel-state/profile parsing and budget resolution logic.
The communication executor should call these helpers instead of keeping all
channel-budget rules inline.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


def channel_profiles_cfg(
    arce_cfg: Dict[str, Any],
    default_channel_profiles: Dict[str, Dict[str, Any]],
    normalize_state_name_fn: Callable[[Any], str],
) -> Dict[str, Dict[str, Any]]:
    channel_cfg = arce_cfg.get("channel", {}) or {}
    profiles = channel_cfg.get("profiles", None)

    out = copy.deepcopy(default_channel_profiles)

    if isinstance(profiles, dict):
        for state, profile in profiles.items():
            state_l = normalize_state_name_fn(state)
            if state_l == "ego_or_padding":
                continue
            if isinstance(profile, dict):
                merged = copy.deepcopy(out.get(state_l, {}))
                merged.update(copy.deepcopy(profile))
                merged.setdefault("state_name", state_l)
                out[state_l] = merged

    # Force PLR / fixed-delay defaults unless explicitly overridden.
    for state_name, plr in (("good", 0.05), ("medium", 0.20), ("bad", 0.35)):
        out[state_name]["loss_rate"] = float(
            out[state_name].get("plr", out[state_name].get("loss_rate", plr))
        )
        out[state_name]["plr"] = float(out[state_name]["loss_rate"])

    for state_name, delay_ms in (("good", 10.0), ("medium", 50.0), ("bad", 100.0)):
        out[state_name]["delay_ms"] = float(
            out[state_name].get(
                "fixed_delay_ms",
                out[state_name].get("delay_ms", delay_ms),
            )
        )
        out[state_name]["fixed_delay_ms"] = float(out[state_name]["delay_ms"])

    out["good"]["temporal_source"] = "current"
    out["medium"]["temporal_source"] = "current"
    out["bad"]["temporal_source"] = "previous_frame"

    return out


def profile_for_state(
    state_name: str,
    arce_cfg: Dict[str, Any],
    default_channel_profiles: Dict[str, Dict[str, Any]],
    normalize_state_name_fn: Callable[[Any], str],
) -> Dict[str, Any]:
    state_name = normalize_state_name_fn(state_name)
    profiles = channel_profiles_cfg(
        arce_cfg,
        default_channel_profiles,
        normalize_state_name_fn,
    )
    profile = copy.deepcopy(profiles.get(state_name, profiles.get("medium")))
    profile["state_name"] = state_name
    return profile


def extract_channel_state_ids(
    data_dict: Optional[Dict[str, Any]],
    local_batch_idx: int,
    safe_get_nested_fn: Callable[[Dict[str, Any], List[str]], Any],
) -> Optional[List[int]]:
    if data_dict is None:
        return None

    candidates = [
        safe_get_nested_fn(data_dict, ["ego", "channel_state_ids"]),
        safe_get_nested_fn(data_dict, ["channel_state_ids"]),
    ]

    for x in candidates:
        if x is None:
            continue

        if torch.is_tensor(x):
            arr = x.detach().cpu()
            if arr.dim() == 1:
                return [int(v) for v in arr.tolist()]
            if arr.dim() >= 2:
                idx = min(int(local_batch_idx), int(arr.shape[0]) - 1)
                return [int(v) for v in arr[idx].flatten().tolist()]

        if isinstance(x, (list, tuple)):
            if len(x) > 0 and isinstance(x[0], (list, tuple)):
                idx = min(int(local_batch_idx), len(x) - 1)
                return [int(v) for v in x[idx]]
            return [int(v) for v in x]

    return None


def state_name_for_sender(
    data_dict: Optional[Dict[str, Any]],
    batch_idx: int,
    sender_local_idx: int,
    state_id_to_name: Dict[int, str],
    safe_get_nested_fn: Callable[[Dict[str, Any], List[str]], Any],
) -> str:
    ids = extract_channel_state_ids(data_dict, batch_idx, safe_get_nested_fn)
    if ids is not None and 0 <= int(sender_local_idx) < len(ids):
        return state_id_to_name.get(int(ids[int(sender_local_idx)]), "medium")
    return "medium"


def budget_source_scope(
    arce_cfg: Dict[str, Any],
    default_budget_scope: str,
) -> Tuple[str, str]:
    scheduler_cfg = arce_cfg.get("scheduler", {}) or {}
    if not isinstance(scheduler_cfg, dict):
        scheduler_cfg = {}

    oracle_cfg = arce_cfg.get("ego_oracle", {}) or {}
    if not isinstance(oracle_cfg, dict):
        oracle_cfg = {}

    budget_source = str(
        scheduler_cfg.get(
            "budget_source",
            arce_cfg.get(
                "budget_source",
                oracle_cfg.get("budget_source", "channel_profiles"),
            ),
        )
    ).strip().lower()

    budget_scope = str(
        scheduler_cfg.get(
            "budget_scope",
            arce_cfg.get(
                "budget_scope",
                oracle_cfg.get("budget_scope", default_budget_scope),
            ),
        )
    ).strip().lower()

    return budget_source, budget_scope


def use_channel_profile_budget(
    arce_cfg: Dict[str, Any],
    default_budget_scope: str,
) -> bool:
    source, scope = budget_source_scope(arce_cfg, default_budget_scope)
    return (
        source in ("channel_profiles", "channel_profile", "profiles")
        or scope in ("global_sum_link", "channel_profiles", "channel_profile")
    )


def system_budget_bytes(
    system_budget_mbps: float,
    tx_window_ms: float,
    budget_bytes_from_bandwidth_fn: Callable[[float, float], float],
) -> float:
    return float(
        budget_bytes_from_bandwidth_fn(
            float(system_budget_mbps),
            float(tx_window_ms),
        )
    )


def channel_profile_budget_bytes(
    profile: Dict[str, Any],
    system_budget_mbps: float,
    tx_window_ms: float,
    profile_scalar_fn: Callable[[Any, float], float],
    budget_bytes_from_bandwidth_fn: Callable[[float, float], float],
) -> float:
    bandwidth_mbps = profile_scalar_fn(
        profile.get("bandwidth_mbps", system_budget_mbps),
        float(system_budget_mbps),
    )
    return float(
        budget_bytes_from_bandwidth_fn(
            float(bandwidth_mbps),
            float(tx_window_ms),
        )
    )


def per_link_budget_bytes(
    num_collaborators: int,
    system_budget_mbps: float,
    tx_window_ms: float,
    budget_bytes_from_bandwidth_fn: Callable[[float, float], float],
) -> float:
    if int(num_collaborators) <= 0:
        return 0.0
    return float(
        system_budget_bytes(
            float(system_budget_mbps),
            float(tx_window_ms),
            budget_bytes_from_bandwidth_fn,
        )
        / float(num_collaborators)
    )


def prepare_link_channel_budget(
    data_dict: Optional[Dict[str, Any]],
    batch_idx: int,
    sender_idx: int,
    num_collaborators: int,
    arce_cfg: Dict[str, Any],
    default_channel_profiles: Dict[str, Dict[str, Any]],
    state_id_to_name: Dict[int, str],
    system_budget_mbps: float,
    tx_window_ms: float,
    default_budget_scope: str,
    normalize_state_name_fn: Callable[[Any], str],
    safe_get_nested_fn: Callable[[Dict[str, Any], List[str]], Any],
    profile_scalar_fn: Callable[[Any, float], float],
    budget_bytes_from_bandwidth_fn: Callable[[float, float], float],
) -> Tuple[str, Dict[str, Any], float]:
    state_name = state_name_for_sender(
        data_dict,
        batch_idx,
        sender_idx,
        state_id_to_name,
        safe_get_nested_fn,
    )
    profile = profile_for_state(
        state_name,
        arce_cfg,
        default_channel_profiles,
        normalize_state_name_fn,
    )

    if state_name == "ego_or_padding":
        return state_name, profile, 0.0

    if use_channel_profile_budget(arce_cfg, default_budget_scope):
        link_budget_bytes = channel_profile_budget_bytes(
            profile,
            system_budget_mbps,
            tx_window_ms,
            profile_scalar_fn,
            budget_bytes_from_bandwidth_fn,
        )
    else:
        link_budget_bytes = per_link_budget_bytes(
            num_collaborators,
            system_budget_mbps,
            tx_window_ms,
            budget_bytes_from_bandwidth_fn,
        )

    return state_name, profile, float(link_budget_bytes)
