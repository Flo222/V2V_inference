"""Runtime action adapter for PDF-ARCE and fixed ARCE actions.

Why this exists:
    PDFARCEAction has PDF-level fields such as send/cache_enabled/action_id.
    ARCEAction is the low-level executor action and may store extra fields
    either as attributes or inside action.extra.

This adapter provides one canonical access path so execution code does not
depend on ad-hoc getattr/setattr behavior.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional


def _get_extra(action: Any) -> Dict[str, Any]:
    extra = getattr(action, "extra", None)
    if isinstance(extra, dict):
        return extra
    return {}


def get_action_field(action: Any, name: str, default: Any = None) -> Any:
    """Read one action field from dict, attribute, or action.extra.

    For PDF/runtime fields, prefer action.extra because low-level ARCEAction
    may have legacy/default attributes that do not reflect PDFAction values.
    """
    if action is None:
        return default

    if isinstance(action, dict):
        return action.get(name, default)

    extra = _get_extra(action)

    # PDF/runtime fields should not be shadowed by legacy/default attributes.
    # For example, no-send actions may have PDF rho=0.25 in the 48-action
    # space, while low-level ARCEAction correctly uses effective rho=0.0
    # for execution. Policy/logging should still see the PDF rho.
    pdf_runtime_fields = (
        "send",
        "cache_enabled",
        "action_id",
        "pdf_action_id",
        "quant_mode",
        "fec_type",
        "redundancy_ratio",
        "xor_group_size",
        "decode_overhead",
        "channel_state",
    )
    if name in pdf_runtime_fields:
        if name in extra:
            return extra[name]
        if name == "action_id" and "pdf_action_id" in extra:
            return extra["pdf_action_id"]

    if hasattr(action, name):
        return getattr(action, name)

    if name in extra:
        return extra[name]

    return default


def set_action_field(action: Any, name: str, value: Any) -> Any:
    """Write one runtime field to both attribute and extra when possible."""
    if action is None:
        return action

    if isinstance(action, dict):
        action[name] = value
        return action

    try:
        setattr(action, name, value)
    except Exception:
        pass

    if hasattr(action, "extra"):
        extra = getattr(action, "extra", None)
        if not isinstance(extra, dict):
            try:
                setattr(action, "extra", {})
                extra = getattr(action, "extra")
            except Exception:
                extra = None
        if isinstance(extra, dict):
            extra[name] = value

    return action


def normalize_runtime_action(
    action: Any,
    *,
    send: Optional[int] = None,
    cache_enabled: Optional[int] = None,
    action_id: Optional[str] = None,
    default_send: int = 1,
    default_cache_enabled: int = 0,
) -> Any:
    """Ensure runtime action has canonical PDF/runtime fields.

    The returned object is the same object, updated in-place where possible.
    """
    if action is None:
        return action

    if send is None:
        send = int(get_action_field(action, "send", default_send))
    else:
        send = int(send)

    if cache_enabled is None:
        cache_enabled = int(get_action_field(action, "cache_enabled", default_cache_enabled))
    else:
        cache_enabled = int(cache_enabled)

    if action_id is None:
        action_id = get_action_field(action, "action_id", None)

    set_action_field(action, "send", int(send))
    set_action_field(action, "cache_enabled", int(cache_enabled))

    if action_id is not None:
        set_action_field(action, "action_id", str(action_id))

    return action


def runtime_action_as_dict(action: Any) -> Dict[str, Any]:
    """Export runtime action with merged core fields and extra fields."""
    if action is None:
        return {"send": 0}

    if isinstance(action, dict):
        return copy.deepcopy(action)

    if hasattr(action, "as_dict"):
        try:
            d = action.as_dict()
        except Exception:
            d = {}
    else:
        d = {}

    if not isinstance(d, dict):
        d = {}

    extra = _get_extra(action)
    if extra:
        d.setdefault("extra", copy.deepcopy(extra))
        for k, v in extra.items():
            d.setdefault(k, copy.deepcopy(v))

    for key, default in (
        ("send", 1),
        ("cache_enabled", 0),
        ("action_id", None),
        ("quant_mode", None),
        ("fec_type", None),
        ("redundancy_ratio", 0.0),
        ("xor_group_size", None),
        ("decode_overhead", 0.0),
    ):
        value = get_action_field(action, key, default)
        if value is not None:
            d[key] = value

    return d
