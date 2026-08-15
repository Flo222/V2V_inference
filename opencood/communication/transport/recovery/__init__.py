"""
Partial reconstruction package for ARCE communication simulation.

This subpackage contains recovery modules used after packet loss and FEC
decoding in OPV2V / V2X-ViT ARCE experiments.

Planned modules:

1. zero_fill.py
   Fill missing source packets / spatial regions with zeros.

2. spatial_interpolation.py
   Recover missing spatial packets using neighboring packets or neighboring
   BEV regions.

3. temporal_cache.py
   Maintain historical feature packets for each communication link and use
   cached packets to fill missing packets.

4. partial_reconstruction.py
   Unified reconstruction controller:
       FEC recovered packets
       -> temporal cache
       -> spatial interpolation
       -> zero-fill

Design note:
    Keep this __init__.py lightweight.

    Do not eagerly import concrete recovery classes here during early
    development. These modules may depend on torch, packetizer metadata,
    temporal states, or ARCE communication logic.

    Concrete modules should be imported directly where they are used:

        from opencood.communication.transport.recovery.zero_fill import zero_fill_packets
        from opencood.communication.transport.recovery.temporal_cache import TemporalFeatureCache
        from opencood.communication.transport.recovery.partial_reconstruction import PartialReconstructor
"""

RECOVERY_METHOD_NONE = "none"
RECOVERY_METHOD_ZERO = "zero"
RECOVERY_METHOD_ZERO_FILL = "zero_fill"
RECOVERY_METHOD_SPATIAL = "spatial"
RECOVERY_METHOD_SPATIAL_INTERPOLATION = "spatial_interpolation"
RECOVERY_METHOD_TEMPORAL = "temporal"
RECOVERY_METHOD_TEMPORAL_CACHE = "temporal_cache"
RECOVERY_METHOD_ARCE = "arce"

DEFAULT_RECOVERY_METHOD = RECOVERY_METHOD_ZERO_FILL

VALID_RECOVERY_METHODS = (
    RECOVERY_METHOD_NONE,
    RECOVERY_METHOD_ZERO,
    RECOVERY_METHOD_ZERO_FILL,
    RECOVERY_METHOD_SPATIAL,
    RECOVERY_METHOD_SPATIAL_INTERPOLATION,
    RECOVERY_METHOD_TEMPORAL,
    RECOVERY_METHOD_TEMPORAL_CACHE,
    RECOVERY_METHOD_ARCE,
)

CANONICAL_RECOVERY_METHOD = {
    RECOVERY_METHOD_ZERO: RECOVERY_METHOD_ZERO_FILL,
    RECOVERY_METHOD_SPATIAL: RECOVERY_METHOD_SPATIAL_INTERPOLATION,
    RECOVERY_METHOD_TEMPORAL: RECOVERY_METHOD_TEMPORAL_CACHE,
}

# Unified packet mask convention:
#   missing_mask[i] == True
#       source packet i is missing after channel / FEC.
#
#   available_mask[i] == True
#       source packet i is available after direct receive / FEC / recovery.
#
#   recovered_mask[i] == True
#       source packet i has been recovered by a specific recovery method.
MISSING_MASK_TRUE_MEANS_MISSING = True
AVAILABLE_MASK_TRUE_MEANS_AVAILABLE = True

# Recovery priority used by ARCE partial reconstruction.
# FEC is handled before this package and is therefore not listed as a method
# here, but partial_reconstruction.py should treat FEC-recovered packets as
# the highest-priority available packets.
DEFAULT_RECOVERY_PRIORITY = (
    RECOVERY_METHOD_TEMPORAL_CACHE,
    RECOVERY_METHOD_SPATIAL_INTERPOLATION,
    RECOVERY_METHOD_ZERO_FILL,
)


def normalize_recovery_method(method=None):
    """
    Normalize recovery method name.

    Parameters
    ----------
    method : str or None
        Recovery method. Supported values:
            none
            zero / zero_fill
            spatial / spatial_interpolation
            temporal / temporal_cache
            arce

    Returns
    -------
    str
        Canonical recovery method.

    Raises
    ------
    ValueError
        If method is unsupported.
    """
    if method is None:
        return DEFAULT_RECOVERY_METHOD

    method = str(method).strip().lower()

    if method not in VALID_RECOVERY_METHODS:
        raise ValueError(
            f"Unsupported recovery method: {method}. "
            f"Supported methods: {VALID_RECOVERY_METHODS}."
        )

    return CANONICAL_RECOVERY_METHOD.get(method, method)


def is_valid_recovery_method(method):
    """
    Check whether a recovery method is valid.

    Parameters
    ----------
    method : str
        Input recovery method.

    Returns
    -------
    bool
        True if valid, otherwise False.
    """
    try:
        normalize_recovery_method(method)
        return True
    except ValueError:
        return False


def normalize_recovery_priority(priority=None):
    """
    Normalize recovery priority list.

    Parameters
    ----------
    priority : list / tuple / str / None
        Recovery priority.

        Examples:
            None
                -> default priority:
                   temporal_cache -> spatial_interpolation -> zero_fill

            ["temporal_cache", "spatial_interpolation", "zero_fill"]

            "temporal_cache,spatial_interpolation,zero_fill"

    Returns
    -------
    tuple
        Canonical recovery method tuple.
    """
    if priority is None:
        return DEFAULT_RECOVERY_PRIORITY

    if isinstance(priority, str):
        priority = [
            item.strip()
            for item in priority.split(",")
            if item.strip()
        ]

    if not isinstance(priority, (list, tuple)):
        raise ValueError(
            "recovery priority should be a list, tuple, comma-separated "
            f"string, or None, got {type(priority)}."
        )

    normalized = []
    for method in priority:
        method = normalize_recovery_method(method)

        if method == RECOVERY_METHOD_NONE:
            continue

        if method == RECOVERY_METHOD_ARCE:
            # "arce" means use the default ARCE reconstruction chain.
            for default_method in DEFAULT_RECOVERY_PRIORITY:
                if default_method not in normalized:
                    normalized.append(default_method)
            continue

        if method not in normalized:
            normalized.append(method)

    return tuple(normalized)


def _as_bool(value):
    """
    Convert common config values to bool.

    Supports:
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


def _as_non_negative_float(value, name):
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


def _as_positive_int(value, name):
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


def missing_mask_to_available_mask(missing_mask):
    """
    Convert missing mask to available mask.

    Convention:
        missing_mask[i] == True means source packet i is missing.
        available_mask[i] == True means source packet i is available.
    """
    return ~missing_mask


def available_mask_to_missing_mask(available_mask):
    """
    Convert available mask to missing mask.

    Convention:
        available_mask[i] == True means source packet i is available.
        missing_mask[i] == True means source packet i is missing.
    """
    return ~available_mask


def normalize_recovery_config(cfg=None):
    """
    Normalize recovery config.

    Supported input:
        normalize_recovery_config(arce_cfg)
        normalize_recovery_config(arce_cfg["recovery"])

    YAML style:

        arce:
          recovery:
            zero_fill: true
            spatial_interpolation: true
            temporal_cache: true
            cache_tau: 5.0
            max_cache_age: 5
            priority:
              - temporal_cache
              - spatial_interpolation
              - zero_fill

    Returns
    -------
    dict
        Normalized recovery config:
            {
                "zero_fill": bool,
                "spatial_interpolation": bool,
                "temporal_cache": bool,
                "cache_tau": float,
                "max_cache_age": int,
                "priority": tuple
            }
    """
    cfg = cfg or {}

    if "recovery" in cfg and isinstance(cfg["recovery"], dict):
        cfg = cfg["recovery"]

    zero_fill = _as_bool(cfg.get("zero_fill", True))
    spatial_interpolation = _as_bool(
        cfg.get("spatial_interpolation", cfg.get("spatial", False))
    )
    temporal_cache = _as_bool(
        cfg.get("temporal_cache", cfg.get("temporal", False))
    )

    cache_tau = _as_non_negative_float(
        cfg.get("cache_tau", 5.0),
        "cache_tau",
    )

    max_cache_age = _as_positive_int(
        cfg.get("max_cache_age", 5),
        "max_cache_age",
    )

    priority = cfg.get("priority", None)

    if priority is None:
        priority_list = []

        if temporal_cache:
            priority_list.append(RECOVERY_METHOD_TEMPORAL_CACHE)

        if spatial_interpolation:
            priority_list.append(RECOVERY_METHOD_SPATIAL_INTERPOLATION)

        if zero_fill:
            priority_list.append(RECOVERY_METHOD_ZERO_FILL)

        priority = priority_list

    priority = normalize_recovery_priority(priority)

    return {
        "zero_fill": bool(zero_fill),
        "spatial_interpolation": bool(spatial_interpolation),
        "temporal_cache": bool(temporal_cache),
        "cache_tau": float(cache_tau),
        "max_cache_age": int(max_cache_age),
        "priority": priority,
    }


def get_recovery_config_summary(cfg=None):
    """
    Return JSON-serializable recovery config summary.

    Parameters
    ----------
    cfg : dict or None
        Full ARCE config or direct recovery config.

    Returns
    -------
    dict
        Recovery config summary.
    """
    cfg = normalize_recovery_config(cfg)

    return {
        "zero_fill": bool(cfg["zero_fill"]),
        "spatial_interpolation": bool(cfg["spatial_interpolation"]),
        "temporal_cache": bool(cfg["temporal_cache"]),
        "cache_tau": float(cfg["cache_tau"]),
        "max_cache_age": int(cfg["max_cache_age"]),
        "priority": tuple(cfg["priority"]),
    }


def build_recovery_count_dict(
    num_source_packets=0,
    num_fec_recovered=0,
    num_temporal_filled=0,
    num_spatial_filled=0,
    num_zero_filled=0,
    num_still_missing=0,
):
    """
    Build a standard recovery count dictionary.

    This function is mainly used by partial_reconstruction.py and logging.

    Returns
    -------
    dict
        Standard recovery count fields.
    """
    num_source_packets = int(num_source_packets)
    num_fec_recovered = int(num_fec_recovered)
    num_temporal_filled = int(num_temporal_filled)
    num_spatial_filled = int(num_spatial_filled)
    num_zero_filled = int(num_zero_filled)
    num_still_missing = int(num_still_missing)

    num_total_available = (
        num_source_packets
        - num_still_missing
    )

    return {
        "num_source_packets": num_source_packets,
        "num_fec_recovered_packets": num_fec_recovered,
        "num_temporal_filled_packets": num_temporal_filled,
        "num_spatial_filled_packets": num_spatial_filled,
        "num_zero_filled_packets": num_zero_filled,
        "num_still_missing_packets": num_still_missing,
        "num_total_available_packets": num_total_available,
        "recovery_ratio": (
            float(num_total_available / num_source_packets)
            if num_source_packets > 0
            else 1.0
        ),
    }


__all__ = [
    "RECOVERY_METHOD_NONE",
    "RECOVERY_METHOD_ZERO",
    "RECOVERY_METHOD_ZERO_FILL",
    "RECOVERY_METHOD_SPATIAL",
    "RECOVERY_METHOD_SPATIAL_INTERPOLATION",
    "RECOVERY_METHOD_TEMPORAL",
    "RECOVERY_METHOD_TEMPORAL_CACHE",
    "RECOVERY_METHOD_ARCE",
    "DEFAULT_RECOVERY_METHOD",
    "VALID_RECOVERY_METHODS",
    "CANONICAL_RECOVERY_METHOD",
    "MISSING_MASK_TRUE_MEANS_MISSING",
    "AVAILABLE_MASK_TRUE_MEANS_AVAILABLE",
    "DEFAULT_RECOVERY_PRIORITY",
    "normalize_recovery_method",
    "is_valid_recovery_method",
    "normalize_recovery_priority",
    "missing_mask_to_available_mask",
    "available_mask_to_missing_mask",
    "normalize_recovery_config",
    "get_recovery_config_summary",
    "build_recovery_count_dict",
]