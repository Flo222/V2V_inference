"""
FEC / redundancy coding package for ARCE communication simulation.

This subpackage contains packet-level forward error correction modules used
by the ARCE communication layer in OPV2V / V2X-ViT experiments.

Planned modules:

1. fec_base.py
   Common base classes and dataclasses for FEC encoding / decoding.

2. fec_none.py
   No-FEC baseline. Source packets are transmitted directly.

3. fec_xor.py
   Real XOR parity recovery over integer quantized packets.
   One parity packet is generated for each packet group.

4. fec_raptor_sim.py
   Raptor / fountain-code style abstract simulation.
   This is a recovery-threshold simulator, not a real Raptor codec.

5. fec_raptorq.py
   RFC 6330 RaptorQ adapter backed by the pinned ``raptorq`` package.

Design note:
    Keep this __init__.py lightweight.

    Do not eagerly import concrete FEC classes here during early development.
    Those modules may depend on torch and other ARCE submodules. Import them
    directly where they are used:

        from opencood.communication.transport.fec.fec_xor import XORFEC
        from opencood.communication.transport.fec.fec_none import NoFEC
        from opencood.communication.transport.fec.fec_raptor_sim import RaptorSimFEC
"""

import math
from typing import Any, Dict, Optional, Tuple


FEC_TYPE_NONE = "none"
FEC_TYPE_XOR = "xor"
FEC_TYPE_RAPTOR_SIM = "raptor_sim"
FEC_TYPE_RAPTOR = "raptor"
FEC_TYPE_RAPTORQ = "raptorq"

DEFAULT_FEC_TYPE = FEC_TYPE_NONE
DEFAULT_REDUNDANCY_RATIO = 0.0
DEFAULT_XOR_GROUP_SIZE = 4
DEFAULT_RAPTOR_DECODE_OVERHEAD = 0.0

VALID_FEC_TYPES = (
    FEC_TYPE_NONE,
    FEC_TYPE_XOR,
    FEC_TYPE_RAPTOR_SIM,
    FEC_TYPE_RAPTOR,
    FEC_TYPE_RAPTORQ,
)

CANONICAL_FEC_TYPE = {
    # Preserve legacy experiment semantics. Standard RFC 6330 RaptorQ must
    # always be requested explicitly as ``raptorq``.
    FEC_TYPE_RAPTOR: FEC_TYPE_RAPTOR_SIM,
}

# Mask convention used across ARCE:
#   loss_mask[i] == True  means encoded packet i is lost.
#   loss_mask[i] == False means encoded packet i is received.
#
#   receive_mask[i] == True  means encoded packet i is received.
#   receive_mask[i] == False means encoded packet i is lost.
LOSS_MASK_TRUE_MEANS_LOST = True


def normalize_fec_type(fec_type: Optional[str] = None) -> str:
    """
    Normalize FEC type.

    Parameters
    ----------
    fec_type : str or None
        FEC type. Supported:
            none
            xor
            raptor_sim
            raptor / raptorq

    Returns
    -------
    str
        Canonical FEC type.
        "raptor" is retained as a compatibility alias for "raptor_sim".

    Raises
    ------
    ValueError
        If FEC type is unsupported.
    """
    if fec_type is None:
        return DEFAULT_FEC_TYPE

    fec_type = str(fec_type).strip().lower()

    if fec_type not in VALID_FEC_TYPES:
        raise ValueError(
            f"Unsupported FEC type: {fec_type}. "
            f"Supported FEC types: {VALID_FEC_TYPES}."
        )

    return CANONICAL_FEC_TYPE.get(fec_type, fec_type)


def is_valid_fec_type(fec_type: Optional[str]) -> bool:
    """
    Check whether a FEC type is valid.

    Parameters
    ----------
    fec_type : str or None
        FEC type.

    Returns
    -------
    bool
        True if valid, otherwise False.
    """
    try:
        normalize_fec_type(fec_type)
        return True
    except ValueError:
        return False


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


def normalize_redundancy_ratio(value: Any = DEFAULT_REDUNDANCY_RATIO) -> float:
    """
    Normalize redundancy ratio.

    Parameters
    ----------
    value : float
        Redundancy ratio rho. For example:
            0.0  -> no redundancy
            0.25 -> 25% additional parity / redundant packets
            0.50 -> 50% additional parity / redundant packets

    Returns
    -------
    float
        Non-negative redundancy ratio.
    """
    return _as_non_negative_float(value, "redundancy_ratio")


def normalize_group_size(value: Any = DEFAULT_XOR_GROUP_SIZE) -> int:
    """
    Normalize XOR group size.

    Parameters
    ----------
    value : int
        Number of source packets per XOR group.

    Returns
    -------
    int
        Positive group size.
    """
    return _as_positive_int(value, "group_size")


def normalize_decode_overhead(value: Any = DEFAULT_RAPTOR_DECODE_OVERHEAD) -> float:
    """
    Normalize Raptor-sim decode overhead.

    For Raptor-sim, full recovery can require:

        num_received >= ceil(K * (1 + decode_overhead))

    If decode_overhead = 0.02, the decoder requires about 2% overhead.

    Parameters
    ----------
    value : float
        Non-negative decode overhead.

    Returns
    -------
    float
        Non-negative decode overhead.
    """
    return _as_non_negative_float(value, "decode_overhead")


def estimate_parity_packets(
    num_source_packets: Any,
    fec_type: Optional[str] = DEFAULT_FEC_TYPE,
    redundancy_ratio: Any = DEFAULT_REDUNDANCY_RATIO,
    group_size: Optional[int] = None,
) -> int:
    """
    Estimate number of parity / redundant packets.

    Rules
    -----
    none:
        parity = 0

    xor:
        if group_size is provided:
            parity = ceil(K / group_size)
        else:
            parity = ceil(K * redundancy_ratio)

    raptor_sim:
        parity = ceil(K * redundancy_ratio)

    Parameters
    ----------
    num_source_packets : int
        Number of source packets K.

    fec_type : str
        none / xor / raptor_sim.

    redundancy_ratio : float
        Redundancy ratio rho.

    group_size : int, optional
        XOR group size.

    Returns
    -------
    int
        Number of parity / redundant packets.
    """
    k = _as_non_negative_int(num_source_packets, "num_source_packets")
    fec_type = normalize_fec_type(fec_type)
    redundancy_ratio = normalize_redundancy_ratio(redundancy_ratio)

    if k == 0:
        return 0

    if fec_type == FEC_TYPE_NONE:
        return 0

    if fec_type == FEC_TYPE_XOR:
        if group_size is None:
            return int(math.ceil(k * redundancy_ratio))

        group_size = normalize_group_size(group_size)
        return int(math.ceil(k / group_size))

    if fec_type in (FEC_TYPE_RAPTOR_SIM, FEC_TYPE_RAPTORQ):
        return int(math.ceil(k * redundancy_ratio))

    raise ValueError(f"Unsupported canonical FEC type: {fec_type}")


def estimate_encoded_packets(
    num_source_packets: Any,
    fec_type: Optional[str] = DEFAULT_FEC_TYPE,
    redundancy_ratio: Any = DEFAULT_REDUNDANCY_RATIO,
    group_size: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Estimate encoded packet count.

    Parameters
    ----------
    num_source_packets : int
        Number of source packets K.

    fec_type : str
        none / xor / raptor_sim.

    redundancy_ratio : float
        Redundancy ratio rho.

    group_size : int, optional
        XOR group size.

    Returns
    -------
    tuple
        (num_encoded_packets, num_parity_packets)
    """
    k = _as_non_negative_int(num_source_packets, "num_source_packets")

    parity = estimate_parity_packets(
        num_source_packets=k,
        fec_type=fec_type,
        redundancy_ratio=redundancy_ratio,
        group_size=group_size,
    )

    return int(k + parity), int(parity)


def effective_redundancy_ratio(
    num_source_packets: Any,
    num_parity_packets: Any,
) -> float:
    """
    Compute effective redundancy ratio.

    Formula:
        rho_eff = num_parity_packets / num_source_packets

    Parameters
    ----------
    num_source_packets : int
        Number of source packets K.

    num_parity_packets : int
        Number of parity packets.

    Returns
    -------
    float
        Effective redundancy ratio.
    """
    k = _as_non_negative_int(num_source_packets, "num_source_packets")
    parity = _as_non_negative_int(num_parity_packets, "num_parity_packets")

    if k == 0:
        return 0.0

    return float(parity / k)


def estimate_raptor_required_packets(
    num_source_packets: Any,
    decode_overhead: Any = DEFAULT_RAPTOR_DECODE_OVERHEAD,
) -> int:
    """
    Estimate number of received encoded packets required for Raptor-sim full recovery.

    Formula:
        required = ceil(K * (1 + decode_overhead))

    Parameters
    ----------
    num_source_packets : int
        Number of source packets K.

    decode_overhead : float
        Non-negative decode overhead.

    Returns
    -------
    int
        Required received encoded packet count.
    """
    k = _as_non_negative_int(num_source_packets, "num_source_packets")
    overhead = normalize_decode_overhead(decode_overhead)

    return int(math.ceil(k * (1.0 + overhead)))


def normalize_fec_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Normalize a FEC config dictionary.

    Supported input:
        normalize_fec_config(arce_cfg)
        normalize_fec_config(arce_cfg["fec"])

    YAML style:

        arce:
          fec:
            enabled: true
            type: xor
            group_size: 4
            redundancy_ratio: 0.25
            decode_overhead: 0.0

    Returns
    -------
    dict
        Normalized FEC config:
            {
                "enabled": bool,
                "type": str,
                "redundancy_ratio": float,
                "group_size": int or None,
                "decode_overhead": float
            }
    """
    cfg = cfg or {}

    if "fec" in cfg and isinstance(cfg["fec"], dict):
        cfg = cfg["fec"]

    fec_type = normalize_fec_type(cfg.get("type", DEFAULT_FEC_TYPE))

    if "enabled" in cfg:
        enabled = bool(cfg.get("enabled"))
    else:
        enabled = fec_type != FEC_TYPE_NONE

    if not enabled:
        fec_type = FEC_TYPE_NONE

    redundancy_ratio = normalize_redundancy_ratio(
        cfg.get("redundancy_ratio", DEFAULT_REDUNDANCY_RATIO)
    )

    group_size = cfg.get("group_size", None)
    if group_size is not None:
        group_size = normalize_group_size(group_size)

    decode_overhead = normalize_decode_overhead(
        cfg.get("decode_overhead", DEFAULT_RAPTOR_DECODE_OVERHEAD)
    )

    return {
        "enabled": bool(enabled),
        "type": fec_type,
        "redundancy_ratio": float(redundancy_ratio),
        "group_size": group_size,
        "decode_overhead": float(decode_overhead),
    }


def get_fec_config_summary(
    fec_type: Optional[str] = DEFAULT_FEC_TYPE,
    redundancy_ratio: Any = DEFAULT_REDUNDANCY_RATIO,
    group_size: Optional[int] = None,
    decode_overhead: Any = DEFAULT_RAPTOR_DECODE_OVERHEAD,
    num_source_packets: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Return a JSON-serializable FEC configuration summary.

    If num_source_packets is provided, this also estimates:
        num_parity_packets
        num_encoded_packets
        effective_redundancy_ratio
    """
    fec_type = normalize_fec_type(fec_type)
    redundancy_ratio = normalize_redundancy_ratio(redundancy_ratio)

    if group_size is not None:
        group_size = normalize_group_size(group_size)

    decode_overhead = normalize_decode_overhead(decode_overhead)

    summary = {
        "fec_type": fec_type,
        "redundancy_ratio": float(redundancy_ratio),
        "group_size": group_size,
        "decode_overhead": float(decode_overhead),
    }

    if num_source_packets is not None:
        k = _as_non_negative_int(num_source_packets, "num_source_packets")
        num_encoded, num_parity = estimate_encoded_packets(
            num_source_packets=k,
            fec_type=fec_type,
            redundancy_ratio=redundancy_ratio,
            group_size=group_size,
        )

        summary.update(
            {
                "num_source_packets": int(k),
                "num_parity_packets": int(num_parity),
                "num_encoded_packets": int(num_encoded),
                "effective_redundancy_ratio": float(
                    effective_redundancy_ratio(k, num_parity)
                ),
            }
        )

        if fec_type == FEC_TYPE_RAPTOR_SIM:
            summary["raptor_required_packets"] = int(
                estimate_raptor_required_packets(
                    num_source_packets=k,
                    decode_overhead=decode_overhead,
                )
            )

    return summary


def loss_mask_to_receive_mask(loss_mask):
    """
    Convert packet loss mask to receive mask.

    Convention:
        loss_mask[i] == True means packet i is lost.
        receive_mask[i] == True means packet i is received.
    """
    return ~loss_mask


def receive_mask_to_loss_mask(receive_mask):
    """
    Convert packet receive mask to loss mask.

    Convention:
        receive_mask[i] == True means packet i is received.
        loss_mask[i] == True means packet i is lost.
    """
    return ~receive_mask


__all__ = [
    "FEC_TYPE_NONE",
    "FEC_TYPE_XOR",
    "FEC_TYPE_RAPTOR_SIM",
    "FEC_TYPE_RAPTOR",
    "FEC_TYPE_RAPTORQ",
    "DEFAULT_FEC_TYPE",
    "DEFAULT_REDUNDANCY_RATIO",
    "DEFAULT_XOR_GROUP_SIZE",
    "DEFAULT_RAPTOR_DECODE_OVERHEAD",
    "VALID_FEC_TYPES",
    "CANONICAL_FEC_TYPE",
    "LOSS_MASK_TRUE_MEANS_LOST",
    "normalize_fec_type",
    "is_valid_fec_type",
    "normalize_redundancy_ratio",
    "normalize_group_size",
    "normalize_decode_overhead",
    "estimate_parity_packets",
    "estimate_encoded_packets",
    "effective_redundancy_ratio",
    "estimate_raptor_required_packets",
    "normalize_fec_config",
    "get_fec_config_summary",
    "loss_mask_to_receive_mask",
    "receive_mask_to_loss_mask",
]
