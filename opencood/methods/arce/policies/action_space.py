"""
Final-setting 48-dimensional ARCE action space.

This module implements the action definition in the final GRACE / C2MAB design:

    a = (a1, a2, a3, a4)

    a1: collaboration trigger / send flag {0, 1}
    a2: compression / quantization {fp32, fp16, int8, int4}
    a3: redundancy ratio {0.0, 0.25, 0.50}
    a4: temporal fusion/cache flag {0, 1}

Therefore, the full action space size is:

    2 x 4 x 3 x 2 = 48

The FEC type is an engineering realization of rho. It is not an additional
The configured ``fec_main`` selects the codec for rho > 0. The formal GRACE
configuration uses RFC 6330 RaptorQ; ``raptor_sim`` remains available only for
legacy reproduction.
XOR should be used through fec_mode="xor" for FEC ablations.

Important:
    This file only defines PDF/C2MAB-level actions and cost estimates.
    The actual communication execution is handled by ARCEFixedComm through
    action_override.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from opencood.methods.arce.fixed_policy import ARCEAction
    from opencood.communication.transport.recovery import (
        RECOVERY_METHOD_TEMPORAL_CACHE,
        RECOVERY_METHOD_SPATIAL_INTERPOLATION,
        RECOVERY_METHOD_ZERO_FILL,
    )
except Exception:  # pragma: no cover
    ARCEAction = None
    RECOVERY_METHOD_TEMPORAL_CACHE = "temporal_cache"
    RECOVERY_METHOD_SPATIAL_INTERPOLATION = "spatial_interpolation"
    RECOVERY_METHOD_ZERO_FILL = "zero_fill"


from opencood.methods.arce.policies.action_adapter import normalize_runtime_action


# ----------------------------------------------------------------------
# Final action dimensions
# ----------------------------------------------------------------------

SUPPORTED_QUANT_MODES: Tuple[str, ...] = ("fp32", "fp16", "int8", "int4")
ONLINE_QUANT_MODES: Tuple[str, ...] = ("fp16", "int8", "int4")
QUANT_MODES: Tuple[str, ...] = SUPPORTED_QUANT_MODES
ONLINE_RHO_VALUES: Tuple[float, ...] = (0.0, 0.10, 0.25, 0.60)
RHO_VALUES: Tuple[float, ...] = ONLINE_RHO_VALUES
CACHE_VALUES: Tuple[int, ...] = (0, 1)
SEND_VALUES: Tuple[int, ...] = (0, 1)
NO_SEND_ACTION_ID = "send0_none_rho0_cache0_none"


QUANT_ALIASES: Dict[str, str] = {
    "none": "none",
    "no_send": "none",
    "nosend": "none",
    "fp32": "fp32",
    "float32": "fp32",
    "torch.float32": "fp32",
    "float": "fp32",

    "fp16": "fp16",
    "float16": "fp16",
    "torch.float16": "fp16",
    "half": "fp16",

    "int8": "int8",
    "uint8": "int8",

    "int4": "int4",
}


QUANT_BITS: Dict[str, int] = {
    "none": 0,
    "fp32": 32,
    "fp16": 16,
    "int8": 8,
    "int4": 4,
}


def normalize_pdf_quant_mode(mode: Any) -> str:
    """Normalize quantization mode names used in YAML / runtime records."""
    q = str(mode).strip().lower()
    q = QUANT_ALIASES.get(q, q)

    if q not in QUANT_BITS:
        raise ValueError(
            f"Unsupported quant_mode={q!r}; final action space uses "
            "fp32/fp16/int8/int4."
        )

    return q


# ----------------------------------------------------------------------
# Action dataclass
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PDFARCEAction:
    """One PDF-level ARCE action."""

    action_id: str
    send: int
    quant_mode: str
    redundancy_ratio: float
    cache_enabled: int
    fec_type: str = "none"
    xor_group_size: int = 4
    decode_overhead: float = 0.0
    channel_state: str = "medium"

    @property
    def is_no_send(self) -> bool:
        return int(self.send) == 0

    @property
    def quant_bits(self) -> int:
        q = normalize_pdf_quant_mode(self.quant_mode)
        return int(QUANT_BITS[q])

    @property
    def compression_ratio(self) -> float:
        """
        Ratio relative to FP32 byte size.

        fp32 -> 1.000
        fp16 -> 0.500
        int8 -> 0.250
        int4 -> 0.125
        """
        if self.is_no_send or normalize_pdf_quant_mode(self.quant_mode) == "none":
            return 0.0
        return float(self.quant_bits) / 32.0

    def as_dict(self) -> Dict[str, Any]:
        q = normalize_pdf_quant_mode(self.quant_mode)
        out = asdict(self)
        out["quant_mode"] = q
        out["is_no_send"] = self.is_no_send
        out["quant_bits"] = self.quant_bits
        out["compression_ratio"] = self.compression_ratio
        return out

    def recovery_priority(self) -> Tuple[str, ...]:
        """
        Recovery order used by the ARCE executor.

        cache_enabled=1:
            temporal cache -> spatial interpolation -> zero-fill

        cache_enabled=0:
            spatial interpolation -> zero-fill
        """
        if int(self.cache_enabled):
            return (
                RECOVERY_METHOD_TEMPORAL_CACHE,
                RECOVERY_METHOD_SPATIAL_INTERPOLATION,
                RECOVERY_METHOD_ZERO_FILL,
            )

        return (
            RECOVERY_METHOD_SPATIAL_INTERPOLATION,
            RECOVERY_METHOD_ZERO_FILL,
        )

    def to_arce_action(self):
        """Convert PDF action to the existing ARCEAction executor format."""
        if ARCEAction is None:
            raise ImportError("ARCEAction is unavailable; check OpenCOOD import path.")

        q = normalize_pdf_quant_mode(self.quant_mode)

        action = ARCEAction(
            name=self.action_id,
            channel_state=self.channel_state,
            quant_mode=q,
            fec_type=self.fec_type,
            redundancy_ratio=float(self.redundancy_ratio),
            xor_group_size=int(self.xor_group_size),
            decode_overhead=float(self.decode_overhead),
            recovery="arce" if int(self.send) == 1 else "zero_fill",
            recovery_priority=self.recovery_priority(),
            extra={
                "pdf_action_id": self.action_id,
                "action_id": self.action_id,
                "send": int(self.send),
                "cache_enabled": int(self.cache_enabled),
                "quant_mode": q,
                "fec_type": self.fec_type,
                "redundancy_ratio": float(self.redundancy_ratio),
                "xor_group_size": int(self.xor_group_size),
                "decode_overhead": float(self.decode_overhead),
                "channel_state": self.channel_state,
            },
        )

        return normalize_runtime_action(
            action,
            send=int(self.send),
            cache_enabled=int(self.cache_enabled),
            action_id=str(self.action_id),
        )


# ----------------------------------------------------------------------
# Action construction utilities
# ----------------------------------------------------------------------

def _canonical_float_text(x: float) -> str:
    x = float(x)
    if abs(x) < 1e-12:
        return "0"
    if abs(x - 0.10) < 1e-12:
        return "0p10"
    if abs(x - 0.25) < 1e-12:
        return "0p25"
    if abs(x - 0.60) < 1e-12:
        return "0p60"
    return str(x).replace(".", "p")


def infer_fec_type(rho: float, fec_mode: str = "raptor_sim") -> str:
    """
    Infer executor-level FEC type from redundancy ratio.

    rho = 0:
        no redundancy, fec_type = none

    rho > 0:
        fec_type follows fec_mode.
    """
    rho = float(rho)

    if rho <= 0.0:
        return "none"

    fec_mode = str(fec_mode).strip().lower()

    if fec_mode == "raptorq":
        return "raptorq"

    if fec_mode in ("raptor", "raptor_sim", "fountain"):
        return "raptor_sim"

    if fec_mode == "xor":
        return "xor"

    raise ValueError(
        f"Unsupported fec_mode={fec_mode!r}; expected raptorq, "
        "raptor_sim, or xor."
    )


def is_valid_pdf_action_combination(send: int, quant_mode: str, rho: float) -> bool:
    """Validate system-supported PDF action fields.

    This broad validator intentionally keeps FP32 support for fixed baselines
    and ARCEFixedComm. Online ARCE-C2MAB uses
    is_valid_online_pdf_action_combination().
    """
    _ = int(send)
    _ = float(rho)
    normalize_pdf_quant_mode(quant_mode)
    return True


def is_valid_online_pdf_action_combination(
    send: int,
    quant_mode: str,
    rho: float,
    allow_fp32_send: bool = False,
) -> bool:
    """Validate online ARCE-C2MAB actions.

    FP32 is still supported by the system, but excluded from online send arms
    unless allow_fp32_send=True.
    """
    send_i = int(send)
    rho_f = float(rho)
    q = normalize_pdf_quant_mode(quant_mode)

    if send_i == 0:
        return q == "none" and abs(rho_f) < 1e-12

    if send_i != 1:
        return False

    allowed_quant_modes = set(ONLINE_QUANT_MODES)
    if bool(allow_fp32_send):
        allowed_quant_modes.add("fp32")

    if q not in allowed_quant_modes:
        return False

    return any(abs(rho_f - float(x)) < 1e-12 for x in ONLINE_RHO_VALUES)



def build_pdf_action_space(
    fec_mode: str = "raptor_sim",
    send_values: Sequence[int] = SEND_VALUES,
    quant_modes: Sequence[str] = ONLINE_QUANT_MODES,
    redundancy_ratios: Sequence[float] = ONLINE_RHO_VALUES,
    cache_values: Sequence[int] = CACHE_VALUES,
    channel_state: str = "medium",
    xor_group_size: int = 4,
    decode_overhead: float = 0.0,
    allow_fp32_send: bool = False,
) -> List[PDFARCEAction]:
    """Build the online ARCE-C2MAB action space.

    Default size is 25:
        1 no-send + 3 quant modes x 4 rho values x 2 cache flags.

    FP32 remains available for fixed baselines and ARCEFixedComm, but is
    excluded from online C2MAB send arms unless allow_fp32_send=True.
    """
    actions: List[PDFARCEAction] = []

    if 0 in [int(x) for x in send_values]:
        actions.append(PDFARCEAction(
            action_id=NO_SEND_ACTION_ID,
            send=0,
            quant_mode="none",
            redundancy_ratio=0.0,
            cache_enabled=0,
            fec_type="none",
            xor_group_size=int(xor_group_size),
            decode_overhead=float(decode_overhead),
            channel_state=str(channel_state),
        ))

    if 1 not in [int(x) for x in send_values]:
        return actions

    normalized_quant_modes: List[str] = []
    for q in quant_modes:
        qn = normalize_pdf_quant_mode(q)
        if qn == "none":
            continue
        if qn == "fp32" and not bool(allow_fp32_send):
            continue
        if qn not in normalized_quant_modes:
            normalized_quant_modes.append(qn)

    for q in normalized_quant_modes:
        for rho in redundancy_ratios:
            rho_f = float(rho)
            for cache in cache_values:
                cache_i = int(cache)

                if not is_valid_online_pdf_action_combination(
                    1,
                    q,
                    rho_f,
                    allow_fp32_send=allow_fp32_send,
                ):
                    continue

                fec_type = "none" if abs(rho_f) < 1e-12 else infer_fec_type(rho_f, fec_mode)
                action_id = f"send1_{q}_rho{_canonical_float_text(rho_f)}_cache{cache_i}_{fec_type}"

                actions.append(PDFARCEAction(
                    action_id=action_id,
                    send=1,
                    quant_mode=q,
                    redundancy_ratio=rho_f,
                    cache_enabled=cache_i,
                    fec_type=fec_type,
                    xor_group_size=int(xor_group_size),
                    decode_overhead=float(decode_overhead),
                    channel_state=str(channel_state),
                ))

    return actions


# ----------------------------------------------------------------------
# Cost utilities
# ----------------------------------------------------------------------

def raw_feature_bytes_fp32(feature_shape: Sequence[int]) -> float:
    """
    Estimate raw FP32 feature size in bytes.

    feature_shape:
        Usually [C, H, W].
    """
    n = 1
    for v in feature_shape:
        n *= int(v)
    return float(n * 4)


def estimate_action_cost_bytes(raw_fp32_bytes: float, action: PDFARCEAction) -> float:
    """
    Estimate action cost in bytes.

    Cost model:
        cost = FP32_bytes × quant_ratio × (1 + rho)

    where:
        fp32: 1.000
        fp16: 0.500
        int8: 0.250
        int4: 0.125
    """
    if action.is_no_send:
        return 0.0

    compressed = float(raw_fp32_bytes) * action.compression_ratio
    return float(compressed * (1.0 + float(action.redundancy_ratio)))


def budget_bytes_from_bandwidth(
    bandwidth_mbps: float,
    tau_trans_ms: float = 100.0,
) -> float:
    """
    Convert bandwidth budget to byte budget.

    bandwidth_mbps:
        Mbps.

    tau_trans_ms:
        Transmission window in milliseconds.
    """
    return float(
        float(bandwidth_mbps) * 1e6 / 8.0 * (float(tau_trans_ms) / 1000.0)
    )


def feasible_action_costs(
    actions: Iterable[PDFARCEAction],
    raw_fp32_bytes: float,
    budget_bytes: float,
    include_no_send: bool = True,
) -> List[Tuple[PDFARCEAction, float]]:
    """
    Return actions whose estimated cost fits the given budget.
    """
    feasible: List[Tuple[PDFARCEAction, float]] = []

    for action in actions:
        if action.is_no_send and not include_no_send:
            continue

        cost = estimate_action_cost_bytes(raw_fp32_bytes, action)

        if action.is_no_send or cost <= float(budget_bytes):
            feasible.append((action, float(cost)))

    return feasible


def action_by_id(actions: Sequence[PDFARCEAction]) -> Dict[str, PDFARCEAction]:
    return {a.action_id: a for a in actions}


__all__ = [
    "PDFARCEAction",
    "SUPPORTED_QUANT_MODES",
    "ONLINE_QUANT_MODES",
    "ONLINE_RHO_VALUES",
    "NO_SEND_ACTION_ID",
    "is_valid_online_pdf_action_combination",
    "QUANT_MODES",
    "RHO_VALUES",
    "CACHE_VALUES",
    "SEND_VALUES",
    "QUANT_BITS",
    "normalize_pdf_quant_mode",
    "build_pdf_action_space",
    "estimate_action_cost_bytes",
    "raw_feature_bytes_fp32",
    "budget_bytes_from_bandwidth",
    "feasible_action_costs",
    "action_by_id",
    "is_valid_pdf_action_combination",
]
