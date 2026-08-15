"""
Real fountain / Raptor-like FEC for ARCE communication simulation.

This module implements a real packet-level fountain-code style FEC:

    source_packets [K, ...]
    -> generate repair packets by XORing random subsets of source packets
    -> encoded_packets = systematic source packets + repair packets
    -> GE packet loss is applied to encoded packets
    -> peeling decoder recovers missing source packets from received repair packets

Important distinction:
    This is NOT merely a threshold simulator.

    It really creates repair packets and really decodes them.

    However, this is still called "raptor_sim" in the project because it is
    not a full standardized Raptor/RaptorQ implementation. Standard RaptorQ
    includes a specific precode, symbol schedule, degree distribution, and
    decoding system. This implementation is a practical Raptor-like LT /
    fountain code suitable for packet-level ARCE experiments.

Usage:
    quant_result = quantizer.quantize_packets(packets, mode="int8")

    fec = RaptorSimFEC(arce_cfg)
    encode_result = fec.encode(quant_result.q_tensor)

    loss_mask, channel_info = channel_manager.sample_packet_loss(
        num_packets=encode_result.num_encoded_packets,
        ...
    )

    decode_result = fec.decode(
        encode_result,
        loss_mask=loss_mask,
    )

    recovered_float_packets = quantizer.dequantize(
        decode_result.recovered_packets,
        quant_result.meta,
    )

Supported tensor dtypes:
    torch.int8
    torch.uint8
    torch.int16
    torch.int32
    torch.int64

Do not run this FEC on float feature tensors.
Quantize packets first.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from opencood.communication.transport.fec import (
    FEC_TYPE_RAPTOR_SIM,
    DEFAULT_REDUNDANCY_RATIO,
    DEFAULT_RAPTOR_DECODE_OVERHEAD,
    normalize_redundancy_ratio,
    normalize_decode_overhead,
    effective_redundancy_ratio,
    estimate_raptor_required_packets,
)

from opencood.communication.transport.fec.fec_base import (
    EncodedPacketMeta,
    FECBase,
    FECEncodeResult,
    FECDecodeResult,
    resolve_receive_mask,
    validate_packet_tensor,
)


SUPPORTED_RAPTOR_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _extract_fec_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Accept either full ARCE config or direct fec config.
    """
    cfg = cfg or {}

    if "fec" in cfg and isinstance(cfg["fec"], dict):
        return cfg["fec"]

    return cfg


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


def _validate_raptor_packet_tensor(
    packets: torch.Tensor,
    name: str = "packets",
) -> torch.Tensor:
    """
    Validate packet tensor for fountain / Raptor-like FEC.

    This implementation performs XOR over packet symbols, so packet tensors
    must be integer tensors.
    """
    packets = validate_packet_tensor(
        packets,
        name=name,
        min_dim=1,
    )

    if packets.dtype not in SUPPORTED_RAPTOR_DTYPES:
        raise TypeError(
            f"{name} should be an integer tensor for Raptor-like FEC, "
            f"got dtype={packets.dtype}. "
            "Please quantize packets first, for example with "
            "FeatureQuantizer.quantize_packets(...).q_tensor."
        )

    return packets


def _xor_reduce(packet_group: torch.Tensor) -> torch.Tensor:
    """
    XOR-reduce a group of packets.

    Parameters
    ----------
    packet_group : torch.Tensor
        Shape [D, ...].

    Returns
    -------
    torch.Tensor
        One repair packet with shape [...].
    """
    packet_group = _validate_raptor_packet_tensor(
        packet_group,
        name="packet_group",
    )

    if packet_group.shape[0] <= 0:
        raise ValueError("packet_group should contain at least one packet.")

    out = packet_group[0].clone()

    for idx in range(1, int(packet_group.shape[0])):
        out = torch.bitwise_xor(out, packet_group[idx])

    return out


def _xor_packet_with_known(
    repair_packet: torch.Tensor,
    known_packets: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Remove known packets from a repair equation.

    If:
        repair = p0 XOR p1 XOR p2

    and p0, p2 are known, then:
        repair XOR p0 XOR p2 = p1
    """
    repair_packet = _validate_raptor_packet_tensor(
        repair_packet,
        name="repair_packet",
    )

    out = repair_packet.clone()

    for packet in known_packets:
        packet = _validate_raptor_packet_tensor(
            packet,
            name="known_packet",
        )
        out = torch.bitwise_xor(out, packet)

    return out


def _ideal_soliton_distribution(k: int) -> np.ndarray:
    """
    Build ideal soliton degree distribution for degrees 1..K.

    rho(1) = 1 / K
    rho(d) = 1 / (d * (d - 1)), d = 2..K
    """
    k = _as_positive_int(k, "k")

    rho = np.zeros(k, dtype=np.float64)

    if k == 1:
        rho[0] = 1.0
        return rho

    rho[0] = 1.0 / float(k)

    for d in range(2, k + 1):
        rho[d - 1] = 1.0 / float(d * (d - 1))

    rho = rho / rho.sum()
    return rho


def _robust_soliton_distribution(
    k: int,
    c: float = 0.1,
    delta: float = 0.5,
) -> np.ndarray:
    """
    Build robust soliton degree distribution for degrees 1..K.

    This is the standard LT-code style robust soliton distribution.

    Parameters
    ----------
    k : int
        Number of source packets.

    c : float
        Robust soliton constant. Typical values are around 0.03 to 0.2.

    delta : float
        Failure probability parameter. Smaller delta usually adds more
        low-degree symbols.

    Returns
    -------
    np.ndarray
        Probability vector of length K.
        index 0 corresponds to degree 1.
    """
    k = _as_positive_int(k, "k")

    if k == 1:
        return np.array([1.0], dtype=np.float64)

    c = float(c)
    delta = float(delta)

    if c <= 0.0:
        raise ValueError(f"robust_soliton_c should be positive, got {c}.")

    if not (0.0 < delta < 1.0):
        raise ValueError(
            f"robust_soliton_delta should be in (0, 1), got {delta}."
        )

    rho = _ideal_soliton_distribution(k)

    r = c * math.log(k / delta) * math.sqrt(k)
    if r <= 0.0:
        return rho

    s = int(math.floor(k / r))
    s = max(1, min(s, k))

    tau = np.zeros(k, dtype=np.float64)

    for d in range(1, s):
        tau[d - 1] = r / float(d * k)

    tau[s - 1] = r * math.log(r / delta) / float(k)

    beta = float((rho + tau).sum())

    if beta <= 0.0 or not np.isfinite(beta):
        return rho

    mu = (rho + tau) / beta
    mu = np.maximum(mu, 0.0)
    mu = mu / mu.sum()

    return mu


def _uniform_degree_distribution(k: int, max_degree: Optional[int] = None) -> np.ndarray:
    """
    Uniform distribution over degrees 1..max_degree.
    """
    k = _as_positive_int(k, "k")

    if max_degree is None:
        max_degree = k

    max_degree = max(1, min(int(max_degree), k))

    probs = np.zeros(k, dtype=np.float64)
    probs[:max_degree] = 1.0 / float(max_degree)

    return probs


def _fixed_degree_distribution(k: int, degree: int) -> np.ndarray:
    """
    Fixed degree distribution.
    """
    k = _as_positive_int(k, "k")
    degree = max(1, min(int(degree), k))

    probs = np.zeros(k, dtype=np.float64)
    probs[degree - 1] = 1.0

    return probs


def _build_degree_distribution(
    k: int,
    mode: str,
    robust_soliton_c: float,
    robust_soliton_delta: float,
    fixed_degree: int,
    max_degree: Optional[int],
) -> np.ndarray:
    """
    Build degree distribution according to config.
    """
    mode = str(mode).strip().lower()

    if mode in ("robust_soliton", "robust", "rs"):
        return _robust_soliton_distribution(
            k=k,
            c=robust_soliton_c,
            delta=robust_soliton_delta,
        )

    if mode in ("ideal_soliton", "ideal", "is"):
        return _ideal_soliton_distribution(k)

    if mode in ("uniform",):
        return _uniform_degree_distribution(k, max_degree=max_degree)

    if mode in ("fixed", "constant"):
        return _fixed_degree_distribution(k, degree=fixed_degree)

    raise ValueError(
        f"Unsupported degree_distribution: {mode}. "
        "Expected robust_soliton / ideal_soliton / uniform / fixed."
    )


class RaptorSimFEC(FECBase):
    """
    Real fountain / Raptor-like FEC with systematic source packets.

    Encoded packet order:
        0 ... K-1:
            original source packets

        K ... K+P-1:
            repair packets generated by XORing random source subsets

    Decode method:
        1. Copy directly received source packets.
        2. For each received repair packet, create an equation:
              repair = XOR(source_ids)
        3. Remove known source packets from equations.
        4. If an equation has exactly one unknown source, recover it.
        5. Repeat until no more packets can be recovered.

    This is a real peeling decoder, not a threshold-only simulator.
    """

    fec_type = FEC_TYPE_RAPTOR_SIM

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        cfg : dict, optional
            Can be full ARCE config or direct fec config.

            YAML style:

                arce:
                  fec:
                    enabled: true
                    type: raptor_sim
                    redundancy_ratio: 0.25
                    decode_overhead: 0.0

                    # Real fountain-code parameters
                    degree_distribution: robust_soliton
                    robust_soliton_c: 0.1
                    robust_soliton_delta: 0.5
                    seed: 2026

                    # Optional override
                    num_repair_packets: null

                    # Decoder
                    max_decode_iters: 10000
        """
        super().__init__(cfg or {})

        self.raw_cfg = _extract_fec_cfg(cfg)

        self.enabled = True
        self.fec_type = FEC_TYPE_RAPTOR_SIM

        self.redundancy_ratio = normalize_redundancy_ratio(
            self.raw_cfg.get(
                "redundancy_ratio",
                self.redundancy_ratio
                if self.redundancy_ratio > 0.0
                else DEFAULT_REDUNDANCY_RATIO,
            )
        )

        self.decode_overhead = normalize_decode_overhead(
            self.raw_cfg.get(
                "decode_overhead",
                self.decode_overhead
                if self.decode_overhead is not None
                else DEFAULT_RAPTOR_DECODE_OVERHEAD,
            )
        )

        self.degree_distribution = str(
            self.raw_cfg.get("degree_distribution", "robust_soliton")
        ).strip().lower()

        self.robust_soliton_c = float(
            self.raw_cfg.get("robust_soliton_c", 0.1)
        )
        self.robust_soliton_delta = float(
            self.raw_cfg.get("robust_soliton_delta", 0.5)
        )

        self.fixed_degree = int(
            self.raw_cfg.get("fixed_degree", 2)
        )

        max_degree = self.raw_cfg.get("max_degree", None)
        self.max_degree = None if max_degree is None else int(max_degree)

        self.num_repair_packets_override = self.raw_cfg.get(
            "num_repair_packets",
            self.raw_cfg.get("repair_packets", None),
        )
        if self.num_repair_packets_override is not None:
            self.num_repair_packets_override = _as_non_negative_int(
                self.num_repair_packets_override,
                "num_repair_packets",
            )

        self.seed = int(self.raw_cfg.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        self.max_decode_iters = int(
            self.raw_cfg.get("max_decode_iters", 10000)
        )

        if self.max_decode_iters <= 0:
            raise ValueError(
                f"max_decode_iters should be positive, got {self.max_decode_iters}."
            )

    def _estimate_num_repair_packets(self, k: int) -> int:
        """
        Estimate number of repair packets P.
        """
        k = _as_non_negative_int(k, "num_source_packets")

        if k == 0:
            return 0

        if self.num_repair_packets_override is not None:
            return int(self.num_repair_packets_override)

        return int(math.ceil(k * float(self.redundancy_ratio)))

    def _sample_degree_distribution(self, k: int) -> np.ndarray:
        """
        Return degree probability vector for current K.
        """
        return _build_degree_distribution(
            k=k,
            mode=self.degree_distribution,
            robust_soliton_c=self.robust_soliton_c,
            robust_soliton_delta=self.robust_soliton_delta,
            fixed_degree=self.fixed_degree,
            max_degree=self.max_degree,
        )

    def _sample_source_ids(
        self,
        k: int,
        degree_probs: np.ndarray,
    ) -> Tuple[int, Tuple[int, ...]]:
        """
        Sample a repair equation degree and source ids.

        Returns
        -------
        degree : int
            Number of source packets included in this repair packet.

        source_ids : tuple
            Sorted source ids.
        """
        k = _as_positive_int(k, "k")

        if k == 1:
            return 1, (0,)

        degrees = np.arange(1, k + 1, dtype=np.int64)

        degree = int(self.rng.choice(degrees, p=degree_probs))
        degree = max(1, min(degree, k))

        source_ids = self.rng.choice(
            np.arange(k, dtype=np.int64),
            size=degree,
            replace=False,
        )

        source_ids = tuple(int(x) for x in sorted(source_ids.tolist()))

        return degree, source_ids

    def _build_repair_packets_and_metas(
        self,
        source_packets: torch.Tensor,
        start_encoded_id: int,
    ) -> Tuple[torch.Tensor, List[EncodedPacketMeta], List[int]]:
        """
        Build repair packets and metadata.

        Parameters
        ----------
        source_packets : torch.Tensor
            Shape [K, ...].

        start_encoded_id : int
            First encoded id for repair packets.

        Returns
        -------
        repair_packets : torch.Tensor
            Shape [P, ...].

        repair_metas : list of EncodedPacketMeta

        degree_list : list of int
            Degree of each repair packet.
        """
        source_packets = _validate_raptor_packet_tensor(
            source_packets,
            name="source_packets",
        )

        k = int(source_packets.shape[0])
        p = self._estimate_num_repair_packets(k)

        if p == 0:
            empty_shape = (0,) + tuple(source_packets.shape[1:])
            repair_tensor = torch.empty(
                empty_shape,
                dtype=source_packets.dtype,
                device=source_packets.device,
            )
            return repair_tensor, [], []

        degree_probs = self._sample_degree_distribution(k)

        repair_packets = []
        repair_metas: List[EncodedPacketMeta] = []
        degree_list: List[int] = []

        for repair_id in range(p):
            degree, source_ids = self._sample_source_ids(
                k=k,
                degree_probs=degree_probs,
            )

            packet_group = source_packets[list(source_ids)]
            repair_packet = _xor_reduce(packet_group)

            encoded_id = int(start_encoded_id + repair_id)

            repair_packets.append(repair_packet)
            repair_metas.append(
                EncodedPacketMeta(
                    encoded_id=encoded_id,
                    kind="repair",
                    source_id=None,
                    group_id=repair_id,
                    source_ids=source_ids,
                    note=(
                        "Raptor-like fountain repair packet generated by XORing "
                        "a random subset of source packets."
                    ),
                )
            )
            degree_list.append(int(degree))

        repair_tensor = torch.stack(repair_packets, dim=0)

        return repair_tensor, repair_metas, degree_list

    def encode(self, source_packets: torch.Tensor, **kwargs) -> FECEncodeResult:
        """
        Encode integer source packets with real fountain repair packets.

        Parameters
        ----------
        source_packets : torch.Tensor
            Integer tensor with shape [K, ...].

        Returns
        -------
        FECEncodeResult
            encoded_packets has shape [K + P, ...].
        """
        source_packets = _validate_raptor_packet_tensor(
            source_packets,
            name="source_packets",
        )

        k = int(source_packets.shape[0])

        source_metas = self._build_source_metas(k)

        repair_packets, repair_metas, degree_list = (
            self._build_repair_packets_and_metas(
                source_packets=source_packets,
                start_encoded_id=k,
            )
        )

        encoded_packets = torch.cat(
            [source_packets.clone(), repair_packets],
            dim=0,
        )

        encoded_metas = source_metas + repair_metas

        p = int(repair_packets.shape[0])
        n = int(encoded_packets.shape[0])

        if degree_list:
            degree_min = int(min(degree_list))
            degree_max = int(max(degree_list))
            degree_mean = float(sum(degree_list) / len(degree_list))
        else:
            degree_min = 0
            degree_max = 0
            degree_mean = 0.0

        required_packets = estimate_raptor_required_packets(
            num_source_packets=k,
            decode_overhead=self.decode_overhead,
        )

        info = {
            "fec_type": FEC_TYPE_RAPTOR_SIM,
            "enabled": True,
            "implementation": "real_fountain_peeling_decoder",
            "is_threshold_only_simulation": False,
            "is_standard_raptorq": False,
            "num_source_packets": int(k),
            "num_repair_packets": int(p),
            "num_parity_packets": int(p),
            "num_encoded_packets": int(n),
            "redundancy_ratio_config": float(self.redundancy_ratio),
            "effective_redundancy_ratio": float(
                effective_redundancy_ratio(k, p)
            ),
            "decode_overhead": float(self.decode_overhead),
            "raptor_required_packets_reference": int(required_packets),
            "degree_distribution": self.degree_distribution,
            "robust_soliton_c": float(self.robust_soliton_c),
            "robust_soliton_delta": float(self.robust_soliton_delta),
            "fixed_degree": int(self.fixed_degree),
            "max_degree": self.max_degree,
            "degree_min": int(degree_min),
            "degree_max": int(degree_max),
            "degree_mean": float(degree_mean),
            "seed": int(self.seed),
            "note": (
                "This encoder really creates XOR repair packets over random "
                "source subsets. Decoding is done by a peeling decoder. "
                "It is Raptor-like / LT-style, not standardized RaptorQ."
            ),
        }

        result = FECEncodeResult(
            source_packets=source_packets,
            encoded_packets=encoded_packets,
            encoded_metas=encoded_metas,
            fec_type=FEC_TYPE_RAPTOR_SIM,
            redundancy_ratio_config=float(self.redundancy_ratio),
            group_size=None,
            decode_overhead=float(self.decode_overhead),
            info=info,
        )

        result.validate()
        return result

    @staticmethod
    def _build_initial_equations(
        encode_result: FECEncodeResult,
        receive_mask: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        Build received repair equations.

        Each equation stores:
            encoded_id
            source_ids
            value
            active
        """
        equations: List[Dict[str, Any]] = []

        encoded_packets = encode_result.encoded_packets

        for meta in encode_result.encoded_metas:
            if not meta.is_repair:
                continue

            enc_id = int(meta.encoded_id)

            if not bool(receive_mask[enc_id].item()):
                continue

            source_ids = set(int(x) for x in meta.source_ids)

            if len(source_ids) == 0:
                continue

            equations.append(
                {
                    "encoded_id": enc_id,
                    "source_ids": source_ids,
                    "value": encoded_packets[enc_id].clone(),
                    "active": True,
                    "original_degree": int(len(source_ids)),
                }
            )

        return equations

    @staticmethod
    def _reduce_equation_with_known_sources(
        equation: Dict[str, Any],
        recovered_packets: torch.Tensor,
        recovered_source_mask: torch.Tensor,
    ) -> bool:
        """
        Remove currently known source packets from one equation.

        Returns
        -------
        bool
            True if the equation changed.
        """
        if not equation.get("active", True):
            return False

        source_ids = set(equation["source_ids"])
        known_ids = [
            src_id
            for src_id in source_ids
            if bool(recovered_source_mask[src_id].item())
        ]

        if not known_ids:
            return False

        known_packets = [recovered_packets[src_id] for src_id in known_ids]
        equation["value"] = _xor_packet_with_known(
            repair_packet=equation["value"],
            known_packets=known_packets,
        )

        for src_id in known_ids:
            equation["source_ids"].remove(src_id)

        if len(equation["source_ids"]) == 0:
            equation["active"] = False

        return True

    def decode(
        self,
        encode_result: FECEncodeResult,
        receive_mask: Optional[Any] = None,
        loss_mask: Optional[Any] = None,
        fill_value: int = 0,
        **kwargs,
    ) -> FECDecodeResult:
        """
        Decode received source + repair packets.

        This is a real peeling decoder:

            1. Directly received systematic source packets are known.
            2. Received repair equations are reduced by known sources.
            3. If an equation has exactly one unknown source, recover it.
            4. Repeat until no equation can recover more packets.

        Parameters
        ----------
        encode_result : FECEncodeResult
            Result returned by encode().

        receive_mask : tensor-like, optional
            Shape [N]. True means encoded packet received.

        loss_mask : tensor-like, optional
            Shape [N]. True means encoded packet lost.

        fill_value : int
            Temporary fill value for packets that remain missing.

        Returns
        -------
        FECDecodeResult
        """
        encode_result.validate()

        if encode_result.fec_type != FEC_TYPE_RAPTOR_SIM:
            raise ValueError(
                "RaptorSimFEC.decode expects encode_result.fec_type "
                f"== 'raptor_sim', got {encode_result.fec_type}."
            )

        _validate_raptor_packet_tensor(
            encode_result.encoded_packets,
            name="encode_result.encoded_packets",
        )

        receive_mask = resolve_receive_mask(
            num_encoded_packets=encode_result.num_encoded_packets,
            receive_mask=receive_mask,
            loss_mask=loss_mask,
            device=encode_result.encoded_packets.device,
        )

        recovered_packets, direct_received_source_mask = self._initial_decode_tensors(
            encode_result=encode_result,
            receive_mask=receive_mask,
            fill_value=fill_value,
        )

        k = encode_result.num_source_packets

        fec_recovered_source_mask = torch.zeros(
            k,
            dtype=torch.bool,
            device=encode_result.encoded_packets.device,
        )

        recovered_source_mask = direct_received_source_mask | fec_recovered_source_mask

        equations = self._build_initial_equations(
            encode_result=encode_result,
            receive_mask=receive_mask,
        )

        iteration = 0
        num_equation_reductions = 0
        num_degree_one_events = 0
        recovered_events: List[Dict[str, Any]] = []

        changed = True

        while changed and iteration < self.max_decode_iters:
            iteration += 1
            changed = False

            recovered_source_mask = direct_received_source_mask | fec_recovered_source_mask

            for equation in equations:
                if not equation.get("active", True):
                    continue

                reduced = self._reduce_equation_with_known_sources(
                    equation=equation,
                    recovered_packets=recovered_packets,
                    recovered_source_mask=recovered_source_mask,
                )

                if reduced:
                    num_equation_reductions += 1
                    changed = True

                unknown_ids = equation["source_ids"]

                if len(unknown_ids) == 1:
                    source_id = int(next(iter(unknown_ids)))

                    if not bool(recovered_source_mask[source_id].item()):
                        recovered_packets[source_id] = equation["value"]
                        fec_recovered_source_mask[source_id] = True

                        recovered_events.append(
                            {
                                "iteration": int(iteration),
                                "source_id": int(source_id),
                                "encoded_id": int(equation["encoded_id"]),
                                "original_degree": int(equation["original_degree"]),
                            }
                        )

                        num_degree_one_events += 1
                        changed = True

                    equation["source_ids"].clear()
                    equation["active"] = False

            if bool((direct_received_source_mask | fec_recovered_source_mask).all().item()):
                break

        recovered_source_mask = direct_received_source_mask | fec_recovered_source_mask
        missing_source_mask = ~recovered_source_mask

        num_direct = int(direct_received_source_mask.sum().item())
        num_fec_recovered = int(fec_recovered_source_mask.sum().item())
        num_missing = int(missing_source_mask.sum().item())
        num_received_encoded = int(receive_mask.sum().item())
        num_lost_encoded = int((~receive_mask).sum().item())

        received_repair_count = 0
        total_repair_count = 0
        for meta in encode_result.encoded_metas:
            if meta.is_repair:
                total_repair_count += 1
                if bool(receive_mask[int(meta.encoded_id)].item()):
                    received_repair_count += 1

        active_equations_left = sum(
            1 for equation in equations if equation.get("active", True)
        )

        required_packets_reference = estimate_raptor_required_packets(
            num_source_packets=k,
            decode_overhead=self.decode_overhead,
        )

        info = {
            "fec_type": FEC_TYPE_RAPTOR_SIM,
            "enabled": True,
            "implementation": "real_fountain_peeling_decoder",
            "is_threshold_only_simulation": False,
            "is_standard_raptorq": False,
            "num_source_packets": int(k),
            "num_encoded_packets": int(encode_result.num_encoded_packets),
            "num_received_encoded_packets": int(num_received_encoded),
            "num_lost_encoded_packets": int(num_lost_encoded),
            "num_repair_packets": int(total_repair_count),
            "num_received_repair_packets": int(received_repair_count),
            "num_direct_received_source_packets": int(num_direct),
            "num_fec_recovered_source_packets": int(num_fec_recovered),
            "num_missing_source_packets_after_raptor": int(num_missing),
            "num_initial_equations": int(len(equations)),
            "num_active_equations_left": int(active_equations_left),
            "num_decode_iterations": int(iteration),
            "max_decode_iters": int(self.max_decode_iters),
            "num_equation_reductions": int(num_equation_reductions),
            "num_degree_one_events": int(num_degree_one_events),
            "required_packets_reference": int(required_packets_reference),
            "received_enough_by_threshold_reference": bool(
                num_received_encoded >= required_packets_reference
            ),
            "degree_distribution": self.degree_distribution,
            "decode_overhead": float(self.decode_overhead),
            "recovered_events": recovered_events,
            "note": (
                "Decoder uses real peeling over received repair equations. "
                "Full recovery depends on equation graph structure, not only "
                "on the number of received packets."
            ),
        }

        result = self._finalize_decode_result(
            recovered_packets=recovered_packets,
            direct_received_source_mask=direct_received_source_mask,
            fec_recovered_source_mask=fec_recovered_source_mask,
            receive_mask=receive_mask,
            info=info,
        )

        result.validate()
        return result

    def get_config(self) -> Dict[str, Any]:
        """
        Return JSON-friendly config.
        """
        return {
            "enabled": True,
            "fec_type": FEC_TYPE_RAPTOR_SIM,
            "implementation": "real_fountain_peeling_decoder",
            "is_threshold_only_simulation": False,
            "is_standard_raptorq": False,
            "redundancy_ratio": float(self.redundancy_ratio),
            "decode_overhead": float(self.decode_overhead),
            "degree_distribution": self.degree_distribution,
            "robust_soliton_c": float(self.robust_soliton_c),
            "robust_soliton_delta": float(self.robust_soliton_delta),
            "fixed_degree": int(self.fixed_degree),
            "max_degree": self.max_degree,
            "num_repair_packets_override": self.num_repair_packets_override,
            "seed": int(self.seed),
            "max_decode_iters": int(self.max_decode_iters),
        }

    def __repr__(self) -> str:
        return (
            "RaptorSimFEC("
            "implementation=real_fountain_peeling_decoder, "
            f"redundancy_ratio={self.redundancy_ratio}, "
            f"decode_overhead={self.decode_overhead}, "
            f"degree_distribution={self.degree_distribution}, "
            f"seed={self.seed})"
        )


# Compatibility aliases.
# The project still names the file fec_raptor_sim.py, but these aliases make
# the intended usage explicit.
RaptorFEC = RaptorSimFEC
RaptorLikeFEC = RaptorSimFEC
FountainFEC = RaptorSimFEC


__all__ = [
    "RaptorSimFEC",
    "RaptorFEC",
    "RaptorLikeFEC",
    "FountainFEC",
]