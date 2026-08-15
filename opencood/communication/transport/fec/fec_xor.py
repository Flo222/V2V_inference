"""
XOR FEC for ARCE communication simulation.

This module implements real XOR parity coding over integer quantized packets.

Workflow:
    source_packets [K, ...]
    -> group consecutive source packets
    -> one XOR parity packet per group
    -> encoded_packets = source_packets + parity_packets
    -> GE packet loss samples encoded packets
    -> decoder recovers a missing source packet if:
           1. the parity packet of its group is received;
           2. exactly one source packet in that group is missing;
           3. all other source packets in that group are received.

Important:
    XOR should be performed on integer quantized packets, not float tensors.

    Recommended usage:
        quant_result = quantizer.quantize_packets(packets, mode="int8")
        encode_result = xor_fec.encode(quant_result.q_tensor)

        loss_mask, channel_info = channel_manager.sample_packet_loss(
            num_packets=encode_result.num_encoded_packets,
            ...
        )

        decode_result = xor_fec.decode(
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

For INT4:
    INT4 values are usually stored in torch.int8 with range [-7, 7].
    XOR parity values may fall outside [-7, 7], because parity is a byte-level
    representation. That is fine. The recovered source packet will match the
    original source packet exactly if the XOR recovery condition is satisfied.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from opencood.communication.transport.fec import (
    FEC_TYPE_XOR,
    DEFAULT_XOR_GROUP_SIZE,
    normalize_group_size,
    effective_redundancy_ratio,
)

from opencood.communication.transport.fec.fec_base import (
    EncodedPacketMeta,
    FECBase,
    FECEncodeResult,
    FECDecodeResult,
    resolve_receive_mask,
    validate_packet_tensor,
)


SUPPORTED_XOR_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _validate_xor_packet_tensor(
    packets: torch.Tensor,
    name: str = "packets",
) -> torch.Tensor:
    """
    Validate packet tensor for XOR coding.

    XOR requires integer tensors. Float tensors are intentionally rejected,
    because XOR over float feature values is not meaningful for real FEC.
    """
    packets = validate_packet_tensor(
        packets,
        name=name,
        min_dim=1,
    )

    if packets.dtype not in SUPPORTED_XOR_DTYPES:
        raise TypeError(
            f"{name} should be an integer tensor for XOR FEC, "
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
        Shape [G, ...], integer dtype.

    Returns
    -------
    torch.Tensor
        XOR parity packet with shape [...].
    """
    packet_group = _validate_xor_packet_tensor(
        packet_group,
        name="packet_group",
    )

    if packet_group.shape[0] <= 0:
        raise ValueError("packet_group should contain at least one packet.")

    parity = packet_group[0].clone()

    for idx in range(1, int(packet_group.shape[0])):
        parity = torch.bitwise_xor(parity, packet_group[idx])

    return parity


def _xor_recover_from_group(
    parity_packet: torch.Tensor,
    known_packets: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Recover one missing packet from parity and all known packets.

    Formula:
        missing = parity XOR known_1 XOR known_2 XOR ...

    Parameters
    ----------
    parity_packet : torch.Tensor
        XOR parity packet.

    known_packets : sequence of torch.Tensor
        All received source packets in the group except the missing one.

    Returns
    -------
    torch.Tensor
        Recovered missing packet.
    """
    parity_packet = _validate_xor_packet_tensor(
        parity_packet,
        name="parity_packet",
    )

    recovered = parity_packet.clone()

    for packet in known_packets:
        packet = _validate_xor_packet_tensor(packet, name="known_packet")
        recovered = torch.bitwise_xor(recovered, packet)

    return recovered


class XORFEC(FECBase):
    """
    Real XOR parity FEC.

    One parity packet is generated for each consecutive source-packet group.

    Example:
        K = 10
        group_size = 4

        groups:
            group 0: source 0, 1, 2, 3 -> parity 0
            group 1: source 4, 5, 6, 7 -> parity 1
            group 2: source 8, 9       -> parity 2

        encoded packet order:
            source 0
            source 1
            ...
            source 9
            parity 0
            parity 1
            parity 2
    """

    fec_type = FEC_TYPE_XOR

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
                    type: xor
                    group_size: 4
                    redundancy_ratio: 0.25
        """
        super().__init__(cfg or {})

        self.enabled = True
        self.fec_type = FEC_TYPE_XOR

        if self.group_size is None:
            self.group_size = DEFAULT_XOR_GROUP_SIZE

        self.group_size = normalize_group_size(self.group_size)

        # For XOR, actual redundancy is determined by group_size:
        #     parity = ceil(K / group_size)
        # redundancy_ratio is kept as config/logging information.
        if self.redundancy_ratio <= 0.0:
            self.redundancy_ratio = 1.0 / float(self.group_size)

    def _build_groups(self, num_source_packets: int) -> List[Tuple[int, List[int]]]:
        """
        Build consecutive source packet groups.

        Returns
        -------
        list
            [
                (group_id, [source_id_0, source_id_1, ...]),
                ...
            ]
        """
        k = int(num_source_packets)

        if k < 0:
            raise ValueError(f"num_source_packets should be non-negative, got {k}.")

        groups: List[Tuple[int, List[int]]] = []

        group_id = 0
        for start in range(0, k, self.group_size):
            end = min(start + self.group_size, k)
            source_ids = list(range(start, end))
            groups.append((group_id, source_ids))
            group_id += 1

        return groups

    def _build_parity_packets_and_metas(
        self,
        source_packets: torch.Tensor,
        start_encoded_id: int,
    ) -> Tuple[torch.Tensor, List[EncodedPacketMeta], Dict[int, List[int]]]:
        """
        Build XOR parity packets and their metadata.

        Parameters
        ----------
        source_packets : torch.Tensor
            Shape [K, ...].

        start_encoded_id : int
            Encoded id of the first parity packet.

        Returns
        -------
        parity_packets : torch.Tensor
            Shape [P, ...].

        parity_metas : list of EncodedPacketMeta

        group_to_source_ids : dict
            Mapping group_id -> source_ids.
        """
        source_packets = _validate_xor_packet_tensor(
            source_packets,
            name="source_packets",
        )

        k = int(source_packets.shape[0])
        groups = self._build_groups(k)

        parity_packets = []
        parity_metas: List[EncodedPacketMeta] = []
        group_to_source_ids: Dict[int, List[int]] = {}

        for parity_idx, (group_id, source_ids) in enumerate(groups):
            group_tensor = source_packets[source_ids]
            parity = _xor_reduce(group_tensor)

            encoded_id = int(start_encoded_id + parity_idx)

            parity_packets.append(parity)
            parity_metas.append(
                EncodedPacketMeta(
                    encoded_id=encoded_id,
                    kind="parity",
                    source_id=None,
                    group_id=group_id,
                    source_ids=tuple(int(x) for x in source_ids),
                    note=(
                        "XOR parity packet. Can recover the group if exactly "
                        "one source packet in this group is missing and parity "
                        "is received."
                    ),
                )
            )

            group_to_source_ids[group_id] = [int(x) for x in source_ids]

        if len(parity_packets) == 0:
            empty_shape = (0,) + tuple(source_packets.shape[1:])
            parity_tensor = torch.empty(
                empty_shape,
                dtype=source_packets.dtype,
                device=source_packets.device,
            )
        else:
            parity_tensor = torch.stack(parity_packets, dim=0)

        return parity_tensor, parity_metas, group_to_source_ids

    def encode(self, source_packets: torch.Tensor, **kwargs) -> FECEncodeResult:
        """
        Encode integer source packets with XOR parity.

        Parameters
        ----------
        source_packets : torch.Tensor
            Integer tensor with shape [K, ...].

        Returns
        -------
        FECEncodeResult
            encoded_packets has shape [K + ceil(K / group_size), ...].
        """
        source_packets = _validate_xor_packet_tensor(
            source_packets,
            name="source_packets",
        )

        num_source_packets = int(source_packets.shape[0])

        source_metas = self._build_source_metas(num_source_packets)

        parity_packets, parity_metas, group_to_source_ids = (
            self._build_parity_packets_and_metas(
                source_packets=source_packets,
                start_encoded_id=num_source_packets,
            )
        )

        encoded_packets = torch.cat(
            [source_packets.clone(), parity_packets],
            dim=0,
        )

        encoded_metas = source_metas + parity_metas

        num_parity_packets = int(parity_packets.shape[0])
        num_encoded_packets = int(encoded_packets.shape[0])

        info = {
            "fec_type": FEC_TYPE_XOR,
            "enabled": True,
            "group_size": int(self.group_size),
            "num_source_packets": int(num_source_packets),
            "num_parity_packets": int(num_parity_packets),
            "num_encoded_packets": int(num_encoded_packets),
            "redundancy_ratio_config": float(self.redundancy_ratio),
            "effective_redundancy_ratio": float(
                effective_redundancy_ratio(
                    num_source_packets,
                    num_parity_packets,
                )
            ),
            "group_to_source_ids": {
                int(k): [int(v) for v in values]
                for k, values in group_to_source_ids.items()
            },
            "note": (
                "XOR FEC creates one parity packet per source-packet group. "
                "A group can recover exactly one missing source packet if "
                "the parity packet is received."
            ),
        }

        result = FECEncodeResult(
            source_packets=source_packets,
            encoded_packets=encoded_packets,
            encoded_metas=encoded_metas,
            fec_type=FEC_TYPE_XOR,
            redundancy_ratio_config=float(self.redundancy_ratio),
            group_size=int(self.group_size),
            decode_overhead=0.0,
            info=info,
        )

        result.validate()
        return result

    def _decode_one_group(
        self,
        encode_result: FECEncodeResult,
        receive_mask: torch.Tensor,
        recovered_packets: torch.Tensor,
        direct_received_source_mask: torch.Tensor,
        fec_recovered_source_mask: torch.Tensor,
        parity_meta: EncodedPacketMeta,
    ) -> Dict[str, Any]:
        """
        Try to recover one XOR group.

        Recovery condition:
            parity received
            exactly one source in the group missing
            all other source packets are directly received or already recovered

        Returns
        -------
        dict
            Group-level decoding info.
        """
        encoded_packets = encode_result.encoded_packets
        k = encode_result.num_source_packets

        parity_encoded_id = int(parity_meta.encoded_id)
        group_id = int(parity_meta.group_id)
        source_ids = [int(x) for x in parity_meta.source_ids]

        group_info = {
            "group_id": group_id,
            "parity_encoded_id": parity_encoded_id,
            "source_ids": source_ids,
            "parity_received": False,
            "num_missing_in_group": 0,
            "recovered_source_id": None,
            "success": False,
            "reason": "",
        }

        if not bool(receive_mask[parity_encoded_id].item()):
            group_info["reason"] = "parity packet lost"
            return group_info

        group_info["parity_received"] = True

        if len(source_ids) == 0:
            group_info["reason"] = "empty source group"
            return group_info

        available_mask = direct_received_source_mask | fec_recovered_source_mask

        missing_source_ids = [
            src_id
            for src_id in source_ids
            if not bool(available_mask[src_id].item())
        ]

        group_info["num_missing_in_group"] = int(len(missing_source_ids))

        if len(missing_source_ids) == 0:
            group_info["reason"] = "all source packets already received"
            return group_info

        if len(missing_source_ids) > 1:
            group_info["reason"] = "more than one source packet missing in group"
            return group_info

        missing_source_id = int(missing_source_ids[0])

        known_packets = []
        for src_id in source_ids:
            src_id = int(src_id)

            if src_id == missing_source_id:
                continue

            if src_id < 0 or src_id >= k:
                raise ValueError(
                    f"Invalid source_id={src_id} for K={k}."
                )

            if not bool(available_mask[src_id].item()):
                group_info["reason"] = (
                    "internal inconsistency: a known packet is unavailable"
                )
                return group_info

            known_packets.append(recovered_packets[src_id])

        parity_packet = encoded_packets[parity_encoded_id]

        recovered = _xor_recover_from_group(
            parity_packet=parity_packet,
            known_packets=known_packets,
        )

        recovered_packets[missing_source_id] = recovered
        fec_recovered_source_mask[missing_source_id] = True

        group_info["recovered_source_id"] = int(missing_source_id)
        group_info["success"] = True
        group_info["reason"] = "exactly one source packet missing and parity received"

        return group_info

    def decode(
        self,
        encode_result: FECEncodeResult,
        receive_mask: Optional[Any] = None,
        loss_mask: Optional[Any] = None,
        fill_value: int = 0,
        **kwargs,
    ) -> FECDecodeResult:
        """
        Decode XOR FEC.

        Parameters
        ----------
        encode_result : FECEncodeResult
            Result from encode().

        receive_mask : tensor-like, optional
            Shape [N]. True means encoded packet received.

        loss_mask : tensor-like, optional
            Shape [N]. True means encoded packet lost.

        fill_value : int
            Temporary value for packets that remain missing after XOR recovery.

        Returns
        -------
        FECDecodeResult
            Source-packet recovery result.
        """
        encode_result.validate()

        if encode_result.fec_type != FEC_TYPE_XOR:
            raise ValueError(
                "XORFEC.decode expects encode_result.fec_type == 'xor', "
                f"got {encode_result.fec_type}."
            )

        _validate_xor_packet_tensor(
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

        fec_recovered_source_mask = torch.zeros(
            encode_result.num_source_packets,
            dtype=torch.bool,
            device=encode_result.encoded_packets.device,
        )

        group_infos = []

        # Each XOR group has only one parity packet, so one pass is enough.
        # The implementation still allows available_mask to include earlier
        # recovered packets, which is harmless and useful if future variants add
        # multiple parity layers.
        for meta in encode_result.encoded_metas:
            if not meta.is_parity:
                continue

            group_info = self._decode_one_group(
                encode_result=encode_result,
                receive_mask=receive_mask,
                recovered_packets=recovered_packets,
                direct_received_source_mask=direct_received_source_mask,
                fec_recovered_source_mask=fec_recovered_source_mask,
                parity_meta=meta,
            )
            group_infos.append(group_info)

        num_groups = len(group_infos)
        num_success_groups = sum(1 for g in group_infos if g.get("success", False))
        num_fec_recovered = int(fec_recovered_source_mask.sum().item())
        num_direct = int(direct_received_source_mask.sum().item())
        num_missing_after_xor = int(
            encode_result.num_source_packets
            - num_direct
            - num_fec_recovered
        )

        info = {
            "fec_type": FEC_TYPE_XOR,
            "enabled": True,
            "group_size": int(self.group_size),
            "num_source_packets": int(encode_result.num_source_packets),
            "num_encoded_packets": int(encode_result.num_encoded_packets),
            "num_parity_packets": int(
                encode_result.num_encoded_packets
                - encode_result.num_source_packets
            ),
            "num_groups": int(num_groups),
            "num_success_groups": int(num_success_groups),
            "num_direct_received_source_packets": int(num_direct),
            "num_fec_recovered_source_packets": int(num_fec_recovered),
            "num_missing_source_packets_after_xor": int(num_missing_after_xor),
            "group_infos": group_infos,
            "note": (
                "XOR recovers a source packet only when exactly one source "
                "packet in its group is missing and the parity packet is received."
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
        Return JSON-friendly XOR FEC config.
        """
        return {
            "enabled": True,
            "fec_type": FEC_TYPE_XOR,
            "redundancy_ratio": float(self.redundancy_ratio),
            "group_size": int(self.group_size),
            "decode_overhead": 0.0,
        }

    def __repr__(self) -> str:
        return (
            "XORFEC("
            f"enabled=True, "
            f"fec_type=xor, "
            f"group_size={self.group_size}, "
            f"redundancy_ratio={self.redundancy_ratio})"
        )