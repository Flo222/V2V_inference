"""
No-FEC baseline for ARCE communication simulation.

This module implements the simplest packet transmission scheme:

    source_packets
    -> encoded_packets = source_packets
    -> GE packet loss
    -> directly received source packets are kept
    -> lost source packets are marked missing and filled with zero temporarily

No parity packets are generated.

Mask convention:
    loss_mask[i] == True
        encoded/source packet i is lost.

    receive_mask[i] == True
        encoded/source packet i is received.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from opencood.communication.transport.fec import FEC_TYPE_NONE
from opencood.communication.transport.fec.fec_base import (
    EncodedPacketMeta,
    FECBase,
    FECEncodeResult,
    FECDecodeResult,
    resolve_receive_mask,
    validate_packet_tensor,
)


class NoFEC(FECBase):
    """
    No-FEC packet transmission baseline.

    In NoFEC:
        K source packets are transmitted as K encoded packets.
        No parity or repair packets are added.

    This module is useful for:
        1. no-protection baseline;
        2. verifying that GE packet loss really damages V2X-ViT features;
        3. measuring how much ARCE gains from adding XOR / Raptor-sim later.
    """

    fec_type = FEC_TYPE_NONE

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        cfg : dict, optional
            Can be full ARCE config or direct fec config.

            Even if cfg says another FEC type, NoFEC forces:
                enabled = False
                fec_type = none
                redundancy_ratio = 0.0
        """
        super().__init__(cfg or {})

        self.enabled = False
        self.fec_type = FEC_TYPE_NONE
        self.redundancy_ratio = 0.0
        self.group_size = None
        self.decode_overhead = 0.0

    def encode(self, source_packets: torch.Tensor, **kwargs) -> FECEncodeResult:
        """
        Encode source packets without adding redundancy.

        Parameters
        ----------
        source_packets : torch.Tensor
            Source packet tensor with shape [K, ...].

        Returns
        -------
        FECEncodeResult
            encoded_packets are identical to source_packets.
        """
        source_packets = validate_packet_tensor(
            source_packets,
            name="source_packets",
            min_dim=1,
        )

        num_source_packets = int(source_packets.shape[0])

        # Clone to make encoded_packets independent from source_packets.
        # This is safer because later modules may modify encoded_packets.
        encoded_packets = source_packets.clone()

        encoded_metas = []

        for packet_id in range(num_source_packets):
            encoded_metas.append(
                EncodedPacketMeta(
                    encoded_id=packet_id,
                    kind="source",
                    source_id=packet_id,
                    group_id=None,
                    source_ids=(packet_id,),
                    note="NoFEC direct source packet",
                )
            )

        info = {
            "fec_type": FEC_TYPE_NONE,
            "enabled": False,
            "num_source_packets": int(num_source_packets),
            "num_parity_packets": 0,
            "num_encoded_packets": int(num_source_packets),
            "redundancy_ratio_config": 0.0,
            "effective_redundancy_ratio": 0.0,
            "note": "No redundancy is added. Encoded packets equal source packets.",
        }

        result = FECEncodeResult(
            source_packets=source_packets,
            encoded_packets=encoded_packets,
            encoded_metas=encoded_metas,
            fec_type=FEC_TYPE_NONE,
            redundancy_ratio_config=0.0,
            group_size=None,
            decode_overhead=0.0,
            info=info,
        )

        result.validate()
        return result

    def decode(
        self,
        encode_result: FECEncodeResult,
        receive_mask: Optional[Any] = None,
        loss_mask: Optional[Any] = None,
        fill_value: float = 0.0,
        **kwargs,
    ) -> FECDecodeResult:
        """
        Decode received packets under NoFEC.

        Since there is no redundancy:
            - directly received source packets are recovered;
            - lost source packets remain missing;
            - fec_recovered_source_mask is always all False.

        Parameters
        ----------
        encode_result : FECEncodeResult
            Result from encode().

        receive_mask : tensor-like, optional
            Shape [K]. True means packet received.

        loss_mask : tensor-like, optional
            Shape [K]. True means packet lost.

        fill_value : float
            Temporary value used for missing packets in recovered_packets.
            Later partial reconstruction can overwrite these missing packets.

        Returns
        -------
        FECDecodeResult
            Decoding result with missing_source_mask indicating lost packets.
        """
        encode_result.validate()

        if encode_result.fec_type != FEC_TYPE_NONE:
            raise ValueError(
                "NoFEC.decode expects encode_result.fec_type == 'none', "
                f"got {encode_result.fec_type}."
            )

        num_encoded_packets = encode_result.num_encoded_packets

        receive_mask = resolve_receive_mask(
            num_encoded_packets=num_encoded_packets,
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

        num_direct = int(direct_received_source_mask.sum().item())
        num_missing = int(encode_result.num_source_packets - num_direct)

        info = {
            "fec_type": FEC_TYPE_NONE,
            "enabled": False,
            "num_source_packets": int(encode_result.num_source_packets),
            "num_encoded_packets": int(encode_result.num_encoded_packets),
            "num_direct_received_source_packets": int(num_direct),
            "num_fec_recovered_source_packets": 0,
            "num_missing_source_packets": int(num_missing),
            "note": (
                "NoFEC cannot recover lost source packets. "
                "Missing packets are temporarily filled and should be handled "
                "by partial reconstruction."
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
        Return JSON-friendly NoFEC config.
        """
        return {
            "enabled": False,
            "fec_type": FEC_TYPE_NONE,
            "redundancy_ratio": 0.0,
            "group_size": None,
            "decode_overhead": 0.0,
        }

    def __repr__(self) -> str:
        return "NoFEC(enabled=False, fec_type=none, redundancy_ratio=0.0)"