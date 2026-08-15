"""Priority-preserving, budget-aware block scheduling for packet FEC."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Tuple

import torch

from opencood.communication.transport.fec.fec_raptorq import (
    RaptorQBlockCodec,
    RaptorQEncodedBlock,
)


SCHEDULING_MODE = "exact_ratio_protected_prefix_with_best_effort_tail"
REDUNDANCY_POLICY = "exact_protected_prefix_with_best_effort_tail"


def exact_redundancy_group(rho: float) -> tuple:
    """Return the smallest exact ``(source, repair)`` group for ``rho``."""
    rho = float(rho)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho must be finite and positive")
    ratio = Fraction(str(rho)).limit_denominator(1000)
    if abs(float(ratio) - rho) > 1e-9:
        raise ValueError("rho cannot be represented by a stable packet ratio")
    return int(ratio.denominator), int(ratio.numerator)


def exact_repair_packets_for_block(
    num_source_packets: int,
    rho: float,
) -> int:
    """Return an exact repair count; reject a non-integral protection block."""
    num_source_packets = int(num_source_packets)
    if num_source_packets <= 0:
        return 0
    source_unit, repair_unit = exact_redundancy_group(rho)
    if num_source_packets % source_unit != 0:
        raise ValueError(
            "Protected source count {} is not a multiple of the exact group {}"
            .format(num_source_packets, source_unit)
        )
    return int(num_source_packets // source_unit * repair_unit)


def largest_exact_protected_source_block(
    remaining_wire_packets: int,
    remaining_source_packets: int,
    target_source_packets: int,
    rho: float,
) -> int:
    upper = min(
        int(remaining_wire_packets),
        int(remaining_source_packets),
        int(target_source_packets),
    )
    source_unit, _ = exact_redundancy_group(rho)
    upper -= upper % source_unit
    for source_count in range(upper, 0, -source_unit):
        repair_count = exact_repair_packets_for_block(source_count, rho)
        if source_count + repair_count <= int(remaining_wire_packets):
            return source_count
    return 0


def plan_exact_protected_prefix(
    total_source_packets: int,
    wire_packet_slots: int,
    block_source_packets: int,
    rho: float,
) -> Tuple[List[int], int]:
    """Plan exact-ratio protection blocks and an unprotected source tail."""
    remaining_source = max(0, int(total_source_packets))
    remaining_slots = max(0, int(wire_packet_slots))
    protected_blocks: List[int] = []

    while remaining_source > 0 and remaining_slots > 0:
        source_count = largest_exact_protected_source_block(
            remaining_wire_packets=remaining_slots,
            remaining_source_packets=remaining_source,
            target_source_packets=block_source_packets,
            rho=rho,
        )
        if source_count <= 0:
            break
        repair_count = exact_repair_packets_for_block(source_count, rho)
        protected_blocks.append(source_count)
        remaining_source -= source_count
        remaining_slots -= source_count + repair_count

    tail_source_packets = min(remaining_source, remaining_slots)
    return protected_blocks, int(tail_source_packets)


@dataclass(frozen=True)
class ScheduledRaptorQBlock:
    block_id: int
    role: str
    global_source_start: int
    encoded_start: int
    encoded_end: int
    encoded: RaptorQEncodedBlock

    @property
    def num_source_packets(self) -> int:
        return self.encoded.num_source_packets

    @property
    def num_repair_packets(self) -> int:
        return self.encoded.num_repair_packets

    def as_dict(self) -> Dict[str, Any]:
        result = self.encoded.as_dict()
        result.update({
            "block_role": str(self.role),
            "protected": bool(self.role == "protected"),
            "global_source_start": int(self.global_source_start),
            "global_source_end": int(
                self.global_source_start + self.num_source_packets
            ),
            "encoded_start": int(self.encoded_start),
            "encoded_end": int(self.encoded_end),
        })
        return result


@dataclass
class PriorityBlockFECPlan:
    source_packets: torch.Tensor
    wire_packets: torch.Tensor
    blocks: List[ScheduledRaptorQBlock]
    admitted_source_mask: torch.Tensor
    protected_source_mask: torch.Tensor
    tail_source_mask: torch.Tensor
    source_wire_positions: torch.Tensor
    repair_wire_positions: torch.Tensor
    budget_bytes: float
    source_packet_bytes: int
    wire_packet_bytes: int
    redundancy_ratio: float
    block_source_packets: int

    @property
    def num_source_packets(self) -> int:
        return int(self.source_packets.shape[0])

    @property
    def num_admitted_source_packets(self) -> int:
        return int(self.admitted_source_mask.sum().item())

    @property
    def num_source_dropped_by_budget(self) -> int:
        return self.num_source_packets - self.num_admitted_source_packets

    @property
    def num_protected_source_packets(self) -> int:
        return int(self.protected_source_mask.sum().item())

    @property
    def num_tail_source_packets(self) -> int:
        return int(self.tail_source_mask.sum().item())

    @property
    def num_repair_packets(self) -> int:
        return int(self.repair_wire_positions.numel())

    @property
    def num_protected_blocks(self) -> int:
        return sum(block.role == "protected" for block in self.blocks)

    @property
    def num_tail_blocks(self) -> int:
        return sum(block.role == "best_effort_tail" for block in self.blocks)

    @property
    def num_parity_packets(self) -> int:
        return self.num_repair_packets

    @property
    def num_encoded_packets(self) -> int:
        return int(self.wire_packets.shape[0])

    @property
    def protected_redundancy_ratio(self) -> float:
        if self.num_protected_source_packets <= 0:
            return 0.0
        return float(
            self.num_repair_packets / self.num_protected_source_packets
        )

    @property
    def overall_redundancy_ratio(self) -> float:
        if self.num_admitted_source_packets <= 0:
            return 0.0
        return float(
            self.num_repair_packets / self.num_admitted_source_packets
        )

    @property
    def encoded_packets(self) -> torch.Tensor:
        return self.wire_packets

    @property
    def actual_transmitted_bytes(self) -> int:
        return int(self.num_encoded_packets * self.wire_packet_bytes)

    @property
    def metadata_bytes(self) -> int:
        return int(
            self.num_encoded_packets
            * (self.wire_packet_bytes - self.source_packet_bytes)
        )

    @property
    def unused_budget_bytes(self) -> float:
        return float(self.budget_bytes - self.actual_transmitted_bytes)

    def repair_receive_mask(self, receive_mask: torch.Tensor) -> torch.Tensor:
        receive_mask = torch.as_tensor(
            receive_mask,
            dtype=torch.bool,
            device=self.wire_packets.device,
        ).flatten()
        return receive_mask[self.repair_wire_positions]

    def as_dict(
        self,
        include_blocks: bool = True,
        include_metas: bool = False,
    ) -> Dict[str, Any]:
        _ = bool(include_metas)
        block_records = []
        if include_blocks:
            for block in self.blocks:
                block_record = block.as_dict()
                is_protected = block.role == "protected"
                block_record.update({
                    "action_redundancy_ratio_target": float(
                        self.redundancy_ratio
                    ),
                    "redundancy_ratio_target": float(
                        self.redundancy_ratio if is_protected else 0.0
                    ),
                    "redundancy_ratio_effective": float(
                        block.num_repair_packets
                        / max(1, block.num_source_packets)
                    ),
                    "redundancy_rounding": (
                        "exact_rational_group" if is_protected else "none"
                    ),
                })
                block_records.append(block_record)

        result = {
            "enabled": True,
            "fec_type": "raptorq",
            "codec": "raptorq",
            "standard": "RFC6330",
            "scheduling": SCHEDULING_MODE,
            "redundancy_policy": REDUNDANCY_POLICY,
            "num_source_packets": int(self.num_source_packets),
            "num_admitted_source_packets": int(
                self.num_admitted_source_packets
            ),
            "num_protected_source_packets": int(
                self.num_protected_source_packets
            ),
            "num_tail_source_packets": int(self.num_tail_source_packets),
            "num_source_dropped_by_budget": int(
                self.num_source_dropped_by_budget
            ),
            "num_repair_packets": int(self.num_repair_packets),
            "num_parity_packets": int(self.num_repair_packets),
            "num_encoded_packets": int(self.num_encoded_packets),
            "num_transmitted_packets": int(self.num_encoded_packets),
            "redundancy_ratio_config": float(self.redundancy_ratio),
            "redundancy_ratio_target": float(self.redundancy_ratio),
            "effective_redundancy_ratio": float(
                self.overall_redundancy_ratio
            ),
            "protected_redundancy_ratio": float(
                self.protected_redundancy_ratio
            ),
            "overall_redundancy_ratio": float(
                self.overall_redundancy_ratio
            ),
            "redundancy_rounding": "exact_rational_groups",
            "tail_policy": "best_effort_source_without_repair",
            "block_source_packets": int(self.block_source_packets),
            "num_blocks": int(len(self.blocks)),
            "num_protected_blocks": int(self.num_protected_blocks),
            "num_tail_blocks": int(self.num_tail_blocks),
            "source_packet_bytes": int(self.source_packet_bytes),
            "wire_packet_bytes": int(self.wire_packet_bytes),
            "metadata_bytes_per_packet": int(
                self.wire_packet_bytes - self.source_packet_bytes
            ),
            "metadata_bytes": int(self.metadata_bytes),
            "budget_bytes": float(self.budget_bytes),
            "actual_transmitted_bytes": int(self.actual_transmitted_bytes),
            "unused_budget_bytes": float(self.unused_budget_bytes),
        }
        if include_blocks:
            result["blocks"] = block_records
        return result


@dataclass
class PriorityBlockFECDecodeResult:
    recovered_packets: torch.Tensor
    recovered_source_mask: torch.Tensor
    direct_received_source_mask: torch.Tensor
    fec_recovered_source_mask: torch.Tensor
    missing_source_mask: torch.Tensor
    receive_mask: torch.Tensor
    loss_mask: torch.Tensor
    fec_type: str = "raptorq"
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_source_packets(self) -> int:
        return int(self.recovered_packets.shape[0])

    @property
    def num_encoded_packets(self) -> int:
        return int(self.receive_mask.numel())

    @property
    def num_recovered_source_packets(self) -> int:
        return int(self.recovered_source_mask.sum().item())

    @property
    def num_direct_received_source_packets(self) -> int:
        return int(self.direct_received_source_mask.sum().item())

    @property
    def num_fec_recovered_source_packets(self) -> int:
        return int(self.fec_recovered_source_mask.sum().item())

    @property
    def num_missing_source_packets(self) -> int:
        return int(self.missing_source_mask.sum().item())

    @property
    def full_recovery(self) -> bool:
        return bool(self.recovered_source_mask.all().item())

    @property
    def recovery_ratio(self) -> float:
        return float(
            self.num_recovered_source_packets / max(1, self.num_source_packets)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fec_type": self.fec_type,
            "num_source_packets": int(self.num_source_packets),
            "num_encoded_packets": int(self.num_encoded_packets),
            "num_received_encoded_packets": int(self.receive_mask.sum().item()),
            "num_lost_encoded_packets": int(self.loss_mask.sum().item()),
            "num_recovered_source_packets": int(
                self.num_recovered_source_packets
            ),
            "num_direct_received_source_packets": int(
                self.num_direct_received_source_packets
            ),
            "num_fec_recovered_source_packets": int(
                self.num_fec_recovered_source_packets
            ),
            "num_missing_source_packets": int(
                self.num_missing_source_packets
            ),
            "full_recovery": bool(self.full_recovery),
            "recovery_ratio": float(self.recovery_ratio),
            "info": dict(self.info),
        }


class PriorityBlockFECTransport:
    def __init__(
        self,
        source_packet_bytes: int = 1024,
        block_source_packets: int = 20,
    ):
        self.source_packet_bytes = int(source_packet_bytes)
        self.block_source_packets = int(block_source_packets)
        if self.source_packet_bytes <= 0:
            raise ValueError("source_packet_bytes must be positive")
        if self.block_source_packets <= 0:
            raise ValueError("block_source_packets must be positive")
        self.codec = RaptorQBlockCodec(self.source_packet_bytes)

    @property
    def wire_packet_bytes(self) -> int:
        return self.codec.wire_packet_bytes

    def encode_under_budget(
        self,
        source_packets: torch.Tensor,
        budget_bytes: float,
        redundancy_ratio: float,
    ) -> PriorityBlockFECPlan:
        if not torch.is_tensor(source_packets) or source_packets.dim() != 2:
            raise ValueError("source_packets must have shape [K, packet_bytes]")
        if int(source_packets.shape[1]) != self.source_packet_bytes:
            raise ValueError("source packet width does not match transport config")
        budget_bytes = float(budget_bytes)
        if not math.isfinite(budget_bytes) or budget_bytes < 0.0:
            raise ValueError("budget_bytes must be finite and non-negative")
        redundancy_ratio = float(redundancy_ratio)
        if not math.isfinite(redundancy_ratio) or redundancy_ratio <= 0.0:
            raise ValueError("Block RaptorQ transport requires rho > 0")

        total_source = int(source_packets.shape[0])
        remaining_slots = int(math.floor(budget_bytes / self.wire_packet_bytes))
        source_cursor = 0
        encoded_cursor = 0
        encoded_rows = []
        blocks: List[ScheduledRaptorQBlock] = []
        source_positions: List[int] = []
        repair_positions: List[int] = []

        protected_blocks, tail_source_packets = plan_exact_protected_prefix(
            total_source_packets=total_source,
            wire_packet_slots=remaining_slots,
            block_source_packets=self.block_source_packets,
            rho=redundancy_ratio,
        )
        for k in protected_blocks:
            p = exact_repair_packets_for_block(k, redundancy_ratio)
            block_id = len(blocks)
            encoded = self.codec.encode_block(
                source_packets=source_packets[source_cursor:source_cursor + k],
                num_repair_packets=p,
                block_id=block_id,
            )
            n = encoded.num_encoded_packets
            if n != k + p or n > remaining_slots:
                raise RuntimeError("RaptorQ block violates admission plan")

            encoded_rows.append(encoded.wire_packets)
            source_positions.extend(range(encoded_cursor, encoded_cursor + k))
            repair_positions.extend(
                range(encoded_cursor + k, encoded_cursor + n)
            )
            blocks.append(ScheduledRaptorQBlock(
                block_id=block_id,
                role="protected",
                global_source_start=source_cursor,
                encoded_start=encoded_cursor,
                encoded_end=encoded_cursor + n,
                encoded=encoded,
            ))
            source_cursor += k
            encoded_cursor += n
            remaining_slots -= n

        protected_source_end = source_cursor

        # An incomplete exact-ratio group is not over-protected. Remaining
        # packet slots carry the next lower-priority source symbols without
        # repair, preserving payload coverage and the selected prefix order.
        tail_remaining = int(tail_source_packets)
        while tail_remaining > 0:
            k = min(
                tail_remaining,
                self.block_source_packets,
            )
            if k <= 0:
                break
            block_id = len(blocks)
            encoded = self.codec.encode_block(
                source_packets=source_packets[source_cursor:source_cursor + k],
                num_repair_packets=0,
                block_id=block_id,
            )
            n = encoded.num_encoded_packets
            if n != k or n > remaining_slots:
                raise RuntimeError("RaptorQ tail block violates admission plan")

            encoded_rows.append(encoded.wire_packets)
            source_positions.extend(range(encoded_cursor, encoded_cursor + k))
            blocks.append(ScheduledRaptorQBlock(
                block_id=block_id,
                role="best_effort_tail",
                global_source_start=source_cursor,
                encoded_start=encoded_cursor,
                encoded_end=encoded_cursor + n,
                encoded=encoded,
            ))
            source_cursor += k
            encoded_cursor += n
            remaining_slots -= n
            tail_remaining -= k

        if encoded_rows:
            wire_packets = torch.cat(encoded_rows, dim=0)
        else:
            wire_packets = torch.empty(
                (0, self.wire_packet_bytes),
                dtype=torch.uint8,
                device=source_packets.device,
            )
        admitted = torch.zeros(
            total_source,
            dtype=torch.bool,
            device=source_packets.device,
        )
        admitted[:source_cursor] = True
        protected = torch.zeros_like(admitted)
        protected[:protected_source_end] = True
        tail = admitted & ~protected
        return PriorityBlockFECPlan(
            source_packets=source_packets,
            wire_packets=wire_packets,
            blocks=blocks,
            admitted_source_mask=admitted,
            protected_source_mask=protected,
            tail_source_mask=tail,
            source_wire_positions=torch.tensor(
                source_positions,
                dtype=torch.long,
                device=source_packets.device,
            ),
            repair_wire_positions=torch.tensor(
                repair_positions,
                dtype=torch.long,
                device=source_packets.device,
            ),
            budget_bytes=budget_bytes,
            source_packet_bytes=self.source_packet_bytes,
            wire_packet_bytes=self.wire_packet_bytes,
            redundancy_ratio=redundancy_ratio,
            block_source_packets=self.block_source_packets,
        )

    def decode(
        self,
        plan: PriorityBlockFECPlan,
        receive_mask: torch.Tensor,
        fill_value: int = 0,
    ) -> PriorityBlockFECDecodeResult:
        receive_mask = torch.as_tensor(
            receive_mask,
            dtype=torch.bool,
            device=plan.wire_packets.device,
        ).flatten()
        if int(receive_mask.numel()) != plan.num_encoded_packets:
            raise ValueError("Block transport receive mask length mismatch")

        recovered = torch.full_like(plan.source_packets, int(fill_value))
        direct = torch.zeros(
            plan.num_source_packets,
            dtype=torch.bool,
            device=recovered.device,
        )
        fec_recovered = torch.zeros_like(direct)
        block_records = []

        for scheduled in plan.blocks:
            local_receive = receive_mask[
                scheduled.encoded_start:scheduled.encoded_end
            ]
            local = self.codec.decode_block(
                scheduled.encoded,
                receive_mask=local_receive,
                fill_value=fill_value,
            )
            start = scheduled.global_source_start
            end = start + scheduled.num_source_packets
            recovered[start:end] = local.recovered_packets
            direct[start:end] = local.direct_received_source_mask
            fec_recovered[start:end] = local.fec_recovered_source_mask
            block_records.append({
                "block_id": int(scheduled.block_id),
                "block_role": str(scheduled.role),
                "num_source_packets": int(scheduled.num_source_packets),
                "num_repair_packets": int(scheduled.num_repair_packets),
                "num_received_encoded_packets": int(local_receive.sum().item()),
                "num_direct_received_source_packets": int(
                    local.direct_received_source_mask.sum().item()
                ),
                "num_fec_recovered_source_packets": int(
                    local.fec_recovered_source_mask.sum().item()
                ),
                "full_recovery": bool(local.full_recovery),
            })

        recovered_mask = direct | fec_recovered
        return PriorityBlockFECDecodeResult(
            recovered_packets=recovered,
            recovered_source_mask=recovered_mask,
            direct_received_source_mask=direct,
            fec_recovered_source_mask=fec_recovered,
            missing_source_mask=~recovered_mask,
            receive_mask=receive_mask,
            loss_mask=~receive_mask,
            info={
                "codec": "raptorq",
                "standard": "RFC6330",
                "scheduling": SCHEDULING_MODE,
                "redundancy_policy": REDUNDANCY_POLICY,
                "blocks": block_records,
            },
        )
