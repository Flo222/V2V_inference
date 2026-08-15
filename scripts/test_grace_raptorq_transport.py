#!/usr/bin/env python
from __future__ import annotations

import unittest

import torch

from opencood.methods.arce.priority_block_fec_transport import (
    PriorityBlockFECTransport,
)
from opencood.communication.transport.fec.fec_raptorq import (
    GRACE_BLOCK_HEADER_BYTES,
    RAPTORQ_PAYLOAD_ID_BYTES,
    RaptorQBlockCodec,
    require_raptorq_backend,
)


def source_packets(count: int, width: int = 1024) -> torch.Tensor:
    values = torch.arange(count * width, dtype=torch.int64) % 251
    return values.to(torch.uint8).reshape(count, width)


class RaptorQTransportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_raptorq_backend()

    def test_systematic_packets_preserve_source_order(self):
        source = source_packets(20)
        encoded = RaptorQBlockCodec(1024).encode_block(source, 5, block_id=3)
        self.assertEqual(tuple(encoded.wire_packets.shape), (25, 1032))
        payload_start = GRACE_BLOCK_HEADER_BYTES + RAPTORQ_PAYLOAD_ID_BYTES
        self.assertTrue(torch.equal(
            encoded.wire_packets[:20, payload_start:],
            source,
        ))

    def test_rfc6330_recovers_lost_source_symbols_exactly(self):
        source = source_packets(20)
        codec = RaptorQBlockCodec(1024)
        encoded = codec.encode_block(source, 12, block_id=0)
        receive = torch.ones(encoded.num_encoded_packets, dtype=torch.bool)
        receive[1] = False
        receive[7] = False
        result = codec.decode_block(encoded, receive)
        self.assertTrue(result.full_recovery)
        self.assertTrue(torch.equal(result.recovered_packets, source))
        self.assertEqual(result.num_fec_recovered_source_packets, 2)

    def test_priority_blocks_share_budget_with_repair_packets(self):
        source = source_packets(100)
        transport = PriorityBlockFECTransport(
            source_packet_bytes=1024,
            block_source_packets=20,
        )
        budget = 38 * transport.wire_packet_bytes
        plan = transport.encode_under_budget(source, budget, 0.25)

        self.assertEqual(plan.actual_transmitted_bytes, budget)
        self.assertEqual(plan.num_protected_source_packets, 28)
        self.assertEqual(plan.num_tail_source_packets, 3)
        self.assertEqual(plan.num_admitted_source_packets, 31)
        self.assertEqual(plan.num_repair_packets, 7)
        self.assertEqual(len(plan.blocks), 3)
        self.assertEqual(plan.num_protected_blocks, 2)
        self.assertEqual(plan.num_tail_blocks, 1)
        self.assertEqual(
            [block.role for block in plan.blocks],
            ["protected", "protected", "best_effort_tail"],
        )
        self.assertAlmostEqual(plan.protected_redundancy_ratio, 0.25)
        self.assertAlmostEqual(plan.overall_redundancy_ratio, 7.0 / 31.0)
        self.assertEqual(
            plan.source_wire_positions.tolist(),
            list(range(20)) + list(range(25, 33)) + list(range(35, 38)),
        )
        self.assertEqual(
            plan.repair_wire_positions.tolist(),
            list(range(20, 25)) + list(range(33, 35)),
        )

        receive = torch.ones(plan.num_encoded_packets, dtype=torch.bool)
        receive[0] = False
        receive[25] = False
        receive[35] = False
        decoded = transport.decode(plan, receive)
        self.assertTrue(torch.equal(decoded.recovered_packets[:28], source[:28]))
        self.assertTrue(decoded.recovered_source_mask[:28].all().item())
        self.assertFalse(decoded.recovered_source_mask[28].item())
        self.assertTrue(decoded.recovered_source_mask[29:31].all().item())
        self.assertFalse(decoded.recovered_source_mask[31:].any().item())

    def test_bad_budget_keeps_exact_protection_and_uses_tail_slots(self):
        source = source_packets(100)
        transport = PriorityBlockFECTransport(1024, 20)
        budget = 12 * transport.wire_packet_bytes
        expected = {
            0.10: (10, 1, 1),
            0.25: (8, 2, 2),
            0.60: (5, 4, 3),
        }
        for rho, (protected, tail, repair) in expected.items():
            plan = transport.encode_under_budget(source, budget, rho)
            self.assertEqual(plan.num_protected_source_packets, protected)
            self.assertEqual(plan.num_tail_source_packets, tail)
            self.assertEqual(plan.num_repair_packets, repair)
            self.assertEqual(plan.num_encoded_packets, 12)
            self.assertAlmostEqual(plan.protected_redundancy_ratio, rho)
            self.assertLess(plan.overall_redundancy_ratio, rho)

    def test_markov_budgets_are_hard_upper_bounds(self):
        source = source_packets(1000)
        transport = PriorityBlockFECTransport(1024, 20)
        for budget in (337500.0, 62500.0, 12500.0):
            for rho in (0.10, 0.25, 0.60):
                plan = transport.encode_under_budget(source, budget, rho)
                self.assertLessEqual(plan.actual_transmitted_bytes, budget)
                self.assertGreater(plan.num_repair_packets, 0)
                self.assertEqual(
                    plan.num_encoded_packets,
                    plan.num_admitted_source_packets + plan.num_repair_packets,
                )
                self.assertAlmostEqual(
                    plan.protected_redundancy_ratio,
                    rho,
                )
                self.assertLessEqual(plan.overall_redundancy_ratio, rho)
                self.assertTrue(plan.admitted_source_mask[:plan.num_admitted_source_packets].all())
                self.assertFalse(plan.admitted_source_mask[plan.num_admitted_source_packets:].any())


if __name__ == "__main__":
    unittest.main()
