#!/usr/bin/env python
from __future__ import annotations

import unittest
from types import SimpleNamespace

from opencood.methods.arce.cost.communication_cost_estimator import (
    estimate_byte_stream_fec_cost,
)
from opencood.methods.arce.transport_policy.priority_fec_scheduler import (
    exact_redundancy_group,
    exact_repair_packets_for_block,
    largest_exact_protected_source_block,
    plan_exact_protected_prefix,
)


class RaptorQPlannerTest(unittest.TestCase):
    def test_exact_packet_groups_match_configured_ratio(self):
        self.assertEqual(exact_redundancy_group(0.10), (10, 1))
        self.assertEqual(exact_redundancy_group(0.25), (4, 1))
        self.assertEqual(exact_redundancy_group(0.60), (5, 3))
        self.assertEqual(exact_repair_packets_for_block(20, 0.10), 2)
        self.assertEqual(exact_repair_packets_for_block(20, 0.25), 5)
        self.assertEqual(exact_repair_packets_for_block(20, 0.60), 12)

    def test_block_admission_uses_only_complete_exact_groups(self):
        self.assertEqual(
            largest_exact_protected_source_block(12, 100, 20, 0.60),
            5,
        )
        self.assertEqual(
            largest_exact_protected_source_block(25, 100, 20, 0.25),
            20,
        )
        self.assertEqual(
            largest_exact_protected_source_block(10, 100, 20, 0.10),
            0,
        )

    def test_non_integral_protection_block_is_rejected(self):
        with self.assertRaises(ValueError):
            exact_repair_packets_for_block(9, 0.25)
        with self.assertRaises(ValueError):
            exact_repair_packets_for_block(7, 0.60)

    def test_bad_budget_uses_remaining_slots_for_source_tail(self):
        expected = {
            0.10: ([10], 1),
            0.25: ([8], 2),
            0.60: ([5], 4),
        }
        for rho, expected_plan in expected.items():
            self.assertEqual(
                plan_exact_protected_prefix(100, 12, 20, rho),
                expected_plan,
            )

    def test_estimator_counts_rfc6330_wire_metadata(self):
        action = SimpleNamespace(
            is_no_send=False,
            send=1,
            quant_mode="int8",
            fec_type="raptorq",
            redundancy_ratio=0.25,
        )
        result = estimate_byte_stream_fec_cost(
            feature_shape=(64, 320, 1),
            action=action,
            budget_bytes=62500.0,
            packet_size_bytes=1024,
            metadata_bytes_per_packet=0,
            raw_feature_bytes_fp32_fn=lambda shape: float(
                shape[0] * shape[1] * shape[2] * 4
            ),
            quant_ratio_to_fp32={"int8": 0.25},
            raptorq_block_source_packets=20,
            raptorq_metadata_bytes_per_packet=8,
        )
        self.assertEqual(result["fec_type"], "raptorq")
        self.assertEqual(result["wire_packet_size_bytes"], 1032)
        self.assertEqual(result["source_packets"], 20)
        self.assertEqual(result["parity_packets"], 5)
        self.assertEqual(result["encoded_packets"], 25)


if __name__ == "__main__":
    unittest.main()
