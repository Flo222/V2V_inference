#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from opencood.methods.arce.executors.fixed_executor import ARCEFixedComm


def make_cfg(out_dir: str):
    return {
        'arce': {
            'enabled': True,
            'mode': 'fixed',
            'policy': 'fixed',
            'link_scope': 'non_ego',
            'transport_mode': 'compact_sparse',
            'compact_sparse': {
                'enabled': True,
                'threshold': 0.5,
                'budget_aware_topk': False,
                'max_tokens': -1,
            },
            'fixed_action': {
                'send': 1,
                'quant': 'fp32',
                'quant_mode': 'fp32',
                'rho': 0.0,
                'redundancy_ratio': 0.0,
                'cache': 0,
                'cache_enabled': 0,
                'fec_type': 'none',
            },
            'quantization': {
                'enabled': False,
                'mode': 'fp32',
                'granularity': 'per_tensor',
                'compute_error': True,
            },
            'packetizer': {'packet_size_bytes': 32},
            'scheduler': {
                'budget_source': 'system_budget',
                'budget_scope': 'system_equal_split',
                'system_budget_mbps': 100000.0,
                'tx_window_ms': 100.0,
            },
            'channel': {
                'mode': 'fixed',
                'bernoulli_loss_rates': {'good': 0.0, 'medium': 0.0, 'bad': 0.0},
            },
            'compression_audit': {
                'enabled': True,
                'strict': True,
                'experiment_name': 'experiment2_compact_accounting_unit',
                'output_dir': out_dir,
                'file_name': 'compression_budget_audit.jsonl',
                'save_tensors': False,
                'require_no_budget_drop': False,
                'require_no_bernoulli_loss': True,
                'require_no_fec_parity': True,
                'require_all_source_transmitted': False,
                'require_quant_equals_recovered': False,
            },
        }
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix='arce_exp2_compact_') as out_dir:
        comm = ARCEFixedComm(make_cfg(out_dir))
        feature = torch.arange(4 * 4 * 5, dtype=torch.float32).reshape(4, 4, 5)
        # Select exactly ten of twenty spatial tokens, producing [4, 10, 1].
        mask = torch.zeros(4, 5)
        mask.reshape(-1)[:10] = 1.0
        _, record = comm.communicate_feature(
            feature,
            message_mask=mask,
            link_id=(0, 0, 1),
            frame_id='scene/frame0001',
            agent_index=1,
            ego_index=0,
            channel_state='good',
            # Compact payload: 4*10*4=160 bytes -> 5 source packets.
            # Send only first three source packets.
            budget_bytes=3 * 32,
            update_cache=False,
        )

        assert record['compact_sparse']['num_tokens'] == 10
        size = record['size']
        assert size['actual_num_source_packets'] == 5, size
        assert size['actual_num_transmitted_source_packets'] == 3, size
        assert size['num_source_dropped_by_budget'] == 2, size
        assert size['actual_num_transmitted_parity_packets'] == 0, size
        assert size['num_parity_dropped_by_budget'] == 0, size

        audit_path = os.path.join(out_dir, 'compression_budget_audit.jsonl')
        with open(audit_path, 'r', encoding='utf-8') as f:
            audit = json.loads(next(line for line in f if line.strip()))

        assert audit['source_payload_before_quant']['shape'] == [4, 10, 1], audit
        assert audit['budget_accounting']['source'] == 'runtime_locals', audit
        assert audit['budget_accounting']['runtime_complete'] is True, audit
        assert audit['sizes']['bandwidth_budget_bytes'] == 96.0, audit
        assert audit['packet_outcome']['num_source_packets'] == 5, audit
        assert audit['packet_outcome']['num_transmitted_source_packets'] == 3, audit
        assert audit['packet_outcome']['num_source_dropped_by_budget'] == 2, audit
        assert audit['packet_outcome']['source_tx_ratio'] == 0.6, audit
        assert audit['sanity']['budget_packet_accounting_valid'] is True, audit
        assert audit['sanity']['runtime_budget_accounting_complete'] is True, audit
        assert audit['sanity']['passed'] is True, audit

    print('Experiment 2 compact-sparse budget-accounting smoke test passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
