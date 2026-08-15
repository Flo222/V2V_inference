#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy
from pathlib import Path
import yaml


def final_arce_extension_cfg(backbone: str):
    source = 'where2comm_mask' if backbone == 'where2comm' else 'feature_activation'
    return {
        'enabled': True,
        'mode': 'dc2mab',
        'policy': 'dc2mab_sender_ego',
        'link_scope': 'non_ego',
        'seed': 2026,
        'action_space': {'type': 'final_36', 'send': [0,1], 'quant': ['fp16','int8','int4'], 'rho': [0.0,0.25,0.5], 'cache': [0,1], 'proposal_send_only': True},
        'context': {'dim': 6, 'use_complementarity': backbone == 'where2comm'},
        'c2mab': {'context_dim': 6, 'ridge_lambda': 1.0, 'discount': 0.97, 'exploration_beta': 1.0},
        'ego_oracle': {'type': 'diversity_greedy_knapsack', 'lambda_comp': 0.5 if backbone == 'where2comm' else 0.0, 'lambda_red': 0.5 if backbone == 'where2comm' else 0.0, 'budget_scope': 'global_sum_link', 'max_budget_cavs': 4},
        'channel': {'mode': 'markov', 'initial_state': 'medium', 'seed': 2026, 'states': ['good','medium','bad'], 'transition_matrix': [[0.90,0.09,0.01],[0.12,0.82,0.06],[0.02,0.23,0.75]], 'profiles': {'good': {'bandwidth_mbps': 27.0, 'loss_rate': 0.03}, 'medium': {'bandwidth_mbps': 5.0, 'loss_rate': 0.12}, 'bad': {'bandwidth_mbps': 1.0, 'loss_rate': 0.28}}},
        'scheduler': {'fps': 10, 'tx_window_ms': 100.0, 'frame_interval_ms': 100.0, 'budget_scope': 'global_sum_link', 'per_link_budget': True},
        'packetizer': {'mode': 'block', 'block_size': [4,4], 'pad_boundary': True},
        'patch_selection': {'enabled': True, 'source': source, 'mask_threshold': 0.05, 'patch_selector': 'score_per_byte', 'ranking': 'score_per_byte', 'min_effective_patch_ratio': 0.2, 'score': {'lambda_mask': 1.0 if source == 'where2comm_mask' else 0.0, 'lambda_activation': 0.2 if source == 'where2comm_mask' else 1.0, 'lambda_complementarity': 0.3 if source == 'where2comm_mask' else 0.0}},
        'latency': {'enabled': True, 'deadline_ms': 100.0, 'frame_interval_ms': 100.0, 'late_policy': 'allow_stale', 'tau_stale_ms': 300.0},
        'recovery': {'temporal_cache': True, 'spatial_interpolation': True, 'zero_fill': True, 'temporal_fusion': {'enabled': True, 'beta': 5.0, 'tau_stale_ms': 300.0, 'use_delay_penalty': True}},
        'fec': {'apply_to': 'selected_patches_only'},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--backbone', choices=['v2xvit','rocooper','coopdiff','where2comm'], required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    with open(args.base, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg['name'] = f"{cfg.get('name','model')}_{args.backbone}_ours_extension"
    arce = final_arce_extension_cfg(args.backbone)
    cfg.setdefault('model', {}).setdefault('args', {})['arce'] = arce
    cfg['arce'] = copy.deepcopy(arce)
    # For V2X-ViT use existing ARCE wrapper if available. RoCooper/CoopDiff require their own wrapper integration.
    if args.backbone == 'v2xvit':
        cfg['model']['core_method'] = 'point_pillar_transformer_opv2v_arce'
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(args.out)

if __name__ == '__main__':
    main()
