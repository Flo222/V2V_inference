#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy
from pathlib import Path
import yaml


def get_arce(cfg):
    return cfg.setdefault('model', {}).setdefault('args', {}).setdefault('arce', cfg.setdefault('arce', {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    with open(args.base, 'r', encoding='utf-8') as f:
        base = yaml.safe_load(f)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    variants = {}
    # no comp
    cfg = copy.deepcopy(base); cfg['name'] = 'point_pillar_where2comm_arce_ablate_no_comp'
    arce = get_arce(cfg); arce.setdefault('context', {})['use_complementarity'] = False; arce['context']['dim'] = 5; arce.setdefault('c2mab', {})['context_dim'] = 5; arce.setdefault('ego_oracle', {})['lambda_comp'] = 0.0
    variants['ablate_no_comp'] = cfg
    # no div
    cfg = copy.deepcopy(base); cfg['name'] = 'point_pillar_where2comm_arce_ablate_no_div'
    arce = get_arce(cfg); arce.setdefault('ego_oracle', {})['lambda_red'] = 0.0
    variants['ablate_no_div'] = cfg
    # no cache
    cfg = copy.deepcopy(base); cfg['name'] = 'point_pillar_where2comm_arce_ablate_no_cache'
    arce = get_arce(cfg); arce.setdefault('action_space', {})['cache'] = [0]; arce.setdefault('recovery', {})['temporal_cache'] = False; arce.setdefault('recovery', {}).setdefault('temporal_fusion', {})['enabled'] = False
    variants['ablate_no_cache'] = cfg
    # no redundancy
    cfg = copy.deepcopy(base); cfg['name'] = 'point_pillar_where2comm_arce_ablate_no_red'
    arce = get_arce(cfg); arce.setdefault('action_space', {})['rho'] = [0.0]; arce.setdefault('fec', {})['force_none'] = True
    variants['ablate_no_red'] = cfg
    # full
    cfg = copy.deepcopy(base); cfg['name'] = 'point_pillar_where2comm_arce_ours_full'
    variants['ours_full'] = cfg

    for name, cfg in variants.items():
        p = out / f'point_pillar_where2comm_arce_{name}.yaml'
        with open(p, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(p)

if __name__ == '__main__':
    main()
