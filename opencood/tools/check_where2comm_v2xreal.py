"""Static and optional one-batch checks for Where2Comm on V2X-Real.

Usage examples:
  python opencood/tools/check_where2comm_v2xreal.py \
    --hypes_yaml opencood/hypes_yaml/v2xreal/point_pillar_where2comm_v2xreal_vc.yaml

  python opencood/tools/check_where2comm_v2xreal.py \
    --hypes_yaml opencood/hypes_yaml/v2xreal/point_pillar_where2comm_arce_c2mab_v2xreal_vc.yaml \
    --check_dataset --forward --device cuda
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes_yaml', required=True)
    parser.add_argument('--check_dataset', action='store_true',
                        help='Build V2X-Real dataset and inspect one collated batch.')
    parser.add_argument('--forward', action='store_true',
                        help='Run one forward pass. Implies --check_dataset.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--dataset_mode', default=None,
                        help='Optional override: vc/ic/v2v/i2i')
    return parser.parse_args()


def main():
    args = parse_args()
    hypes = yaml_utils.load_yaml(args.hypes_yaml)
    if args.dataset_mode is not None:
        hypes['dataset_mode'] = args.dataset_mode

    model_cfg = hypes['model']
    model_args = model_cfg['args']
    num_class = int(model_args.get('num_class', 3))
    anchor_number = int(model_args.get('anchor_number', model_args.get('anchor_num', 2)))
    expected_psm = anchor_number * num_class * num_class
    expected_rm = 7 * anchor_number * num_class

    print('yaml:', args.hypes_yaml)
    print('model.core_method:', model_cfg['core_method'])
    print('fusion.core_method:', hypes['fusion']['core_method'])
    print('postprocess.core_method:', hypes['postprocess']['core_method'])
    print('loss.core_method:', hypes['loss']['core_method'])
    print('expected psm/rm channels:', expected_psm, expected_rm)

    model = train_utils.create_model(hypes)
    print('actual cls_head/reg_head out:', model.cls_head.out_channels,
          model.reg_head.out_channels)
    assert model.cls_head.out_channels == expected_psm
    assert model.reg_head.out_channels == expected_rm

    if 'arce' in model_args:
        arce = model_args['arce']
        print('arce enabled/mode/policy:', arce.get('enabled'),
              arce.get('mode'), arce.get('policy'))
        print('arce context:', arce.get('context', {}))
        print('arce c2mab:', arce.get('c2mab', {}))
        if arce.get('enabled'):
            assert bool(arce.get('context', {}).get('include_cav_confidence', False))
            assert int(arce.get('c2mab', {}).get('context_dim', -1)) == 7

    if not (args.check_dataset or args.forward):
        print('STATIC_CHECK_OK')
        return

    root = hypes['validate_dir']
    if not os.path.exists(root):
        raise FileNotFoundError(
            'validate_dir does not exist: {}. Edit root_dir/validate_dir first.'.format(root)
        )

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=0,
                        collate_fn=dataset.collate_batch_train,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)
    batch_data = next(iter(loader))
    ego = batch_data['ego']
    print('batch record_len:', ego['record_len'].tolist())
    print('prior_encoding shape:', tuple(ego['prior_encoding'].shape))
    if 'channel_state_ids' in ego:
        print('channel_state_ids:', ego['channel_state_ids'].tolist())
        print('channel_delay_ms:', ego.get('channel_delay_ms').tolist())
        print('channel_delay_slots:', ego.get('channel_delay_slots').tolist())
    else:
        raise KeyError('channel_state_ids missing from V2X-Real batch')

    if args.forward:
        device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()
        batch_data = train_utils.to_device(batch_data, device)
        with torch.no_grad():
            out = model(batch_data['ego'])
        print('forward psm shape:', tuple(out['psm'].shape))
        print('forward rm shape:', tuple(out['rm'].shape))
        assert out['psm'].shape[1] == expected_psm
        assert out['rm'].shape[1] == expected_rm
        if 'comm_info' in out:
            print('comm_info keys:', sorted(out['comm_info'].keys()))
    print('BATCH_CHECK_OK')


if __name__ == '__main__':
    main()
