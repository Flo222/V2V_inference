# -*- coding: utf-8 -*-
"""Static and one-batch checks for CoopDiff + V2X-Real."""

import argparse

import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes_yaml', required=True)
    parser.add_argument('--check_dataset', action='store_true')
    parser.add_argument('--forward', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--sample_idx', type=int, default=0)
    args = parser.parse_args()

    hypes = yaml_utils.load_yaml(args.hypes_yaml, None)
    print('yaml:', args.hypes_yaml)
    print('model.core_method:', hypes['model']['core_method'])
    print('fusion.core_method:', hypes['fusion']['core_method'])
    print('postprocess.core_method:', hypes['postprocess']['core_method'])
    print('loss.core_method:', hypes['loss']['core_method'])

    num_class = int(hypes['model']['args'].get('num_class', 3))
    anchor_number = int(hypes['model']['args'].get('anchor_number', 2))
    expected_psm = anchor_number * num_class * num_class
    expected_rm = 7 * anchor_number * num_class
    print('expected psm/rm channels:', expected_psm, expected_rm)

    model = train_utils.create_model(hypes)

    # CoopDiff teacher-only model has only cls_head_teacher/reg_head_teacher.
    # CoopDiff student model has both student heads and teacher heads.
    checked_any_head = False

    if hasattr(model, 'cls_head') and hasattr(model, 'reg_head'):
        print('actual student cls/reg out:',
              model.cls_head.out_channels,
              model.reg_head.out_channels)
        assert model.cls_head.out_channels == expected_psm
        assert model.reg_head.out_channels == expected_rm
        checked_any_head = True

    if hasattr(model, 'cls_head_teacher') and hasattr(model, 'reg_head_teacher'):
        print('actual teacher cls/reg out:',
              model.cls_head_teacher.out_channels,
              model.reg_head_teacher.out_channels)
        assert model.cls_head_teacher.out_channels == expected_psm
        assert model.reg_head_teacher.out_channels == expected_rm
        checked_any_head = True

    assert checked_any_head, 'No detection head found in CoopDiff model.'

    if not args.check_dataset:
        print('STATIC_CHECK_OK')
        return

    dataset = build_dataset(hypes, visualize=False, train=False)
    print('dataset:', type(dataset))
    print('len:', len(dataset))
    sample = dataset[args.sample_idx]
    batch = dataset.collate_batch_test([sample])
    ego = batch['ego']
    print('batch record_len:', ego['record_len'].tolist())
    print('object_bbx_center shape:', tuple(ego['object_bbx_center'].shape))
    print('processed_lidar keys:', ego['processed_lidar'].keys())
    print('processed_lidar_paint keys:', ego['processed_lidar_paint'].keys())
    print('processed_lidar voxel_features:', tuple(ego['processed_lidar']['voxel_features'].shape))
    print('processed_lidar_paint voxel_features:', tuple(ego['processed_lidar_paint']['voxel_features'].shape))
    if 'channel_state_ids' in ego:
        print('channel_state_ids:', ego['channel_state_ids'].tolist())

    if args.forward:
        device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        if hasattr(model, 'update_epoch'):
            # Use epoch 20 for a real CoopDiff inference-style pass so diffuser path is enabled.
            model.update_epoch(20)
        batch = train_utils.to_device(batch, device)
        with torch.no_grad():
            out = model(batch['ego'])
        print('forward psm shape:', tuple(out['psm'].shape))
        print('forward rm shape:', tuple(out['rm'].shape))
        if 'comm_info' in out:
            print('comm_info keys:', sorted(out['comm_info'].keys()))

    print('BATCH_CHECK_OK')


if __name__ == '__main__':
    main()
