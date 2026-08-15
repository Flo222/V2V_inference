# -*- coding: utf-8 -*-
"""Create a CoopDiff-V2XReal student initialization from a trained teacher.

Usage:
  python scripts/make_mapped_student_v2xreal_from_teacher.py \
    --teacher_dir opencood/logs/point_pillar_diffteacher_v2xreal_vc_xxx \
    --out_dir opencood/logs/point_pillar_diffstudent_v2xreal_vc_mapped \
    --epoch 20

The script copies teacher branch weights exactly and maps *_teacher modules to
student modules when tensor shapes match.  It saves net_epoch1.pth so OpenCOOD's
train.py --model_dir can load it and continue training.
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict

import torch
import yaml


def load_state(path):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict):
        for key in ['state_dict', 'model_state_dict', 'model']:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def can_copy(src, dst):
    return torch.is_tensor(src) and torch.is_tensor(dst) and tuple(src.shape) == tuple(dst.shape)


def try_partial_copy_teacher_vfe_to_student(src_key, src_val, dst_key, dst_val):
    """Copy 5D teacher VFE weights to 4D student VFE when possible.

    Teacher CoopDiff uses painted point features:
        [x, y, z, intensity, paint_mask]  -> 5 dims
    Student uses raw point features:
        [x, y, z, intensity]              -> 4 dims

    For the first PFN linear/kernel weight, copy the overlapping first 4 input
    channels and keep the remaining destination initialization unchanged.
    """
    if not (torch.is_tensor(src_val) and torch.is_tensor(dst_val)):
        return None

    if 'pillar_vfe_teacher' not in src_key or 'pillar_vfe' not in dst_key:
        return None

    if src_val.ndim != dst_val.ndim:
        return None

    src_shape = tuple(src_val.shape)
    dst_shape = tuple(dst_val.shape)

    # Common Linear layout: [out_dim, in_dim], teacher in_dim=5, student in_dim=4.
    if src_val.ndim == 2 and src_shape[0] == dst_shape[0] and src_shape[1] >= dst_shape[1]:
        out = dst_val.detach().clone()
        out[:, :dst_shape[1]] = src_val[:, :dst_shape[1]]
        return out

    # Generic fallback: same leading dims, last dim teacher=5/student=4.
    if src_shape[:-1] == dst_shape[:-1] and src_shape[-1] >= dst_shape[-1]:
        out = dst_val.detach().clone()
        out[..., :dst_shape[-1]] = src_val[..., :dst_shape[-1]]
        return out

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--validate_dir', default=None)
    parser.add_argument('--student_epoches', type=int, default=20)
    args = parser.parse_args()

    teacher_dir = Path(args.teacher_dir)
    if args.epoch is None:
        ckpts = sorted(teacher_dir.glob('net_epoch*.pth'))
        if not ckpts:
            raise FileNotFoundError('no net_epoch*.pth in %s' % teacher_dir)
        def ep(p): return int(p.stem.split('epoch')[-1])
        teacher_ckpt = max(ckpts, key=ep)
    else:
        teacher_ckpt = teacher_dir / ('net_epoch%d.pth' % args.epoch)
    teacher_cfg = teacher_dir / 'config.yaml'
    out_dir = Path(args.out_dir)

    if not teacher_ckpt.exists():
        raise FileNotFoundError(teacher_ckpt)
    if not teacher_cfg.exists():
        raise FileNotFoundError(teacher_cfg)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.load(teacher_cfg.read_text(), Loader=yaml.Loader)
    cfg['name'] = out_dir.name
    cfg['model']['core_method'] = 'point_pillar_diff_stu_v2xreal'
    cfg['fusion']['core_method'] = 'IntermediateFusionDatasetCoopdiffV2XReal'
    cfg['loss']['core_method'] = 'point_pillar_loss_v2xreal'
    cfg['postprocess']['core_method'] = 'VoxelPostprocessorV2XReal'
    cfg.setdefault('dataset_mode', 'vc')
    if args.root_dir is not None:
        cfg['root_dir'] = args.root_dir
    if args.validate_dir is not None:
        cfg['validate_dir'] = args.validate_dir
    cfg.setdefault('train_params', {})
    cfg['train_params']['epoches'] = int(args.student_epoches)
    cfg['train_params'].setdefault('late_loss_weight', 10.0)
    cfg['train_params']['batch_size'] = 1
    cfg['train_params']['max_cav'] = 4
    cfg.setdefault('model', {}).setdefault('args', {})
    cfg['model']['args']['num_class'] = 3
    cfg['model']['args']['anchor_number'] = 2
    cfg['model']['args']['anchor_num'] = 2
    cfg['model']['args']['max_cav'] = 4

    cfg_path = out_dir / 'config.yaml'
    cfg_path.write_text(yaml.dump(cfg, sort_keys=False, default_flow_style=False))

    # Instantiate target student state.
    import importlib
    module = importlib.import_module('opencood.models.' + cfg['model']['core_method'])
    model_cls = getattr(module, 'PointPillarDiffStuV2xreal')
    student_model = model_cls(cfg['model']['args'])
    student_state = student_model.state_dict()
    teacher_state = load_state(teacher_ckpt)

    new_state = {}
    for k, v in student_state.items():
        new_state[k] = v.detach().clone() if torch.is_tensor(v) else v

    copied_exact = []
    copied_mapped = []
    shape_mismatch = []
    missing_target = []
    copied_keys = set()

    for src_key, src_val in teacher_state.items():
        if src_key not in new_state:
            continue
        if can_copy(src_val, new_state[src_key]):
            new_state[src_key] = src_val.detach().clone()
            copied_exact.append(src_key)
            copied_keys.add(src_key)
        elif torch.is_tensor(src_val) and torch.is_tensor(new_state[src_key]):
            shape_mismatch.append((src_key, src_key, tuple(src_val.shape), tuple(new_state[src_key].shape), 'exact'))

    for src_key, src_val in teacher_state.items():
        if '.' not in src_key:
            continue
        top, rest = src_key.split('.', 1)
        if not top.endswith('_teacher'):
            continue
        dst_key = top[:-len('_teacher')] + '.' + rest
        if dst_key not in new_state:
            missing_target.append((src_key, dst_key))
            continue
        if can_copy(src_val, new_state[dst_key]):
            new_state[dst_key] = src_val.detach().clone()
            copied_mapped.append((src_key, dst_key))
            copied_keys.add(dst_key)
        else:
            partial = try_partial_copy_teacher_vfe_to_student(
                src_key, src_val, dst_key, new_state[dst_key])
            if partial is not None:
                new_state[dst_key] = partial
                copied_mapped.append((src_key, dst_key + ' [partial_vfe_5d_to_4d]'))
                copied_keys.add(dst_key)
            elif torch.is_tensor(src_val) and torch.is_tensor(new_state[dst_key]):
                shape_mismatch.append((src_key, dst_key, tuple(src_val.shape), tuple(new_state[dst_key].shape), 'teacher_to_student'))

    # Save as net_epoch1.pth because OpenCOOD load_saved_model ignores epoch 0.
    out_ckpt = out_dir / 'net_epoch1.pth'
    torch.save(new_state, out_ckpt)

    prefix_total = defaultdict(int)
    prefix_copied = defaultdict(int)
    for k in student_state.keys():
        top = k.split('.', 1)[0]
        prefix_total[top] += 1
        if k in copied_keys:
            prefix_copied[top] += 1

    lines = []
    lines.append('teacher_ckpt: %s' % teacher_ckpt)
    lines.append('teacher_cfg: %s' % teacher_cfg)
    lines.append('student_cfg: %s' % cfg_path)
    lines.append('out_ckpt: %s' % out_ckpt)
    lines.append('')
    lines.append('student model keys: %d' % len(student_state))
    lines.append('teacher checkpoint keys: %d' % len(teacher_state))
    lines.append('copied exact: %d' % len(copied_exact))
    lines.append('copied teacher_to_student: %d' % len(copied_mapped))
    lines.append('shape mismatches: %d' % len(shape_mismatch))
    lines.append('missing mapping targets: %d' % len(missing_target))
    lines.append('')
    lines.append('Top-level copied coverage:')
    for top in sorted(prefix_total.keys()):
        lines.append('  %s: copied %d / total %d' % (top, prefix_copied[top], prefix_total[top]))
    lines.append('')
    lines.append('First 30 shape mismatches:')
    for item in shape_mismatch[:30]:
        lines.append('  %s' % (item,))
    lines.append('')
    lines.append('First 30 missing targets:')
    for item in missing_target[:30]:
        lines.append('  %s' % (item,))

    report = out_dir / 'checkpoint_mapping_report.txt'
    report.write_text('\n'.join(lines))
    print('\n'.join(lines[:120]))
    print('\n[OK] saved config:', cfg_path)
    print('[OK] saved ckpt:', out_ckpt)
    print('[OK] saved report:', report)


if __name__ == '__main__':
    main()
