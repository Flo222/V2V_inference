# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import random
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets.cosdh_dataset_builder import build_dataset_cosdh as build_dataset
from opencood.utils import eval_utils
from opencood.utils import eval_utils_v2xreal
import opencood.data_utils
from opencood.visualization import vis_utils_cosdh as vis_utils, my_vis, simple_vis_cosdh as simple_vis
from opencood.tools import train_utils_cosdh as train_utils, inference_utils_cosdh as inference_utils
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--epoch', default=-1, type=int, help="epoch used")
    opt = parser.parse_args()
    return opt


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_vis_range(hypes):
    postprocess_cfg = hypes.get('postprocess', {})
    if 'gt_range' in postprocess_cfg and postprocess_cfg['gt_range']:
        return postprocess_cfg['gt_range']

    anchor_args = postprocess_cfg.get('anchor_args', {})
    if 'cav_lidar_range' in anchor_args and anchor_args['cav_lidar_range']:
        return anchor_args['cav_lidar_range']

    model_args = hypes.get('model', {}).get('args', {})
    return model_args.get('lidar_range', None)


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single', 'intermediatelate'] 

    hypes = yaml_utils.load_yaml(None, opt)

    # Prefer test_dir for formal testing.  Some older CoSDH/V2X-Real configs
    # only contain validate_dir; keep them runnable instead of crashing with a
    # KeyError.
    if 'test_dir' in hypes and hypes['test_dir']:
        hypes['validate_dir'] = hypes['test_dir']
        eval_data_dir = hypes['test_dir']
    else:
        eval_data_dir = hypes.get('validate_dir', hypes.get('root_dir', ''))
        print('[CoSDH inference] test_dir not found; using validate_dir/root_dir:', eval_data_dir)

    if "OPV2V" in str(eval_data_dir) or "opv2v" in str(eval_data_dir) or "v2xsim" in str(eval_data_dir):
        if 'test_dir' in hypes and hypes['test_dir']:
            assert "test" in hypes['validate_dir'].lower()
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model, epoch=opt.epoch)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting random seed
    set_random_seed(303)
    
    if hypes['fusion']['core_method'] != opt.fusion_method and opt.fusion_method == "intermediatelate":
        print(f"Change the fusion method in dataset config to {opt.fusion_method}")
        hypes['fusion']['core_method'] = opt.fusion_method
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation.
    # V2X-Real is multi-class and should report per-class AP + mAP,
    # matching inference_v2xreal.py / inference_coopdiff_v2xreal.py.
    post_core = str(hypes.get("postprocess", {}).get("core_method", ""))
    num_class = int(hypes.get("model", {}).get("args", {}).get("num_class", 1))
    is_v2xreal_eval = ("V2XReal" in post_core) or (num_class > 1)

    if is_v2xreal_eval:
        result_stat = {}
        for class_name in opencood.data_utils.SUPER_CLASS_MAP.keys():
            result_stat[class_name] = {}
            for iou_threshold in [0.3, 0.5, 0.7]:
                result_stat[class_name][iou_threshold] = {
                    "tp": [], "fp": [], "gt": 0
                }
        print("[CoSDH inference] Using V2X-Real multi-class evaluation.")
    else:
        result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                       0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                       0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
        print("[CoSDH inference] Using OPV2V single-class evaluation.")

    infer_info = opt.fusion_method + opt.note
    vis_range = _resolve_vis_range(hypes)

    for i, batch_data in tqdm(enumerate(data_loader)):

        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            if opt.fusion_method in ('late', 'intermediatelate'):
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            
            if is_v2xreal_eval:
                gt_label_tensor = infer_result.get("gt_label_tensor", None)
                if pred_box_tensor is not None and pred_score is not None and gt_label_tensor is not None:
                    # V2X-Real pred_score is expected to be [N, 2]:
                    # column 0 = score, column 1 = 1-indexed class id.
                    if pred_score.dim() == 1:
                        raise RuntimeError("V2X-Real eval expects pred_score with shape [N, 2], got 1-D score.")
                    for class_id, class_name in enumerate(result_stat.keys()):
                        class_id += 1
                        keep_index_pred = pred_score[:, -1] == class_id
                        keep_index_gt = gt_label_tensor == class_id
                        for iou_threshold in result_stat[class_name].keys():
                            eval_utils_v2xreal.caluclate_tp_fp(
                                pred_box_tensor[keep_index_pred, ...],
                                pred_score[keep_index_pred, 0],
                                gt_box_tensor[keep_index_gt, ...],
                                result_stat[class_name],
                                iou_threshold)
                elif gt_label_tensor is None:
                    raise RuntimeError("V2X-Real eval needs gt_label_tensor, but inference result does not contain it.")
            else:
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "lidar_agent_record": lidar_agent_record})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                """
                If you want 3D visualization, uncomment lines below
                """
                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     vis_range,
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                if vis_range is not None:
                    simple_vis.visualize(infer_result,
                                        batch_data['ego'][
                                            'origin_lidar'][0],
                                        vis_range,
                                        vis_save_path,
                                        method='bev',
                                        left_hand=left_hand)
        torch.cuda.empty_cache()
    
    print(f"total frame: {i + 1}")
    if is_v2xreal_eval:
        eval_utils_v2xreal.eval_final_results(result_stat, opt.model_dir)
    else:
        eval_utils.eval_final_results(result_stat, opt.model_dir, infer_info)

if __name__ == '__main__':
    main()
