# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat,
                    iou_thresh):
    """
    Calculate TP/FP for one frame.

    Robust for mixed CoSDH / V2X-Real cases where det_score may be [N, 1]
    or may contain more entries than final decoded det_boxes.
    """
    fp = []
    tp = []
    gt = gt_boxes.shape[0]

    if det_boxes is not None:
        # convert bounding boxes and scores to numpy
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        det_score = np.asarray(det_score).reshape(-1)

        # Number of final decoded boxes. Scores must correspond to these boxes.
        num_det = int(det_boxes.shape[0])

        # V2X-Real mixed fusion may return score count != box count.
        # Align to the available final boxes to avoid invalid indexing.
        if det_score.shape[0] != num_det:
            if not hasattr(caluclate_tp_fp, "_warned_score_box_mismatch"):
                print("[eval_utils] warning: det_score length ({}) != det_boxes length ({}). "
                      "Aligning scores to final boxes for evaluation.".format(
                          det_score.shape[0], num_det))
                caluclate_tp_fp._warned_score_box_mismatch = True

            n = min(det_score.shape[0], num_det)
            det_score = det_score[:n]
            det_boxes = det_boxes[:n]
            num_det = n

        if num_det == 0:
            result_stat[iou_thresh]['gt'] += gt
            return

        # sort prediction boxes by score
        score_order_descend = np.argsort(-det_score).reshape(-1)
        det_score = det_score[score_order_descend]

        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_idx = int(score_order_descend[i])

            # Extra guard: should not happen after alignment, but keeps eval robust.
            if det_idx < 0 or det_idx >= len(det_polygon_list):
                continue

            det_polygon = det_polygon_list[det_idx]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

        result_stat[iou_thresh]['score'] += det_score.tolist()
        result_stat[iou_thresh]['fp'] += fp
        result_stat[iou_thresh]['tp'] += tp

    result_stat[iou_thresh]['gt'] += gt


def calculate_ap(result_stat, iou, global_sort_detections):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
        
    iou : float
        The threshold of iou.

    global_sort_detections : bool
        Whether to sort the detection results globally.
    """
    iou_5 = result_stat[iou]

    if global_sort_detections:
        fp = np.array(iou_5['fp'])
        tp = np.array(iou_5['tp'])
        score = np.array(iou_5['score'])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()
        
    else:
        fp = iou_5['fp']
        tp = iou_5['tp']
        assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, global_sort_detections):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30, global_sort_detections)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50, global_sort_detections)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70, global_sort_detections)

    dump_dict.update({'ap30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    
    output_file = 'eval.yaml' if not global_sort_detections else 'eval_global_sort.yaml'
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))

    print('The Average Precision at IOU 0.3 is %.3f, '
          'The Average Precision at IOU 0.5 is %.3f, '
          'The Average Precision at IOU 0.7 is %.3f' % (ap_30, ap_50, ap_70))
