# -*- coding: utf-8 -*-
"""Legacy-equivalent candidate-level Late post-processing for OPV2V CoSDH.

The helper reproduces VoxelPostprocessor.post_process in two stages:
1. sender-side anchor decode and confidence filtering;
2. receiver-side projection, cleanup, global NMS, and range filtering.

Only non-ego candidates inside the dataset communication range cross the
Ideal byte boundary.  No beta change, new threshold, top-k, or extra NMS is
introduced.
"""
from __future__ import print_function

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F

from opencood.utils import box_utils


def _distance_to_ego(cav_content):
    matrix = cav_content["transformation_matrix"]
    x = float(matrix[0][3])
    y = float(matrix[1][3])
    return math.sqrt(x * x + y * y)


def communication_range_cav_ids(dataset, data_dict):
    """Return CAV ids used by the original dataset post-process."""
    result = []
    limit = float(dataset.params["comm_range"])
    for cav_id, cav_content in data_dict.items():
        if _distance_to_ego(cav_content) <= limit:
            result.append(str(cav_id))
    return result


def _ensure_output_aliases(output_dict):
    for output in output_dict.values():
        if not isinstance(output, dict):
            continue
        if "psm" not in output and "cls_preds" in output:
            output["psm"] = output["cls_preds"]
        if "rm" not in output and "reg_preds" in output:
            output["rm"] = output["reg_preds"]
        if "dm" not in output and "dir_preds" in output:
            output["dm"] = output["dir_preds"]


def _prepare_local_candidates(post_processor, cav_content, cav_output):
    """Match VoxelPostprocessor lines before corner conversion/projection."""
    anchor_box = cav_content["anchor_box"]
    prob = cav_output["psm"]
    prob = torch.sigmoid(prob.permute(0, 2, 3, 1))
    prob = prob.reshape(1, -1)
    reg = cav_output["rm"]
    batch_box3d = post_processor.delta_to_boxes3d(reg, anchor_box)
    threshold = post_processor.params["target_args"]["score_threshold"]
    mask = torch.gt(prob, threshold).view(1, -1)
    mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
    if int(batch_box3d.shape[0]) != 1:
        raise RuntimeError(
            "candidate Late inference expects batch size 1, got {}".format(
                int(batch_box3d.shape[0])
            )
        )
    boxes_local = torch.masked_select(
        batch_box3d[0], mask_reg[0]
    ).view(-1, 7)
    scores = torch.masked_select(prob[0], mask[0])
    return boxes_local, scores


def _finalize_candidates(post_processor, projected_boxes, scores):
    """Match the original concatenation, cleanup, NMS, and range filtering."""
    if not projected_boxes:
        return None, None
    pred_box3d_tensor = torch.vstack(projected_boxes)
    scores = torch.cat(scores, dim=0)

    keep_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
    keep_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
    keep = torch.logical_and(keep_1, keep_2)
    pred_box3d_tensor = pred_box3d_tensor[keep]
    scores = scores[keep]

    keep = box_utils.nms_rotated(
        pred_box3d_tensor,
        scores,
        post_processor.params["nms_thresh"],
    )
    pred_box3d_tensor = pred_box3d_tensor[keep]
    scores = scores[keep]

    keep = box_utils.get_mask_for_boxes_within_range_torch(
        pred_box3d_tensor
    )
    pred_box3d_tensor = pred_box3d_tensor[keep, :, :]
    scores = scores[keep]
    if int(scores.shape[0]) != int(pred_box3d_tensor.shape[0]):
        raise RuntimeError("candidate boxes/scores count mismatch")
    return pred_box3d_tensor, scores


def candidate_post_process_ideal(dataset, data_dict, output_dict, transport):
    """Run exact candidate-level Ideal transport and original final NMS."""
    _ensure_output_aliases(output_dict)
    selected_ids = communication_range_cav_ids(dataset, data_dict)
    projected_boxes = []
    projected_scores = []

    for cav_id in selected_ids:
        if cav_id not in output_dict:
            raise KeyError("CAV {} missing from output_dict".format(cav_id))
        cav_content = data_dict[cav_id]
        boxes_local, scores = _prepare_local_candidates(
            dataset.post_processor,
            cav_content,
            output_dict[cav_id],
        )
        if cav_id != "ego":
            boxes_local, scores = transport.roundtrip_late_candidates(
                boxes_local,
                scores,
                cav_id=cav_id,
            )
        if int(boxes_local.shape[0]) == 0:
            continue
        corners = box_utils.boxes_to_corners_3d(
            boxes_local,
            order=dataset.post_processor.params["order"],
        )
        projected = box_utils.project_box3d(
            corners,
            cav_content["transformation_matrix"],
        )
        projected_boxes.append(projected)
        projected_scores.append(scores)

    pred_box_tensor, pred_score = _finalize_candidates(
        dataset.post_processor,
        projected_boxes,
        projected_scores,
    )
    data_dict_gt = OrderedDict()
    data_dict_gt["ego"] = data_dict["ego"]
    gt_box_tensor = dataset.post_processor.generate_gt_bbx(data_dict_gt)
    return pred_box_tensor, pred_score, gt_box_tensor
