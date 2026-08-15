# -*- coding: utf-8 -*-
"""Candidate preparation/finalization for CoSDH fixed-Markov transport."""
from __future__ import print_function

from collections import OrderedDict

import torch

from opencood.models.baselines.cosdh.transport.cosdh_legacy_candidate_postprocess import (
    _ensure_output_aliases,
    _finalize_candidates,
    _prepare_local_candidates,
    communication_range_cav_ids,
)
from opencood.utils import box_utils


def prepare_non_ego_late_candidates(
    dataset, data_dict, output_dict, transport
):
    _ensure_output_aliases(output_dict)
    selected = communication_range_cav_ids(dataset, data_dict)
    for cav_id in selected:
        if cav_id == "ego":
            continue
        if cav_id not in output_dict:
            raise KeyError("CAV {} missing from output_dict".format(cav_id))
        boxes, scores = _prepare_local_candidates(
            dataset.post_processor,
            data_dict[cav_id],
            output_dict[cav_id],
        )
        transport.set_late_candidates(cav_id, boxes, scores)


def candidate_post_process_fixed_markov(
    dataset, data_dict, output_dict, transport
):
    _ensure_output_aliases(output_dict)
    selected = communication_range_cav_ids(dataset, data_dict)
    projected_boxes = []
    projected_scores = []

    for cav_id in selected:
        cav_content = data_dict[cav_id]
        if cav_id == "ego":
            boxes_local, scores = _prepare_local_candidates(
                dataset.post_processor,
                cav_content,
                output_dict[cav_id],
            )
        else:
            reference = output_dict[cav_id]["psm"]
            boxes_local, scores = transport.get_received_late_candidates(
                cav_id, device=reference.device
            )
        if int(boxes_local.shape[0]) == 0:
            continue
        corners = box_utils.boxes_to_corners_3d(
            boxes_local,
            order=dataset.post_processor.params["order"],
        )
        projected = box_utils.project_box3d(
            corners, cav_content["transformation_matrix"]
        )
        projected_boxes.append(projected)
        projected_scores.append(scores)

    pred_box_tensor, pred_score = _finalize_candidates(
        dataset.post_processor, projected_boxes, projected_scores
    )
    data_dict_gt = OrderedDict()
    data_dict_gt["ego"] = data_dict["ego"]
    gt_box_tensor = dataset.post_processor.generate_gt_bbx(data_dict_gt)
    return pred_box_tensor, pred_score, gt_box_tensor
