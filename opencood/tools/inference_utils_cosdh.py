# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils_cosdh import torch_tensor_to_numpy
from opencood.utils.transformation_utils_cosdh import get_relative_transformation
from opencood.utils.box_utils import create_bbx, project_box3d, nms_rotated
from opencood.utils.camera_utils import indices_to_depth
from sklearn.metrics import mean_squared_error


def _paper_native_intermediate_late_outputs(batch_data, model, dataset):
    """Build native late heads first, then make one joint frame UCB call.

    Computing non-ego local heads before ego intermediate fusion does not
    change CoSDH's mathematical inputs. It only makes all sender payload
    segments available so the existing UCB executor can be invoked exactly
    once and enforce one shared frame budget.
    """
    output_dict = OrderedDict()

    # IntermediateLateFusionDataset keeps one input entry for each
    # scene agent. The main ego's record_len, however, includes only
    # agents within comm_range. Use the same distance filter as the
    # dataset's native post_process before constructing late messages.
    comm_range = float(
        getattr(dataset, "params", {}).get(
            "comm_range", float("inf")
        )
    )

    non_ego_items = []
    filtered_out_ids = []

    for cav_id, cav_content in batch_data.items():
        if cav_id == "ego":
            continue

        transformation = cav_content.get(
            "transformation_matrix", None
        )
        if transformation is None:
            raise KeyError(
                "Non-ego CAV {} has no transformation_matrix"
                .format(cav_id)
            )

        if torch.is_tensor(transformation):
            matrix = transformation.detach().cpu().numpy()
        else:
            matrix = np.asarray(transformation)

        # Be tolerant of an optional leading batch dimension.
        while matrix.ndim > 2:
            matrix = matrix[0]

        if matrix.shape != (4, 4):
            raise ValueError(
                "Unexpected transformation_matrix shape {} for CAV {}"
                .format(matrix.shape, cav_id)
            )

        tx = float(matrix[0, 3])
        ty = float(matrix[1, 3])
        distance = float((tx * tx + ty * ty) ** 0.5)

        if distance <= comm_range:
            non_ego_items.append((cav_id, cav_content))
        else:
            filtered_out_ids.append(str(cav_id))

    # The selected Late senders must exactly match the collaborators
    # represented in the ego Intermediate input.
    ego_record_len = batch_data["ego"].get("record_len", None)
    if ego_record_len is not None:
        if torch.is_tensor(ego_record_len):
            cav_num = int(
                ego_record_len.detach().reshape(-1)[0].item()
            )
        else:
            cav_num = int(
                np.asarray(ego_record_len).reshape(-1)[0]
            )

        expected_collaborators = max(0, cav_num - 1)

        if len(non_ego_items) != expected_collaborators:
            raise RuntimeError(
                "Communication-range filtering selected {} Late "
                "senders, but ego record_len expects {}. "
                "filtered_out_ids={}"
                .format(
                    len(non_ego_items),
                    expected_collaborators,
                    filtered_out_ids,
                )
            )

    late_outputs = []

    for local_idx, (cav_id, cav_content) in enumerate(
        non_ego_items, start=1
    ):
        cav_content["_ego_flag"] = False
        cav_content["_comm_link_key"] = str(cav_id)
        cav_content["_comm_local_idx"] = int(local_idx)
        late_outputs.append(model(cav_content))

    ego_content = batch_data["ego"]
    ego_content["_ego_flag"] = True
    ego_content["_comm_link_key"] = "ego"
    ego_content["_comm_local_idx"] = 0
    ego_content["_cosdh_paper_late_outputs"] = late_outputs
    ego_content["_cosdh_paper_late_cav_ids"] = [
        str(cav_id) for cav_id, _ in non_ego_items
    ]

    try:
        ego_output = model(ego_content)
    finally:
        ego_content.pop("_cosdh_paper_late_outputs", None)
        ego_content.pop("_cosdh_paper_late_cav_ids", None)

    recovered_late = ego_output.pop(
        "_cosdh_recovered_late_outputs", late_outputs
    )
    if len(recovered_late) != len(non_ego_items):
        raise RuntimeError(
            "Recovered late output count {} does not match non-ego count {}"
            .format(len(recovered_late), len(non_ego_items))
        )

    output_dict["ego"] = ego_output
    for (cav_id, _), cav_output in zip(
        non_ego_items, recovered_late
    ):
        output_dict[cav_id] = cav_output
    return output_dict

def inference_late_fusion(batch_data, model, dataset):
    """Model inference for intermediate-late CoSDH fusion.

    In Markov models, intermediate features are damaged inside the model's
    CoSDH fusion module.  Non-ego late dense detection messages are damaged
    here, after each non-ego CAV produces its single-agent output.
    """
    if bool(getattr(model, "cosdh_paper_native_enabled", False)) \
            and not model.training:
        output_dict = _paper_native_intermediate_late_outputs(
            batch_data, model, dataset
        )
    else:
        output_dict = OrderedDict()

        try:
            from opencood.models.baselines.cosdh.transport.cosdh_late_message_channel import \
                apply_late_markov_to_output_dict
        except Exception:
            apply_late_markov_to_output_dict = None

        # Dedicated late-message channels need an explicit per-sample frame reset.
        # Shared late/intermediate channels keep the frame started inside ego
        # forward(), so start_late_comm_frame() is a no-op in that case.
        if hasattr(model, "start_late_comm_frame"):
            model.start_late_comm_frame()

        # COSDH_OFFICIAL_FIXED_MARKOV_INFERENCE
    fixed_markov_transport = getattr(
        model, 'cosdh_official_fixed_markov_transport', None
    )
    fixed_markov_enabled = bool(
        fixed_markov_transport is not None
        and bool(getattr(fixed_markov_transport, 'enabled', False))
    )
    if fixed_markov_enabled:
        from opencood.models.baselines.cosdh.transport.\
            cosdh_official_fixed_markov_postprocess import (
                prepare_non_ego_late_candidates,
                candidate_post_process_fixed_markov,
            )

        ego_content = batch_data['ego']
        fixed_markov_transport.start_frame(
            record_len=ego_content.get('record_len', None),
            link_key_aliases=ego_content.get('cav_id_list', None),
            data_dict=ego_content,
        )

        # Prepare sender-side Late candidates before the ego call so all
        # Intermediate scales and Late records share one per-link byte stream.
        local_idx = 1
        for cav_id, cav_content in batch_data.items():
            if cav_id == 'ego':
                continue
            cav_content['_ego_flag'] = False
            cav_content['_comm_link_key'] = str(cav_id)
            cav_content['_comm_local_idx'] = local_idx
            output_dict[cav_id] = model(cav_content)
            local_idx += 1

        prepare_non_ego_late_candidates(
            dataset, batch_data, output_dict, fixed_markov_transport
        )

        ego_content['_ego_flag'] = True
        ego_content['_comm_link_key'] = 'ego'
        ego_content['_comm_local_idx'] = 0
        output_dict['ego'] = model(ego_content)

        post_process_result = candidate_post_process_fixed_markov(
            dataset,
            batch_data,
            output_dict,
            fixed_markov_transport,
        )
        if len(post_process_result) == 4:
            pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                post_process_result
        else:
            pred_box_tensor, pred_score, gt_box_tensor = post_process_result
            gt_label_tensor = None
        return_dict = {
            'pred_box_tensor': pred_box_tensor,
            'pred_score': pred_score,
            'gt_box_tensor': gt_box_tensor,
        }
        if gt_label_tensor is not None:
            return_dict['gt_label_tensor'] = gt_label_tensor
        return return_dict

    # Make the order explicit: ego first starts the intermediate Markov frame,
        # then non-ego late messages optionally share the same physical link
        # session with intermediate fusion when configured to do so.
        cav_items = []
        if 'ego' in batch_data:
            cav_items.append(('ego', batch_data['ego']))
        for cav_id, cav_content in batch_data.items():
            if cav_id != 'ego':
                cav_items.append((cav_id, cav_content))

        for local_idx, (cav_id, cav_content) in enumerate(cav_items):
            cav_content["_ego_flag"] = cav_id == 'ego'
            cav_content["_comm_link_key"] = str(cav_id)
            cav_content["_comm_local_idx"] = local_idx

            cav_output = model(cav_content)

            legacy_transport = getattr(
                model, 'cosdh_legacy_native_transport', None
            )
            if (
                cav_id != 'ego'
                and legacy_transport is not None
                and bool(getattr(legacy_transport, 'enabled', False))
                and bool(getattr(legacy_transport, 'late_enabled', False))
                # COSDH_LEGACY_DENSE_LATE_GUARD
                and str(getattr(
                    legacy_transport, 'late_payload_type', 'dense_heads'
                )).lower() != 'candidate_records'
            ):
                cav_output = legacy_transport.roundtrip_late_output(
                    cav_output,
                    cav_id=cav_id,
                )

            if cav_id != 'ego' and apply_late_markov_to_output_dict is not None:
                share_with_intermediate = bool(
                    getattr(model, "cosdh_late_markov_share_profile", False)
                )
                channel = getattr(model, "cosdh_late_markov_channel", None)
                if share_with_intermediate and hasattr(model, "cosdh_markov_channel"):
                    channel = getattr(model, "cosdh_markov_channel")
                elif channel is None:
                    channel = getattr(model, "cosdh_markov_channel", None)

                if channel is not None and bool(getattr(channel, "enabled", False)):
                    model_name = model.__class__.__name__.lower()
                    if "v2xreal" in model_name:
                        verbose_prefix = "CoSDH-Markov-Late-V2XReal"
                    else:
                        verbose_prefix = "CoSDH-Markov-Late-OPV2V"

                    if share_with_intermediate and \
                            hasattr(channel, "_resolve_link_key"):
                        link_key = channel._resolve_link_key(0, local_idx)
                    elif share_with_intermediate:
                        link_key = 'link_' + str(cav_id)
                    else:
                        link_key = 'late_' + str(cav_id)

                    cav_output = apply_late_markov_to_output_dict(
                        cav_output,
                        channel,
                        link_key=link_key,
                        verbose_prefix=verbose_prefix,
                    )

            output_dict[cav_id] = cav_output

    # COSDH_LEGACY_CANDIDATE_IDEAL
    legacy_transport = getattr(
        model, 'cosdh_legacy_native_transport', None
    )
    candidate_ideal = bool(
        legacy_transport is not None
        and bool(getattr(legacy_transport, 'enabled', False))
        and bool(getattr(legacy_transport, 'late_enabled', False))
        and str(getattr(
            legacy_transport, 'late_payload_type', 'dense_heads'
        )).lower() == 'candidate_records'
    )
    if candidate_ideal:
        from opencood.models.baselines.cosdh.transport.\
            cosdh_legacy_candidate_postprocess import \
            candidate_post_process_ideal
        post_process_result = candidate_post_process_ideal(
            dataset,
            batch_data,
            output_dict,
            legacy_transport,
        )
    else:
        post_process_result = dataset.post_process(batch_data, output_dict)
    if not isinstance(post_process_result, tuple):
        raise RuntimeError(
            "dataset.post_process is expected to return a tuple, got "
            f"{type(post_process_result)}"
        )

    gt_label_tensor = None
    if len(post_process_result) == 4:
        pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
            post_process_result
    elif len(post_process_result) == 3:
        pred_box_tensor, pred_score, gt_box_tensor = post_process_result
    else:
        raise RuntimeError(
            "dataset.post_process returned an unexpected tuple length: "
            f"{len(post_process_result)}"
        )

    return_dict = {"pred_box_tensor": pred_box_tensor,
                   "pred_score": pred_score,
                   "gt_box_tensor": gt_box_tensor}
    if gt_label_tensor is not None:
        return_dict["gt_label_tensor"] = gt_label_tensor
    return return_dict

def inference_no_fusion(batch_data, model, dataset, single_gt=False):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    single_gt : bool
        if True, only use ego agent's label.
        else, use all agent's merged labels.
    """
    output_dict_ego = OrderedDict()
    if single_gt:
        batch_data = {'ego': batch_data['ego']}
        
    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_no_fusion(batch_data,  # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict

def inference_no_fusion_w_uncertainty(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
        dataset.post_process_no_fusion_uncertainty(batch_data, # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor, \
                    "uncertainty_tensor" : uncertainty_tensor}

    return return_dict


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    return return_dict


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return_dict = inference_early_fusion(batch_data, model, dataset)
    return return_dict


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)


def depth_metric(depth_items, grid_conf):
    # depth logdit: [N, D, H, W]
    # depth gt indices: [N, H, W]
    depth_logit, depth_gt_indices = depth_items
    depth_pred_indices = torch.argmax(depth_logit, 1)
    depth_pred = indices_to_depth(depth_pred_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    depth_gt = indices_to_depth(depth_gt_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    rmse = mean_squared_error(depth_gt.cpu(), depth_pred.cpu(), squared=False)
    return rmse


def fix_cavs_box(pred_box_tensor, gt_box_tensor, pred_score, batch_data):
    """
    Fix the missing pred_box and gt_box for ego and cav(s).
    Args:
        pred_box_tensor : tensor
            shape (N1, 8, 3), may or may not include ego agent prediction, but it should include
        gt_box_tensor : tensor
            shape (N2, 8, 3), not include ego agent in camera cases, but it should include
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    Returns:
        pred_box_tensor : tensor
            shape (N1+?, 8, 3)
        gt_box_tensor : tensor
            shape (N2+1, 8, 3)
    """
    if pred_box_tensor is None or gt_box_tensor is None:
        return pred_box_tensor, gt_box_tensor, pred_score, 0
    # prepare cav's boxes

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()
        
    extent = [2.45, 1.06, 0.75]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_tensor = torch.tensor(np.concatenate(box_list, axis=0), device=pred_box_tensor.device)
    
    pred_box_tensor_ = torch.cat((cav_box_tensor, pred_box_tensor), dim=0)
    gt_box_tensor_ = torch.cat((cav_box_tensor, gt_box_tensor), dim=0)

    pred_score_ = torch.cat((torch.ones(N, device=pred_score.device), pred_score))

    gt_score_ = torch.ones(gt_box_tensor_.shape[0], device=pred_box_tensor.device)
    gt_score_[N:] = 0.5

    keep_index = nms_rotated(pred_box_tensor_,
                            pred_score_,
                            0.01)
    pred_box_tensor = pred_box_tensor_[keep_index]
    pred_score = pred_score_[keep_index]

    keep_index = nms_rotated(gt_box_tensor_,
                            gt_score_,
                            0.01)
    gt_box_tensor = gt_box_tensor_[keep_index]

    return pred_box_tensor, gt_box_tensor, pred_score, N


def get_cav_box(batch_data):
    """
    Args:
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    """

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
        lidar_agent_record = batch_data['ego']['lidar_agent_record'].cpu().numpy()

    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        lidar_agent_record = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
            lidar_agent_record.append(1 if 'processed_lidar' in cav_data else 0)
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()

        

    extent = [0.2, 0.2, 0.2]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_np = np.concatenate(box_list, axis=0)


    return cav_box_np, lidar_agent_record
