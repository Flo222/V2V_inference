# -*- coding: utf-8 -*-
"""V2X-Real dataset adapter for CoopDiff.

It extends the already-working IntermediateFusionDatasetV2XReal with one extra
branch required by CoopDiff:
    processed_lidar_paint

The painted branch has 5 point features.  For each CAV we produce two painted
voxel entries, matching the original CoopDiff convention:
    1) supervise_feature entry
    2) object_feature entry
Therefore the teacher branch receives 2 * record_len features and the inherited
CoopDiff model can split them with ``double_record_len``.
"""

import math
import warnings
from collections import OrderedDict

import numpy as np
import torch

import opencood.data_utils.datasets
from opencood.data_utils.datasets.intermediate_fusion_dataset_v2xreal import (
    IntermediateFusionDatasetV2XReal,
    V2XREAL_COM_RANGE,
)
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_points_by_range


class IntermediateFusionDatasetCoopdiffV2XReal(IntermediateFusionDatasetV2XReal):
    """Intermediate fusion dataset for CoopDiff on V2X-Real."""

    @staticmethod
    def _boxes_to_corners_np(boxes_center, order):
        if boxes_center is None or len(boxes_center) == 0:
            return np.zeros((0, 8, 3), dtype=np.float32)
        boxes7 = np.asarray(boxes_center)[:, :7]
        if boxes7.shape[0] == 0:
            return np.zeros((0, 8, 3), dtype=np.float32)
        return box_utils.boxes_to_corners_3d(boxes7, order)

    @staticmethod
    def object_all_inside_points(points, boxes_corners):
        points = np.asarray(points)
        if points.size == 0:
            return points.reshape(0, points.shape[-1] if points.ndim > 1 else 4)
        boxes_corners = np.asarray(boxes_corners)
        if boxes_corners.size == 0:
            return points[:0]
        expanded_points = np.expand_dims(points[:, :3], axis=1)
        inside_mask = box_utils.is_point_inside_any_box(expanded_points, boxes_corners)
        inside_indices = np.any(inside_mask, axis=1)
        return points[inside_indices]

    @staticmethod
    def object_all_outside_points(points, boxes_corners):
        points = np.asarray(points)
        if points.size == 0:
            return points.reshape(0, points.shape[-1] if points.ndim > 1 else 4)
        boxes_corners = np.asarray(boxes_corners)
        if boxes_corners.size == 0:
            return points
        expanded_points = np.expand_dims(points[:, :3], axis=1)
        inside_mask = box_utils.is_point_inside_any_box(expanded_points, boxes_corners)
        inside_indices = np.any(inside_mask, axis=1)
        return points[~inside_indices]

    @staticmethod
    def _append_paint(points, value):
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(0, 4)
        if points.shape[0] == 0:
            return np.zeros((0, 5), dtype=np.float32)
        if points.shape[1] >= 5:
            out = points[:, :5].copy()
            out[:, 4] = value
            return out.astype(np.float32, copy=False)
        paint = np.full((points.shape[0], 1), value, dtype=points.dtype)
        return np.concatenate([points[:, :4], paint], axis=1).astype(np.float32, copy=False)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(
            idx,
            cur_ego_pose_flag=self.cur_ego_pose_flag)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        flip, noise_rotation, noise_scale = self.generate_augment()

        ego_id = -1
        ego_lidar_pose = []
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(base_data_dict.keys())[0], \
            "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = self.get_pairwise_transformation(
            base_data_dict, self.max_cav)

        processed_features = []
        object_stack = []
        object_id_stack = []

        velocity = []
        time_delay = []
        infra = []
        channel_state_ids = []
        channel_delay_ms = []
        channel_delay_slots = []
        spatial_correction_matrix = []

        early_fusion_lidar_stack = []
        object_bbx_corner_stack = []

        if self.visualize:
            projected_lidar_stack = []

        for cav_id, selected_cav_base in base_data_dict.items():
            distance = math.sqrt(
                (selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 +
                (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            if distance > V2XREAL_COM_RANGE:
                continue

            selected_cav_base['flip'] = flip
            selected_cav_base['noise_rotation'] = noise_rotation
            selected_cav_base['noise_scale'] = noise_scale

            selected_cav_processed, void_lidar = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)
            if void_lidar:
                continue

            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            processed_features.append(selected_cav_processed['processed_features'])

            lidar_np = selected_cav_processed['projected_lidar']
            early_fusion_lidar_stack.append(lidar_np)
            object_bbx_corner_stack.append(
                self._boxes_to_corners_np(
                    selected_cav_processed['object_bbx_center'],
                    self.params['postprocess']['order']))

            velocity.append(selected_cav_processed['velocity'])
            time_delay.append(float(selected_cav_base['time_delay']))
            channel_state_ids.append(
                int(selected_cav_base.get(
                    'channel_state_id',
                    -1 if selected_cav_base.get('ego', False) else -2)))
            channel_delay_ms.append(float(selected_cav_base.get('channel_delay_ms', 0.0)))
            channel_delay_slots.append(
                int(selected_cav_base.get(
                    'channel_delay_slots',
                    selected_cav_base.get('time_delay', 0))))
            spatial_correction_matrix.append(
                selected_cav_base['params']['spatial_correction_matrix'])
            infra.append(1 if int(cav_id) < 0 else 0)

            if self.visualize:
                from opencood.data_utils.datasets import GT_RANGE
                projected_lidar_vis = mask_points_by_range(
                    selected_cav_processed['projected_lidar_original'],
                    GT_RANGE)
                projected_lidar_stack.append(projected_lidar_vis)

        if len(processed_features) == 0:
            raise RuntimeError("No valid CAV after V2X-Real filtering.")

        # Build CoopDiff painted teacher branch.  The first record_len entries
        # are supervise features; the following record_len entries are object
        # features.  This matches point_pillar_diff_stu.forward().
        early_fusion_all_lidar = np.vstack(early_fusion_lidar_stack)
        processed_features_paint = []
        object_paint_entries = []
        for lidar_np, boxes_corners in zip(early_fusion_lidar_stack,
                                           object_bbx_corner_stack):
            inside_global = self.object_all_inside_points(
                early_fusion_all_lidar, boxes_corners)
            outside_local = self.object_all_outside_points(
                lidar_np, boxes_corners)

            inside_paint = self._append_paint(inside_global, 1.0)
            outside_paint = self._append_paint(outside_local, 0.0)
            if inside_paint.shape[0] == 0:
                # Keep the voxel generator non-empty and keep this CAV aligned.
                supervise_points = self._append_paint(lidar_np, 0.0)
                object_points = self._append_paint(lidar_np, 0.0)
            else:
                supervise_points = np.concatenate([inside_paint, outside_paint], axis=0)
                object_points = inside_paint

            processed_features_paint.append(
                self.pre_processor.preprocess_paint(supervise_points))
            object_paint_entries.append(
                self.pre_processor.preprocess_paint(object_points))

        processed_features_paint.extend(object_paint_entries)

        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 8))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        cav_num = len(processed_features)
        merged_feature_dict = self.merge_features_to_dict(processed_features)
        merged_feature_dict_paint = self.merge_features_to_dict(processed_features_paint)

        all_anchors, num_anchors_per_location = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center,
            anchors=all_anchors,
            num_anchors_per_location=num_anchors_per_location,
            mask=mask)

        velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        channel_state_ids = channel_state_ids + (self.max_cav - len(channel_state_ids)) * [-1]
        channel_delay_ms = channel_delay_ms + (self.max_cav - len(channel_delay_ms)) * [0.0]
        channel_delay_slots = channel_delay_slots + (self.max_cav - len(channel_delay_slots)) * [0]

        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(
            np.eye(4)[None],
            (self.max_cav - len(spatial_correction_matrix), 1, 1))
        spatial_correction_matrix = np.concatenate(
            [spatial_correction_matrix, padding_eye], axis=0)

        processed_data_dict['ego'].update({
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': mask,
            'object_ids': [object_id_stack[i] for i in unique_indices],
            'all_anchors': all_anchors,
            'num_anchors_per_location': num_anchors_per_location,
            'processed_lidar': merged_feature_dict,
            'processed_lidar_paint': merged_feature_dict_paint,
            'label_dict': label_dict,
            'cav_num': cav_num,
            'velocity': velocity,
            'time_delay': time_delay,
            'infra': infra,
            'channel_state_ids': channel_state_ids,
            'channel_delay_ms': channel_delay_ms,
            'channel_delay_slots': channel_delay_slots,
            'spatial_correction_matrix': spatial_correction_matrix,
            'pairwise_t_matrix': pairwise_t_matrix,
        })

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar': projected_lidar_stack})

        return processed_data_dict

    def collate_batch_train(self, batch):
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        processed_lidar_paint_list = []
        record_len = []
        label_dict_list = []
        velocity = []
        time_delay = []
        infra = []
        channel_state_ids = []
        channel_delay_ms = []
        channel_delay_slots = []
        pairwise_t_matrix_list = []
        spatial_correction_matrix_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            processed_lidar_paint_list.append(ego_dict['processed_lidar_paint'])
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            channel_state_ids.append(ego_dict.get('channel_state_ids', [-1] * self.max_cav))
            channel_delay_ms.append(ego_dict.get('channel_delay_ms', [0.0] * self.max_cav))
            channel_delay_slots.append(ego_dict.get('channel_delay_slots', [0] * self.max_cav))
            spatial_correction_matrix_list.append(ego_dict['spatial_correction_matrix'])
            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_feature_dict)

        merged_feature_dict_paint = self.merge_features_to_dict(processed_lidar_paint_list)
        processed_lidar_paint_torch_dict = self.pre_processor.collate_batch(
            merged_feature_dict_paint)

        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)

        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        channel_state_ids = torch.from_numpy(np.array(channel_state_ids, dtype=np.int64))
        channel_delay_ms = torch.from_numpy(np.array(channel_delay_ms, dtype=np.float32))
        channel_delay_slots = torch.from_numpy(np.array(channel_delay_slots, dtype=np.int64))
        spatial_correction_matrix_list = torch.from_numpy(
            np.array(spatial_correction_matrix_list))
        prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        output_dict['ego'].update({
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': object_bbx_mask,
            'processed_lidar': processed_lidar_torch_dict,
            'processed_lidar_paint': processed_lidar_paint_torch_dict,
            'record_len': record_len,
            'label_dict': label_torch_dict,
            'object_ids': object_ids[0],
            'prior_encoding': prior_encoding,
            'channel_state_ids': channel_state_ids,
            'channel_delay_ms': channel_delay_ms,
            'channel_delay_slots': channel_delay_slots,
            'spatial_correction_matrix': spatial_correction_matrix_list,
            'pairwise_t_matrix': pairwise_t_matrix,
        })

        if self.visualize:
            origin_lidar = [torch.from_numpy(lidar) for lidar in origin_lidar[0]]
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict
