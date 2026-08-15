# -*- coding: utf-8 -*-
"""Intermediate-late CoSDH dataset for V2X-Real.

This mirrors the OPV2V CoSDH ``intermediatelate`` evaluation semantics:
- ``ego`` contains the multi-CAV intermediate-fusion input.
- each non-ego key contains a single-CAV late dense-detection input.

The V2X-Real ideal and Markov runs can therefore share the same mixed-fusion
protocol.  Markov damage for late dense maps is applied in the model helper.
"""

import math
from collections import OrderedDict

import numpy as np
import torch

from opencood.data_utils.datasets.intermediate_fusion_dataset_v2xreal import (
    IntermediateFusionDatasetV2XReal,
    V2XREAL_COM_RANGE,
)


class IntermediateLateFusionDatasetV2XReal(IntermediateFusionDatasetV2XReal):
    """V2X-Real mixed intermediate+late dataset for CoSDH inference."""

    def __init__(self, params, visualize, train=True):
        super().__init__(params, visualize, train)
        postprocess_cfg = params.get("postprocess", {})
        # Match the original CoSDH late-message confidence calibration while
        # keeping it configurable for V2X-Real experiments.
        self.confidence_beta = postprocess_cfg.get("confidence_beta", 0.9)
        self.confidence_threshold = postprocess_cfg.get(
            "confidence_threshold", 0.3
        )
        self.class_aware_nms = bool(
            postprocess_cfg.get("class_aware_nms", True)
        )
        print("confidence_beta:", self.confidence_beta)
        print("confidence_threshold:", self.confidence_threshold)
        print("class_aware_nms:", self.class_aware_nms)

    def __getitem__(self, idx):
        if self.train:
            return super().__getitem__(idx)

        base_data_dict = self.retrieve_base_data(
            idx, cur_ego_pose_flag=self.cur_ego_pose_flag
        )
        processed_data_dict = OrderedDict()

        flip, noise_rotation, noise_scale = self.generate_augment()

        main_ego_id = None
        main_ego_pose = None
        for cav_id, cav_content in base_data_dict.items():
            if cav_content["ego"]:
                main_ego_id = cav_id
                main_ego_pose = cav_content["params"]["lidar_pose"]
                break
        if main_ego_id is None:
            raise RuntimeError("V2X-Real mixed CoSDH cannot find ego CAV")

        # Ego entry: multi-CAV intermediate feature fusion.
        ego_entry = self._build_intermediate_entry(
            base_data_dict=base_data_dict,
            ego_pose=main_ego_pose,
            idx=idx,
            flip=flip,
            noise_rotation=noise_rotation,
            noise_scale=noise_scale,
        )
        processed_data_dict["ego"] = ego_entry

        # Late entries: one single-agent dense detection message per non-ego CAV.
        for cav_id, cav_content in base_data_dict.items():
            if cav_id == main_ego_id:
                continue
            distance = math.sqrt(
                (cav_content["params"]["lidar_pose"][0] - main_ego_pose[0]) ** 2
                + (cav_content["params"]["lidar_pose"][1] - main_ego_pose[1]) ** 2
            )
            if distance > V2XREAL_COM_RANGE:
                continue
            late_entry = self._build_late_entry(
                selected_cav_base=cav_content,
                main_ego_pose=main_ego_pose,
                idx=idx,
                flip=flip,
                noise_rotation=noise_rotation,
                noise_scale=noise_scale,
            )
            if late_entry is not None:
                processed_data_dict[str(cav_id)] = late_entry

        return processed_data_dict

    def _build_intermediate_entry(
        self,
        base_data_dict,
        ego_pose,
        idx,
        flip,
        noise_rotation,
        noise_scale,
    ):
        pairwise_t_matrix = self.get_pairwise_transformation(
            base_data_dict, self.max_cav
        )

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
        projected_lidar_stack = [] if self.visualize else None

        for cav_id, selected_cav_base in base_data_dict.items():
            distance = math.sqrt(
                (selected_cav_base["params"]["lidar_pose"][0] - ego_pose[0]) ** 2
                + (selected_cav_base["params"]["lidar_pose"][1] - ego_pose[1]) ** 2
            )
            if distance > V2XREAL_COM_RANGE:
                continue

            selected_cav_base["flip"] = flip
            selected_cav_base["noise_rotation"] = noise_rotation
            selected_cav_base["noise_scale"] = noise_scale
            selected_cav_processed, void_lidar = self.get_item_single_car(
                selected_cav_base, ego_pose
            )
            if void_lidar:
                continue

            object_stack.append(selected_cav_processed["object_bbx_center"])
            object_id_stack += selected_cav_processed["object_ids"]
            processed_features.append(selected_cav_processed["processed_features"])
            velocity.append(selected_cav_processed["velocity"])
            time_delay.append(float(selected_cav_base["time_delay"]))
            infra.append(1 if int(cav_id) < 0 else 0)
            channel_state_ids.append(
                int(
                    selected_cav_base.get(
                        "channel_state_id",
                        -1 if selected_cav_base.get("ego", False) else -2,
                    )
                )
            )
            channel_delay_ms.append(float(selected_cav_base.get("channel_delay_ms", 0.0)))
            channel_delay_slots.append(
                int(
                    selected_cav_base.get(
                        "channel_delay_slots", selected_cav_base.get("time_delay", 0)
                    )
                )
            )
            spatial_correction_matrix.append(
                selected_cav_base["params"].get("spatial_correction_matrix", np.eye(4))
            )
            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed["projected_lidar_original"]
                )

        if not processed_features:
            return None

        object_bbx_center, mask, object_ids = self._merge_objects(
            object_stack, object_id_stack
        )
        merged_feature_dict = self.merge_features_to_dict(processed_features)
        all_anchors, num_anchors_per_location = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center,
            anchors=all_anchors,
            num_anchors_per_location=num_anchors_per_location,
            mask=mask,
        )

        cav_num = len(processed_features)
        velocity = velocity + (self.max_cav - len(velocity)) * [0.0]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.0]
        infra = infra + (self.max_cav - len(infra)) * [0]
        channel_state_ids = channel_state_ids + (self.max_cav - len(channel_state_ids)) * [-1]
        channel_delay_ms = channel_delay_ms + (self.max_cav - len(channel_delay_ms)) * [0.0]
        channel_delay_slots = channel_delay_slots + (self.max_cav - len(channel_delay_slots)) * [0]

        if spatial_correction_matrix:
            spatial_correction_matrix = np.stack(spatial_correction_matrix)
        else:
            spatial_correction_matrix = np.zeros((0, 4, 4))
        padding_eye = np.tile(
            np.eye(4)[None],
            (self.max_cav - len(spatial_correction_matrix), 1, 1),
        )
        spatial_correction_matrix = np.concatenate(
            [spatial_correction_matrix, padding_eye], axis=0
        )

        entry = {
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": mask,
            "object_ids": object_ids,
            "all_anchors": all_anchors,
            "num_anchors_per_location": num_anchors_per_location,
            "processed_lidar": merged_feature_dict,
            "label_dict": label_dict,
            "cav_num": cav_num,
            "velocity": velocity,
            "time_delay": time_delay,
            "infra": infra,
            "channel_state_ids": channel_state_ids,
            "channel_delay_ms": channel_delay_ms,
            "channel_delay_slots": channel_delay_slots,
            "spatial_correction_matrix": spatial_correction_matrix,
            "pairwise_t_matrix": pairwise_t_matrix,
            "transformation_matrix": np.eye(4),
            "sample_idx": idx,
        }
        if self.visualize:
            entry["origin_lidar"] = projected_lidar_stack
        return entry

    def _build_late_entry(
        self,
        selected_cav_base,
        main_ego_pose,
        idx,
        flip,
        noise_rotation,
        noise_scale,
    ):
        selected_cav_base["flip"] = flip
        selected_cav_base["noise_rotation"] = noise_rotation
        selected_cav_base["noise_scale"] = noise_scale
        selected_cav_processed, void_lidar = self.get_item_single_car(
            selected_cav_base, main_ego_pose
        )
        if void_lidar:
            return None

        object_bbx_center, mask, object_ids = self._merge_objects(
            [selected_cav_processed["object_bbx_center"]],
            selected_cav_processed["object_ids"],
        )
        all_anchors, num_anchors_per_location = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center,
            anchors=all_anchors,
            num_anchors_per_location=num_anchors_per_location,
            mask=mask,
        )

        pairwise_t_matrix = np.tile(np.eye(4)[None, None], (self.max_cav, self.max_cav, 1, 1))
        spatial_correction_matrix = np.tile(np.eye(4)[None], (self.max_cav, 1, 1))

        # get_item_single_car projects points to main ego when proj_first=True.
        # In that common V2X-Real setting the dense late map is already in ego
        # coordinates, so postprocess should use identity.  If proj_first=False,
        # keep the dataset-provided CAV->ego transform.
        transformation_matrix = (
            np.eye(4)
            if self.proj_first
            else selected_cav_base["params"].get("transformation_matrix", np.eye(4))
        )

        entry = {
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": mask,
            "object_ids": object_ids,
            "all_anchors": all_anchors,
            "num_anchors_per_location": num_anchors_per_location,
            "processed_lidar": selected_cav_processed["processed_features"],
            "label_dict": label_dict,
            "cav_num": 1,
            "velocity": [selected_cav_processed["velocity"]] + (self.max_cav - 1) * [0.0],
            "time_delay": [float(selected_cav_base.get("time_delay", 0.0))]
            + (self.max_cav - 1) * [0.0],
            "infra": [1 if int(selected_cav_base.get("cav_id", 1)) < 0 else 0]
            + (self.max_cav - 1) * [0],
            "channel_state_ids": [
                int(selected_cav_base.get("channel_state_id", -2))
            ]
            + (self.max_cav - 1) * [-1],
            "channel_delay_ms": [float(selected_cav_base.get("channel_delay_ms", 0.0))]
            + (self.max_cav - 1) * [0.0],
            "channel_delay_slots": [
                int(selected_cav_base.get("channel_delay_slots", 0))
            ]
            + (self.max_cav - 1) * [0],
            "spatial_correction_matrix": spatial_correction_matrix,
            "pairwise_t_matrix": pairwise_t_matrix,
            "transformation_matrix": transformation_matrix,
            "sample_idx": idx,
        }
        if self.visualize:
            entry["origin_lidar"] = [selected_cav_processed["projected_lidar_original"]]
        return entry

    def _merge_objects(self, object_stack, object_id_stack):
        object_bbx_center = np.zeros((self.params["postprocess"]["max_num"], 8))
        mask = np.zeros(self.params["postprocess"]["max_num"])
        if len(object_stack) == 0:
            return object_bbx_center, mask, []

        valid_stack = [x for x in object_stack if x is not None and len(x) > 0]
        if len(valid_stack) == 0:
            return object_bbx_center, mask, []

        stacked = np.vstack(valid_stack)
        if len(object_id_stack) > 0:
            unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            unique_indices = [i for i in unique_indices if i < stacked.shape[0]]
            stacked = stacked[unique_indices]
            object_ids = [object_id_stack[i] for i in unique_indices]
        else:
            object_ids = []

        num = min(stacked.shape[0], self.params["postprocess"]["max_num"])
        object_bbx_center[:num, :] = stacked[:num]
        mask[:num] = 1
        return object_bbx_center, mask, object_ids

    def _feature_dict_for_collate(self, processed_lidar):
        out = OrderedDict()
        for key, value in processed_lidar.items():
            if isinstance(value, list):
                out[key] = value
            else:
                out[key] = [value]
        return out

    def _collate_one_entry(self, entry):
        output = {}
        object_bbx_center = torch.from_numpy(np.array([entry["object_bbx_center"]]))
        object_bbx_mask = torch.from_numpy(np.array([entry["object_bbx_mask"]]))

        processed_lidar_torch = self.pre_processor.collate_batch(
            self._feature_dict_for_collate(entry["processed_lidar"])
        )
        record_len = torch.from_numpy(np.array([entry["cav_num"]], dtype=int))
        label_torch_dict = self.post_processor.collate_batch([entry["label_dict"]])

        velocity = torch.from_numpy(np.array([entry["velocity"]]))
        time_delay = torch.from_numpy(np.array([entry["time_delay"]]))
        infra = torch.from_numpy(np.array([entry["infra"]]))
        prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()

        channel_state_ids = torch.from_numpy(
            np.array([entry["channel_state_ids"]], dtype=np.int64)
        )
        channel_delay_ms = torch.from_numpy(
            np.array([entry["channel_delay_ms"]], dtype=np.float32)
        )
        channel_delay_slots = torch.from_numpy(
            np.array([entry["channel_delay_slots"]], dtype=np.int64)
        )
        spatial_correction_matrix = torch.from_numpy(
            np.array([entry["spatial_correction_matrix"]])
        )
        pairwise_t_matrix = torch.from_numpy(np.array([entry["pairwise_t_matrix"]]))
        all_anchors = torch.from_numpy(np.array(entry["all_anchors"]))
        transformation_matrix = torch.from_numpy(
            np.array(entry.get("transformation_matrix", np.eye(4)))
        ).float()

        output.update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "processed_lidar": processed_lidar_torch,
                "record_len": record_len,
                "label_dict": label_torch_dict,
                "object_ids": entry.get("object_ids", []),
                "prior_encoding": prior_encoding,
                "channel_state_ids": channel_state_ids,
                "channel_delay_ms": channel_delay_ms,
                "channel_delay_slots": channel_delay_slots,
                "spatial_correction_matrix": spatial_correction_matrix,
                "pairwise_t_matrix": pairwise_t_matrix,
                "all_anchors": all_anchors,
                "num_anchors_per_location": entry["num_anchors_per_location"],
                "transformation_matrix": transformation_matrix,
                "sample_idx": entry.get("sample_idx", -1),
            }
        )

        if self.visualize and "origin_lidar" in entry:
            output["origin_lidar"] = [torch.from_numpy(x) for x in entry["origin_lidar"]]

        return output

    def collate_batch_train(self, batch):
        # Training keeps the normal V2X-Real intermediate behaviour.
        return super().collate_batch_train(batch)

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = OrderedDict()
        for cav_id, cav_content in batch[0].items():
            if cav_content is None:
                continue
            output_dict[cav_id] = self._collate_one_entry(cav_content)
        return output_dict

    def post_process(self, data_dict, output_dict):
        # CoSDH OPV2V compatibility uses cls_preds/reg_preds; V2X-Real
        # postprocessor consumes psm/rm.
        for _cav_id, _out in output_dict.items():
            if isinstance(_out, dict):
                if "psm" not in _out and "cls_preds" in _out:
                    _out["psm"] = _out["cls_preds"]
                if "rm" not in _out and "reg_preds" in _out:
                    _out["rm"] = _out["reg_preds"]
                if "dm" not in _out and "dir_preds" in _out:
                    _out["dm"] = _out["dir_preds"]

        pred_box_tensor, pred_score = self.post_processor.post_process(
            data_dict,
            output_dict,
            confidence_beta=self.confidence_beta,
            confidence_threshold=self.confidence_threshold,
            class_aware_nms=self.class_aware_nms,
        )

        gt_result = self.post_processor.generate_gt_bbx({"ego": data_dict["ego"]})
        if isinstance(gt_result, tuple):
            gt_box_tensor = gt_result[0]
            gt_label_tensor = gt_result[1] if len(gt_result) > 1 else None
        else:
            gt_box_tensor = gt_result
            gt_label_tensor = None

        if gt_label_tensor is not None:
            return pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor
        return pred_box_tensor, pred_score, gt_box_tensor


def getIntermediatelateFusionDatasetV2XReal(cls=None):
    # Keep the same factory style as OPV2V's CoSDH dataset builder.
    return IntermediateLateFusionDatasetV2XReal
