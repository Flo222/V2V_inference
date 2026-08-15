from __future__ import annotations

import torch

from opencood.models.baselines.v2xvit.point_pillar_transformer_v2xreal import (
    PointPillarTransformerV2xreal,
)
from opencood.models.common.fuse_modules.fuse_utils import regroup
from opencood.models.baselines.v2xvit.v2xvit_native_payload_adapter import (
    V2XViTNativePayloadAdapter,
)


class PointPillarTransformerV2xrealArce(PointPillarTransformerV2xreal):
    """V2X-Real V2X-ViT with ARCE at the correct native payload boundary.

    The clean V2X-Real baseline class remains untouched. This wrapper preserves
    all original trainable module names, so its original checkpoint can be
    loaded with ``strict=False`` without retraining merely because of the
    communication bridge.
    """

    def __init__(self, args):
        super(PointPillarTransformerV2xrealArce, self).__init__(args)
        self.arce_cfg = args.get("arce", {}) or {}
        self.payload_transport = V2XViTNativePayloadAdapter(
            self.arce_cfg,
            dataset_name="V2X-Real",
        )
        self.arce_enabled = self.payload_transport.enabled
        self.arce_comm = self.payload_transport.executor
        self.arce_comm_type = self.payload_transport.executor_type

    def _run_native_payload_channel(
        self,
        spatial_features_2d,
        record_len,
        data_dict,
    ):
        return self.payload_transport.communicate(
            features=spatial_features_2d,
            record_len=record_len,
            data_dict=data_dict,
        )

    def forward(self, data_dict):
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        record_len = data_dict["record_len"]
        spatial_correction_matrix = data_dict["spatial_correction_matrix"]

        # Keep the three priors as three scalars per link until after the
        # physical feature communication. They are repeated over HxW only for
        # the local Transformer input and are never counted as feature values.
        prior_encoding = data_dict["prior_encoding"].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "record_len": record_len,
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]

        # Baseline-native processing must finish before the channel.
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # Correct communication boundary:
        # post native compressor, pre regroup/padding/prior repetition.
        spatial_features_2d, comm_info = self._run_native_payload_channel(
            spatial_features_2d,
            record_len,
            data_dict,
        )

        regroup_feature, mask = regroup(
            spatial_features_2d,
            record_len,
            self.max_cav,
        )

        prior_encoding = prior_encoding.repeat(
            1,
            1,
            1,
            regroup_feature.shape[3],
            regroup_feature.shape[4],
        )
        regroup_feature = torch.cat(
            [regroup_feature, prior_encoding],
            dim=2,
        )
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)

        fused_feature = self.fusion_net(
            regroup_feature,
            mask,
            spatial_correction_matrix,
        )
        fused_feature = fused_feature.permute(0, 3, 1, 2)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        return {
            "psm": psm,
            "rm": rm,
            "comm_info": comm_info,
        }


# Defensive aliases for OpenCOOD's underscore-insensitive class lookup.
PointPillarTransformerV2XRealArce = PointPillarTransformerV2xrealArce
PointPillarTransformerV2xRealArce = PointPillarTransformerV2xrealArce
