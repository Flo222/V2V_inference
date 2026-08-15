import torch
import torch.nn as nn

from opencood.models.common.sub_modules.pillar_vfe import PillarVFE
from opencood.models.common.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common.fuse_modules.fuse_utils import regroup
from opencood.models.common.sub_modules.downsample_conv import DownsampleConv
from opencood.models.common.sub_modules.naive_compress import NaiveCompressor
from opencood.models.baselines.v2xvit.v2xvit_basic import V2XTransformer

from opencood.models.baselines.v2xvit.v2xvit_native_payload_adapter import V2XViTNativePayloadAdapter

ID_TO_CHANNEL_STATE = {
    0: "good",
    1: "medium",
    2: "bad",
}

class PointPillarTransformerOpv2vArce(nn.Module):
    """
    OPV2V-compatible V2X-ViT model with an optional ARCE communication layer.

    Difference from point_pillar_transformer_opv2v.py:
    1. Keep the original OPV2V-compatible V2X-ViT backbone unchanged.
    2. Insert ARCE communication simulation after shrink/compression and before regroup().
    3. If ARCE is disabled, this model should behave like the original V2X-ViT OPV2V wrapper.
    4. If OPV2V dataloader does not provide prior_encoding, create zero prior.
       prior_encoding channels mean [velocity, time_delay, infra].
       For OPV2V, all agents are vehicles, so infra=0.
    5. If OPV2V dataloader does not provide spatial_correction_matrix, use identity matrices.
    """

    def __init__(self, args):
        super(PointPillarTransformerOpv2vArce, self).__init__()

        self.max_cav = args['max_cav']

        # -------------------------
        # PointPillar feature encoder
        # -------------------------
        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range']
        )

        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # -------------------------
        # Optional shrink header
        # -------------------------
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        # -------------------------
        # Original V2X-ViT naive compressor
        # Keep this disabled when using ARCE quantization,
        # unless you intentionally want to test double compression.
        # -------------------------
        self.compression = False
        if args.get('compression', 0) > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        # -------------------------
        # Unified baseline-native payload bridge.
        #
        # V2X-ViT must finish shrink/native compression first. The adapter
        # transports only [sum(record_len), C, H, W], before regroup padding
        # and before the 3 priors are repeated over HxW.
        # -------------------------
        self.arce_cfg = args.get('arce', {}) or {}
        self.payload_transport = V2XViTNativePayloadAdapter(
            self.arce_cfg,
            dataset_name="OPV2V",
        )
        self.arce_enabled = self.payload_transport.enabled
        self.arce_comm = self.payload_transport.executor
        self.arce_comm_type = self.payload_transport.executor_type

        # -------------------------
        # V2X-ViT fusion transformer
        # -------------------------
        self.fusion_net = V2XTransformer(args['transformer'])

        # -------------------------
        # Detection heads
        # -------------------------
        self.cls_head = nn.Conv2d(
            128 * 2,
            args['anchor_number'],
            kernel_size=1
        )
        self.reg_head = nn.Conv2d(
            128 * 2,
            7 * args['anchor_number'],
            kernel_size=1
        )

        if args.get('backbone_fix', False):
            self.backbone_fix()

    def backbone_fix(self):
        """
        Freeze perception backbone and heads.

        This follows the original OpenCOOD style.
        ARCE itself is normally non-learnable in the fixed-policy stage,
        so it does not need special freezing here.
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False

        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False

        for p in self.reg_head.parameters():
            p.requires_grad = False

    def _make_default_prior_encoding(self, record_len, device, dtype):
        """
        Return shape: [B, max_cav, 3]

        channel 0: velocity / delta-related prior, set to 0
        channel 1: time delay, set to 0
        channel 2: infra indicator, set to 0 because OPV2V is pure V2V
        """
        batch_size = int(record_len.shape[0])

        return torch.zeros(
            batch_size,
            self.max_cav,
            3,
            device=device,
            dtype=dtype
        )

    def _make_default_spatial_correction_matrix(self, record_len, device, dtype):
        """
        Return shape: [B, max_cav, 4, 4]

        Identity matrix means no additional STTF correction.
        For OPV2V intermediate fusion, features/points are already aligned by
        the dataset pipeline, so identity is the safest default.
        """
        batch_size = int(record_len.shape[0])
        eye = torch.eye(4, device=device, dtype=dtype)

        return eye.view(1, 1, 4, 4).repeat(
            batch_size,
            self.max_cav,
            1,
            1
        )

    def _pad_or_crop_prior(self, prior_encoding, device, dtype):
        """
        Make sure prior_encoding is [B, max_cav, 3].
        """
        prior_encoding = prior_encoding.to(device=device, dtype=dtype)

        if prior_encoding.dim() != 3:
            raise ValueError(
                "prior_encoding should have shape [B, L, 3], "
                f"got {tuple(prior_encoding.shape)}."
            )

        batch_size, num_cav, channels = prior_encoding.shape

        if channels != 3:
            raise ValueError(
                f"prior_encoding should have 3 channels, got {channels}."
            )

        if num_cav == self.max_cav:
            return prior_encoding

        if num_cav > self.max_cav:
            return prior_encoding[:, :self.max_cav, :]

        pad = torch.zeros(
            batch_size,
            self.max_cav - num_cav,
            3,
            device=device,
            dtype=dtype
        )

        return torch.cat([prior_encoding, pad], dim=1)

    def _pad_or_crop_spatial_matrix(self, matrix, device, dtype):
        """
        Make sure spatial_correction_matrix is [B, max_cav, 4, 4].
        """
        matrix = matrix.to(device=device, dtype=dtype)

        if matrix.dim() != 4 or matrix.shape[-2:] != (4, 4):
            raise ValueError(
                "spatial_correction_matrix should have shape [B, L, 4, 4], "
                f"got {tuple(matrix.shape)}."
            )

        batch_size, num_cav = matrix.shape[:2]

        if num_cav == self.max_cav:
            return matrix

        if num_cav > self.max_cav:
            return matrix[:, :self.max_cav, :, :]

        eye = torch.eye(4, device=device, dtype=dtype)
        pad = eye.view(1, 1, 4, 4).repeat(
            batch_size,
            self.max_cav - num_cav,
            1,
            1
        )

        return torch.cat([matrix, pad], dim=1)

    def _run_arce_comm(self, spatial_features_2d, record_len, data_dict):
        """Transport V2X-ViT's post-compressor method-native payload."""
        return self.payload_transport.communicate(
            features=spatial_features_2d,
            record_len=record_len,
            data_dict=data_dict,
        )

    def forward(self, data_dict):
        """
        Forward pipeline:

        processed_lidar
        -> PillarVFE
        -> PointPillarScatter
        -> BaseBEVBackbone
        -> shrink_conv / original naive_compressor
        -> ARCE communication layer
        -> regroup to [B, max_cav, C, H, W]
        -> concatenate prior_encoding
        -> V2XTransformer fusion
        -> detection heads
        """
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        device = voxel_features.device
        dtype = voxel_features.dtype

        # -------------------------
        # Prepare prior_encoding
        # -------------------------
        if 'prior_encoding' in data_dict:
            prior_encoding = self._pad_or_crop_prior(
                data_dict['prior_encoding'],
                device,
                dtype
            )
        else:
            prior_encoding = self._make_default_prior_encoding(
                record_len,
                device,
                dtype
            )

        # -------------------------
        # Prepare spatial_correction_matrix
        # -------------------------
        if 'spatial_correction_matrix' in data_dict:
            spatial_correction_matrix = self._pad_or_crop_spatial_matrix(
                data_dict['spatial_correction_matrix'],
                device,
                dtype
            )
        else:
            spatial_correction_matrix = self._make_default_spatial_correction_matrix(
                record_len,
                device,
                dtype
            )

        # [B, max_cav, 3] -> [B, max_cav, 3, 1, 1]
        prior_encoding = prior_encoding.unsqueeze(-1).unsqueeze(-1)

        # -------------------------
        # PointPillar backbone
        # -------------------------
        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len
        }

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        # -------------------------
        # Original V2X-ViT shrink/compression
        # -------------------------
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # -------------------------
        # ARCE communication impairment / recovery
        # This is the only newly inserted part.
        # It should keep tensor shape as [N, C, H, W].
        # -------------------------
        spatial_features_2d, comm_info = self._run_arce_comm(
            spatial_features_2d,
            record_len,
            data_dict
        )

        # -------------------------
        # Regroup CAV features
        # [N, C, H, W] -> [B, max_cav, C, H, W]
        # -------------------------
        regroup_feature, mask = regroup(
            spatial_features_2d,
            record_len,
            self.max_cav
        )

        # -------------------------
        # Concatenate prior_encoding
        # prior_encoding: [B, max_cav, 3, H, W]
        # regroup_feature: [B, max_cav, C, H, W]
        # output: [B, max_cav, C+3, H, W]
        # -------------------------
        prior_encoding = prior_encoding.repeat(
            1,
            1,
            1,
            regroup_feature.shape[3],
            regroup_feature.shape[4]
        )

        regroup_feature = torch.cat(
            [regroup_feature, prior_encoding],
            dim=2
        )

        # [B, L, C, H, W] -> [B, L, H, W, C]
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)

        # -------------------------
        # V2X-ViT fusion
        # -------------------------
        fused_feature = self.fusion_net(
            regroup_feature,
            mask,
            spatial_correction_matrix
        )

        # [B, H, W, C] -> [B, C, H, W]
        fused_feature = fused_feature.permute(0, 3, 1, 2)

        # -------------------------
        # Detection heads
        # -------------------------
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {
            'psm': psm,
            'rm': rm,
            'comm_info': comm_info
        }

        return output_dict


# Defensive aliases for possible OpenCOOD class-name matching variants.
PointPillarTransformerOPV2VArce = PointPillarTransformerOpv2vArce
PointPillarTransformerOpv2VArce = PointPillarTransformerOpv2vArce
PointPillarTransformerOPV2VARCE = PointPillarTransformerOpv2vArce