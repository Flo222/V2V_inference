import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.common.sub_modules.pillar_vfe import PillarVFE
from opencood.models.baselines.cosdh.components.point_pillar_scatter_cosdh import PointPillarScatter, SimplePointPillarScatter
from opencood.models.baselines.cosdh.components.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.baselines.cosdh.components.base_bev_backbone_cosdh import BaseBEVBackbone 
from opencood.models.common.sub_modules.downsample_conv import DownsampleConv
from opencood.models.baselines.cosdh.components.naive_compress_cosdh import NaiveCompressor
from opencood.models.baselines.cosdh.fusion.fusion_in_one_cosdh_markov import Where2comm
from opencood.models.baselines.cosdh.transport.cosdh_markov_byte_channel import CosDHMarkovByteChannel
from opencood.models.baselines.cosdh.transport.cosdh_legacy_native_transport import CosDHLegacyNativeTransport
from opencood.models.baselines.cosdh.transport.cosdh_official_fixed_markov_transport import CosDHOfficialFixedMarkovTransport
from opencood.models.baselines.cosdh.components.cosdh_paper_native_adapter import CosDHPaperNativeFrameTransport, run_cosdh_paper_native_ego
from opencood.utils.transformation_utils_cosdh import normalize_pairwise_tfm


def _merge_markov_cfg(base_cfg, override_cfg):
    merged = copy.deepcopy(base_cfg) if isinstance(base_cfg, dict) else {}
    if not isinstance(override_cfg, dict):
        return merged

    for key, value in override_cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_markov_cfg(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class PointPillarCosdhMarkov(nn.Module):
    """
    Where2comm implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarCosdhMarkov, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        
        self.simple_scatter = SimplePointPillarScatter(
            feature_dim=1, grid_size=args['point_pillar_scatter']['grid_size'])
        
        self.req_points_threshold = -1
        if 'req_points_threshold' in args:
            self.req_points_threshold = args['req_points_threshold']
            print(f"req_points_threshold = {args['req_points_threshold']}")
        
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.voxel_size = args['voxel_size']

        # Pass CoSDH-Markov byte-stream channel config into CoSDH fusion.
        base_markov_cfg = copy.deepcopy(args.get('cosdh_markov', {}))
        if 'where2comm' in args:
            args['where2comm']['cosdh_markov'] = base_markov_cfg
        self.cosdh_markov_channel = CosDHMarkovByteChannel(base_markov_cfg)

        late_markov_cfg = _merge_markov_cfg(
            base_markov_cfg, args.get('cosdh_late_markov', {})
        )
        self.cosdh_late_markov_share_profile = bool(
            late_markov_cfg.get('share_profile_with_intermediate', False)
        )
        if self.cosdh_late_markov_share_profile:
            self.cosdh_late_markov_channel = self.cosdh_markov_channel
        else:
            self.cosdh_late_markov_channel = CosDHMarkovByteChannel(
                late_markov_cfg
            )
        # COSDH_OFFICIAL_FAITHFUL_TRANSPORT_INIT
        # Preserve the released checkpoint graph.  Only the original encoder
        # output / decoder input is exposed as a physical byte boundary.
        self.cosdh_legacy_native_cfg = copy.deepcopy(
            args.get('cosdh_legacy_native', {})
        )
        self.cosdh_legacy_native_transport = CosDHLegacyNativeTransport(
            self.cosdh_legacy_native_cfg
        )

        # COSDH_OFFICIAL_FIXED_MARKOV_INIT
        self.cosdh_official_fixed_markov_cfg = copy.deepcopy(
            args.get('cosdh_official_fixed_markov', {})
        )
        self.cosdh_official_fixed_markov_transport =             CosDHOfficialFixedMarkovTransport(
                self.cosdh_official_fixed_markov_cfg,
                arce_cfg=copy.deepcopy(args.get('arce', {})),
            )

        self.fusion_net = nn.ModuleList()
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            fuse_module = Where2comm(args['where2comm'], dim=args['feat_dim'][i])
            fuse_module.cosdh_markov_channel = self.cosdh_markov_channel
            self.fusion_net.append(fuse_module)
        
        if 'k_ratio' in args['where2comm']['communication']:
            print(f"k_ratio: {args['where2comm']['communication']['k_ratio']}")
        elif 'threshold' in args['where2comm']['communication']:
            print(f"threshold: {args['where2comm']['communication']['threshold']}")
        
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]
        
        self.compression = False
        self.compression_ratio = 1
        if "compression" in args:
            self.compression = True
            self.compression_ratio = args['compression']
            self.naive_compressor_list = nn.ModuleList()
            for i in range(len(args['feat_dim'])):
                self.naive_compressor_list.append(NaiveCompressor(args['feat_dim'][i],
                                                                    args['compression']))
            print(f"compression_ratio: {self.compression_ratio}")

        # Paper-native CoSDH bridge. This adapter is the only component that
        # knows how to serialize CoSDH messages; ARCE/UCB remains unchanged.
        self.cosdh_paper_native_cfg = copy.deepcopy(
            args.get('cosdh_paper_native', {})
        )
        self.cosdh_paper_native_enabled = bool(
            self.cosdh_paper_native_cfg.get('enabled', False)
        )
        self.cosdh_paper_native_in_train = bool(
            self.cosdh_paper_native_cfg.get('paper_native_in_train', False)
        )
        self.cosdh_output_style = "opv2v"
        self.cosdh_paper_transport = CosDHPaperNativeFrameTransport(
            arce_cfg=args.get('arce', {}),
            paper_cfg=self.cosdh_paper_native_cfg,
            dataset_name="OPV2V",
        )
        self.latest_paper_native_info = {}

        # Legacy-native bridge: preserve the checkpoint's original execution
        # order and expose only the encoder-output/decoder-input boundary.
        self.cosdh_legacy_native_cfg = copy.deepcopy(
            args.get('cosdh_legacy_native', {})
        )
        self.cosdh_legacy_native_transport = CosDHLegacyNativeTransport(
            self.cosdh_legacy_native_cfg
        )

        # Both datasets use one physical sender-to-ego frame budget for all
        # three intermediate scales and the late message.
        if self.cosdh_paper_native_enabled:
            self.cosdh_late_markov_share_profile = True
            self.cosdh_late_markov_channel = self.cosdh_markov_channel


        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
 
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for compressor in self.naive_compressor_list:
                for p in compressor.parameters():
                    p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
        
        if self.use_dir:
            for p in self.dir_head.parameters():
                p.requires_grad = False
        print("Backbone fixed.")
    
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def start_late_comm_frame(self):
        """Reset the late-message channel once per inference sample."""
        if self.cosdh_late_markov_channel is self.cosdh_markov_channel:
            return
        if hasattr(self.cosdh_late_markov_channel, 'start_frame'):
            self.cosdh_late_markov_channel.start_frame()
    

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        
        ego_flag = True
        if '_ego_flag' in data_dict:
            # ego_flag indicates whether the data is from ego vehicle
            # in reference, if not ego_flag, no need to fuse feature, just use the single detection results
            ego_flag = data_dict['_ego_flag']
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        ###############################################
        # for intermediate-late fusion, get the single detection results
        if not ego_flag and not self.training:
            no_fusion_feature_list = self.backbone.get_multiscale_feature(batch_dict['spatial_features'])
            no_fusion_feature_after_compress = []
            for i, fuse_module in enumerate(self.fusion_net):
                feature_i = no_fusion_feature_list[i]
                if self.compression:
                    feature_i = self.naive_compressor_list[i](feature_i, use_fp16=False)
                no_fusion_feature_after_compress.append(feature_i)
            spatial_features_2d = self.backbone.decode_multiscale_feature(no_fusion_feature_after_compress)
            
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            
            spatial_features_2d = spatial_features_2d[0].unsqueeze(0)
        
            psm = self.cls_head(spatial_features_2d)
            rm = self.reg_head(spatial_features_2d)

            output_dict = {'cls_preds': psm,
                        'reg_preds': rm}

            if self.use_dir:
                output_dict.update({'dir_preds': self.dir_head(spatial_features_2d)})
            
            return output_dict
        ###############################################
        
        req_mask = None
        if not self.training and self.req_points_threshold > 0:
            # n, -> N, 1, H, W
            points_map = self.simple_scatter(voxel_num_points.unsqueeze(1), voxel_coords).float()
            smoothed_points_map = points_map
            req_mask = (smoothed_points_map < self.req_points_threshold).float()
        
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features = batch_dict['spatial_features']
        spatial_features_2d_single = batch_dict['spatial_features_2d']
        
        if self.shrink_flag:
            spatial_features_2d_single = self.shrink_conv(spatial_features_2d_single)
        
        psm_single = self.cls_head(spatial_features_2d_single)
        

        if (
            self.cosdh_paper_native_enabled
            and ego_flag
            and (
                not self.training
                or self.cosdh_paper_native_in_train
            )
        ):
            return run_cosdh_paper_native_ego(
                model=self,
                data_dict=data_dict,
                spatial_features=spatial_features,
                psm_single=psm_single,
                record_len=record_len,
                normalized_affine_matrix=normalized_affine_matrix,
                req_mask=req_mask,
            )
        
        # multiscale fusion
        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        if hasattr(self.cosdh_markov_channel, 'start_frame'):
            self.cosdh_markov_channel.start_frame(
                link_key_aliases=data_dict.get('cav_id_list', None)
            )
        num_fusion_scales = len(self.fusion_net)
        # COSDH_OFFICIAL_FAITHFUL_FRAME_RESET
        if self.cosdh_legacy_native_transport.enabled:
            self.cosdh_legacy_native_transport.start_frame(
                record_len=record_len,
                link_key_aliases=data_dict.get('cav_id_list', None),
            )
        if self.cosdh_legacy_native_transport.enabled:
            self.cosdh_legacy_native_transport.start_frame(
                record_len=record_len,
                link_key_aliases=data_dict.get('cav_id_list', None),
            )
        
        # COSDH_OFFICIAL_FIXED_MARKOV_JOINT_STREAM
        fixed_markov_transport = getattr(
            self, 'cosdh_official_fixed_markov_transport', None
        )
        use_fixed_markov = bool(
            not self.training
            and fixed_markov_transport is not None
            and bool(getattr(fixed_markov_transport, 'enabled', False))
        )
        if use_fixed_markov:
            encoded_scales = []
            for scale_idx in range(num_fusion_scales):
                if not self.compression:
                    raise RuntimeError(
                        'CoSDH fixed-Markov requires the official compressors'
                    )
                encoded_scales.append(
                    self.naive_compressor_list[scale_idx].encode_for_wire(
                        feature_list[scale_idx], use_fp16=True
                    )
                )
            recovered_encoded_scales =                 fixed_markov_transport.communicate_joint_frame(
                    encoded_scales,
                    record_len=record_len,
                    data_dict=data_dict,
                    link_key_aliases=data_dict.get('cav_id_list', None),
                )
            for i, fuse_module in enumerate(self.fusion_net):
                feature_i = self.naive_compressor_list[i].decode_from_wire(
                    recovered_encoded_scales[i], use_fp16=True
                )
                x_out, _ = fuse_module(
                    feature_i,
                    psm_single,
                    record_len,
                    normalized_affine_matrix,
                    req_mask,
                    scale_idx=i,
                    num_scales=num_fusion_scales,
                )
                fused_feature_list.append(x_out)
        else:
            for i, fuse_module in enumerate(self.fusion_net):
                feature_i = feature_list[i]
                if self.compression:
                    compressor = self.naive_compressor_list[i]
                    if (
                        not self.training
                        and self.cosdh_legacy_native_transport.enabled
                        and self.cosdh_legacy_native_transport.intermediate_enabled
                    ):
                        encoded_i = compressor.encode_for_wire(
                            feature_i,
                            use_fp16=True,
                        )
                        encoded_i = (
                            self.cosdh_legacy_native_transport
                            .roundtrip_intermediate(
                                encoded_i,
                                record_len=record_len,
                                scale_idx=i,
                                link_key_aliases=data_dict.get(
                                    'cav_id_list', None
                                ),
                            )
                        )
                        feature_i = compressor.decode_from_wire(
                            encoded_i,
                            use_fp16=True,
                        )
                    else:
                        feature_i = compressor(
                            feature_i,
                            use_fp16=not self.training,
                        )
                x_out, _ = fuse_module(
                    feature_i,
                    psm_single,
                    record_len,
                    normalized_affine_matrix,
                    req_mask,
                    scale_idx=i,
                    num_scales=num_fusion_scales,
                )

                fused_feature_list.append(x_out)
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)
        
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
        
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}

        if hasattr(self.cosdh_markov_channel, 'latest_info'):
            output_dict['comm_info'] = {
                'cosdh_markov': self.cosdh_markov_channel.latest_info,
                'cosdh_markov_enabled': getattr(self.cosdh_markov_channel, 'enabled', False),
            }

        # COSDH_OFFICIAL_FIXED_MARKOV_COMM_INFO
        fixed_markov_transport = getattr(
            self, 'cosdh_official_fixed_markov_transport', None
        )
        if (
            fixed_markov_transport is not None
            and bool(getattr(fixed_markov_transport, 'enabled', False))
        ):
            output_dict.setdefault('comm_info', {})[
                'cosdh_official_fixed_markov'
            ] = fixed_markov_transport.latest_info

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})
        
        return output_dict
