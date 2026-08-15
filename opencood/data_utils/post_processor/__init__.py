# -*- coding: utf-8 -*-
# Author: Runsheng Xu
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.data_utils.post_processor.bev_postprocessor import BevPostprocessor
from opencood.data_utils.post_processor.ciassd_postprocessor import CiassdPostprocessor
from opencood.data_utils.post_processor.fpvrcnn_postprocessor import FpvrcnnPostprocessor
from opencood.data_utils.post_processor.voxel_postprocessor_v2xreal import VoxelPostprocessorV2XReal

__all__ = {
    'VoxelPostprocessor': VoxelPostprocessor,
    'BevPostprocessor': BevPostprocessor,
    'CiassdPostprocessor': CiassdPostprocessor,
    'FpvrcnnPostprocessor': FpvrcnnPostprocessor,
    'VoxelPostprocessorV2XReal': VoxelPostprocessorV2XReal,
}


def build_postprocessor(anchor_cfg, train=True, class_names=None):
    process_method_name = anchor_cfg['core_method']
    assert process_method_name in __all__.keys(), \
        f"{process_method_name} is not registered in post_processor/__init__.py"

    postprocessor_cls = __all__[process_method_name]

    if process_method_name == 'VoxelPostprocessorV2XReal':
        return postprocessor_cls(
            anchor_params=anchor_cfg,
            class_names=class_names,
            train=train
        )

    return postprocessor_cls(
        anchor_params=anchor_cfg,
        train=train
    )
