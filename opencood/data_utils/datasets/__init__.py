# -*- coding: utf-8 -*-
# Author: Runsheng Xu
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset_v2 import IntermediateFusionDatasetV2
from opencood.data_utils.datasets.intermediate_fusion_dataset_v2xreal import IntermediateFusionDatasetV2XReal
from opencood.data_utils.datasets.intermediate_fusion_dataset_coopdiff import IntermediateFusionDatasetCoopdiff
from opencood.data_utils.datasets.intermediate_fusion_dataset_coopdiff_v2xreal import IntermediateFusionDatasetCoopdiffV2XReal

__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'IntermediateFusionDatasetV2': IntermediateFusionDatasetV2,
    'IntermediateFusionDatasetV2XReal': IntermediateFusionDatasetV2XReal,
    'IntermediateFusionDatasetCoopdiff': IntermediateFusionDatasetCoopdiff,
    'IntermediateFusionDatasetCoopdiffV2XReal': IntermediateFusionDatasetCoopdiffV2XReal,
}

# Keep OPV2V original global range. V2X-Real uses V2XREAL_GT_RANGE inside its own dataset file.
GT_RANGE = [-140, -40, -3, 140, 40, 1]
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/__init__.py"
    assert dataset_name in __all__.keys(), error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
