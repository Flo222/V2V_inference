from opencood.data_utils.datasets.intermediate_late_fusion_dataset_cosdh import getIntermediatelateFusionDataset
from opencood.data_utils.datasets.opv2v_basedataset_cosdh import OPV2VBaseDataset
from opencood.data_utils.datasets.intermediate_late_fusion_dataset_cosdh_v2xreal import getIntermediatelateFusionDatasetV2XReal
from opencood.data_utils.datasets import build_dataset


def _is_v2xreal_cfg(dataset_cfg):
    if "dataset_mode" in dataset_cfg:
        return True
    root_dir = str(dataset_cfg.get("root_dir", "")).lower()
    val_dir = str(dataset_cfg.get("validate_dir", "")).lower()
    test_dir = str(dataset_cfg.get("test_dir", "")).lower()
    return "v2x-real" in root_dir or "v2xreal" in root_dir or \
           "v2x-real" in val_dir or "v2xreal" in val_dir or \
           "v2x-real" in test_dir or "v2xreal" in test_dir


def build_dataset_cosdh(dataset_cfg, visualize=False, train=True):
    fusion_name = dataset_cfg["fusion"]["core_method"]

    if fusion_name == "intermediatelate":
        if _is_v2xreal_cfg(dataset_cfg):
            dataset_cls = getIntermediatelateFusionDatasetV2XReal()
            return dataset_cls(params=dataset_cfg, visualize=visualize, train=train)

        dataset_cls = getIntermediatelateFusionDataset(OPV2VBaseDataset)
        return dataset_cls(params=dataset_cfg, visualize=visualize, train=train)

    return build_dataset(dataset_cfg, visualize=visualize, train=train)
