import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def _shape(x):
    return tuple(x.shape) if hasattr(x, "shape") else type(x)


def main():
    parser = argparse.ArgumentParser("Check RoCooper + V2X-Real config/model/dataset/forward")
    parser.add_argument("--hypes_yaml", required=True)
    parser.add_argument("--check_dataset", action="store_true")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sample_idx", type=int, default=0)
    args = parser.parse_args()

    hypes = yaml_utils.load_yaml(args.hypes_yaml, None)
    print("yaml:", args.hypes_yaml)
    print("model.core_method:", hypes["model"]["core_method"])
    print("fusion.core_method:", hypes["fusion"].get("core_method"))
    print("postprocess.core_method:", hypes["postprocess"].get("core_method"))
    print("loss.core_method:", hypes["loss"].get("core_method"))

    model_args = hypes["model"]["args"]
    anchor_number = int(model_args.get("anchor_number", model_args.get("anchor_num", 2)))
    num_class = int(model_args.get("num_class", 3))
    exp_psm = anchor_number * num_class * num_class
    exp_rm = 7 * anchor_number * num_class
    print("expected psm/rm channels:", exp_psm, exp_rm)

    model = train_utils.create_model(hypes)
    print("model class:", model.__class__.__name__)
    print("actual cls_head/reg_head out:", model.cls_head.out_channels, model.reg_head.out_channels)
    assert int(model.cls_head.out_channels) == exp_psm
    assert int(model.reg_head.out_channels) == exp_rm

    if hasattr(model, "comm_module"):
        comm = model.comm_module
        print("comm_module:", comm.__class__.__name__)
        print("comm enabled:", getattr(comm, "enabled", None))
        if hasattr(comm, "markov_enabled"):
            print("markov_enabled:", getattr(comm, "markov_enabled", None))
        if hasattr(comm, "states"):
            print("markov states:", getattr(comm, "states", None))

    if not args.check_dataset and not args.forward:
        print("STATIC_CHECK_OK")
        return

    dataset = build_dataset(hypes, visualize=False, train=False)
    print("dataset:", type(dataset))
    print("len:", len(dataset))

    sample = dataset[args.sample_idx]
    ego = sample["ego"]
    print("sample_idx:", args.sample_idx)
    print("sample ego keys:", sorted(list(ego.keys())))
    if "cav_num" in ego:
        print("sample cav_num:", ego["cav_num"])
    if "object_bbx_center" in ego:
        print("object_bbx_center shape:", _shape(ego["object_bbx_center"]))
    if "channel_state_ids" in ego:
        print("channel_state_ids:", ego["channel_state_ids"])

    batch = dataset.collate_batch_test([sample])
    batch_ego = batch["ego"]
    print("batch record_len:", batch_ego["record_len"].detach().cpu().tolist())
    if "object_bbx_center" in batch_ego:
        print("batch object_bbx_center shape:", _shape(batch_ego["object_bbx_center"]))

    if not args.forward:
        print("DATASET_CHECK_OK")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    batch = train_utils.to_device(batch, device)

    with torch.no_grad():
        output = model(batch["ego"])

    print("forward output keys:", sorted(list(output.keys())))
    print("forward psm shape:", _shape(output["psm"]))
    print("forward rm shape:", _shape(output["rm"]))
    assert output["psm"].shape[1] == exp_psm
    assert output["rm"].shape[1] == exp_rm

    if "comm_info" in output:
        ci = output["comm_info"]
        print("comm_info keys:", sorted(list(ci.keys())))
        if "link_records" in ci:
            print("link_records:", len(ci["link_records"]))
            if len(ci["link_records"]) > 0:
                print("first link_record:", ci["link_records"][0])
    if "fusion_info" in output:
        print("fusion_info keys:", sorted(list(output["fusion_info"].keys())))

    print("BATCH_CHECK_OK")


if __name__ == "__main__":
    main()
