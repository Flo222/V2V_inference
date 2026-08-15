import argparse
import os
import sys
from pathlib import Path

import torch

# Allow running as: python opencood/tools/check_cosdh_v2xreal.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def _as_shape(x):
    if hasattr(x, "shape"):
        return tuple(x.shape)
    return type(x)


def main():
    parser = argparse.ArgumentParser(
        description="Static / one-batch checker for CoSDH + V2X-Real configs."
    )
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
    cls_out = int(model.cls_head.out_channels)
    reg_out = int(model.reg_head.out_channels)
    print("actual cls_head/reg_head out:", cls_out, reg_out)
    assert cls_out == exp_psm, (cls_out, exp_psm)
    assert reg_out == exp_rm, (reg_out, exp_rm)

    if hasattr(model, "cosdh_markov_channel"):
        ch = model.cosdh_markov_channel
        print("cosdh_markov enabled:", getattr(ch, "enabled", None))
        print("cosdh_markov states:", getattr(ch, "states", None))
        print("cosdh_markov initial_state:", getattr(ch, "initial_state", None))

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
        print("object_bbx_center shape:", _as_shape(ego["object_bbx_center"]))
    if "channel_state_ids" in ego:
        print("channel_state_ids:", ego["channel_state_ids"])

    batch = dataset.collate_batch_test([sample])
    batch_ego = batch["ego"]
    print("batch record_len:", batch_ego["record_len"].detach().cpu().tolist())
    if "prior_encoding" in batch_ego:
        print("prior_encoding shape:", _as_shape(batch_ego["prior_encoding"]))
    if "object_bbx_center" in batch_ego:
        print("batch object_bbx_center shape:", _as_shape(batch_ego["object_bbx_center"]))

    if not args.forward:
        print("DATASET_CHECK_OK")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    batch = train_utils.to_device(batch, device)

    with torch.no_grad():
        output = model(batch["ego"])

    print("forward output keys:", sorted(list(output.keys())))
    print("forward psm shape:", _as_shape(output["psm"]))
    print("forward rm shape:", _as_shape(output["rm"]))
    assert output["psm"].shape[1] == exp_psm, output["psm"].shape
    assert output["rm"].shape[1] == exp_rm, output["rm"].shape

    if "comm_info" in output:
        comm_info = output["comm_info"]
        print("comm_info keys:", sorted(list(comm_info.keys())))
        if "cosdh_markov" in comm_info:
            info = comm_info["cosdh_markov"]
            print("cosdh_markov link records:", len(info))
            if len(info) > 0:
                print("first cosdh_markov link record:", info[0])

    print("BATCH_CHECK_OK")


if __name__ == "__main__":
    main()
