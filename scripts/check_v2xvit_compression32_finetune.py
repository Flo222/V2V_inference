#!/usr/bin/env python3
"""Check a V2X-ViT compression-32 fine-tuning config before training."""

import argparse
from types import SimpleNamespace

import torch

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    hypes = yaml_utils.load_yaml(args.config, SimpleNamespace(model_dir=""))
    model_args = hypes["model"]["args"]
    wild = hypes["wild_setting"]
    assert model_args["compression"] == 32
    assert "arce" not in model_args
    assert wild["async"] is False
    assert wild["loc_err"] is False
    assert "channel_state_markov" not in wild

    model = train_utils.create_model(hypes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint, strict=False)
    assert getattr(model, "compression", False)
    assert hasattr(model, "naive_compressor")

    print("CONFIG_OK compression=32 arce=off async=off loc_err=off")
    print("CHECKPOINT_OK missing={} unexpected={}".format(
        len(incompatible.missing_keys), len(incompatible.unexpected_keys)
    ))
    print("MISSING_KEYS={}".format(",".join(incompatible.missing_keys)))


if __name__ == "__main__":
    main()
