#!/usr/bin/env python3
"""Make a reproducible V2X-ViT compression-32 perfect-communication config.

The resulting config deliberately keeps the checkpoint's data split and model
architecture, but removes ARCE and all Markov settings so the short second
stage is the native V2X-ViT compression fine-tuning stage.
"""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epoches", required=True, type=int)
    args = parser.parse_args()

    with open(args.source_config, "r", encoding="utf-8") as stream:
        hypes = yaml.safe_load(stream)

    hypes["name"] = args.name

    wild_setting = hypes.setdefault("wild_setting", {})
    wild_setting["async"] = False
    wild_setting["async_mode"] = "sim"
    wild_setting["async_overhead"] = 0
    wild_setting["loc_err"] = False
    wild_setting["xyz_std"] = 0
    wild_setting["ryp_std"] = 0
    wild_setting.pop("channel_state_markov", None)

    model_args = hypes.setdefault("model", {}).setdefault("args", {})
    model_args["compression"] = 32
    # ARCE can contain a private legacy transport definition in old run
    # directories. Removing it makes this explicitly native perfect comm.
    model_args.pop("arce", None)
    hypes.pop("communication_environment", None)

    train_params = hypes.setdefault("train_params", {})
    train_params["epoches"] = args.epoches
    train_params["save_freq"] = 1
    train_params["eval_freq"] = 1

    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(hypes, stream, sort_keys=False)

    print("Wrote {}".format(output_path))
    print("  compression={}".format(model_args["compression"]))
    print("  arce_present={}".format("arce" in model_args))
    print("  async={}, loc_err={}".format(
        wild_setting["async"], wild_setting["loc_err"]
    ))
    print("  total_epoches={}".format(train_params["epoches"]))


if __name__ == "__main__":
    main()
