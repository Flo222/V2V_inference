#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit checks for communication record normalization."""

from __future__ import print_function

import importlib.util
import sys
import types
from pathlib import Path


def load_module():
    # Parser tests do not need model construction. Stub the heavy OpenCOOD
    # imports so this unit test can run even outside the full conda env.
    opencood = types.ModuleType("opencood")
    hypes_yaml = types.ModuleType("opencood.hypes_yaml")
    yaml_utils = types.ModuleType("opencood.hypes_yaml.yaml_utils")
    tools = types.ModuleType("opencood.tools")
    train_utils = types.ModuleType("opencood.tools.train_utils")
    data_utils = types.ModuleType("opencood.data_utils")
    datasets = types.ModuleType("opencood.data_utils.datasets")
    datasets.build_dataset = lambda *args, **kwargs: None
    hypes_yaml.yaml_utils = yaml_utils
    tools.train_utils = train_utils

    sys.modules.setdefault("opencood", opencood)
    sys.modules.setdefault("opencood.hypes_yaml", hypes_yaml)
    sys.modules.setdefault("opencood.hypes_yaml.yaml_utils", yaml_utils)
    sys.modules.setdefault("opencood.tools", tools)
    sys.modules.setdefault("opencood.tools.train_utils", train_utils)
    sys.modules.setdefault("opencood.data_utils", data_utils)
    sys.modules.setdefault("opencood.data_utils.datasets", datasets)

    path = Path(__file__).resolve().parent / "markov_bw_audit.py"
    spec = importlib.util.spec_from_file_location("markov_bw_audit", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    m = load_module()

    arce = {
        "frame_id": 1,
        "link_id": "(0,0,1)",
        "channel_state": "medium",
        "tx_bytes": 7168,
        "rx_bytes": 5120,
        "raw_bytes": 8650752,
        "size": {"bandwidth_budget_bytes": 7224},
    }
    row = m.normalize_record(arce, 1, 0, "test")
    assert row["tx_bytes"] == 7168
    assert row["state"] == "medium"

    coopdiff = {
        "frame_id": 2,
        "link_key": "b0_cav1",
        "state": "good",
        "scale_idx": 1,
        "message_bytes": 100000,
        "consumed_bytes": 337000,
        "received_units": 10,
        "initial_budget_bytes": 337000,
    }
    row = m.normalize_record(coopdiff, 2, 1, "test")
    assert row["tx_bytes"] == 337000
    assert row["scale_idx"] == 1

    cosdh = {
        "channel_links": [{
            "frame_id": 3,
            "link_key": "1054",
            "state": "medium",
            "source_bytes": 3352276,
            "sent_bytes_before_loss": 62464,
            "received_valid_bytes": 53248,
            "budget_bytes": 62464,
            "budget_truncated_bytes": 3289812,
        }]
    }
    rows = m.extract_records_from_structure(cosdh, 3, 2, "test")
    assert len(rows) == 1
    assert rows[0]["tx_bytes"] == 62464
    assert rows[0]["rx_bytes"] == 53248

    rocooper = {
        "enabled": True,
        "state": "bad",
        "actual_tx_bytes": 12288,
        "actual_rx_bytes": 8192,
        "source_bytes": 9011200,
        "bandwidth_budget_bytes": 12500,
    }
    row = m.normalize_record(rocooper, 4, 3, "test")
    assert row["tx_bytes"] == 12288
    assert row["state"] == "bad"

    duplicate = m.deduplicate_records([rows[0], dict(rows[0])])
    assert len(duplicate) == 1

    print("Markov BW record parser unit checks passed.")


if __name__ == "__main__":
    main()
