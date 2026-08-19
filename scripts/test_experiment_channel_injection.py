"""Static construction checks for the shared experiment channel adapters."""
from __future__ import annotations

from pathlib import Path

import torch

from opencood.communication.channel.channel_manager import ChannelManager
from opencood.communication.experiment_channel import (
    validate_experiment_channel_configuration,
)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.methods.arce.executors.c2mab_executor import ARCEC2MABComm
from opencood.models.baselines.rocooper.components.rocooper_markov_comm import (
    RoCooperMarkovComm,
)
from opencood.tools.train_utils import create_model


TARGETS = (
    "opencood/hypes_yaml/v2xreal/point_pillar_rocooper_markov_v2xreal_vc.yaml",
    "opencood/hypes_yaml/v2xreal/point_pillar_cosdh_markov_v2xreal_vc.yaml",
    "opencood/hypes_yaml/v2xreal/point_pillar_diff_student_markov_v2xreal_vc.yaml",
    "opencood/hypes_yaml/v2xreal/point_pillar_v2xvit_native_payload_arce_markov_v2xreal_vc.yaml",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in TARGETS:
        hypes = load_yaml(str(root / relative), None)
        model = create_model(hypes)
        manager = getattr(model, "experiment_channel_manager", None)
        adapter_count = int(getattr(model, "experiment_channel_adapter_count", 0))
        assert manager is not None and manager.mode == "markov"
        assert adapter_count > 0
        if "rocooper" in relative:
            assert model.comm_module.__class__.__name__ == "RoCooperMarkovComm"
        print("OK", relative, "adapters={}".format(adapter_count))

    # Exercise RoCooper's actual adapter path without data/model dependencies.
    manager = ChannelManager({
        "seed": 7,
        "channel": {
            "mode": "markov",
            "initial_state": "medium",
            "loss_model": "bernoulli",
            "profiles": {
                "good": {"bandwidth_mbps": 27.0, "packet_loss_rate": 0.05, "delay_ms": 10.0},
                "medium": {"bandwidth_mbps": 5.0, "packet_loss_rate": 0.20, "delay_ms": 50.0},
                "bad": {"bandwidth_mbps": 1.0, "packet_loss_rate": 0.35, "delay_ms": 100.0},
            },
        },
    })
    comm = RoCooperMarkovComm({"enabled": True, "channel_state_mode": "markov"})
    comm.set_channel_manager(manager)
    output, info = comm(torch.ones(2, 4, 8, 8), [2])
    assert output.shape == (2, 4, 8, 8)
    assert info["packet_loss_enabled"] and info["delay_enabled"]
    print("OK rocooper_public_transport")

    # C2MAB configuration/reward parsing is tested elsewhere.  This verifies
    # the requested public injection interface without changing UCB policy.
    class Recorder:
        received = None

        def set_channel_manager(self, value):
            self.received = value

    recorder = Recorder()
    c2mab = ARCEC2MABComm.__new__(ARCEC2MABComm)
    c2mab.executor = recorder
    c2mab.set_channel_manager(manager)
    assert recorder.received is manager
    print("OK where2comm_c2mab_set_channel_manager")

    invalid = {
        "communication_environment": {"enabled": True, "strict": True},
        "model": {"args": {"cosdh_markov": {"bandwidth_mbps": 9.0}}},
    }
    try:
        validate_experiment_channel_configuration(invalid)
    except ValueError:
        print("OK strict_private_physics_rejected")
    else:
        raise AssertionError("strict validation accepted a private bandwidth setting")


if __name__ == "__main__":
    main()
