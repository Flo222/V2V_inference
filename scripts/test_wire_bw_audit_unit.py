#!/usr/bin/env python3
from __future__ import print_function

import torch
import torch.nn as nn

from opencood.communication.metrics.ideal_wire_auditor import IdealWireAuditor
from opencood.models.baselines.coopdiff.transport.coopdiff_markov_feature_channel import (
    CoopDiffMarkovFeatureChannel,
)
from opencood.models.baselines.cosdh.transport.cosdh_markov_byte_channel import (
    CosDHMarkovByteChannel,
)
from opencood.models.baselines.rocooper.components.rocooper_comm import RoCooperComm


class _V2XFusion(nn.Module):
    def forward(self, x, mask, spatial):
        del mask, spatial
        return x[:, 0]


class _V2XModel(nn.Module):
    def __init__(self):
        super(_V2XModel, self).__init__()
        self.fusion_net = _V2XFusion()

    def forward(self):
        feature = torch.zeros(1, 2, 2, 2, 7)
        valid = torch.tensor([[1, 1]], dtype=torch.bool)
        return self.fusion_net(feature, valid, None)


def _channel_cfg():
    return {
        "enabled": True,
        "fps": 10,
        "states": ["good"],
        "initial_state": "good",
        "transition_matrix": {"good": {"good": 1.0}},
        "state_profiles": {
            "good": {
                "bandwidth_mbps": 100,
                "packet_loss_rate": 0.0,
                "delay_ms": 0,
                "temporal_source": "current",
            }
        },
        "packetization": {
            "packet_size_bytes": 16,
            "bytes_per_value": 4,
        },
    }


def main():
    model = _V2XModel()
    auditor = IdealWireAuditor(model, "v2xvit", packet_size_bytes=16)
    auditor.start_frame()
    model()
    records = auditor.finish_frame("frame", 0)
    auditor.close()
    assert len(records) == 1
    assert records[0]["source_payload_bytes"] == 64
    assert records[0]["transmitted_wire_bytes"] == 64

    channel = CoopDiffMarkovFeatureChannel(_channel_cfg())
    channel.start_frame("frame")
    _, records = channel(
        torch.ones(2, 3, 2, 2), torch.tensor([2]), frame_id="frame"
    )
    assert records[0]["transmitted_wire_bytes"] == 48

    channel = CosDHMarkovByteChannel(_channel_cfg())
    channel.start_frame("frame")
    _, records = channel(
        torch.ones(2, 3, 2, 2),
        torch.tensor([2]),
        torch.ones(2, 1, 2, 2),
        "frame",
    )
    assert records[0]["transmitted_wire_bytes"] == 48

    channel = RoCooperComm({
        "enabled": True,
        "test_with_impairment": True,
        "network_loss": {},
        "wire_audit": {"packet_size_bytes": 16},
    })
    channel.eval()
    _, info = channel(torch.ones(2, 3, 2, 2), torch.tensor([2]))
    assert info["wire_records"][0]["transmitted_wire_bytes"] == 48

    print("wire BW audit unit tests: PASS")


if __name__ == "__main__":
    main()
