"""Regression test: CoopDiff must use ARCE's ByteStreamPacketizer."""
import torch

from opencood.communication.channel.channel_manager import ChannelManager
from opencood.models.baselines.coopdiff.transport.coopdiff_markov_feature_channel import (
    CoopDiffMarkovFeatureChannel,
)


def main() -> None:
    adapter = CoopDiffMarkovFeatureChannel({
        "enabled": True,
        "fps": 10.0,
        "packetization": {"packet_size_bytes": 16, "zero_fill_missing": True},
    })
    manager = ChannelManager({
        "seed": 9,
        "channel": {
            "mode": "fixed",
            "fixed_state": "medium",
            "loss_model": "bernoulli",
            "bernoulli_loss_rates": {"good": 0.0, "medium": 0.0, "bad": 0.0},
            "profiles": {
                state: {"bandwidth_mbps": 1000.0, "packet_loss_rate": 0.0, "delay_ms": 0.0}
                for state in ("good", "medium", "bad")
            },
        },
    })
    adapter.set_channel_manager(manager)
    scale_a = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).view(2, 2, 3, 4)
    scale_b = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).view(2, 3, 2, 2)
    outputs, records = adapter.forward_multiscale([scale_a, scale_b], torch.tensor([2]), frame_id=1)
    assert len(records) == 1
    assert records[0]["packetizer_mode"] == "byte_stream"
    assert records[0]["packetizer"]["mode"] == "byte_stream"
    assert not hasattr(adapter, "_serialize_feature")
    assert torch.equal(outputs[0], scale_a)
    assert torch.equal(outputs[1], scale_b)
    print("CoopDiff ARCE byte-stream packetizer: PASS")


if __name__ == "__main__":
    main()
