"""Regression checks for the Bernoulli-only public packet-loss model."""
import torch

from opencood.communication.channel.channel_manager import ChannelManager


def main() -> None:
    cfg = {
        "seed": 2026,
        "channel": {
            "mode": "markov",
            "initial_state": "medium",
            "loss_model": "bernoulli",
            "bernoulli_loss_rates": {"good": 0.05, "medium": 0.20, "bad": 0.35},
            "profiles": {
                "good": {"bandwidth_mbps": 27.0, "packet_loss_rate": 0.05},
                "medium": {"bandwidth_mbps": 5.0, "packet_loss_rate": 0.20},
                "bad": {"bandwidth_mbps": 1.0, "packet_loss_rate": 0.35},
            },
        },
    }
    manager = ChannelManager(cfg)
    mask_a, info_a = manager.sample_packet_loss(128, link_id=(0, 1), frame_id=3)
    mask_b, info_b = manager.sample_packet_loss(128, link_id=(0, 1), frame_id=3)
    assert info_a["model"] == "bernoulli"
    assert info_b["model"] == "bernoulli"
    assert torch.equal(mask_a, mask_b), "same link/frame sample must be deterministic"
    try:
        ChannelManager({"channel": {"loss_model": "ge"}})
    except ValueError as exc:
        assert "Gilbert-Elliott" in str(exc)
    else:
        raise AssertionError("obsolete GE loss_model was accepted")
    print("channel manager Bernoulli-only checks: PASS")


if __name__ == "__main__":
    main()
