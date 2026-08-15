from __future__ import annotations

from typing import Any, Dict, List


class RewardBuffer:
    """FIFO buffer for delayed C2MAB reward updates."""

    def __init__(self):
        self.items: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]) -> None:
        self.items.append(dict(item))

    def pop_all(self) -> List[Dict[str, Any]]:
        items = list(self.items)
        self.items.clear()
        return items

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)


def effective_receive_quality(
    packet_loss_rate: float = 0.0,
    delay_ms: float = 0.0,
    stale_ms: float = 0.0,
    tau_stale_ms: float = 300.0,
) -> float:
    """Estimate feedback quality for bookkeeping; this is not a positive reward term."""
    plr = min(max(float(packet_loss_rate), 0.0), 1.0)
    stale = max(float(stale_ms), max(float(delay_ms), 0.0))
    tau = max(float(tau_stale_ms), 1e-6)
    stale_quality = max(0.0, min(1.0, 1.0 - stale / tau))
    return float((1.0 - plr) * stale_quality)


__all__ = ["RewardBuffer", "effective_receive_quality"]
