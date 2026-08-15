"""Method-to-transport contract.

A communication method/controller decides *what configuration to use*; the
shared communication layer implements the mechanics. This dataclass is kept
small on purpose so ARCE and future controllers can share the same contract.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CommunicationAction:
    transmit: bool = True
    quantization: str = "fp32"
    fec: str = "none"
    redundancy_ratio: float = 0.0
    recovery: str = "zero_fill"
    packet_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
