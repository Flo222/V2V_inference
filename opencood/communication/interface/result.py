from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CommunicationResult:
    payload: Any
    stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
