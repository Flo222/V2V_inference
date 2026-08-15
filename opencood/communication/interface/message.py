"""Baseline-agnostic native message representation.

``NativePayload`` is retained for exact compatibility with current dense NCHW
experiments. ``NativeMessage``/``MessageSegment`` are the structured interface
for new multi-segment baselines (for example CoSDH and CoopDiff).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MessageSegment:
    name: str
    values: Any
    codec: Optional[str] = None
    layout: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NativeMessage:
    baseline: str
    segments: List[MessageSegment]
    record_len: Any = None
    stage: str = "native_tx"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.segments:
            raise ValueError("NativeMessage must contain at least one MessageSegment")
        names = [s.name for s in self.segments]
        if len(names) != len(set(names)):
            raise ValueError("MessageSegment names must be unique within one NativeMessage")

    def get(self, name: str) -> MessageSegment:
        for segment in self.segments:
            if segment.name == name:
                return segment
        raise KeyError(name)
