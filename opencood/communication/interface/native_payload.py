from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Sequence, Union

import torch


RecordLen = Union[torch.Tensor, Sequence[int]]


def record_len_to_list(record_len: RecordLen) -> List[int]:
    if torch.is_tensor(record_len):
        values = record_len.detach().cpu().reshape(-1).tolist()
    elif isinstance(record_len, (list, tuple)):
        values = list(record_len)
    else:
        raise TypeError(
            "record_len must be a torch.Tensor, list, or tuple, got {}".format(
                type(record_len)
            )
        )

    result = [int(value) for value in values]
    if not result or any(value <= 0 for value in result):
        raise ValueError("record_len must contain positive integers")
    return result


@dataclass
class NativePayload:
    """Baseline-native payload passed to the physical communication layer.

    This object is deliberately small and baseline-agnostic. The baseline is
    responsible for completing its own native processing first, such as:
      * shrink/native compressor;
      * spatial mask or block selection;
      * intermediate/late branch selection.

    ARCE then operates on ``values`` without reconstructing a larger tensor or
    repeating auxiliary priors over every BEV cell.
    """

    values: torch.Tensor
    record_len: RecordLen
    payload_type: str
    stage: str
    layout: str = "NCHW"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "NativePayload":
        if not torch.is_tensor(self.values):
            raise TypeError("NativePayload.values must be a torch.Tensor")
        if self.layout != "NCHW":
            raise ValueError(
                "Current ARCE bridge supports NCHW payloads, got {}".format(
                    self.layout
                )
            )
        if self.values.dim() != 4:
            raise ValueError(
                "NCHW payload must have shape [sum(record_len), C, H, W], got {}".format(
                    tuple(self.values.shape)
                )
            )

        lengths = record_len_to_list(self.record_len)
        expected = sum(lengths)
        if expected != int(self.values.shape[0]):
            raise ValueError(
                "sum(record_len)={} does not match values.shape[0]={}".format(
                    expected, int(self.values.shape[0])
                )
            )
        if not self.payload_type:
            raise ValueError("payload_type must not be empty")
        if not self.stage:
            raise ValueError("stage must not be empty")
        return self

    @property
    def record_len_list(self) -> List[int]:
        return record_len_to_list(self.record_len)

    @property
    def collaborator_count(self) -> int:
        return sum(max(0, value - 1) for value in self.record_len_list)

    @property
    def value_bytes(self) -> int:
        """Complete model tensor bytes, including ego features."""
        return int(
            self.values.numel()
            * self.values.element_size()
        )

    @property
    def bytes_per_agent(self) -> int:
        """Native dense feature bytes for one real CAV."""
        total_agents = sum(self.record_len_list)
        if total_agents <= 0:
            return 0
        if int(self.values.shape[0]) != total_agents:
            raise ValueError(
                "values.shape[0]={} does not match "
                "sum(record_len)={}".format(
                    int(self.values.shape[0]),
                    total_agents,
                )
            )
        return int(
            self.values[0].numel()
            * self.values.element_size()
        )

    @property
    def non_ego_value_bytes(self) -> int:
        """Feature bytes actually offered to sender-to-ego wireless links."""
        return int(
            self.collaborator_count
            * self.bytes_per_agent
        )

    @property
    def total_wire_bytes_estimate(self) -> int:
        """Non-ego feature values plus the declared per-link auxiliaries."""
        return int(
            self.non_ego_value_bytes
            + self.total_aux_bytes_estimate
        )

    @property
    def per_link_aux_bytes(self) -> int:
        return int(self.metadata.get("per_link_aux_bytes", 0) or 0)

    @property
    def total_aux_bytes_estimate(self) -> int:
        return int(self.collaborator_count * self.per_link_aux_bytes)

    def with_values(self, values: torch.Tensor) -> "NativePayload":
        return replace(self, values=values).validate()

    def summary(self) -> Dict[str, Any]:
        self.validate()
        return {
            "interface": "native_payload_v1",
            "payload_type": str(self.payload_type),
            "stage": str(self.stage),
            "layout": str(self.layout),
            "shape": [int(v) for v in self.values.shape],
            "dtype": str(self.values.dtype),
            "record_len": self.record_len_list,
            "collaborator_count": int(self.collaborator_count),
            "model_tensor_bytes_including_ego": int(
                self.value_bytes
            ),
            "bytes_per_agent": int(
                self.bytes_per_agent
            ),
            "non_ego_feature_bytes_before_channel": int(
                self.non_ego_value_bytes
            ),
            "per_link_aux_bytes_estimate": int(
                self.per_link_aux_bytes
            ),
            "total_aux_bytes_estimate": int(
                self.total_aux_bytes_estimate
            ),
            "total_wire_bytes_estimate_before_quant": int(
                self.total_wire_bytes_estimate
            ),
            "metadata": copy.deepcopy(self.metadata),
        }


__all__ = ["NativePayload", "record_len_to_list"]
