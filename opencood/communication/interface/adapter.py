"""Baseline communication adapter interface.

Adapters preserve each paper/baseline's native send semantics while exposing a
uniform boundary to the shared communication stack.
"""
from abc import ABC, abstractmethod
from typing import Any


class BaselineCommunicationAdapter(ABC):
    @abstractmethod
    def build_message(self, *args, **kwargs) -> Any:
        """Convert baseline-native tensors/metadata into a communication message."""
        raise NotImplementedError

    @abstractmethod
    def restore(self, message: Any, *args, **kwargs) -> Any:
        """Convert a received/recovered message back to baseline-native form."""
        raise NotImplementedError
