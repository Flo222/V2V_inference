"""Composable shared communication pipeline interface.

Current experiments keep their exact, validated ARCE execution path. This
facade is the stable boundary for progressively moving transport mechanics out
of method code without changing experiment semantics.
"""
from typing import Any, Callable, Optional
from opencood.communication.interface import CommunicationAction, CommunicationResult


class CommunicationPipeline:
    def __init__(self, executor: Optional[Callable] = None):
        self.executor = executor

    def run(self, payload: Any, action: CommunicationAction, **kwargs) -> CommunicationResult:
        if self.executor is None:
            raise RuntimeError(
                "CommunicationPipeline requires an executor. Existing validated ARCE "
                "experiments use opencood.methods.arce.ARCEFixedComm/ARCEC2MABComm; "
                "pass the desired executor when adopting this facade."
            )
        out = self.executor(payload=payload, action=action, **kwargs)
        if isinstance(out, CommunicationResult):
            return out
        return CommunicationResult(payload=out)
