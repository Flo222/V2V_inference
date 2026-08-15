"""Read-only ARCE diagnostic helpers.

All audit helpers are disabled by default and must never modify tensors used by
normal inference.
"""

from .compression_auditor import CompressionAuditor
from .fec_recovery_auditor import FECRecoveryAuditor

__all__ = ["CompressionAuditor", "FECRecoveryAuditor"]
