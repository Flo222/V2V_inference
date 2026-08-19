#!/usr/bin/env python
from __future__ import annotations

import json

try:
    from importlib.metadata import version
except ImportError:  # Python 3.7
    from importlib_metadata import version

from opencood.methods.arce.policy.c2mab.action_space import build_pdf_action_space
from opencood.communication.transport.fec.fec_raptorq import require_raptorq_backend


def main() -> None:
    require_raptorq_backend()
    backend_version = version("raptorq")
    if backend_version != "1.6.3":
        raise RuntimeError(
            "Expected raptorq==1.6.3, got {}".format(backend_version)
        )

    actions = build_pdf_action_space(
        fec_mode="raptorq",
        quant_modes=("fp16", "int8", "int4"),
        redundancy_ratios=(0.0, 0.10, 0.25, 0.60),
        cache_values=(0, 1),
    )
    if len(actions) != 25:
        raise RuntimeError("Expected 25 online actions, got {}".format(len(actions)))
    invalid = [
        action.action_id
        for action in actions
        if float(action.redundancy_ratio) > 0.0
        and str(action.fec_type) != "raptorq"
    ]
    if invalid:
        raise RuntimeError("Non-RaptorQ redundant actions: {}".format(invalid))

    print(json.dumps({
        "pass": True,
        "backend": "raptorq",
        "backend_version": backend_version,
        "standard": "RFC6330",
        "action_count": len(actions),
        "redundant_action_count": sum(
            float(action.redundancy_ratio) > 0.0 for action in actions
        ),
        "scheduling": "exact_ratio_protected_prefix_with_best_effort_tail",
        "redundancy_policy": (
            "exact_protected_prefix_with_best_effort_tail"
        ),
        "source_symbol_bytes": 1024,
        "wire_packet_bytes": 1032,
    }, indent=2))


if __name__ == "__main__":
    main()
