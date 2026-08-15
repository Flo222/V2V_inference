#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencood.methods.arce.policies.action_space import build_pdf_action_space


def rho_values(actions):
    return sorted({
        round(float(a.redundancy_ratio), 8)
        for a in actions
        if not a.is_no_send
    })


def main():
    actions = build_pdf_action_space(
        send_values=[0, 1],
        quant_modes=["fp16", "int8", "int4"],
        redundancy_ratios=[0.0, 0.10, 0.25, 0.60],
        cache_values=[0, 1],
    )
    assert len(actions) == 25, len(actions)
    assert rho_values(actions) == [0.0, 0.1, 0.25, 0.6]
    assert sum(1 for a in actions if a.is_no_send) == 1
    assert len({a.action_id for a in actions}) == 25
    print("Final Markov+C2MAB current 25-arm action-space test passed.")


if __name__ == "__main__":
    main()
