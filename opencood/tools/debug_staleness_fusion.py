#!/usr/bin/env python3
from __future__ import annotations
import torch
from opencood.communication.transport.recovery.staleness_fusion import blend_current_cache, compute_staleness_quality


def main():
    cur = torch.ones(2, 3)
    cache = torch.zeros(2, 3)
    for delay in [20, 100, 300, 500]:
        out, info = blend_current_cache(cur, cache, q_recv=0.8, q_cache=0.5, delay_ms=delay)
        print(delay, info.as_dict(), 'mean=', float(out.mean()))
    print('q_eff bad:', compute_staleness_quality(0.8, 400, 300))

if __name__ == '__main__':
    main()
