#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
import numpy as np

STATES = ['good', 'medium', 'bad']
P = np.array([[0.90,0.09,0.01],[0.12,0.82,0.06],[0.02,0.23,0.75]], dtype=float)
PROFILES = {
    'good': {'bandwidth_mbps': 27.0, 'loss_rate': 0.03},
    'medium': {'bandwidth_mbps': 5.0, 'loss_rate': 0.12},
    'bad': {'bandwidth_mbps': 1.0, 'loss_rate': 0.28},
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-frames', type=int, required=True)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--initial-state', choices=STATES, default='medium')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cur = STATES.index(args.initial_state)
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_id','channel_state','bandwidth_mbps','loss_rate','B_link_bytes'])
        for t in range(args.num_frames):
            s = STATES[cur]
            bw = PROFILES[s]['bandwidth_mbps']
            b_link = bw * 1e6 * 0.1 / 8.0
            w.writerow([t, s, bw, PROFILES[s]['loss_rate'], int(round(b_link))])
            cur = int(rng.choice(len(STATES), p=P[cur]))
    print(args.out)

if __name__ == '__main__':
    main()
