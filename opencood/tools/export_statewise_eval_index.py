#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from opencood.communication.metrics.final_alignment_metrics import read_jsonl, infer_frame_id, _get


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comm-jsonl', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = read_jsonl(args.comm_jsonl)
    seen = {}
    for r in rows:
        fid = infer_frame_id(r)
        if fid not in seen:
            seen[fid] = r.get('channel_state') or _get(r, 'channel', 'state', default='unknown')
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_id', 'channel_state'])
        for fid, state in sorted(seen.items()):
            w.writerow([fid, state])
    print(args.out)

if __name__ == '__main__':
    main()
