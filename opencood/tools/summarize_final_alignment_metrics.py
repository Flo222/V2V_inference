#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from opencood.communication.metrics.final_alignment_metrics import read_jsonl, summarize_records, summarize_by_channel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comm-jsonl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--q-min', type=float, default=0.3)
    ap.add_argument('--eff-patch-min', type=float, default=0.2)
    args = ap.parse_args()
    rows = read_jsonl(args.comm_jsonl)
    summary = {
        'overall': summarize_records(rows, fps=args.fps, q_min=args.q_min, eff_patch_min=args.eff_patch_min),
        'by_channel': summarize_by_channel(rows, fps=args.fps, q_min=args.q_min, eff_patch_min=args.eff_patch_min),
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(args.out)
    print(json.dumps(summary['overall'], indent=2, ensure_ascii=False)[:3000])

if __name__ == '__main__':
    main()
