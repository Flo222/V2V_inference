#!/usr/bin/env python
"""Summarize DC2MAB communication jsonl logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = Path(args.jsonl)
    action_counter = Counter()
    sender_counter = Counter()
    num_records = 0
    num_no_send = 0
    tx = 0.0
    rx = 0.0
    rewards = []
    superarms = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            num_records += 1
            if item.get("no_send"):
                num_no_send += 1
            if "pdf_action" in item:
                aid = item["pdf_action"].get("action_id", "unknown")
                action_counter[aid] += 1
            if "agent_index" in item:
                sender_counter[str(item["agent_index"])] += 1
            tx += float(item.get("actual_transmitted_bytes", item.get("transmitted_bytes", 0.0)) or 0.0)
            rx += float(item.get("actual_received_bytes", item.get("received_bytes", 0.0)) or 0.0)
            if "reward_update" in item:
                rewards.append(float(item["reward_update"].get("reward", 0.0)))
            if "dc2mab_superarm" in item:
                superarms += 1

    summary = {
        "num_records": num_records,
        "num_superarm_records": superarms,
        "num_no_send": num_no_send,
        "tx_mb": tx / 1_000_000.0,
        "rx_mb": rx / 1_000_000.0,
        "num_reward_updates": len(rewards),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "action_counter": dict(action_counter.most_common()),
        "sender_counter": dict(sender_counter.most_common()),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
