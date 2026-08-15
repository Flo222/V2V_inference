#!/usr/bin/env python3
"""Validate exact decision/update contexts in an ARCE counterfactual CSV."""

from __future__ import print_function

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


NAMES = (
    "B_norm",
    "p_loss",
    "d_norm",
    "ego_confidence",
    "cache_quality",
    "complementarity",
    "cav_confidence",
)


def number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} is not numeric: {!r}".format(label, value))
    if not math.isfinite(result):
        raise ValueError("{} is not finite: {!r}".format(label, value))
    return result


def truth(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def stats(values):
    values = [float(value) for value in values]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
        "nonzero": sum(value > 1e-12 for value in values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    with Path(args.csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("CSV is empty")

    required = {
        "sequence_id",
        "sequence_frame_idx",
        "sender_index",
        "action_id",
        "no_send",
        "decision_context_available",
        "update_context_available",
        "decision_update_context_max_abs_diff",
    }
    for prefix in ("decision_context_", "update_context_"):
        required.update(prefix + name for name in NAMES)
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise RuntimeError("Missing runtime context columns: {}".format(missing))

    groups = defaultdict(list)
    for row in rows:
        key = (
            row["sequence_id"],
            row["sequence_frame_idx"],
            row["sender_index"],
        )
        groups[key].append(row)

    unavailable = []
    group_variation = []
    send_diffs = []
    no_send_diffs = []
    context_values = defaultdict(list)
    action_counter = Counter()

    for key, group in groups.items():
        if len(group) != 7:
            raise RuntimeError("Group {} contains {} rows".format(key, len(group)))
        vectors = []
        for row in group:
            action_counter[row["action_id"]] += 1
            if not truth(row["decision_context_available"]):
                unavailable.append((key, row["action_id"], "decision"))
                continue
            vector = [
                number(row["decision_context_" + name], name) for name in NAMES
            ]
            vectors.append(vector)
            for name, value in zip(NAMES, vector):
                context_values[name].append(value)

            if truth(row["update_context_available"]):
                diff = number(
                    row["decision_update_context_max_abs_diff"],
                    "decision_update_context_max_abs_diff",
                )
                if truth(row["no_send"]):
                    no_send_diffs.append(diff)
                else:
                    send_diffs.append(diff)
            else:
                unavailable.append((key, row["action_id"], "update"))

        if vectors:
            reference = vectors[0]
            group_variation.append(
                max(
                    abs(a - b)
                    for vector in vectors[1:]
                    for a, b in zip(reference, vector)
                )
                if len(vectors) > 1
                else 0.0
            )

    print("rows:", len(rows))
    print("groups:", len(groups))
    print("actions:", dict(action_counter))
    print("unavailable contexts:", len(unavailable))
    print("group decision-context variation:", stats(group_variation))
    print("send decision/update difference:", stats(send_diffs))
    print("no-send decision/update difference:", stats(no_send_diffs))
    print("context distributions:")
    for name in NAMES:
        print(" ", name, stats(context_values[name]))

    if unavailable:
        print("unavailable examples:", unavailable[:10])
        raise RuntimeError("Some runtime contexts are unavailable")
    if max(group_variation or [0.0]) > 1e-9:
        raise RuntimeError("Decision context is not action-invariant within a group")
    if max(send_diffs or [0.0]) > 1e-9:
        raise RuntimeError("Send decision and update contexts do not match")
    if all(
        max((abs(value) for value in context_values[name]), default=0.0) <= 1e-12
        for name in NAMES
    ):
        raise RuntimeError(
            "Every recorded decision-context component is zero; context "
            "serialization is invalid"
        )

    print("Runtime context dataset audit: PASS")


if __name__ == "__main__":
    main()
