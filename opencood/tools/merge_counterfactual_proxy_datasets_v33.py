#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from opencood.methods.arce.policies.decoded_box_proxy_features import (
    PAIRED_DECODED_MATCH_FEATURES,
    RICH_DECODED_BOX_FEATURES,
    delta_decoded_feature_names,
    no_send_decoded_feature_names,
)


KEY_COLUMNS = (
    "sequence_id",
    "frame_idx",
    "sender_index",
    "action_id",
)
V33_COLUMNS = (
    RICH_DECODED_BOX_FEATURES
    + no_send_decoded_feature_names()
    + delta_decoded_feature_names()
    + PAIRED_DECODED_MATCH_FEATURES
)


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _key(row: Dict[str, str]) -> Tuple[int, int, int, str]:
    return (
        int(row["sequence_id"]),
        int(row["frame_idx"]),
        int(row["sender_index"]),
        str(row["action_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_meta", required=True)
    args = parser.parse_args()

    fieldnames = None
    merged: Dict[Tuple[int, int, int, str], Dict[str, str]] = {}
    source_counts = {}
    duplicate_count = 0

    for raw_path in args.inputs:
        path = Path(raw_path)
        current_fields, rows = _read_csv(path)
        missing_keys = sorted(set(KEY_COLUMNS).difference(current_fields))
        if missing_keys:
            raise RuntimeError(
                "{} missing key columns {}".format(path, missing_keys)
            )
        missing_v33 = sorted(set(V33_COLUMNS).difference(current_fields))
        if missing_v33:
            raise RuntimeError(
                "{} is not a v3.3 rich decoded-box dataset; missing {}."
                .format(path, missing_v33)
            )
        if fieldnames is None:
            fieldnames = current_fields
        elif current_fields != fieldnames:
            raise RuntimeError(
                "CSV schema mismatch for {}.".format(path)
            )
        source_counts[str(path)] = len(rows)
        for row in rows:
            key = _key(row)
            previous = merged.get(key)
            if previous is None:
                merged[key] = row
            elif previous == row:
                duplicate_count += 1
            else:
                raise RuntimeError(
                    "Conflicting duplicate counterfactual row: {}".format(key)
                )

    if not fieldnames or not merged:
        raise RuntimeError("No counterfactual rows were loaded.")

    sorted_rows = [merged[key] for key in sorted(merged)]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

    frame_sender_keys = {
        (
            int(row["sequence_id"]),
            int(row["frame_idx"]),
            int(row["sender_index"]),
        )
        for row in sorted_rows
    }
    meta = {
        "feature_definition": (
            "canonical_psm_rm_head_plus_rich_paired_decoded_box_v33"
        ),
        "decoded_box_feature_schema": "v3.3_rich_paired_aabb_iou",
        "sources": source_counts,
        "rows": len(sorted_rows),
        "identical_duplicates_skipped": duplicate_count,
        "sequences": sorted(
            {int(row["sequence_id"]) for row in sorted_rows}
        ),
        "sender_counter": dict(
            Counter(str(row["sender_index"]) for row in sorted_rows)
        ),
        "action_counter": dict(
            Counter(str(row["action_id"]) for row in sorted_rows)
        ),
        "frame_sender_groups": len(frame_sender_keys),
    }
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("saved:", out_csv)
    print("saved:", out_meta)


if __name__ == "__main__":
    main()
