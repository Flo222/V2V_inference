#!/usr/bin/env python
"""Add receiver-observable transport features to the ARCE CF collector."""

from __future__ import print_function

import argparse
from pathlib import Path


CONSTANT_AND_HELPER = r'''

RECEIVER_TRANSPORT_FEATURES = (
    "rx_q_recv_unit",
    "rx_q_cache_unit",
    "rx_q_eff_unit",
    "rx_q_recv_packet",
    "rx_q_cache_packet",
    "rx_q_eff_packet",
    "rx_num_source_packets",
    "rx_num_transmitted_source_packets",
    "rx_num_source_dropped_by_budget",
    "rx_num_received_packets",
    "rx_num_direct_received_source_packets",
    "rx_num_fec_recovered_source_packets",
    "rx_num_missing_source_packets",
    "rx_num_temporal_filled_packets",
    "rx_num_zero_filled_packets",
    "rx_num_total_units",
    "rx_num_current_recovered_units",
    "rx_num_temporal_filled_units",
    "rx_num_effective_recovered_units",
    "rx_cache_hit",
    "rx_tx_source_ratio",
    "rx_direct_receive_ratio",
    "rx_effective_unit_ratio",
)


def _finite_number(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _first_number(candidates, default=0.0):
    for mapping, key in candidates:
        if isinstance(mapping, dict) and mapping.get(key) is not None:
            return _finite_number(mapping.get(key), default)
    return float(default)


def _receiver_transport_features(record):
    """Return post-action receiver measurements; never use as bandit context."""
    record = record if isinstance(record, dict) else {}
    quality = record.get("quality")
    quality = quality if isinstance(quality, dict) else {}
    packet = record.get("packet")
    packet = packet if isinstance(packet, dict) else {}
    partial = record.get("partial_reconstruction")
    partial = partial if isinstance(partial, dict) else {}
    temporal = partial.get("temporal_cache")
    temporal = temporal if isinstance(temporal, dict) else {}

    def pick(*candidates):
        return _first_number(candidates, 0.0)

    source_packets = pick(
        (packet, "num_source_packets"),
        (quality, "num_source_packets"),
    )
    transmitted_source = pick(
        (packet, "num_transmitted_source_packets"),
    )
    direct_received = pick(
        (packet, "num_direct_received_source_packets"),
        (partial, "num_direct_received_packets"),
    )
    total_units = pick(
        (temporal, "num_total_units"),
    )
    effective_units = pick(
        (temporal, "num_effective_recovered_units"),
        (partial, "num_effective_recovered_units"),
    )

    result = {
        "rx_q_recv_unit": pick(
            (quality, "q_recv_unit"),
            (temporal, "q_recv_unit"),
        ),
        "rx_q_cache_unit": pick(
            (quality, "q_cache_unit"),
            (temporal, "q_cache_unit"),
        ),
        "rx_q_eff_unit": pick(
            (quality, "q_eff_unit"),
            (temporal, "q_eff_unit"),
        ),
        "rx_q_recv_packet": pick(
            (quality, "q_recv_packet"),
            (temporal, "q_recv_packet"),
        ),
        "rx_q_cache_packet": pick(
            (quality, "q_cache_packet"),
            (temporal, "q_cache_packet"),
        ),
        "rx_q_eff_packet": pick(
            (quality, "q_eff_packet"),
            (temporal, "q_eff_packet"),
        ),
        "rx_num_source_packets": source_packets,
        "rx_num_transmitted_source_packets": transmitted_source,
        "rx_num_source_dropped_by_budget": pick(
            (packet, "num_source_dropped_by_budget"),
        ),
        "rx_num_received_packets": pick(
            (packet, "num_received_packets"),
        ),
        "rx_num_direct_received_source_packets": direct_received,
        "rx_num_fec_recovered_source_packets": pick(
            (packet, "num_fec_recovered_source_packets"),
            (partial, "num_fec_recovered_packets"),
        ),
        "rx_num_missing_source_packets": pick(
            (packet, "num_missing_source_packets"),
            (quality, "num_still_missing"),
            (partial, "num_still_missing"),
        ),
        "rx_num_temporal_filled_packets": pick(
            (quality, "num_temporal_filled_packets"),
            (temporal, "num_temporal_filled_packets"),
            (partial, "num_temporal_filled_packets"),
        ),
        "rx_num_zero_filled_packets": pick(
            (partial, "num_zero_filled_packets"),
        ),
        "rx_num_total_units": total_units,
        "rx_num_current_recovered_units": pick(
            (temporal, "num_current_recovered_units"),
            (partial, "num_current_recovered_units"),
        ),
        "rx_num_temporal_filled_units": pick(
            (temporal, "num_temporal_filled_units"),
            (partial, "num_temporal_filled_units"),
        ),
        "rx_num_effective_recovered_units": effective_units,
        "rx_cache_hit": 1.0 if bool(temporal.get("cache_hit", False)) else 0.0,
        "rx_tx_source_ratio": (
            transmitted_source / source_packets if source_packets > 0.0 else 0.0
        ),
        "rx_direct_receive_ratio": (
            direct_received / source_packets if source_packets > 0.0 else 0.0
        ),
        "rx_effective_unit_ratio": (
            effective_units / total_units if total_units > 0.0 else 0.0
        ),
    }
    return {name: _finite_number(result.get(name), 0.0)
            for name in RECEIVER_TRANSPORT_FEATURES}
'''


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            "{}: expected exactly one anchor, found {}".format(label, count)
        )
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default=".",
        help="OPV2V repository root",
    )
    args = parser.parse_args()

    path = (
        Path(args.repo)
        / "opencood"
        / "tools"
        / "audit_arce_counterfactual.py"
    )
    text = path.read_text(encoding="utf-8")

    if "RECEIVER_TRANSPORT_FEATURES = (" in text:
        print("already patched:", path)
        return

    helper_anchor = "\n\ndef _core_model(model):\n"
    text = replace_once(
        text,
        helper_anchor,
        CONSTANT_AND_HELPER + helper_anchor,
        "receiver helper insertion",
    )

    feature_anchor = (
        "    feature_cols = head_feature_cols + decoded_feature_cols\n"
    )
    feature_replacement = (
        "    feature_cols = (\n"
        "        head_feature_cols\n"
        "        + decoded_feature_cols\n"
        "        + list(RECEIVER_TRANSPORT_FEATURES)\n"
        "    )\n"
    )
    text = replace_once(
        text,
        feature_anchor,
        feature_replacement,
        "CSV feature columns",
    )

    row_anchor = (
        "                        row.update(proxy_features)\n"
        "                        row.update(\n"
        "                            decoded_box_features(pred_boxes, pred_scores)\n"
        "                        )\n"
    )
    row_replacement = (
        "                        row.update(proxy_features)\n"
        "                        row.update(\n"
        "                            _receiver_transport_features(raw_target_record)\n"
        "                        )\n"
        "                        row.update(\n"
        "                            decoded_box_features(pred_boxes, pred_scores)\n"
        "                        )\n"
    )
    text = replace_once(
        text,
        row_anchor,
        row_replacement,
        "counterfactual row receiver features",
    )

    path.write_text(text, encoding="utf-8")
    print("patched:", path)


if __name__ == "__main__":
    main()
