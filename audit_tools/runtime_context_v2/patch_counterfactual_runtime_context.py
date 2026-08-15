#!/usr/bin/env python3
"""Patch the ARCE counterfactual collector to retain exact runtime context."""

from __future__ import print_function

import argparse
from pathlib import Path


CONTEXT_HELPERS = r'''

RUNTIME_CONTEXT_NAMES = (
    "B_norm",
    "p_loss",
    "d_norm",
    "ego_confidence",
    "cache_quality",
    "complementarity",
    "cav_confidence",
)

RUNTIME_CONTEXT_CSV_FIELDS = (
    "decision_context_available",
    "decision_context_source",
    "decision_context_B_norm",
    "decision_context_p_loss",
    "decision_context_d_norm",
    "decision_context_ego_confidence",
    "decision_context_cache_quality",
    "decision_context_complementarity",
    "decision_context_cav_confidence",
    "update_context_available",
    "update_context_B_norm",
    "update_context_p_loss",
    "update_context_d_norm",
    "update_context_ego_confidence",
    "update_context_cache_quality",
    "update_context_complementarity",
    "update_context_cav_confidence",
    "decision_update_context_max_abs_diff",
)


def _context_from_vector(vector, source):
    if not isinstance(vector, (list, tuple)) or len(vector) != 7:
        return {
            "available": False,
            "source": str(source),
            "vector": None,
        }
    values = [_finite_number(value, float("nan")) for value in vector]
    if not all(math.isfinite(value) for value in values):
        return {
            "available": False,
            "source": str(source),
            "vector": None,
        }
    return {
        "available": True,
        "source": str(source),
        "vector": list(values),
        **dict(zip(RUNTIME_CONTEXT_NAMES, values)),
    }


def _decision_context_from_record(record):
    """Extract the exact context attached to a scored send proposal."""
    record = record if isinstance(record, dict) else {}
    dc2mab = record.get("dc2mab")
    dc2mab = dc2mab if isinstance(dc2mab, dict) else {}
    proposal = dc2mab.get("proposal")
    proposal = proposal if isinstance(proposal, dict) else {}
    context = proposal.get("context")
    context = context if isinstance(context, dict) else {}

    vector = context.get("vector")
    if not isinstance(vector, (list, tuple)):
        if all(context.get(name) is not None for name in RUNTIME_CONTEXT_NAMES):
            vector = [context[name] for name in RUNTIME_CONTEXT_NAMES]

    return _context_from_vector(vector, "dc2mab.proposal.context")


def _update_context_from_record(record, decision_context):
    """Extract the context used for the reward update when it is recorded."""
    record = record if isinstance(record, dict) else {}
    vector = record.get("context_vector")
    update = _context_from_vector(vector, "record.context_vector")
    if update.get("available"):
        return update
    if isinstance(decision_context, dict) and decision_context.get("available"):
        copied = copy.deepcopy(decision_context)
        copied["source"] = "dc2mab.proposal.context"
        return copied
    return update


def _flatten_runtime_contexts(row):
    decision = row.get("decision_context")
    decision = decision if isinstance(decision, dict) else {}
    update = row.get("reward_update_context")
    update = update if isinstance(update, dict) else {}

    row["decision_context_available"] = bool(decision.get("available", False))
    row["decision_context_source"] = str(decision.get("source", "missing"))
    row["update_context_available"] = bool(update.get("available", False))

    for name in RUNTIME_CONTEXT_NAMES:
        row["decision_context_" + name] = decision.get(name)
        row["update_context_" + name] = update.get(name)

    if decision.get("available") and update.get("available"):
        row["decision_update_context_max_abs_diff"] = max(
            abs(float(a) - float(b))
            for a, b in zip(decision["vector"], update["vector"])
        )
    else:
        row["decision_update_context_max_abs_diff"] = None


def _attach_group_decision_context(action_rows):
    """Attach one action-independent decision context to all seven trials."""
    candidates = [
        row.get("decision_context")
        for row in action_rows
        if isinstance(row, dict)
        and not bool(row.get("no_send", False))
        and isinstance(row.get("decision_context"), dict)
        and bool(row["decision_context"].get("available", False))
    ]
    if not candidates:
        raise RuntimeError(
            "No scored send proposal context was recorded for this seven-action group."
        )

    canonical = candidates[0]
    for other in candidates[1:]:
        max_diff = max(
            abs(float(a) - float(b))
            for a, b in zip(canonical["vector"], other["vector"])
        )
        if max_diff > 1e-9:
            raise RuntimeError(
                "Counterfactual send trials do not share one decision context: "
                "max_abs_diff={}".format(max_diff)
            )

    for row in action_rows:
        if not isinstance(row, dict) or row.get("error") is not None:
            continue
        row["decision_context"] = copy.deepcopy(canonical)
        _flatten_runtime_contexts(row)
'''


REPLACEMENTS = (
    (
        "\n\ndef _finite_number(value, default=0.0):\n",
        CONTEXT_HELPERS + "\n\ndef _finite_number(value, default=0.0):\n",
        "runtime context helpers",
    ),
    (
        '''        "proxy_source",\n    ] + feature_cols\n''',
        '''        "proxy_source",\n    ] + list(RUNTIME_CONTEXT_CSV_FIELDS) + feature_cols\n''',
        "runtime context CSV fields",
    ),
    (
        '''                "proxy_source": action.get("proxy_source"),\n            }\n            for name in feature_cols:\n''',
        '''                "proxy_source": action.get("proxy_source"),\n            }\n            for name in RUNTIME_CONTEXT_CSV_FIELDS:\n                row[name] = action.get(name)\n            for name in feature_cols:\n''',
        "runtime context CSV values",
    ),
    (
        '''                        feature_delta = _feature_delta(output)\n                        row = {\n''',
        '''                        feature_delta = _feature_delta(output)\n                        decision_context = _decision_context_from_record(\n                            raw_target_record\n                        )\n                        row = {\n''',
        "decision context extraction",
    ),
    (
        '''                            "policy_update_applied": update.get("policy_update_applied"),\n                        }\n''',
        '''                            "policy_update_applied": update.get("policy_update_applied"),\n                            "decision_context": decision_context,\n                            "reward_update_context": _update_context_from_record(\n                                raw_target_record,\n                                decision_context,\n                            ),\n                        }\n''',
        "decision and update context records",
    ),
    (
        '''                    action_rows.append(row)\n\n                no_send_id = next(\n''',
        '''                    action_rows.append(row)\n\n                _attach_group_decision_context(action_rows)\n\n                no_send_id = next(\n''',
        "group decision context attachment",
    ),
)


def transformed(text):
    if "RUNTIME_CONTEXT_NAMES = (" in text:
        raise RuntimeError("Runtime context collector patch is already installed")
    if "RECEIVER_TRANSPORT_FEATURES = (" not in text:
        raise RuntimeError(
            "Receiver-feature collector patch is required before this patch"
        )

    result = text
    for old, new, label in REPLACEMENTS:
        count = result.count(old)
        if count != 1:
            raise RuntimeError(
                "Expected one {} anchor, found {}".format(label, count)
            )
        result = result.replace(old, new, 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="opencood/tools/audit_arce_counterfactual.py",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.check == args.apply:
        raise SystemExit("Specify exactly one of --check or --apply")

    path = Path(args.target)
    text = path.read_text(encoding="utf-8")
    result = transformed(text)
    if args.check:
        print("Runtime context collector patch preflight: PASS")
        return
    path.write_text(result, encoding="utf-8")
    print("patched:", path)


if __name__ == "__main__":
    main()
