#!/usr/bin/env python3
"""Fix serialization of the PDFContext used by selected ARCE proposals."""

from __future__ import print_function

import argparse
from pathlib import Path


OLD_CONTEXT_BLOCK = '''    if ctx is not None:
        summary["context"] = {
            "B_norm": float(getattr(ctx, "B_norm", 0.0)),
            "p_loss": float(getattr(ctx, "p_loss", 0.0)),
            "d_norm": float(getattr(ctx, "d_norm", 0.0)),
            "ego_confidence": float(getattr(ctx, "ego_confidence", 0.0)),
            "cache_quality": float(getattr(ctx, "cache_quality", 0.0)),
            "complementarity": float(getattr(ctx, "complementarity", 0.0)),
            "cav_confidence": float(getattr(ctx, "cav_confidence", 0.0)),
        }
'''


NEW_CONTEXT_BLOCK = '''    if ctx is not None:
        raw_vector = getattr(ctx, "vector", None)
        if hasattr(raw_vector, "tolist"):
            raw_vector = raw_vector.tolist()
        if not isinstance(raw_vector, (list, tuple)):
            raise RuntimeError(
                "Selected proposal context does not expose a vector."
            )
        vector = [float(value) for value in raw_vector]
        if len(vector) not in (6, 7):
            raise RuntimeError(
                "Selected proposal context must be 6D or 7D, got {}D.".format(
                    len(vector)
                )
            )
        names = (
            "B_norm",
            "p_loss",
            "d_norm",
            "ego_confidence",
            "cache_quality",
            "complementarity",
            "cav_confidence",
        )
        context_summary = {
            "vector": list(vector),
            **dict(zip(names, vector)),
        }
        info = getattr(ctx, "info", None)
        if isinstance(info, dict):
            for key in ("bandwidth_mbps", "latency_ms"):
                if info.get(key) is not None:
                    context_summary[key] = float(info[key])
        summary["context"] = context_summary
'''


RECORD_ANCHOR = '''    record["pdf_action"] = pdf_action.as_dict()
'''


RECORD_REPLACEMENT = '''    proposal_context = proposal_summary.get("context", {})
    if isinstance(proposal_context, dict):
        proposal_context_vector = proposal_context.get("vector")
        if isinstance(proposal_context_vector, (list, tuple)):
            # This is also the context passed to the selected-action reward
            # update. Keep an explicit copy for decision/update audits.
            record["context_vector"] = [
                float(value) for value in proposal_context_vector
            ]
            record["context_vector_source"] = "selected_proposal"

    record["pdf_action"] = pdf_action.as_dict()
'''


def transformed(text):
    if 'context_vector_source"] = "selected_proposal"' in text:
        raise RuntimeError("Runtime context recording patch is already installed")
    if text.count(OLD_CONTEXT_BLOCK) != 1:
        raise RuntimeError(
            "Expected one legacy PDFContext serialization block, found {}".format(
                text.count(OLD_CONTEXT_BLOCK)
            )
        )
    if text.count(RECORD_ANCHOR) != 1:
        raise RuntimeError(
            "Expected one selected record context anchor, found {}".format(
                text.count(RECORD_ANCHOR)
            )
        )
    text = text.replace(OLD_CONTEXT_BLOCK, NEW_CONTEXT_BLOCK, 1)
    return text.replace(RECORD_ANCHOR, RECORD_REPLACEMENT, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default=(
            "opencood/methods/arce/policies/"
            "c2mab_execution_record_builder.py"
        ),
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
        print("Runtime context recording patch preflight: PASS")
        return
    path.write_text(result, encoding="utf-8")
    print("patched:", path)


if __name__ == "__main__":
    main()
