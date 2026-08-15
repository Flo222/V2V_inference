#!/usr/bin/env python3
"""Full-information audit of a shared action-conditioned LinUCB replay.

This imports the validated exact-context dataset/reward reader. Unselected
counterfactual rewards are used only for regret measurement, never updates.
"""

from __future__ import print_function

import argparse
import json
import math
from collections import namedtuple
from pathlib import Path

import numpy as np

import audit_dlinucb_exact_context_replay as replay


Score = namedtuple("Score", "mean bonus ucb")


def csv_values(text, cast):
    values = [cast(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("parameter list cannot be empty")
    return values


def action_descriptor(action_id):
    action_id = str(action_id).lower()
    if "_fp16_" in action_id:
        bits = 16.0
    elif "_int8_" in action_id:
        bits = 8.0
    elif "_int4_" in action_id:
        bits = 4.0
    else:
        raise ValueError("unsupported send action: {}".format(action_id))

    cache_enabled = 1.0 if "_cache1_" in action_id else 0.0
    precision = bits / 16.0
    capacity = (16.0 / bits) / 4.0
    return np.asarray(
        [precision, capacity, cache_enabled], dtype=np.float64
    )


class SharedActionLinUCB(object):
    """One shared linear model over state, action and state-action terms."""

    def __init__(
        self,
        action_ids,
        context_dim,
        lambda_reg,
        discount,
        beta,
        feedback_weight_mode="none",
    ):
        self.action_ids = tuple(str(x) for x in action_ids)
        self.context_dim = int(context_dim)
        self.action_dim = 3
        self.feature_dim = (
            1
            + self.context_dim
            + self.action_dim
            + self.context_dim * self.action_dim
        )
        self.lambda_reg = float(lambda_reg)
        self.discount = float(discount)
        self.beta = float(beta)
        self.feedback_weight_mode = str(feedback_weight_mode)
        if self.feedback_weight_mode != "none":
            raise ValueError(
                "shared audit intentionally supports feedback_weight_mode=none"
            )
        if not (0.0 < self.discount <= 1.0):
            raise ValueError("discount must be in (0, 1]")
        if self.lambda_reg <= 0.0:
            raise ValueError("lambda_reg must be positive")

        self._prior = self.lambda_reg * np.eye(
            self.feature_dim, dtype=np.float64
        )
        self.A = self._prior.copy()
        self.b = np.zeros(self.feature_dim, dtype=np.float64)
        self._inverse = None

    def _feature(self, action_id, context):
        state = np.asarray(context, dtype=np.float64).reshape(-1)
        if state.size != self.context_dim:
            raise ValueError(
                "context dim mismatch: {} != {}".format(
                    state.size, self.context_dim
                )
            )
        action = action_descriptor(action_id)
        interaction = np.outer(state, action).reshape(-1)
        return np.concatenate(
            (
                np.ones(1, dtype=np.float64),
                state,
                action,
                interaction,
            )
        )

    def _inverse_matrix(self):
        if self._inverse is None:
            self._inverse = np.linalg.inv(self.A)
        return self._inverse

    def score(self, action_id, context):
        feature = self._feature(action_id, context)
        inverse = self._inverse_matrix()
        theta = inverse.dot(self.b)
        mean = float(feature.dot(theta))
        variance = max(float(feature.dot(inverse).dot(feature)), 0.0)
        bonus = float(self.beta * math.sqrt(variance))
        return Score(mean=mean, bonus=bonus, ucb=mean + bonus)

    def update(
        self,
        action_id,
        context,
        reward,
        channel_profile=None,
    ):
        del channel_profile
        feature = self._feature(action_id, context)
        gamma = self.discount
        # Preserve the regularization prior while discounting observations.
        self.A = (
            gamma * self.A
            + (1.0 - gamma) * self._prior
            + np.outer(feature, feature)
        )
        self.b = gamma * self.b + float(reward) * feature
        self._inverse = None
        return 1.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--sender_index", default="1")
    parser.add_argument("--reference_bytes", type=float, default=337500.0)
    parser.add_argument("--lambda_abs", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=3.0)
    parser.add_argument("--lambda_cost", type=float, default=0.3)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--tie_tolerance", type=float, default=0.01)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--betas", default="0,0.05,0.1,0.2,0.5,1")
    parser.add_argument("--discounts", default="1,0.999,0.995,0.99")
    return parser.parse_args()


def compact(result, beta, discount, feedback_source):
    return {
        "beta": float(beta),
        "discount": float(discount),
        "feedback_source": str(feedback_source),
        "feedback_weight_mode": "none",
        "late_true_regret": result[
            "last_quarter_mean_true_regret"
        ]["mean"],
        "late_feedback_regret": result[
            "last_quarter_mean_feedback_regret"
        ]["mean"],
        "late_true_top_set_hit_rate": result[
            "last_quarter_true_top_set_hit_rate"
        ]["mean"],
        "late_true_regret_vs_random": result[
            "late_true_regret_vs_random"
        ],
        "late_true_regret_vs_best_fixed": result[
            "late_true_regret_vs_best_fixed"
        ],
    }


def main():
    args = parse_args()
    args.beta = 1.0
    args.markov_discount = 0.995
    args.production_feedback_weight = "none"

    groups, all_actions, send_actions, input_audit = replay.load_groups(
        args.csv, args
    )
    groups = [
        group
        for group in groups
        if group["sender_index"] == str(args.sender_index)
    ]
    if not groups:
        raise ValueError(
            "no groups for sender_index={}".format(args.sender_index)
        )

    seeds = csv_values(args.seeds, int)
    betas = csv_values(args.betas, float)
    discounts = csv_values(args.discounts, float)
    baselines = replay.random_and_fixed_baselines(groups, send_actions)
    rows = []

    for feedback_source in ("true", "proxy"):
        for beta in betas:
            args.beta = float(beta)
            for discount in discounts:
                runs = [
                    replay.replay_once(
                        SharedActionLinUCB,
                        groups,
                        send_actions,
                        feedback_source,
                        float(discount),
                        "none",
                        seed,
                        args,
                    )
                    for seed in seeds
                ]
                result = replay.aggregate_replays(
                    "shared_action_interaction", runs, baselines
                )
                row = compact(result, beta, discount, feedback_source)
                rows.append(row)
                print(
                    "feedback={} beta={:.3f} gamma={:.4f} "
                    "late_true_regret={:.6f} vs_random={:.4f} "
                    "vs_fixed={:.4f}".format(
                        feedback_source,
                        beta,
                        discount,
                        row["late_true_regret"],
                        row["late_true_regret_vs_random"],
                        row["late_true_regret_vs_best_fixed"],
                    ),
                    flush=True,
                )

    key = lambda row: (
        row["late_true_regret"],
        -row["late_true_top_set_hit_rate"],
    )
    true_rows = [x for x in rows if x["feedback_source"] == "true"]
    proxy_rows = [x for x in rows if x["feedback_source"] == "proxy"]
    report = {
        "input_audit": input_audit,
        "scope": {
            "groups": len(groups),
            "sender_index": str(args.sender_index),
            "counterfactual_actions": all_actions,
            "learned_send_actions": send_actions,
            "state_context_fields": list(replay.CONTEXT_FIELDS),
            "action_descriptor_fields": [
                "quant_precision",
                "quant_capacity",
                "cache_enabled",
            ],
            "feature_definition": "[1, x, z, vec(x outer z)]",
            "feature_dim": 1 + 7 + 3 + 7 * 3,
            "bandit_feedback": (
                "Only the selected action reward updates the shared model."
            ),
        },
        "baselines": baselines,
        "grid": {
            "seeds": seeds,
            "betas": betas,
            "discounts": discounts,
        },
        "best_true_feedback": min(true_rows, key=key),
        "best_proxy_feedback": min(proxy_rows, key=key),
        "true_feedback_ranking": sorted(true_rows, key=key),
        "proxy_feedback_ranking": sorted(proxy_rows, key=key),
        "rows": rows,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nBest true:", json.dumps(report["best_true_feedback"]))
    print("Best proxy:", json.dumps(report["best_proxy_feedback"]))
    print("Saved:", out)


if __name__ == "__main__":
    main()
