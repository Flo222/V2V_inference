# -*- coding: utf-8 -*-
"""
Markov channel wrapper for RoCooper communication impairment.

It wraps the existing RoCooperComm and switches among good / medium / bad
communication states according to a Markov transition matrix.
"""

import copy
import random
from collections import Counter

import torch
import torch.nn as nn

from opencood.models.baselines.rocooper.components.rocooper_comm import RoCooperComm


class RoCooperMarkovComm(nn.Module):
    def __init__(self, comm_cfg, *args, **kwargs):
        super().__init__()

        self.base_cfg = copy.deepcopy(comm_cfg)
        self.markov_cfg = self.base_cfg.get("markov_channel", {}) or {}

        self.enabled = bool(self.markov_cfg.get("enabled", True))
        self.states = list(self.markov_cfg.get("states", ["good", "medium", "bad"]))
        self.initial_state = self.markov_cfg.get("initial_state", self.states[0])
        self.current_state = self.initial_state

        self.seed = int(self.markov_cfg.get("seed", 2026))
        self.rng = random.Random(self.seed)

        self.verbose = bool(self.markov_cfg.get("verbose", False))
        self.verbose_every = int(self.markov_cfg.get("verbose_every", 500))
        self.step = 0
        self.counter = Counter()

        transition_matrix = self.markov_cfg.get("transition_matrix", None)
        if transition_matrix is None:
            transition_matrix = {
                "good": [0.92, 0.07, 0.01],
                "medium": [0.10, 0.80, 0.10],
                "bad": [0.02, 0.18, 0.80],
            }

        self.transition_matrix = transition_matrix
        self.state_modules = nn.ModuleDict()

        state_params = self.markov_cfg.get("state_params", {}) or {}
        for state in self.states:
            cfg = self._make_state_cfg(self.base_cfg, state_params.get(state, {}), state)
            self.state_modules[state] = RoCooperComm(cfg, *args, **kwargs)

    def _make_state_cfg(self, base_cfg, params, state_name):
        cfg = copy.deepcopy(base_cfg)

        # Disable nested Markov inside each fixed-state comm.
        cfg["channel_state_mode"] = "fixed"
        cfg.setdefault("markov_channel", {})
        cfg["markov_channel"]["enabled"] = False

        cfg["enabled"] = bool(cfg.get("enabled", True))
        cfg["impair_ego"] = bool(cfg.get("impair_ego", False))
        cfg["train_with_impairment"] = True
        cfg["test_with_impairment"] = True

        # Markov state controls the impairment strength.
        net = cfg.setdefault("network_loss", {})
        net["enabled"] = True

        # Fading: default disabled, because the user asked for Markov impairment,
        # not original fixed RoCooper paper impairment.
        fading = cfg.setdefault("channel_fading", {})
        fading["enabled"] = bool(params.get("fading_enabled", False))
        fading["snr_db"] = float(params.get("snr_db", fading.get("snr_db", 15)))
        fading["model"] = fading.get("model", "rician")
        fading["sigma_h"] = float(params.get("sigma_h", fading.get("sigma_h", 0.1)))
        fading["sigma_w"] = float(params.get("sigma_w", fading.get("sigma_w", 0.01)))
        fading["distance_aware"] = bool(params.get("distance_aware", fading.get("distance_aware", True)))
        fading["path_loss_exp"] = float(params.get("path_loss_exp", fading.get("path_loss_exp", 2)))

        bw = net.setdefault("bandwidth_limit", {})
        bw["enabled"] = True

        # RoCooper native compression first, then strict bandwidth budget check.
        bw["mode"] = params.get(
            "bandwidth_mode",
            bw.get("mode", "rocooper_native_then_budget")
        )
        bw["bandwidth_mbps"] = float(
            params.get("bandwidth_mbps", bw.get("bandwidth_mbps", 27.0))
        )
        bw["frame_interval_ms"] = float(
            params.get("frame_interval_ms", bw.get("frame_interval_ms", 100.0))
        )

        bw["native_compression_policy"] = params.get(
            "native_compression_policy",
            bw.get("native_compression_policy", "auto_to_budget")
        )
        bw["max_native_compression_ratio"] = float(
            params.get(
                "max_native_compression_ratio",
                bw.get("max_native_compression_ratio", 32.0)
            )
        )
        bw["min_native_compression_ratio"] = float(
            params.get(
                "min_native_compression_ratio",
                bw.get("min_native_compression_ratio", 1.0)
            )
        )

        # Keep compression_ratio for fixed policy / compatibility.
        bw["compression_ratio"] = float(
            params.get("compression_ratio", bw.get("compression_ratio", 1.0))
        )

        # For plain RoCooper baseline, use drop.
        # Later GRACE/UCB version can set this to mark_for_ucb.
        bw["fallback_on_exceed"] = params.get(
            "fallback_on_exceed",
            bw.get("fallback_on_exceed", "drop")
        )

        # Kept for backward compatibility with original probabilistic mode.
        bw["mean"] = float(
            params.get("bandwidth_mean", params.get("bandwidth_limit_mean", bw.get("mean", 1.0)))
        )
        bw["std"] = float(
            params.get("bandwidth_std", params.get("bandwidth_limit_std", bw.get("std", 0.0)))
        )

        delay = net.setdefault("delay", {})
        delay["enabled"] = True
        delay["mean_ms"] = float(params.get("delay_mean_ms", delay.get("mean_ms", 60)))
        delay["std_ms"] = float(params.get("delay_std_ms", delay.get("std_ms", 10)))
        delay["frame_interval"] = int(params.get("frame_interval", delay.get("frame_interval", 100)))
        delay["max_delay_frames"] = int(params.get("max_delay_frames", delay.get("max_delay_frames", 1)))

        fd = net.setdefault("frame_drop", {})
        fd["enabled"] = True
        fd["drop_whole_cav_feature"] = bool(params.get("drop_whole_cav_feature", fd.get("drop_whole_cav_feature", True)))
        fd["mean"] = float(params.get("frame_drop_mean", fd.get("mean", 0.05)))
        fd["std"] = float(params.get("frame_drop_std", fd.get("std", 0.02)))

        pl = net.setdefault("packet_loss", {})
        pl["enabled"] = True
        pl["granularity"] = pl.get("granularity", "block")
        pl["block_size"] = int(pl.get("block_size", 4))
        pl["mean"] = float(params.get("packet_loss_mean", pl.get("mean", 0.15)))
        pl["std"] = float(params.get("packet_loss_std", pl.get("std", 0.05)))
        pl["zero_fraction"] = float(params.get("zero_fraction", pl.get("zero_fraction", 1.0)))

        cfg["_markov_fixed_state"] = state_name
        return cfg

    def _next_state(self):
        if not self.enabled:
            return self.current_state

        row = self.transition_matrix.get(self.current_state, None)
        if row is None:
            self.current_state = self.initial_state
            row = self.transition_matrix[self.current_state]

        r = self.rng.random()
        acc = 0.0
        for state, prob in zip(self.states, row):
            acc += float(prob)
            if r <= acc:
                self.current_state = state
                return state

        self.current_state = self.states[-1]
        return self.current_state

    def forward(self, *args, **kwargs):
        state = self._next_state()
        self.step += 1
        self.counter[state] += 1

        if self.verbose and self.step % self.verbose_every == 0:
            total = sum(self.counter.values())
            dist = {k: round(v / max(total, 1), 4) for k, v in self.counter.items()}
            print("[RoCooperMarkovComm] step={} state={} dist={}".format(
                self.step, state, dist
            ), flush=True)

        return self.state_modules[state](*args, **kwargs)
