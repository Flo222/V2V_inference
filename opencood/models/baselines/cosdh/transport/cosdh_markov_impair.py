import math
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.communication.channel.channel_manager import ChannelManager


class CosDHMarkovImpair(nn.Module):
    """
    CoSDH-Markov impairment baseline.

    This module applies the same type of Markov channel limitation used by
    Where2comm-Markov:
      1) Markov state transition;
      2) state-dependent delay slots;
      3) state-dependent bandwidth limit;
      4) state-dependent Bernoulli packet/patch loss.

    It works on BEV feature patches before CoSDH fusion.
    Ego CAV is not impaired by default.

    Input:
        x: [sum_cav, C, H, W]
        record_len: [B]
        score_map: optional confidence map, e.g. psm_single.
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}

        self.enabled = bool(cfg.get("enabled", False))
        self.impair_ego = bool(cfg.get("impair_ego", False))

        # Feature / communication settings.
        self.patch_size = int(cfg.get("patch_size", 8))
        self.fps = float(cfg.get("fps", 10.0))

        # bytes_per_value:
        # fp32 -> 4, fp16 -> 2, int8 -> 1.
        # Set this to the same value as Where2comm-Markov.
        self.bytes_per_value = float(cfg.get("bytes_per_value", 2.0))

        self.delay_fallback = str(cfg.get("delay_fallback", "zero"))
        self.verbose = bool(cfg.get("verbose", False))

        self.states = list(cfg.get("states", ["good", "medium", "bad"]))

        # IMPORTANT:
        # Please set these values equal to your limited Where2comm+Markov config.
        default_state_cfg = {
            "good": {
                "packet_loss_rate": 0.05,
                "bandwidth_mbps": 10.0,
                "delay_slots": 0,
            },
            "medium": {
                "packet_loss_rate": 0.15,
                "bandwidth_mbps": 5.0,
                "delay_slots": 1,
            },
            "bad": {
                "packet_loss_rate": 0.30,
                "bandwidth_mbps": 1.0,
                "delay_slots": 2,
            },
        }

        user_state_cfg = cfg.get("state_cfg", {})
        self.state_cfg = {}
        for state in self.states:
            merged = dict(default_state_cfg.get(state, {}))
            merged.update(user_state_cfg.get(state, {}))
            self.state_cfg[state] = merged

        default_transition = [
            [0.92, 0.07, 0.01],
            [0.15, 0.75, 0.10],
            [0.08, 0.22, 0.70],
        ]
        self.transition = torch.tensor(
            cfg.get("transition", default_transition),
            dtype=torch.float32
        )

        self.init_state = str(cfg.get("init_state", "good"))
        if self.init_state not in self.states:
            self.init_state = self.states[0]

        self._delay_cache = defaultdict(lambda: deque(maxlen=128))
        self._frame_index = -1
        self.channel_manager = ChannelManager({
            "seed": int(cfg.get("seed", 0)),
            "channel": {
                "mode": "markov",
                "initial_state": self.init_state,
                "transition_matrix": cfg.get("transition", default_transition),
                "profiles": {
                    state: {
                        "bandwidth_mbps": float(profile["bandwidth_mbps"]),
                        "packet_loss_rate": float(profile["packet_loss_rate"]),
                        "delay_slots": int(profile.get("delay_slots", 0)),
                    }
                    for state, profile in self.state_cfg.items()
                },
                "loss_model": "bernoulli",
                "bernoulli_loss_rates": {
                    state: float(profile["packet_loss_rate"])
                    for state, profile in self.state_cfg.items()
                },
            },
        })

    def _get_delayed_feature(self, link_key, current_feat, delay_slots):
        """
        Store current feature, then return delayed feature.

        current_feat: [1, C, H, W]
        """
        cache = self._delay_cache[link_key]
        cache.append(current_feat.detach().clone())

        if delay_slots <= 0:
            return current_feat

        target_idx = len(cache) - 1 - int(delay_slots)

        if target_idx >= 0:
            return cache[target_idx].to(current_feat.device)

        if self.delay_fallback == "current":
            return current_feat
        return torch.zeros_like(current_feat)

    def _score_to_patch_score(self, score_map, target_hw, patch_grid):
        """
        Convert psm/confidence map to patch-level scores.

        score_map may be:
            [N, A, H, W] or [N, 1, H, W] or [N, H, W]
        """
        if score_map is None:
            return None

        if score_map.dim() == 3:
            score = score_map.unsqueeze(1)
        elif score_map.dim() == 4:
            score = score_map
        else:
            return None

        # Convert multi-anchor score to single confidence map.
        if score.shape[1] > 1:
            score = torch.sigmoid(score).amax(dim=1, keepdim=True)
        else:
            score = torch.sigmoid(score)

        score = F.interpolate(score, size=target_hw, mode="bilinear", align_corners=False)
        patch_score = F.adaptive_avg_pool2d(score, patch_grid)  # [1, 1, ph, pw]
        return patch_score.flatten()

    def _feature_energy_patch_score(self, feat, patch_grid):
        """
        Fallback patch ranking when confidence map is unavailable.
        """
        energy = feat.abs().mean(dim=1, keepdim=True)  # [1,1,H,W]
        patch_score = F.adaptive_avg_pool2d(energy, patch_grid)
        return patch_score.flatten()

    def _apply_bandwidth_and_loss(self, feat, score_map_one, budget_bytes, link_key, frame_id):
        """
        feat: [1, C, H, W]
        score_map_one: optional confidence map for this CAV.
        """
        _, C, H, W = feat.shape
        ps = max(1, self.patch_size)

        ph = math.ceil(H / ps)
        pw = math.ceil(W / ps)
        num_patches = ph * pw

        patch_bytes = C * ps * ps * self.bytes_per_value
        max_keep = int(budget_bytes // patch_bytes)
        max_keep = max(0, min(num_patches, max_keep))

        # Patch ranking: use CoSDH confidence if available, otherwise feature energy.
        patch_score = self._score_to_patch_score(
            score_map_one, target_hw=(H, W), patch_grid=(ph, pw)
        )

        if patch_score is None:
            patch_score = self._feature_energy_patch_score(feat, patch_grid=(ph, pw))

        keep_patch = torch.zeros(num_patches, device=feat.device, dtype=torch.float32)

        if max_keep > 0:
            _, topk_idx = torch.topk(patch_score, k=max_keep, largest=True)
            keep_patch[topk_idx] = 1.0

            receive = self.channel_manager.sample_receive_mask(
                num_packets=max_keep,
                link_id=link_key,
                frame_id=frame_id,
                device=feat.device,
                return_info=False,
            ).float()
            keep_patch[topk_idx] = keep_patch[topk_idx] * receive

        keep_grid = keep_patch.view(1, 1, ph, pw)
        keep_mask = F.interpolate(keep_grid, size=(H, W), mode="nearest")

        return feat * keep_mask, {
            "num_patches": int(num_patches),
            "max_keep": int(max_keep),
            "received": int(keep_patch.sum().item()),
        }

    def forward(self, x, record_len, score_map=None):
        if not self.enabled:
            return x

        if x is None or record_len is None:
            return x

        out = x.clone()

        if torch.is_tensor(record_len):
            record_len_list = record_len.detach().cpu().tolist()
        else:
            record_len_list = list(record_len)

        self._frame_index += 1
        start = 0

        for b, cav_num in enumerate(record_len_list):
            cav_num = int(cav_num)

            for local_idx in range(cav_num):
                global_idx = start + local_idx

                # Do not impair ego unless explicitly enabled.
                if local_idx == 0 and not self.impair_ego:
                    continue

                link_key = f"batch_{b}_cav_{local_idx}"

                budget = self.channel_manager.get_frame_budget(
                    frame_interval_ms=1000.0 / max(self.fps, 1e-6),
                    link_id=link_key,
                    frame_id=self._frame_index,
                )
                state = budget["channel_state"]
                cfg = budget["profile"]

                delay_slots = int(cfg.get("delay_slots", 0))
                bandwidth_mbps = float(cfg.get("bandwidth_mbps", 0.0))
                packet_loss_rate = float(cfg.get("packet_loss_rate", 0.0))

                current_feat = out[global_idx:global_idx + 1]

                delayed_feat = self._get_delayed_feature(
                    link_key, current_feat, delay_slots
                )

                score_one = None
                if score_map is not None and torch.is_tensor(score_map):
                    if score_map.shape[0] == x.shape[0]:
                        score_one = score_map[global_idx:global_idx + 1]

                impaired_feat, stat = self._apply_bandwidth_and_loss(
                    delayed_feat,
                    score_one,
                    budget_bytes=int(budget["budget_bytes"]),
                    link_key=link_key,
                    frame_id=self._frame_index,
                )

                out[global_idx:global_idx + 1] = impaired_feat

                if self.verbose:
                    print(
                        "[CoSDH-Markov] "
                        f"batch={b}, cav={local_idx}, state={state}, "
                        f"delay={delay_slots}, bw={bandwidth_mbps:.3f}Mbps, "
                        f"plr={packet_loss_rate:.3f}, "
                        f"patches={stat['received']}/{stat['num_patches']}, "
                        f"budget_keep={stat['max_keep']}"
                    )

            start += cav_num

        return out
