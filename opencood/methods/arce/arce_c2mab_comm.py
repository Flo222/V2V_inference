from __future__ import annotations

from opencood.methods.arce.policies.compact_sparse_cost_helper import (
    estimate_compact_sparse_tokens,
)

from opencood.methods.arce.policies.payload_transport import (
    apply_payload_native_transport_to_arce_cfg,
    compact_sparse_cfg_for_transport,
    is_payload_native_transport,
    normalize_transport_mode,
)

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch

from opencood.methods.arce.arce_fixed_comm import ARCEFixedComm
from opencood.methods.arce.policies.action_space import (
    PDFARCEAction,
    build_pdf_action_space,
    raw_feature_bytes_fp32,
    budget_bytes_from_bandwidth,
    NO_SEND_ACTION_ID,
)
from opencood.methods.arce.policies.context_builder import PDFContextBuilder
from opencood.methods.arce.policies.c2mab_policy_bank import (
    C2MABPolicyBank,
    C2MABPolicyConfig,
)
from opencood.methods.arce.policies.c2mab_executor_config import (
    build_c2mab_executor_cfg,
)
from opencood.methods.arce.policies.c2mab_runtime_summary import (
    summarize_dc2mab_runtime_records,
)
from opencood.methods.arce.policies.c2mab_no_send_executor import (
    execute_no_send_sender,
)
from opencood.methods.arce.policies.c2mab_proposal_builder import build_c2mab_proposals
from opencood.methods.arce.policies.c2mab_execution_record_builder import (
    build_budget_consistency,
    enrich_selected_execution_record,
    selected_physical_budget_plan,
    validate_frame_actual_transmitted_bytes,
    selected_transmitted_bytes,
)
from opencood.methods.arce.policies.ego_greedy_oracle import (
    EgoGreedyKnapsackOracle,
)
from opencood.methods.arce.policies.reward import (
    RewardBuffer,
    effective_receive_quality,
)
from opencood.methods.arce.policies.reward_update_manager import update_pending_rewards
from opencood.methods.arce.policies.communication_cost_estimator import (
    estimate_byte_stream_fec_cost as cce_estimate_byte_stream_fec_cost,
)
from opencood.methods.arce.policies.reward_pending_builder import (
    build_selected_pending_reward_item,
)
from opencood.methods.arce.policies.superarm_record_builder import build_dc2mab_superarm_record
from opencood.methods.arce.policies.communication_record_utils import (
    cache_quality as cru_cache_quality,
    make_no_send_record as cru_make_no_send_record,
    update_cache_quality_from_record as cru_update_cache_quality_from_record,
)
from opencood.methods.arce.policies.channel_budget_manager import (
    budget_source_scope as cbm_budget_source_scope,
    channel_profile_budget_bytes as cbm_channel_profile_budget_bytes,
    channel_profiles_cfg as cbm_channel_profiles_cfg,
    per_link_budget_bytes as cbm_per_link_budget_bytes,
    prepare_link_channel_budget as cbm_prepare_link_channel_budget,
    profile_for_state as cbm_profile_for_state,
    system_budget_bytes as cbm_system_budget_bytes,
    use_channel_profile_budget as cbm_use_channel_profile_budget,
)
from opencood.methods.arce.policies.action_adapter import normalize_runtime_action
from opencood.methods.arce.c2mab_local_confidence import get_cav_confidence
from opencood.methods.arce.policies.payload_context import build_payload_agent_confidence


from opencood.methods.arce.c2mab_common import (
    CHANNEL_STATE_ID_TO_NAME,
    DEFAULT_CHANNEL_PROFILES,
    QUANT_RATIO_TO_FP32,
    extract_arce_cfg as _extract_arce_cfg,
    as_list_record_len as _as_list_record_len,
    safe_get_nested as _safe_get_nested,
    profile_scalar as _profile_scalar,
    normalize_state_name as _normalize_state_name,
)




class ARCEC2MABComm:

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.arce_cfg = _extract_arce_cfg(cfg or {})


        self.transport_mode = normalize_transport_mode(self.arce_cfg)
        self.arce_cfg = apply_payload_native_transport_to_arce_cfg(self.arce_cfg)
        self.priority_layout_enabled = True
        spatial_importance_cfg = (
            self.arce_cfg.get("spatial_importance", {}) or {}
        )
        self.uses_arce_spatial_importance = bool(
            spatial_importance_cfg.get("enabled", False)
        )
        self.spatial_importance_method = str(
            spatial_importance_cfg.get("method", "feature_rms")
        ).strip().lower()

        executor_cfg = build_c2mab_executor_cfg(cfg, self.arce_cfg)
        self.executor = ARCEFixedComm(executor_cfg)
        if bool(
            getattr(
                self.executor,
                "uses_arce_spatial_importance",
                False,
            )
        ) != bool(self.uses_arce_spatial_importance):
            raise RuntimeError(
                "ARCE spatial importance configuration differs between "
                "C2MAB and its executor."
            )


        action_cfg = self.arce_cfg.get("action_space", {})
        self.actions = build_pdf_action_space(
            fec_mode=action_cfg.get(
                "fec_main",
                action_cfg.get("fec_mode", "raptor_sim"),
            ),
            send_values=action_cfg.get("send_values", action_cfg.get("send", (0, 1))),
            quant_modes=action_cfg.get("online_quant_modes", ("fp16", "int8", "int4")),
            redundancy_ratios=action_cfg.get("online_redundancy_ratios", (0.0, 0.10, 0.25, 0.60)),
            allow_fp32_send=bool(action_cfg.get("allow_fp32_send", False)),
            cache_values=action_cfg.get("cache_values", action_cfg.get("cache", (0, 1))),
            xor_group_size=int(action_cfg.get("xor_group_size", 4)),
            decode_overhead=float(action_cfg.get("decode_overhead", 0.0)),
        )
        self.action_ids = [a.action_id for a in self.actions]
        self.action_space = self.actions
        self.no_send_action = next(
            (a for a in self.actions if str(a.action_id) == NO_SEND_ACTION_ID),
            None,
        )
        if self.no_send_action is None:
            raise ValueError(f"Missing canonical no-send action: {NO_SEND_ACTION_ID}")

        fp32_online = [
            a.action_id for a in self.actions
            if not getattr(a, "is_no_send", False)
            and str(getattr(a, "quant_mode", "")).lower() == "fp32"
        ]
        if fp32_online:
            raise ValueError(
                "FP32 send actions are not allowed in online ARCE-C2MAB: "
                + ", ".join(fp32_online)
            )


        context_cfg = self.arce_cfg.get("context", {})
        self.include_cav_confidence = bool(context_cfg.get("include_cav_confidence", True))
        self.context_builder = PDFContextBuilder(
            b_max_mbps=float(
                context_cfg.get(
                    "b_max_mbps",
                    context_cfg.get("normalize_bandwidth_by_mbps", 27.0),
                )
            ),
            stale_max_ms=float(
                context_cfg.get(
                    "stale_max_ms",
                    context_cfg.get(
                        "normalize_delay_by_ms",
                        context_cfg.get("deadline_ms", 400.0),
                    ),
                )
            ),
            confidence_threshold=float(
                context_cfg.get("confidence_threshold", 0.3)
            ),
            include_cav_confidence=self.include_cav_confidence,
        )

        c2mab_cfg = self.arce_cfg.get("c2mab", {})
        default_context_dim = 7 if bool(self.include_cav_confidence) else 6
        requested_context_dim = int(c2mab_cfg.get("context_dim", default_context_dim))


        if bool(self.include_cav_confidence) and requested_context_dim != 7:
            self.context_dim_override_reason = (
                "include_cav_confidence=True requires context_dim=7; "
                "old config requested {}".format(requested_context_dim)
            )
            self.context_dim = 7
        else:
            self.context_dim_override_reason = None
            self.context_dim = requested_context_dim
        policy_cfg = copy.deepcopy(c2mab_cfg)
        for key in (
            "feedback_weight_mode",
            "feedback_weight_alpha",
            "feedback_weight_floor",
            "statistical_weight_alpha",
            "feedback_weight",
            "corrupted_feedback",
            "cw_c2ucb",
        ):
            if key not in policy_cfg and key in self.arce_cfg:
                policy_cfg[key] = self.arce_cfg[key]

        self.policy_config = C2MABPolicyConfig.from_mapping(
            policy_cfg,
            context_dim=self.context_dim,
        )

        oracle_cfg = self.arce_cfg.get("ego_oracle", {})
        self.oracle = EgoGreedyKnapsackOracle(
            eps_cost=float(oracle_cfg.get("eps_cost", 1.0)),
            diversity_aware=bool(oracle_cfg.get("diversity_aware", True)),
            min_marginal_coverage=float(
                oracle_cfg.get(
                    "min_marginal_coverage",
                    self.arce_cfg.get("min_marginal_coverage", 0.001),
                )
            ),
            oracle_cost_lambda=float(
                oracle_cfg.get(
                    "oracle_cost_lambda",
                    oracle_cfg.get(
                        "cost_lambda",
                        self.arce_cfg.get("oracle_cost_lambda", 0.12),
                    ),
                )
            ),
            explore_warmup_pulls_per_quant=int(
                oracle_cfg.get("explore_warmup_pulls_per_quant", 6)
            ),
            explore_warmup_pulls_per_rho=int(
                oracle_cfg.get("explore_warmup_pulls_per_rho", 4)
            ),
            explore_warmup_pulls_per_cache=int(
                oracle_cfg.get("explore_warmup_pulls_per_cache", 4)
            ),
            explore_bonus=float(oracle_cfg.get("explore_bonus", 1.0)),
        )


        self.sender_topk_actions = int(
            oracle_cfg.get(
                "sender_topk_actions",
                self.arce_cfg.get("sender_topk_actions", 12),
            )
        )
        self.sender_topk_actions = max(1, int(self.sender_topk_actions))
        self.sender_force_quant_coverage = bool(
            oracle_cfg.get(
                "sender_force_quant_coverage",
                self.arce_cfg.get("sender_force_quant_coverage", True),
            )
        )
        self.sender_include_low_cost = bool(
            oracle_cfg.get(
                "sender_include_low_cost",
                self.arce_cfg.get("sender_include_low_cost", True),
            )
        )


        scheduler_cfg = self.arce_cfg.get("scheduler", {}) or {}
        self.fps = float(scheduler_cfg.get("fps", 10.0))
        self.tx_window_ms = float(
            scheduler_cfg.get(
                "tx_window_ms",
                1000.0 / max(self.fps, 1e-6),
            )
        )


        self.budget_scope = str(
            scheduler_cfg.get(
                "budget_scope",
                self.arce_cfg.get("budget_scope", "global_sum_link"),
            )
        ).strip().lower()


        self.system_budget_mbps = float(
            scheduler_cfg.get(
                "system_budget_mbps",
                scheduler_cfg.get(
                    "total_budget_mbps",
                    oracle_cfg.get(
                        "total_budget_mbps",
                        oracle_cfg.get("fallback_total_budget_mbps", 5.0),
                    ),
                ),
            )
        )


        packet_cfg = self.arce_cfg.get("packetizer", {}) or {}
        self.packet_size_bytes = int(
            packet_cfg.get("packet_size_bytes", packet_cfg.get("Lp", 1024))
        )
        if self.packet_size_bytes <= 0:
            raise ValueError(
                f"packet_size_bytes must be positive, got {self.packet_size_bytes}."
            )

        self.metadata_bytes_per_packet = int(
            packet_cfg.get(
                "metadata_bytes_per_packet",
                self.arce_cfg.get("patch_selection", {}).get(
                    "metadata_bytes_per_packet",
                    0,
                ),
            )
        )
        raptorq_cfg = (self.arce_cfg.get("fec", {}) or {}).get(
            "raptorq", {}
        ) or {}
        self.raptorq_block_source_packets = int(
            raptorq_cfg.get("source_packets_per_block", 20)
        )
        self.raptorq_metadata_bytes_per_packet = int(
            raptorq_cfg.get("grace_block_id_bytes", 2)
            + raptorq_cfg.get("grace_source_count_bytes", 2)
            + raptorq_cfg.get("raptorq_payload_id_bytes", 4)
        )


        reward_cfg = self.arce_cfg.get("reward", {}) or {}
        self._validate_reward_config_schema(reward_cfg)
        self.reward_mode = str(reward_cfg.get("mode", reward_cfg.get("type", "simple_delta")))
        self.reward_lambda_delta = float(
            reward_cfg.get("lambda_delta", reward_cfg.get("lambda_ap", 1.0))
        )
        self.reward_lambda_abs = float(reward_cfg.get("lambda_abs", 0.0))
        self.reward_lambda_ap = float(reward_cfg.get("lambda_ap", self.reward_lambda_delta))
        self.reward_lambda_cost = float(reward_cfg.get("lambda_cost", 0.10))
        self.reward_cost_norm_mode = str(
            reward_cfg.get("cost_norm_mode", "reference")
        ).strip().lower()
        self.reward_cost_reference_source = str(
            reward_cfg.get("cost_reference_source", "channel_profile_max_frame_budget")
        ).strip().lower()
        self.reward_cost_reference_bytes = self._resolve_reward_cost_reference_bytes(
            reward_cfg
        )
        self.reward_lambda_delay = float(reward_cfg.get("lambda_delay", 0.05))
        self.reward_lambda_quant = float(reward_cfg.get("lambda_quant", 0.0))
        self.reward_lambda_violate = float(reward_cfg.get("lambda_violate", 0.0))
        self.reward_tau_stale_ms = float(reward_cfg.get("tau_stale_ms", 300.0))
        self.reward_stale_max_ms = float(reward_cfg.get("stale_max_ms", 100.0))
        self._print_effective_reward_config()

        self.policy_bank = C2MABPolicyBank(
            action_ids=self.action_ids,
            config=self.policy_config,
        )
        self.pending_reward = RewardBuffer()
        self.records: List[Dict[str, Any]] = []
        self.frame_records: Dict[Any, List[Dict[str, Any]]] = {}

        # Counterfactual audit controls. Normal online execution leaves these
        # disabled, so production policy selection is unchanged.
        self.forced_action_id: Optional[str] = None
        self.forced_sender_index: Optional[int] = None
        self.policy_updates_enabled = True

        self.default_ego_confidence = float(
            self.arce_cfg.get("initial_ego_confidence", 0.0)
        )
        self.last_ego_confidence = float(self.default_ego_confidence)
        self.last_cache_quality: Dict[Tuple[str, str], float] = {}


    def _policy_key(self, ego_id: Any, sender_id: Any) -> Tuple[str, str]:
        return self.policy_bank.key(ego_id, sender_id)

    def get_policy(self, ego_id: Any, sender_id: Any):
        return self.policy_bank.get(ego_id, sender_id)

    def set_forced_action(self, action_id: str, sender_index: int = 1) -> None:
        """Force one sender/action and make every other sender no-send.

        This is intended only for matched-state counterfactual audits. The
        caller should restore the communication object before normal online
        execution advances the Markov trace and policy state.
        """
        action_id = str(action_id)
        if action_id not in self.action_ids:
            raise ValueError(
                "Unknown forced action {!r}; available actions: {}".format(
                    action_id, ", ".join(self.action_ids)
                )
            )
        self.forced_action_id = action_id
        self.forced_sender_index = int(sender_index)

    def clear_forced_action(self) -> None:
        self.forced_action_id = None
        self.forced_sender_index = None

    def set_policy_updates_enabled(self, enabled: bool) -> None:
        self.policy_updates_enabled = bool(enabled)

    def _append_record(self, record: Dict[str, Any]) -> None:
        self.records.append(copy.deepcopy(record))
        frame_id = record.get("frame_id", None)
        self.frame_records.setdefault(frame_id, []).append(copy.deepcopy(record))

    def clear_records(self) -> None:
        self.records.clear()
        self.frame_records.clear()
        self.executor.clear_records()

    def reset(self, clear_cache: bool = True, clear_records: bool = True) -> None:
        self.executor.reset(clear_cache=clear_cache, clear_records=clear_records)
        self.policy_bank.clear()
        self.pending_reward = RewardBuffer()
        self.last_cache_quality.clear()
        self.clear_forced_action()
        self.set_policy_updates_enabled(True)
        if clear_records:
            self.clear_records()


    def _resolve_reward_cost_reference_bytes(self, reward_cfg: Dict[str, Any]) -> float:
        fallback = float(reward_cfg.get("cost_reference_bytes", self._system_budget_bytes()))
        source = str(
            reward_cfg.get("cost_reference_source", "channel_profile_max_frame_budget")
        ).strip().lower()
        if source not in ("channel_profile_max_frame_budget", "max_channel_profile", "profiles"):
            return float(max(fallback, 1.0))

        try:
            profiles = cbm_channel_profiles_cfg(
                self.arce_cfg,
                DEFAULT_CHANNEL_PROFILES,
                _normalize_state_name,
            )
            budgets = []
            for profile in profiles.values():
                if not isinstance(profile, dict):
                    continue
                budgets.append(float(self._channel_profile_budget_bytes(profile)))
            if budgets:
                return float(max(max(budgets), 1.0))
        except Exception:
            pass
        return float(max(fallback, 1.0))

    def _profile_for_state(self, state_name: str) -> Dict[str, Any]:
        return cbm_profile_for_state(
            state_name,
            self.arce_cfg,
            DEFAULT_CHANNEL_PROFILES,
            _normalize_state_name,
        )


    def _budget_source_scope(self) -> Tuple[str, str]:
        return cbm_budget_source_scope(
            self.arce_cfg,
            self.budget_scope,
        )

    def _use_channel_profile_budget(self) -> bool:
        return cbm_use_channel_profile_budget(
            self.arce_cfg,
            self.budget_scope,
        )

    def _system_budget_bytes(self) -> float:
        return cbm_system_budget_bytes(
            self.system_budget_mbps,
            self.tx_window_ms,
            budget_bytes_from_bandwidth,
        )

    def _channel_profile_budget_bytes(self, profile: Dict[str, Any]) -> float:
        return cbm_channel_profile_budget_bytes(
            profile,
            self.system_budget_mbps,
            self.tx_window_ms,
            _profile_scalar,
            budget_bytes_from_bandwidth,
        )

    def _per_link_budget_bytes(self, num_collaborators: int) -> float:
        return cbm_per_link_budget_bytes(
            num_collaborators,
            self.system_budget_mbps,
            self.tx_window_ms,
            budget_bytes_from_bandwidth,
        )

    def _prepare_link_channel_budget(
        self,
        data_dict: Optional[Dict[str, Any]],
        batch_idx: int,
        sender_idx: int,
        num_collaborators: int,
    ) -> Tuple[str, Dict[str, Any], float]:
        return cbm_prepare_link_channel_budget(
            data_dict=data_dict,
            batch_idx=batch_idx,
            sender_idx=sender_idx,
            num_collaborators=num_collaborators,
            arce_cfg=self.arce_cfg,
            default_channel_profiles=DEFAULT_CHANNEL_PROFILES,
            state_id_to_name=CHANNEL_STATE_ID_TO_NAME,
            system_budget_mbps=self.system_budget_mbps,
            tx_window_ms=self.tx_window_ms,
            default_budget_scope=self.budget_scope,
            normalize_state_name_fn=_normalize_state_name,
            safe_get_nested_fn=_safe_get_nested,
            profile_scalar_fn=_profile_scalar,
            budget_bytes_from_bandwidth_fn=budget_bytes_from_bandwidth,
        )


    def _estimate_byte_stream_fec_cost(
        self,
        feature_shape,
        action,
        budget_bytes=None,
        message_mask=None,
    ):
        compact_cfg = compact_sparse_cfg_for_transport(self.arce_cfg)
        runtime_message_mask = None if is_payload_native_transport(self.arce_cfg) else message_mask

        first_compact_info = estimate_compact_sparse_tokens(
            feature_shape=feature_shape,
            message_mask=runtime_message_mask,
            action=action,
            budget_bytes=budget_bytes,
            compact_sparse_cfg=compact_cfg,
        )

        first_cost_info = cce_estimate_byte_stream_fec_cost(
            feature_shape=feature_shape,
            action=action,
            budget_bytes=budget_bytes,
            packet_size_bytes=int(self.packet_size_bytes),
            metadata_bytes_per_packet=int(self.metadata_bytes_per_packet),
            raw_feature_bytes_fp32_fn=raw_feature_bytes_fp32,
            quant_ratio_to_fp32=QUANT_RATIO_TO_FP32,
            compact_token_info=first_compact_info,
            raptorq_block_source_packets=int(
                self.raptorq_block_source_packets
            ),
            raptorq_metadata_bytes_per_packet=int(
                self.raptorq_metadata_bytes_per_packet
            ),
        )

        predicted_allocated_budget = float(
            first_cost_info.get("estimated_transmitted_bytes", 0.0) or 0.0
        )

        if bool(first_compact_info.get("compact_enabled", False)) and predicted_allocated_budget > 0.0:
            second_compact_info = estimate_compact_sparse_tokens(
                feature_shape=feature_shape,
                message_mask=runtime_message_mask,
                action=action,
                budget_bytes=predicted_allocated_budget,
                compact_sparse_cfg=compact_cfg,
            )

            second_cost_info = cce_estimate_byte_stream_fec_cost(
                feature_shape=feature_shape,
                action=action,
                budget_bytes=budget_bytes,
                packet_size_bytes=int(self.packet_size_bytes),
                metadata_bytes_per_packet=int(self.metadata_bytes_per_packet),
                raw_feature_bytes_fp32_fn=raw_feature_bytes_fp32,
                quant_ratio_to_fp32=QUANT_RATIO_TO_FP32,
                compact_token_info=second_compact_info,
                raptorq_block_source_packets=int(
                    self.raptorq_block_source_packets
                ),
                raptorq_metadata_bytes_per_packet=int(
                    self.raptorq_metadata_bytes_per_packet
                ),
            )

            second_cost_info["compact_estimator_first_pass"] = dict(first_compact_info)
            second_cost_info["compact_estimator_budget_policy"] = "two_pass_predicted_allocated_budget"
            second_cost_info["compact_estimator_predicted_allocated_budget_bytes"] = float(
                predicted_allocated_budget
            )
            return second_cost_info

        first_cost_info["compact_estimator_budget_policy"] = "single_pass_proposal_budget"
        first_cost_info["compact_estimator_predicted_allocated_budget_bytes"] = float(
            predicted_allocated_budget
        )
        return first_cost_info

    def _decision_receiver_cache_context(
        self,
        batch_idx: Any,
        ego_id: Any,
        sender_idx: int,
        ego_index: int,
        frame_id: Any,
    ) -> Dict[str, Any]:
        return self.executor.receiver_cache_context(
            link_id=(batch_idx, ego_id, int(sender_idx)),
            agent_index=int(sender_idx),
            ego_index=int(ego_index),
            frame_id=frame_id,
        )

    def _cache_quality(self, ego_id: Any, sender_id: Any) -> float:
        return cru_cache_quality(
            self.last_cache_quality,
            self._policy_key(ego_id, sender_id),
        )

    def _update_cache_quality_from_record(
        self,
        ego_id: Any,
        sender_id: Any,
        record: Dict[str, Any],
    ) -> None:
        return cru_update_cache_quality_from_record(
            self.last_cache_quality,
            self._policy_key(ego_id, sender_id),
            record,
            _safe_get_nested,
        )

    def _validate_reward_config_schema(self, reward_cfg):
        """Fail fast when obsolete reward keys are present."""
        obsolete = [
            "alpha_q",
            "alpha_cost",
            "alpha_delay",
            "alpha_violation",
            "stale_norm_ms",
        ]
        present = [k for k in obsolete if k in reward_cfg]
        if present:
            raise ValueError(
                "Obsolete ARCE reward config keys are no longer accepted: "
                + ", ".join(present)
                + ". Use lambda_ap/lambda_cost/lambda_delay/"
                  "lambda_quant/lambda_violate/stale_max_ms instead."
            )

    def _print_effective_reward_config(self):
        if getattr(self, "_effective_reward_config_printed", False):
            return
        self._effective_reward_config_printed = True
        print("===== Effective ARCE reward config =====")
        print("reward_mode    =", str(getattr(self, "reward_mode", "simple_delta")))
        print("lambda_delta   =", float(getattr(self, "reward_lambda_delta", getattr(self, "reward_lambda_ap", 1.0))))
        print("lambda_abs     =", float(getattr(self, "reward_lambda_abs", 0.0)))
        print("lambda_ap      =", float(getattr(self, "reward_lambda_ap", 1.0)))
        print("lambda_cost    =", float(getattr(self, "reward_lambda_cost", 0.10)))
        print("cost_norm_mode =", str(getattr(self, "reward_cost_norm_mode", "reference")))
        print("cost_ref_bytes =", float(getattr(self, "reward_cost_reference_bytes", 0.0)))
        print("lambda_delay   =", float(getattr(self, "reward_lambda_delay", 0.05)))
        print("lambda_quant   =", float(getattr(self, "reward_lambda_quant", 0.0)))
        print("lambda_violate =", float(getattr(self, "reward_lambda_violate", 0.0)))
        print("stale_max_ms   =", float(getattr(self, "reward_stale_max_ms", 100.0)))

    def _make_no_send_record(
        self,
        feature: torch.Tensor,
        frame_id: Any,
        ego_id: Any,
        sender_id: Any,
        action: Optional[PDFARCEAction] = None,
        reason: str = "not_selected_by_oracle",
    ) -> Dict[str, Any]:
        return cru_make_no_send_record(
            feature=feature,
            frame_id=frame_id,
            ego_id=ego_id,
            sender_id=sender_id,
            action=action,
            reason=reason,
        )






    def _prepare_communication_round(
        self,
        features: torch.Tensor,
        ego_index: int,
        data_dict: Optional[Dict[str, Any]],
        batch_idx: int,
        local_cav_confidences: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Prepare per-frame communication context before C2MAB proposal building."""
        n = int(features.shape[0])
        ego_id = int(ego_index)

        collaborator_indices = [
            sender_idx for sender_idx in range(n)
            if int(sender_idx) != int(ego_index)
        ]
        num_collaborators = len(collaborator_indices)

        total_budget_bytes = self._system_budget_bytes()
        per_link_budget_bytes = self._per_link_budget_bytes(num_collaborators)

        budget_source_cfg, budget_scope_cfg = self._budget_source_scope()
        use_channel_profile_budget = self._use_channel_profile_budget()

        payload_context_cfg = (
            (self.arce_cfg.get("context", {}) or {}).get("payload_context", {}) or {}
        )
        use_payload_context = bool(
            payload_context_cfg.get(
                "enabled",
                is_payload_native_transport(self.arce_cfg),
            )
        )
        prefer_payload_conf = bool(
            payload_context_cfg.get("prefer_payload_confidence", use_payload_context)
        )

        local_ego_conf = get_cav_confidence(
            local_cav_confidences,
            int(ego_index),
            default=None,
        )
        if local_ego_conf is not None:
            ego_conf = float(local_ego_conf)
            ego_conf_source = "local_detection_confidence_summary"
        elif prefer_payload_conf:
            ego_payload_ctx = build_payload_agent_confidence(features, int(ego_index))
            ego_conf = float(
                ego_payload_ctx.get("confidence", float(self.default_ego_confidence))
            )
            ego_conf_source = "payload_ego_energy_normalized"
        else:
            ego_conf = float(self.default_ego_confidence)
            ego_conf_source = "default_ego_confidence"

        link_states: Dict[int, str] = {}
        link_profiles: Dict[int, Dict[str, Any]] = {}
        link_budgets: Dict[int, float] = {}

        for sender_idx in collaborator_indices:
            state_name, profile, link_budget_bytes = self._prepare_link_channel_budget(
                data_dict=data_dict,
                batch_idx=batch_idx,
                sender_idx=sender_idx,
                num_collaborators=num_collaborators,
            )
            if state_name == "ego_or_padding":
                continue

            link_states[sender_idx] = state_name
            link_profiles[sender_idx] = profile
            link_budgets[sender_idx] = float(link_budget_bytes)

        if use_channel_profile_budget and link_profiles:
            first_sender = next(iter(link_profiles.keys()))
            global_profile = link_profiles[first_sender]

            total_budget_bytes = float(
                self._channel_profile_budget_bytes(global_profile)
            )
            per_link_budget_bytes = (
                float(total_budget_bytes) / float(max(1, len(link_budgets)))
            )

            link_budgets = {
                int(k): float(per_link_budget_bytes)
                for k in link_budgets.keys()
            }

        return {
            "ego_id": int(ego_id),
            "collaborator_indices": collaborator_indices,
            "num_collaborators": int(num_collaborators),
            "total_budget_bytes": float(total_budget_bytes),
            "per_link_budget_bytes": float(per_link_budget_bytes),
            "budget_source_cfg": str(budget_source_cfg),
            "budget_scope_cfg": str(budget_scope_cfg),
            "use_channel_profile_budget": bool(use_channel_profile_budget),
            "ego_conf": float(ego_conf),
            "ego_conf_source": str(ego_conf_source),
            "link_states": link_states,
            "link_profiles": link_profiles,
            "link_budgets": link_budgets,
        }

    def communicate_agent_features(
        self,
        features: torch.Tensor,
        frame_id: Optional[int] = None,
        ego_index: int = 0,
        data_dict: Optional[Dict[str, Any]] = None,
        batch_idx: int = 0,
        update_cache: bool = True,
        return_records: bool = True,
        message_masks: Optional[torch.Tensor] = None,
        priority_maps: Optional[torch.Tensor] = None,

        local_cav_confidences: Optional[torch.Tensor] = None,
        local_cav_confidence_maps: Optional[torch.Tensor] = None,
    ):
        if features.dim() != 4:
            raise ValueError(f"Expected features [N,C,H,W], got {tuple(features.shape)}")

        round_ctx = self._prepare_communication_round(
            features=features,
            ego_index=int(ego_index),
            data_dict=data_dict,
            batch_idx=batch_idx,
            local_cav_confidences=local_cav_confidences,
        )

        ego_id = round_ctx["ego_id"]
        collaborator_indices = round_ctx["collaborator_indices"]
        num_collaborators = round_ctx["num_collaborators"]
        total_budget_bytes = round_ctx["total_budget_bytes"]
        self.last_frame_budget_bytes = float(total_budget_bytes)
        per_link_budget_bytes = round_ctx["per_link_budget_bytes"]
        budget_source_cfg = round_ctx["budget_source_cfg"]
        budget_scope_cfg = round_ctx["budget_scope_cfg"]
        use_channel_profile_budget = round_ctx["use_channel_profile_budget"]
        ego_conf = round_ctx["ego_conf"]
        ego_conf_source = round_ctx.get("ego_conf_source", "unknown")
        link_states = round_ctx["link_states"]
        link_profiles = round_ctx["link_profiles"]
        link_budgets = round_ctx["link_budgets"]
        runtime_message_masks = (
            None if is_payload_native_transport(self.arce_cfg) else message_masks
        )
        runtime_priority_maps = (
            None if is_payload_native_transport(self.arce_cfg) else priority_maps
        )
        if self.uses_arce_spatial_importance:
            runtime_message_masks = None
            runtime_priority_maps = None

        (
            proposals,
            no_send_candidates,
            decision_contexts,
        ) = build_c2mab_proposals(
            features_shape=features.shape[1:],
            features=features,
            collaborator_indices=collaborator_indices,
            ego_id=ego_id,
            ego_index=int(ego_index),
            ego_confidence=float(ego_conf),
            ego_confidence_source=str(ego_conf_source),
            total_budget_bytes=float(total_budget_bytes),
            per_link_budget_bytes=float(per_link_budget_bytes),
            num_collaborators=int(num_collaborators),
            budget_scope=str(budget_scope_cfg),
            budget_source=str(budget_source_cfg),
            use_channel_profile_budget=bool(use_channel_profile_budget),
            link_states=link_states,
            link_profiles=link_profiles,
            link_budgets=link_budgets,
            actions=self.actions,
            no_send_action=self.no_send_action,
            arce_cfg=self.arce_cfg,
            context_builder=self.context_builder,
            local_cav_confidences=local_cav_confidences,
            local_cav_confidence_maps=local_cav_confidence_maps,
            packet_size_bytes=int(self.packet_size_bytes),
            sender_topk_actions=int(
                len(self.actions)
                if self.forced_action_id is not None
                else self.sender_topk_actions
            ),
            sender_force_quant_coverage=bool(self.sender_force_quant_coverage),
            sender_include_low_cost=bool(self.sender_include_low_cost),
            profile_for_state_fn=self._profile_for_state,
            profile_scalar_fn=_profile_scalar,
            cache_quality_fn=lambda query_ego_id, query_sender_idx: (
                self._decision_receiver_cache_context(
                    batch_idx=batch_idx,
                    ego_id=query_ego_id,
                    sender_idx=int(query_sender_idx),
                    ego_index=int(ego_index),
                    frame_id=frame_id,
                )
            ),
            estimate_cost_fn=self._estimate_byte_stream_fec_cost,
            get_policy_fn=self.get_policy,
        )

        if self.forced_action_id is None:
            oracle_result = self.oracle.select(
                proposals, budget_bytes=total_budget_bytes
            )
        elif self.forced_action_id == NO_SEND_ACTION_ID:
            oracle_result = self.oracle.select([], budget_bytes=total_budget_bytes)
        else:
            forced_candidates = [
                proposal for proposal in proposals
                if int(proposal.sender_id) == int(self.forced_sender_index)
                and str(proposal.action_id) == str(self.forced_action_id)
            ]
            if not forced_candidates:
                available = sorted(
                    str(proposal.action_id) for proposal in proposals
                    if int(proposal.sender_id) == int(self.forced_sender_index)
                )
                raise RuntimeError(
                    "Forced action {!r} is not feasible for sender {}. "
                    "Feasible actions: {}".format(
                        self.forced_action_id,
                        self.forced_sender_index,
                        ", ".join(available) if available else "none",
                    )
                )
            oracle_result = self.oracle.select(
                forced_candidates[:1], budget_bytes=total_budget_bytes
            )
            if not oracle_result.get("selected"):
                raise RuntimeError(
                    "Forced action {!r} was feasible but the oracle did not "
                    "select it under budget {}.".format(
                        self.forced_action_id, total_budget_bytes
                    )
                )
        selected_by_sender = {
            int(p.sender_id): p for p in oracle_result["selected"]
        }
        selected_physical_budgets = selected_physical_budget_plan(
            selected_by_sender,
            total_budget_bytes=float(total_budget_bytes),
        )

        out = features.clone()
        frame_records = []
        used_cost = 0.0

        for sender_idx in collaborator_indices:
            selected = selected_by_sender.get(sender_idx, None)

            if selected is None:
                action = no_send_candidates.get(sender_idx, self.no_send_action)

                rec = execute_no_send_sender(
                    out=out,
                    sender_idx=int(sender_idx),
                    frame_id=frame_id,
                    ego_id=ego_id,
                    action=action,
                    link_states=link_states,
                    link_profiles=link_profiles,
                    link_budgets=link_budgets,
                    per_link_budget_bytes=float(per_link_budget_bytes),
                    budget_scope_cfg=str(budget_scope_cfg),
                    budget_source_cfg=str(budget_source_cfg),
                    system_budget_mbps=float(self.system_budget_mbps),
                    tx_window_ms=float(self.tx_window_ms),
                    total_budget_bytes=float(total_budget_bytes),
                    num_collaborators=int(num_collaborators),
                    ego_conf=float(ego_conf),
                    local_cav_confidences=local_cav_confidences,
                    decision_context=decision_contexts.get(
                        int(sender_idx)
                    ),
                    context_builder=self.context_builder,
                    pending_reward=self.pending_reward,
                    make_no_send_record_fn=self._make_no_send_record,
                    profile_for_state_fn=self._profile_for_state,
                    profile_scalar_fn=_profile_scalar,
                    cache_quality_fn=self._cache_quality,
                )

                frame_records.append(rec)
                self._append_record(rec)
                continue

            pdf_action: PDFARCEAction = selected.action

            arce_action = normalize_runtime_action(
                pdf_action.to_arce_action(),
                send=int(pdf_action.send),
                cache_enabled=int(pdf_action.cache_enabled),
                action_id=str(pdf_action.action_id),
            )

            state_name = selected.record.get("channel_state", "medium")
            allocated_budget_bytes = selected_physical_budgets[int(sender_idx)]

            try:
                recovered, record = self.executor.communicate_feature(
                    feature=features[sender_idx],
                    link_id=(batch_idx, ego_id, sender_idx),
                    frame_id=frame_id,
                    agent_index=sender_idx,
                    ego_index=ego_index,
                    channel_state=state_name,
                    action_override=arce_action,
                    budget_bytes=float(allocated_budget_bytes),
                    message_mask=(
                        runtime_message_masks[sender_idx]
                        if runtime_message_masks is not None
                        else None
                    ),
                    priority_map=(
                        runtime_priority_maps[sender_idx]
                        if runtime_priority_maps is not None
                        else None
                    ),
                    complementarity=float(getattr(selected, "complementarity", 0.0)),
                    update_cache=update_cache,
                    return_result=False,
                )
            except TypeError as exc:
                raise TypeError(
                    "ARCEFixedComm.communicate_feature does not accept "
                    "action_override / budget_bytes / channel_state yet. "
                    "Update arce_fixed_comm.py first."
                ) from exc

            out[sender_idx] = recovered

            record = enrich_selected_execution_record(
                record=record,
                selected=selected,
                pdf_action=pdf_action,
                oracle_result=oracle_result,
                total_budget_bytes=float(total_budget_bytes),
                budget_scope=str(budget_scope_cfg),
                budget_source=str(budget_source_cfg),
                system_budget_mbps=float(self.system_budget_mbps),
                tx_window_ms=float(self.tx_window_ms),
                num_collaborators=int(num_collaborators),
                per_link_budget_bytes=float(per_link_budget_bytes),
                allocated_budget_bytes=float(allocated_budget_bytes),
                link_budgets=link_budgets,
                debug_records=bool(self.arce_cfg.get("debug_records", False)),
            )

            decision_cache_fields = (
                "decision_cache_available",
                "decision_cache_status",
                "decision_cache_valid_unit_ratio",
                "decision_cache_num_valid_units",
                "decision_cache_num_total_units",
                "decision_cache_age_frames",
                "decision_cache_age_norm",
                "decision_cache_context_source",
            )
            for field_name in decision_cache_fields:
                if field_name in selected.record:
                    record[field_name] = selected.record[field_name]

            context_obj = getattr(selected, "context", None)
            context_vector = getattr(context_obj, "vector", None)
            if context_vector is not None:
                if hasattr(context_vector, "detach"):
                    context_values = (
                        context_vector.detach()
                        .cpu()
                        .flatten()
                        .tolist()
                    )
                elif hasattr(context_vector, "tolist"):
                    context_values = context_vector.tolist()
                else:
                    context_values = list(context_vector)

                context_values = [
                    float(value)
                    for value in context_values
                ]
                record["decision_context_vector"] = context_values
                record["decision_context_source"] = (
                    "proposal_time_decision_context"
                )

                if len(context_values) > 4:
                    record[
                        "decision_context_cache_component"
                    ] = float(context_values[4])

                    expected_cache = float(
                        record.get(
                            "decision_cache_valid_unit_ratio",
                            context_values[4],
                        )
                    )
                    record[
                        "decision_context_cache_matches_record"
                    ] = bool(
                        abs(
                            float(context_values[4])
                            - expected_cache
                        ) <= 1e-9
                    )

            tx_bytes = selected_transmitted_bytes(record, selected)

            record["budget_consistency"] = build_budget_consistency(
                selected=selected,
                allocated_budget_bytes=float(allocated_budget_bytes),
                tx_bytes=float(tx_bytes),
                record=record,
            )

            used_cost += tx_bytes

            self._update_cache_quality_from_record(ego_id, sender_idx, record)

            frame_records.append(record)
            self._append_record(record)


            pending_item = build_selected_pending_reward_item(
                selected=selected,
                record=record,
                ego_id=ego_id,
                sender_idx=sender_idx,
                tx_bytes=float(tx_bytes),
                total_budget_bytes=float(total_budget_bytes),
                link_delay_ms=float(link_profiles.get(sender_idx, {}).get("delay_ms", 0.0)),
                fallback_cache_quality=float(self._cache_quality(ego_id, sender_idx)),
                reward_tau_stale_ms=float(self.reward_tau_stale_ms),
                effective_receive_quality_fn=effective_receive_quality,
            )
            self.pending_reward.add(pending_item)

        validate_frame_actual_transmitted_bytes(
            used_cost,
            total_budget_bytes=float(total_budget_bytes),
        )

        superarm_record = build_dc2mab_superarm_record(
            frame_id=frame_id,
            batch_idx=batch_idx,
            ego_id=ego_id,
            total_budget_bytes=float(total_budget_bytes),
            budget_scope=str(budget_scope_cfg),
            budget_source=str(budget_source_cfg),
            system_budget_mbps=float(self.system_budget_mbps),
            tx_window_ms=float(self.tx_window_ms),
            num_collaborators=int(num_collaborators),
            per_link_budget_bytes=float(per_link_budget_bytes),
            link_budgets=link_budgets,
            link_states=link_states,
            used_cost=float(used_cost),
            selected_by_sender=selected_by_sender,
            oracle_result=oracle_result,
            packet_size_bytes=int(self.packet_size_bytes),
            debug_records=bool(self.arce_cfg.get("debug_records", False)),
        )
        self._append_record(superarm_record)

        if return_records:
            return out, frame_records
        return out

    def communicate_flattened_features(
        self,
        features: torch.Tensor,
        record_len: Any,
        data_dict: Optional[Dict[str, Any]] = None,
        frame_id: Optional[int] = None,
        ego_index: Optional[int] = 0,
        update_cache: bool = True,
        return_records: bool = True,
        message_masks: Optional[torch.Tensor] = None,
        priority_maps: Optional[torch.Tensor] = None,

        local_cav_confidences: Optional[torch.Tensor] = None,
        local_cav_confidence_maps: Optional[torch.Tensor] = None,
    ):
        if features.dim() != 4:
            raise ValueError(
                f"Expected flattened features [sumN,C,H,W], got {tuple(features.shape)}"
            )

        record_lens = _as_list_record_len(record_len)

        if is_payload_native_transport(self.arce_cfg):
            message_masks = None
            priority_maps = None

        outputs = []
        all_records = []
        offset = 0

        for b, n in enumerate(record_lens):
            group = features[offset: offset + n]

            group_masks = None
            if message_masks is not None:
                group_masks = message_masks[offset: offset + n]

            group_priority_maps = None
            if priority_maps is not None:
                group_priority_maps = priority_maps[offset: offset + n]

            group_local_cav_confidences = None
            if local_cav_confidences is not None:
                try:
                    group_local_cav_confidences = local_cav_confidences[offset: offset + n]
                except Exception:
                    group_local_cav_confidences = local_cav_confidences

            group_local_cav_confidence_maps = None
            if local_cav_confidence_maps is not None:
                try:
                    group_local_cav_confidence_maps = local_cav_confidence_maps[offset: offset + n]
                except Exception:
                    group_local_cav_confidence_maps = local_cav_confidence_maps

            out_group, records = self.communicate_agent_features(
                group,
                frame_id=frame_id,
                ego_index=int(ego_index or 0),
                data_dict=data_dict,
                batch_idx=b,
                update_cache=update_cache,
                return_records=True,
                message_masks=group_masks,
                priority_maps=group_priority_maps,

                local_cav_confidences=group_local_cav_confidences,
                local_cav_confidence_maps=group_local_cav_confidence_maps,
            )

            outputs.append(out_group)
            all_records.extend(records)

            offset += n

        out = torch.cat(outputs, dim=0) if outputs else features

        if return_records:
            return out, all_records
        return out

    def update_with_proxy_reward(
        self,
        collab_confidence: float,
        ego_confidence: Optional[float] = None,
        budget_bytes: Optional[float] = None,
    ) -> Dict[str, Any]:
        if ego_confidence is None:
            ego_confidence = self.last_ego_confidence
        if ego_confidence is None:
            ego_confidence = self.default_ego_confidence

        pending = self.pending_reward.pop_all()
        frame_budget_bytes = (
            float(budget_bytes)
            if budget_bytes is not None
            else float(getattr(self, "last_frame_budget_bytes", self._system_budget_bytes()))
        )

        summary = update_pending_rewards(
            pending=pending,
            ego_confidence=float(ego_confidence),
            collab_confidence=float(collab_confidence),
            budget_bytes=float(frame_budget_bytes),
            get_policy_fn=self.get_policy,
            reward_lambda_ap=float(getattr(self, "reward_lambda_ap", 1.0)),
            reward_mode=str(getattr(self, "reward_mode", "simple_delta")),
            reward_lambda_delta=float(getattr(self, "reward_lambda_delta", getattr(self, "reward_lambda_ap", 1.0))),
            reward_lambda_abs=float(getattr(self, "reward_lambda_abs", 0.0)),
            reward_lambda_cost=float(getattr(self, "reward_lambda_cost", 0.10)),
            reward_cost_norm_mode=str(getattr(self, "reward_cost_norm_mode", "reference")),
            reward_cost_reference_bytes=float(getattr(self, "reward_cost_reference_bytes", frame_budget_bytes)),
            reward_lambda_delay=float(getattr(self, "reward_lambda_delay", 0.05)),
            reward_lambda_quant=float(getattr(self, "reward_lambda_quant", 0.0)),
            reward_lambda_violate=float(getattr(self, "reward_lambda_violate", 0.0)),
            reward_stale_max_ms=float(self.reward_stale_max_ms),
            quant_quality_cfg={
                "quant_quality_prior": self.arce_cfg.get("quant_quality_prior", {})
            },
            apply_policy_update=bool(self.policy_updates_enabled),
        )

        self.last_ego_confidence = float(ego_confidence)
        self._append_record({"reward_update": summary})
        return summary

    def get_records(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.records)

    def get_frame_records(self, frame_id: Any) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.frame_records.get(frame_id, []))

    def get_summary(self) -> Dict[str, Any]:
        """Return a lightweight runtime compatibility summary."""
        budget_source, budget_scope = self._budget_source_scope()
        return summarize_dc2mab_runtime_records(
            self.records,
            budget_source=str(budget_source),
            budget_scope=str(budget_scope),
            system_budget_mbps=float(self.system_budget_mbps),
            tx_window_ms=float(self.tx_window_ms),
            system_budget_bytes=float(self._system_budget_bytes()),
        )


__all__ = ["ARCEC2MABComm"]
