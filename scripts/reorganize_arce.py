#!/usr/bin/env python3
"""Losslessly reorganize ARCE modules and update in-repository imports."""
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path("opencood/methods/arce")
PROJECT = Path("opencood")

MOVES = {
    "arce_fixed_comm.py": "executors/fixed_executor.py",
    "arce_c2mab_comm.py": "executors/c2mab_executor.py",
    "fixed_policy.py": "policy/fixed_policy.py",
    "random_policy.py": "policy/random_policy.py",
    "c2mab_common.py": "common.py",
    "c2mab_local_confidence.py": "context/local_confidence.py",
    "priority_block_fec_transport.py": "transport_policy/priority_fec_scheduler.py",
    "policies/action_adapter.py": "transport_policy/action_adapter.py",
    "policies/action_space.py": "policy/c2mab/action_space.py",
    "policies/discounted_linucb.py": "policy/c2mab/discounted_linucb.py",
    "policies/c2mab_policy_bank.py": "policy/c2mab/policy_bank.py",
    "policies/c2mab_proposal_builder.py": "policy/c2mab/proposal_builder.py",
    "policies/c2mab_no_send_executor.py": "policy/c2mab/no_send_executor.py",
    "policies/c2mab_executor_config.py": "policy/c2mab/executor_config.py",
    "policies/context_builder.py": "context/context_builder.py",
    "policies/payload_context.py": "context/payload_context.py",
    "policies/complementarity.py": "context/complementarity.py",
    "policies/spatial_importance.py": "context/spatial_importance.py",
    "policies/sender_candidate_selector.py": "context/sender_candidate_selector.py",
    "policies/reward.py": "reward/reward.py",
    "policies/ap_gain_reward.py": "reward/ap_gain_reward.py",
    "policies/reward_pending_builder.py": "reward/reward_pending_builder.py",
    "policies/reward_update_manager.py": "reward/reward_update_manager.py",
    "policies/feedback_corruption.py": "reward/feedback_corruption.py",
    "policies/ap_proxy_features.py": "reward/proxy/ap_proxy_features.py",
    "policies/decoded_box_proxy_features.py": "reward/proxy/decoded_box_proxy_features.py",
    "policies/quant_quality.py": "reward/proxy/quant_quality.py",
    "policies/channel_budget_manager.py": "cost/channel_budget_manager.py",
    "policies/communication_cost_estimator.py": "cost/communication_cost_estimator.py",
    "policies/compact_sparse_cost_helper.py": "cost/compact_sparse_cost_helper.py",
    "policies/bandwidth_patch_selector.py": "cost/bandwidth_patch_selector.py",
    "policies/c2mab_execution_record_builder.py": "runtime/execution_record_builder.py",
    "policies/communication_record_utils.py": "runtime/communication_record_utils.py",
    "policies/communication_volume_summary.py": "runtime/communication_volume_summary.py",
    "policies/c2mab_runtime_summary.py": "runtime/runtime_summary.py",
    "policies/superarm_record_builder.py": "runtime/superarm_record_builder.py",
    "policies/payload_transport.py": "transport_policy/payload_transport.py",
}

MODULE_RENAMES = {
    "opencood.methods.arce.executors.fixed_executor": "opencood.methods.arce.executors.fixed_executor",
    "opencood.methods.arce.executors.c2mab_executor": "opencood.methods.arce.executors.c2mab_executor",
    "opencood.methods.arce.policy.fixed_policy": "opencood.methods.arce.policy.fixed_policy",
    "opencood.methods.arce.policy.random_policy": "opencood.methods.arce.policy.random_policy",
    "opencood.methods.arce.common": "opencood.methods.arce.common",
    "opencood.methods.arce.context.local_confidence": "opencood.methods.arce.context.local_confidence",
    "opencood.methods.arce.transport_policy.priority_fec_scheduler": "opencood.methods.arce.transport_policy.priority_fec_scheduler",
    "opencood.methods.arce.transport_policy.action_adapter": "opencood.methods.arce.transport_policy.action_adapter",
    "opencood.methods.arce.policy.c2mab.action_space": "opencood.methods.arce.policy.c2mab.action_space",
    "opencood.methods.arce.policy.c2mab.discounted_linucb": "opencood.methods.arce.policy.c2mab.discounted_linucb",
    "opencood.methods.arce.policy.c2mab.policy_bank": "opencood.methods.arce.policy.c2mab.policy_bank",
    "opencood.methods.arce.policy.c2mab.proposal_builder": "opencood.methods.arce.policy.c2mab.proposal_builder",
    "opencood.methods.arce.policy.c2mab.no_send_executor": "opencood.methods.arce.policy.c2mab.no_send_executor",
    "opencood.methods.arce.policy.c2mab.executor_config": "opencood.methods.arce.policy.c2mab.executor_config",
    "opencood.methods.arce.policy.c2mab.proposal_builder": "opencood.methods.arce.policy.c2mab.proposal_builder",
    "opencood.methods.arce.context.context_builder": "opencood.methods.arce.context.context_builder",
    "opencood.methods.arce.context.payload_context": "opencood.methods.arce.context.payload_context",
    "opencood.methods.arce.context.complementarity": "opencood.methods.arce.context.complementarity",
    "opencood.methods.arce.context.spatial_importance": "opencood.methods.arce.context.spatial_importance",
    "opencood.methods.arce.context.sender_candidate_selector": "opencood.methods.arce.context.sender_candidate_selector",
    "opencood.methods.arce.reward.reward": "opencood.methods.arce.reward.reward",
    "opencood.methods.arce.reward.ap_gain_reward": "opencood.methods.arce.reward.ap_gain_reward",
    "opencood.methods.arce.reward.reward_pending_builder": "opencood.methods.arce.reward.reward_pending_builder",
    "opencood.methods.arce.reward.reward_update_manager": "opencood.methods.arce.reward.reward_update_manager",
    "opencood.methods.arce.reward.feedback_corruption": "opencood.methods.arce.reward.feedback_corruption",
    "opencood.methods.arce.reward.proxy.ap_proxy_features": "opencood.methods.arce.reward.proxy.ap_proxy_features",
    "opencood.methods.arce.reward.proxy.decoded_box_proxy_features": "opencood.methods.arce.reward.proxy.decoded_box_proxy_features",
    "opencood.methods.arce.reward.proxy.quant_quality": "opencood.methods.arce.reward.proxy.quant_quality",
    "opencood.methods.arce.cost.channel_budget_manager": "opencood.methods.arce.cost.channel_budget_manager",
    "opencood.methods.arce.cost.communication_cost_estimator": "opencood.methods.arce.cost.communication_cost_estimator",
    "opencood.methods.arce.cost.compact_sparse_cost_helper": "opencood.methods.arce.cost.compact_sparse_cost_helper",
    "opencood.methods.arce.cost.bandwidth_patch_selector": "opencood.methods.arce.cost.bandwidth_patch_selector",
    "opencood.methods.arce.runtime.execution_record_builder": "opencood.methods.arce.runtime.execution_record_builder",
    "opencood.methods.arce.runtime.communication_record_utils": "opencood.methods.arce.runtime.communication_record_utils",
    "opencood.methods.arce.runtime.communication_volume_summary": "opencood.methods.arce.runtime.communication_volume_summary",
    "opencood.methods.arce.runtime.runtime_summary": "opencood.methods.arce.runtime.runtime_summary",
    "opencood.methods.arce.runtime.superarm_record_builder": "opencood.methods.arce.runtime.superarm_record_builder",
    "opencood.methods.arce.transport_policy.payload_transport": "opencood.methods.arce.transport_policy.payload_transport",
}


def move(source: str, target: str) -> None:
    src, dst = ROOT / source, ROOT / target
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def write_package_inits() -> None:
    for directory in ("executors", "policy", "policy/c2mab", "context", "reward",
                      "reward/proxy", "cost", "runtime", "transport_policy"):
        path = ROOT / directory / "__init__.py"
        path.touch(exist_ok=True)


def merge_oracle() -> None:
    oracle = ROOT / "policies/ego_greedy_oracle.py"
    proposal = ROOT / "policy/c2mab/proposal_builder.py"
    if not oracle.is_file() or not proposal.is_file():
        raise FileNotFoundError("Expected C2MAB oracle and proposal builder")
    oracle_text = oracle.read_text(encoding="utf-8")
    proposal_text = proposal.read_text(encoding="utf-8")
    proposal_text = proposal_text.replace(
        "from opencood.methods.arce.policy.c2mab.proposal_builder import CAVProposal\n", ""
    )
    proposal.write_text(oracle_text + "\n\n# Proposal construction\n\n" + proposal_text,
                        encoding="utf-8")
    oracle.unlink()


def update_imports() -> None:
    for source_root in (PROJECT, Path("scripts")):
        if not source_root.exists():
            continue
        for path in source_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            changed = text
            for old, new in MODULE_RENAMES.items():
                changed = changed.replace(old, new)
            if path == ROOT / "controller.py":
                changed = changed.replace("from .arce_fixed_comm", "from .executors.fixed_executor")
                changed = changed.replace("from .arce_c2mab_comm", "from .executors.c2mab_executor")
                changed = changed.replace("from .fixed_policy", "from .policy.fixed_policy")
                changed = changed.replace("from .random_policy", "from .policy.random_policy")
            # The merged proposal module now owns CAVProposal; importing itself
            # while it initializes would create a circular import.
            if path == ROOT / "policy/c2mab/proposal_builder.py":
                changed = changed.replace(
                    "from opencood.methods.arce.policy.c2mab.proposal_builder import CAVProposal\n", ""
                )
            if changed != text:
                path.write_text(changed, encoding="utf-8")


def write_readme() -> None:
    (ROOT / "README.md").write_text(
        """# ARCE module layout\n\n- `controller.py`: public facade for ARCE executors and policies.\n- `executors/`: fixed and C2MAB communication execution.\n- `policy/`: fixed/random policies and C2MAB action, UCB, proposal and bank logic.\n- `context/`, `reward/`, `cost/`, `runtime/`: independent decision-support layers.\n- `transport_policy/`: payload transport actions and priority FEC scheduling.\n- `audit/`: compression and FEC recovery auditors.\n\n`common.py` contains shared ARCE/C2MAB helpers.  Import implementations from\nthe role-specific packages above; use `controller.py` for the public facade.\n""",
        encoding="utf-8")


def main() -> None:
    for source, target in MOVES.items():
        move(source, target)
    write_package_inits()
    merge_oracle()
    update_imports()
    write_readme()
    legacy_init = ROOT / "policies/__init__.py"
    if legacy_init.exists():
        legacy_init.unlink()
    for directory in sorted((ROOT / "policies").rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    print("ARCE_REORGANIZED modules={}".format(len(MOVES) + 1))


if __name__ == "__main__":
    main()
