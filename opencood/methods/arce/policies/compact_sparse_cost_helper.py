from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F


def _safe_get_action_field(action: Any, key: str, default: Any = None) -> Any:
    if action is None:
        return default
    if isinstance(action, dict):
        return action.get(key, default)
    return getattr(action, key, default)


def _quant_bytes_per_value(action: Any) -> float:
    q = str(_safe_get_action_field(action, "quant_mode", "fp16")).lower()
    if q in ("fp32", "float32", "none", "raw"):
        return 4.0
    if q in ("fp16", "float16", "half"):
        return 2.0
    if q in ("int8", "8bit", "int8_uniform"):
        return 1.0
    if q in ("int4", "4bit", "int4_uniform"):
        return 0.5
    return 2.0


def _redundancy_ratio(action: Any) -> float:
    try:
        return float(
            _safe_get_action_field(
                action,
                "redundancy_ratio",
                _safe_get_action_field(action, "rho", 0.0),
            )
        )
    except Exception:
        return 0.0


def estimate_compact_sparse_tokens(
    *,
    feature_shape: Sequence[int],
    message_mask: Optional[torch.Tensor],
    action: Any,
    budget_bytes: Optional[float],
    compact_sparse_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = compact_sparse_cfg or {}
    bytes_per_value = _quant_bytes_per_value(action)
    rho = max(0.0, _redundancy_ratio(action))

    def disabled(reason: str) -> Dict[str, Any]:
        return {
            "compact_enabled": False,
            "token_selection_source": reason,
            "num_tokens": None,
            "num_total_tokens": None,
            "candidate_tokens": None,
            "mask_ratio": None,
            "candidate_ratio": None,
            "estimated_payload_budget_bytes": None,
            "bytes_per_value": float(bytes_per_value),
            "redundancy_ratio": float(rho),
            "token_cost_bytes": None,
        }

    if not bool(cfg.get("enabled", False)):
        return disabled("compact_disabled")

    if message_mask is None or not torch.is_tensor(message_mask):
        return disabled("missing_message_mask")

    if feature_shape is None or len(feature_shape) != 3:
        return disabled("invalid_feature_shape")

    C, H, W = [int(x) for x in feature_shape]
    score = message_mask.detach()

    if score.dim() == 3:
        score2d = score[0] if score.shape[0] == 1 else score.float().mean(dim=0)
    elif score.dim() == 2:
        score2d = score
    else:
        return disabled("invalid_message_mask_dim")

    if tuple(score2d.shape[-2:]) != (H, W):
        score2d = F.interpolate(
            score2d.view(1, 1, score2d.shape[-2], score2d.shape[-1]).float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    flat_score = score2d.reshape(-1).float()

    threshold = float(cfg.get("threshold", 0.0))
    cand = torch.nonzero(flat_score > threshold, as_tuple=False).flatten()

    token_selection_source = "mask_threshold"
    if cand.numel() == 0:
        cand = torch.arange(H * W, device=flat_score.device)
        token_selection_source = "empty_mask_fallback_all_tokens"

    scores = flat_score[cand]
    order = torch.argsort(scores, descending=True)
    cand = cand[order]

    dynamic_topk = bool(cfg.get("budget_aware_topk", False))
    max_tokens_cfg = int(cfg.get("max_tokens", -1))

    k_budget = int(cand.numel())
    used_budget_bytes = None

    token_cost = float(C) * float(bytes_per_value) * float(1.0 + rho)
    if dynamic_topk and budget_bytes is not None:
        b = float(max(0.0, budget_bytes))
        if token_cost <= 0:
            token_cost = float(C) * 2.0
        k_budget = int(b // token_cost)
        k_budget = max(1, min(int(cand.numel()), k_budget))
        used_budget_bytes = float(k_budget * token_cost)

    if max_tokens_cfg > 0:
        k_budget = min(k_budget, max_tokens_cfg)

    num_total_tokens = int(H * W)
    num_tokens = int(k_budget)
    candidate_tokens = int(cand.numel())

    return {
        "compact_enabled": True,
        "token_selection_source": token_selection_source,
        "num_tokens": num_tokens,
        "num_total_tokens": num_total_tokens,
        "candidate_tokens": candidate_tokens,
        "mask_ratio": float(num_tokens / max(num_total_tokens, 1)),
        "candidate_ratio": float(candidate_tokens / max(num_total_tokens, 1)),
        "threshold": float(threshold),
        "budget_aware_topk": bool(dynamic_topk),
        "max_tokens": int(max_tokens_cfg),
        "budget_bytes": float(budget_bytes) if budget_bytes is not None else None,
        "estimated_payload_budget_bytes": used_budget_bytes,
        "bytes_per_value": float(bytes_per_value),
        "redundancy_ratio": float(rho),
        "token_cost_bytes": float(token_cost),
    }
