from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch


# Kept unchanged so previously trained v2 models remain loadable.
DENSE_AP_PROXY_FEATURES = [
    "dense_mean_conf",
    "dense_max_conf",
    "dense_sum_conf",
    "dense_std_conf",
    "dense_count_gt_03",
    "dense_count_gt_05",
    "dense_count_gt_07",
    "dense_top10_mean",
    "dense_top50_mean",
]

PSM_V3_EXTRA_FEATURES = [
    "dense_p50",
    "dense_p75",
    "dense_p90",
    "dense_p95",
    "dense_p99",
    "dense_entropy_mean",
    "dense_ratio_gt_03",
    "dense_ratio_gt_05",
    "dense_ratio_gt_07",
    "dense_grad_x_abs_mean",
    "dense_grad_y_abs_mean",
]

REGRESSION_AP_PROXY_FEATURES = [
    "reg_abs_mean",
    "reg_abs_std",
    "reg_abs_max",
    "reg_rms",
    "reg_abs_p50",
    "reg_abs_p90",
    "reg_abs_p99",
    "reg_conf_weighted_abs_mean",
    "reg_conf_weighted_rms",
]

HEAD_AP_PROXY_FEATURES = (
    list(DENSE_AP_PROXY_FEATURES)
    + list(PSM_V3_EXTRA_FEATURES)
    + list(REGRESSION_AP_PROXY_FEATURES)
)

PAIRED_SPATIAL_AP_PROXY_FEATURES = [
    "spatial_prob_l1_mean",
    "spatial_prob_rms",
    "spatial_prob_max_abs",
    "spatial_prob_gain_mean",
    "spatial_prob_loss_mean",
    "spatial_prob_gain_sum",
    "spatial_prob_loss_sum",
    "spatial_prob_change_ratio_001",
    "spatial_prob_change_ratio_005",
    "spatial_prob_change_ratio_010",
    "spatial_prob_cosine",
    "spatial_top50_overlap",
    "reg_diff_abs_mean",
    "reg_diff_rms",
    "reg_diff_abs_max",
]


def _zero_features(names) -> Dict[str, float]:
    return {str(name): 0.0 for name in names}


def _finite_scalar(value: torch.Tensor) -> float:
    result = float(value.detach().cpu().item())
    return result if result == result and abs(result) != float("inf") else 0.0


def _quantiles(
    flat: torch.Tensor,
    values: Sequence[float],
) -> Sequence[float]:
    if flat.numel() == 0:
        return [0.0 for _ in values]
    quantiles = torch.as_tensor(
        list(values),
        dtype=flat.dtype,
        device=flat.device,
    )
    result = torch.quantile(flat, quantiles)
    return [_finite_scalar(item) for item in result]


def _dense_probability_map(psm: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(psm.detach()).float()
    if prob.dim() == 4:
        return prob.max(dim=1)[0]
    if prob.dim() == 3:
        return prob
    if prob.dim() == 2:
        return prob.unsqueeze(0)
    return prob.reshape(1, 1, -1)


def dense_ap_proxy_features(psm: torch.Tensor) -> Dict[str, float]:
    """Extract the canonical v2 class-collapsed dense-head features."""
    with torch.no_grad():
        dense = _dense_probability_map(psm)
        flat = dense.reshape(-1)

        if flat.numel() == 0:
            return _zero_features(DENSE_AP_PROXY_FEATURES)

        top10 = torch.topk(flat, k=min(10, int(flat.numel()))).values
        top50 = torch.topk(flat, k=min(50, int(flat.numel()))).values
        return {
            "dense_mean_conf": _finite_scalar(flat.mean()),
            "dense_max_conf": _finite_scalar(flat.max()),
            "dense_sum_conf": _finite_scalar(flat.sum()),
            "dense_std_conf": _finite_scalar(flat.std(unbiased=False)),
            "dense_count_gt_03": _finite_scalar((flat > 0.3).sum()),
            "dense_count_gt_05": _finite_scalar((flat > 0.5).sum()),
            "dense_count_gt_07": _finite_scalar((flat > 0.7).sum()),
            "dense_top10_mean": _finite_scalar(top10.mean()),
            "dense_top50_mean": _finite_scalar(top50.mean()),
        }


def _psm_v3_extra_features(psm: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        dense = _dense_probability_map(psm)
        flat = dense.reshape(-1)
        if flat.numel() == 0:
            return _zero_features(PSM_V3_EXTRA_FEATURES)

        eps = 1e-6
        clipped = flat.clamp(min=eps, max=1.0 - eps)
        entropy = -(
            clipped * clipped.log()
            + (1.0 - clipped) * (1.0 - clipped).log()
        )
        p50, p75, p90, p95, p99 = _quantiles(
            flat,
            (0.50, 0.75, 0.90, 0.95, 0.99),
        )
        grad_x = (
            (dense[..., 1:] - dense[..., :-1]).abs()
            if dense.shape[-1] > 1
            else dense.new_zeros((0,))
        )
        grad_y = (
            (dense[..., 1:, :] - dense[..., :-1, :]).abs()
            if dense.dim() >= 3 and dense.shape[-2] > 1
            else dense.new_zeros((0,))
        )
        count = float(flat.numel())
        return {
            "dense_p50": p50,
            "dense_p75": p75,
            "dense_p90": p90,
            "dense_p95": p95,
            "dense_p99": p99,
            "dense_entropy_mean": _finite_scalar(entropy.mean()),
            "dense_ratio_gt_03": float((flat > 0.3).sum().item()) / count,
            "dense_ratio_gt_05": float((flat > 0.5).sum().item()) / count,
            "dense_ratio_gt_07": float((flat > 0.7).sum().item()) / count,
            "dense_grad_x_abs_mean": (
                _finite_scalar(grad_x.mean()) if grad_x.numel() else 0.0
            ),
            "dense_grad_y_abs_mean": (
                _finite_scalar(grad_y.mean()) if grad_y.numel() else 0.0
            ),
        }


def _regression_features(
    psm: torch.Tensor,
    rm: Optional[torch.Tensor],
) -> Dict[str, float]:
    if not torch.is_tensor(rm) or rm.numel() == 0:
        return _zero_features(REGRESSION_AP_PROXY_FEATURES)

    with torch.no_grad():
        reg = rm.detach().float()
        abs_reg = reg.abs()
        flat = abs_reg.reshape(-1)
        p50, p90, p99 = _quantiles(flat, (0.50, 0.90, 0.99))
        result = {
            "reg_abs_mean": _finite_scalar(flat.mean()),
            "reg_abs_std": _finite_scalar(flat.std(unbiased=False)),
            "reg_abs_max": _finite_scalar(flat.max()),
            "reg_rms": _finite_scalar(torch.sqrt((reg * reg).mean())),
            "reg_abs_p50": p50,
            "reg_abs_p90": p90,
            "reg_abs_p99": p99,
            "reg_conf_weighted_abs_mean": 0.0,
            "reg_conf_weighted_rms": 0.0,
        }

        if (
            reg.dim() == 4
            and psm.dim() == 4
            and reg.shape[0] == psm.shape[0]
            and reg.shape[-2:] == psm.shape[-2:]
            and psm.shape[1] > 0
            and reg.shape[1] == psm.shape[1] * 7
        ):
            batch, anchors, height, width = psm.shape
            reg_by_anchor = reg.reshape(batch, anchors, 7, height, width)
            weights = torch.sigmoid(psm.detach()).float().unsqueeze(2)
            weight_sum = weights.sum() * 7.0
            if float(weight_sum.cpu().item()) > 1e-12:
                result["reg_conf_weighted_abs_mean"] = _finite_scalar(
                    (reg_by_anchor.abs() * weights).sum() / weight_sum
                )
                result["reg_conf_weighted_rms"] = _finite_scalar(
                    torch.sqrt(
                        (reg_by_anchor.pow(2) * weights).sum() / weight_sum
                    )
                )
        return result


def head_ap_proxy_features(
    psm: torch.Tensor,
    rm: Optional[torch.Tensor],
) -> Dict[str, float]:
    """Features available identically in model forward and offline audit."""
    features = dense_ap_proxy_features(psm)
    features.update(_psm_v3_extra_features(psm))
    features.update(_regression_features(psm, rm))
    return features


def _spatial_pair_features(
    collab_psm: torch.Tensor,
    collab_rm: Optional[torch.Tensor],
    ego_psm: torch.Tensor,
    ego_rm: Optional[torch.Tensor],
) -> Dict[str, float]:
    zeros = _zero_features(PAIRED_SPATIAL_AP_PROXY_FEATURES)
    if tuple(collab_psm.shape) != tuple(ego_psm.shape):
        return zeros

    with torch.no_grad():
        collab = _dense_probability_map(collab_psm)
        ego = _dense_probability_map(ego_psm)
        if tuple(collab.shape) != tuple(ego.shape) or collab.numel() == 0:
            return zeros

        diff = collab - ego
        abs_diff = diff.abs()
        positive = diff.clamp(min=0.0)
        negative = (-diff).clamp(min=0.0)
        collab_flat = collab.reshape(-1)
        ego_flat = ego.reshape(-1)
        denominator = (
            torch.linalg.vector_norm(collab_flat)
            * torch.linalg.vector_norm(ego_flat)
        )
        cosine = (
            _finite_scalar(torch.dot(collab_flat, ego_flat) / denominator)
            if float(denominator.cpu().item()) > 1e-12
            else 0.0
        )

        topk = min(50, int(collab_flat.numel()))
        if topk > 0:
            collab_top = set(
                int(x)
                for x in torch.topk(collab_flat, k=topk).indices.cpu().tolist()
            )
            ego_top = set(
                int(x)
                for x in torch.topk(ego_flat, k=topk).indices.cpu().tolist()
            )
            top_overlap = float(len(collab_top.intersection(ego_top))) / topk
        else:
            top_overlap = 0.0

        result = {
            "spatial_prob_l1_mean": _finite_scalar(abs_diff.mean()),
            "spatial_prob_rms": _finite_scalar(
                torch.sqrt((diff * diff).mean())
            ),
            "spatial_prob_max_abs": _finite_scalar(abs_diff.max()),
            "spatial_prob_gain_mean": _finite_scalar(positive.mean()),
            "spatial_prob_loss_mean": _finite_scalar(negative.mean()),
            "spatial_prob_gain_sum": _finite_scalar(positive.sum()),
            "spatial_prob_loss_sum": _finite_scalar(negative.sum()),
            "spatial_prob_change_ratio_001": _finite_scalar(
                (abs_diff > 0.01).float().mean()
            ),
            "spatial_prob_change_ratio_005": _finite_scalar(
                (abs_diff > 0.05).float().mean()
            ),
            "spatial_prob_change_ratio_010": _finite_scalar(
                (abs_diff > 0.10).float().mean()
            ),
            "spatial_prob_cosine": cosine,
            "spatial_top50_overlap": top_overlap,
            "reg_diff_abs_mean": 0.0,
            "reg_diff_rms": 0.0,
            "reg_diff_abs_max": 0.0,
        }

        if (
            torch.is_tensor(collab_rm)
            and torch.is_tensor(ego_rm)
            and tuple(collab_rm.shape) == tuple(ego_rm.shape)
            and collab_rm.numel() > 0
        ):
            reg_diff = collab_rm.detach().float() - ego_rm.detach().float()
            result["reg_diff_abs_mean"] = _finite_scalar(reg_diff.abs().mean())
            result["reg_diff_rms"] = _finite_scalar(
                torch.sqrt((reg_diff * reg_diff).mean())
            )
            result["reg_diff_abs_max"] = _finite_scalar(reg_diff.abs().max())
        return result


def paired_head_ap_proxy_features(
    collab_psm: torch.Tensor,
    collab_rm: Optional[torch.Tensor],
    ego_psm: torch.Tensor,
    ego_rm: Optional[torch.Tensor],
    collab_head_features: Optional[Dict[str, float]] = None,
    ego_head_features: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    collab = (
        head_ap_proxy_features(collab_psm, collab_rm)
        if collab_head_features is None
        else collab_head_features
    )
    ego = (
        head_ap_proxy_features(ego_psm, ego_rm)
        if ego_head_features is None
        else ego_head_features
    )

    features = {}
    for name in HEAD_AP_PROXY_FEATURES:
        features["collab_" + name] = float(collab[name])
        features["ego_" + name] = float(ego[name])
        features["diff_" + name] = float(collab[name]) - float(ego[name])
    features.update(
        _spatial_pair_features(
            collab_psm,
            collab_rm,
            ego_psm,
            ego_rm,
        )
    )
    return features


def paired_delta_ap_proxy_features(
    collab_psm: torch.Tensor,
    ego_psm: torch.Tensor,
) -> Dict[str, float]:
    """Backward-compatible v2 paired feature API."""
    collab = dense_ap_proxy_features(collab_psm)
    ego = dense_ap_proxy_features(ego_psm)

    features = {}
    for name in DENSE_AP_PROXY_FEATURES:
        features["collab_" + name] = float(collab[name])
        features["ego_" + name] = float(ego[name])
        features["diff_" + name] = float(collab[name]) - float(ego[name])
    return features


def psm_is_identity(
    collab_psm: torch.Tensor,
    ego_psm: torch.Tensor,
    atol: float = 1e-8,
) -> bool:
    if tuple(collab_psm.shape) != tuple(ego_psm.shape):
        return False
    if collab_psm.numel() == 0:
        return True
    with torch.no_grad():
        max_abs = (
            collab_psm.detach().float() - ego_psm.detach().float()
        ).abs().max()
        return bool(float(max_abs.cpu().item()) <= float(atol))
