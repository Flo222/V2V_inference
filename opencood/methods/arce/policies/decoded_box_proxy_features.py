from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch


SIMPLE_DECODED_BOX_FEATURES = (
    "decoded_num_pred_boxes",
    "decoded_has_predictions",
    "decoded_score_mean",
    "decoded_score_max",
    "decoded_score_sum_est",
)

RICH_DECODED_BOX_FEATURES = SIMPLE_DECODED_BOX_FEATURES + (
    "decoded_score_std",
    "decoded_score_p10",
    "decoded_score_p25",
    "decoded_score_p50",
    "decoded_score_p75",
    "decoded_score_p90",
    "decoded_score_count_ge_03",
    "decoded_score_count_ge_05",
    "decoded_score_count_ge_07",
    "decoded_center_x_mean",
    "decoded_center_x_std",
    "decoded_center_y_mean",
    "decoded_center_y_std",
    "decoded_radius_mean",
    "decoded_radius_std",
    "decoded_radius_p50",
    "decoded_radius_p90",
    "decoded_near_count_le_20m",
    "decoded_mid_count_20_40m",
    "decoded_far_count_gt_40m",
    "decoded_quadrant_pp_count",
    "decoded_quadrant_pn_count",
    "decoded_quadrant_np_count",
    "decoded_quadrant_nn_count",
    "decoded_aabb_area_mean",
    "decoded_aabb_area_std",
    "decoded_aabb_area_p50",
    "decoded_aabb_area_p90",
    "decoded_grid_10m_occupancy",
)

PAIRED_DECODED_MATCH_FEATURES = (
    "paired_match_count",
    "paired_added_count",
    "paired_removed_count",
    "paired_match_ratio_current",
    "paired_match_ratio_ego",
    "paired_match_iou_mean",
    "paired_match_iou_p50",
    "paired_match_iou_min",
    "paired_center_shift_mean",
    "paired_center_shift_p90",
    "paired_matched_score_delta_mean",
    "paired_matched_score_delta_std",
    "paired_matched_score_delta_p50",
    "paired_matched_score_delta_p90",
    "paired_matched_score_positive_ratio",
    "paired_added_score_mean",
    "paired_added_score_max",
    "paired_removed_score_mean",
    "paired_removed_score_max",
    "paired_added_high_conf_count",
    "paired_removed_high_conf_count",
)


def no_send_decoded_feature_names(
    names: Sequence[str] = RICH_DECODED_BOX_FEATURES,
) -> Tuple[str, ...]:
    return tuple("no_send_" + str(name) for name in names)


def delta_decoded_feature_names(
    names: Sequence[str] = RICH_DECODED_BOX_FEATURES,
) -> Tuple[str, ...]:
    return tuple("paired_delta_" + str(name) for name in names)


def _zeros(names: Sequence[str]) -> Dict[str, float]:
    return {str(name): 0.0 for name in names}


def _finite(value: torch.Tensor) -> float:
    result = float(value.detach().cpu().item())
    return result if math.isfinite(result) else 0.0


def _quantiles(
    values: torch.Tensor,
    probabilities: Sequence[float],
) -> Tuple[float, ...]:
    if values.numel() == 0:
        return tuple(0.0 for _ in probabilities)
    q = torch.as_tensor(
        list(probabilities),
        dtype=values.dtype,
        device=values.device,
    )
    return tuple(_finite(x) for x in torch.quantile(values, q))


def _normalise_inputs(
    pred_boxes: Optional[torch.Tensor],
    pred_scores: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(pred_boxes) or pred_boxes.numel() == 0:
        return (
            torch.empty((0, 8, 3), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )

    boxes = pred_boxes.detach().float().cpu()
    scores = (
        pred_scores.detach().float().cpu().reshape(-1)
        if torch.is_tensor(pred_scores)
        else torch.empty((0,), dtype=torch.float32)
    )
    if int(boxes.shape[0]) != int(scores.numel()):
        raise ValueError(
            "Decoded boxes/scores count mismatch: {} vs {}.".format(
                int(boxes.shape[0]),
                int(scores.numel()),
            )
        )
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("Decoded boxes and scores must be finite.")
    return boxes, scores


def _box_geometry(
    boxes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = int(boxes.shape[0])
    if count == 0:
        empty2 = torch.empty((0, 2), dtype=torch.float32)
        empty1 = torch.empty((0,), dtype=torch.float32)
        return empty2, empty2.clone(), empty2.clone(), empty1

    if boxes.dim() == 3 and boxes.shape[1] >= 4 and boxes.shape[2] >= 2:
        xy = boxes[..., :2]
        minimum = xy.min(dim=1)[0]
        maximum = xy.max(dim=1)[0]
        center = xy.mean(dim=1)
    elif boxes.dim() == 2 and boxes.shape[1] >= 5:
        center = boxes[:, :2]
        half_extent = boxes[:, 3:5].abs() * 0.5
        minimum = center - half_extent
        maximum = center + half_extent
    else:
        raise ValueError(
            "Unsupported decoded box shape {}; expected [N,8,3] corners "
            "or [N,D>=5] boxes.".format(tuple(boxes.shape))
        )

    extent = (maximum - minimum).clamp(min=0.0)
    area = extent[:, 0] * extent[:, 1]
    return center, minimum, maximum, area


def decoded_box_features(
    pred_boxes: Optional[torch.Tensor],
    pred_scores: Optional[torch.Tensor],
) -> Dict[str, float]:
    boxes, scores = _normalise_inputs(pred_boxes, pred_scores)
    result = _zeros(RICH_DECODED_BOX_FEATURES)
    count = int(boxes.shape[0])
    result["decoded_num_pred_boxes"] = float(count)
    result["decoded_has_predictions"] = float(count > 0)
    if count == 0:
        return result

    center, _, _, area = _box_geometry(boxes)
    radius = torch.linalg.vector_norm(center, dim=1)
    score_p10, score_p25, score_p50, score_p75, score_p90 = _quantiles(
        scores,
        (0.10, 0.25, 0.50, 0.75, 0.90),
    )
    radius_p50, radius_p90 = _quantiles(radius, (0.50, 0.90))
    area_p50, area_p90 = _quantiles(area, (0.50, 0.90))

    result.update({
        "decoded_score_mean": _finite(scores.mean()),
        "decoded_score_max": _finite(scores.max()),
        "decoded_score_sum_est": _finite(scores.sum()),
        "decoded_score_std": _finite(scores.std(unbiased=False)),
        "decoded_score_p10": score_p10,
        "decoded_score_p25": score_p25,
        "decoded_score_p50": score_p50,
        "decoded_score_p75": score_p75,
        "decoded_score_p90": score_p90,
        "decoded_score_count_ge_03": float((scores >= 0.3).sum().item()),
        "decoded_score_count_ge_05": float((scores >= 0.5).sum().item()),
        "decoded_score_count_ge_07": float((scores >= 0.7).sum().item()),
        "decoded_center_x_mean": _finite(center[:, 0].mean()),
        "decoded_center_x_std": _finite(
            center[:, 0].std(unbiased=False)
        ),
        "decoded_center_y_mean": _finite(center[:, 1].mean()),
        "decoded_center_y_std": _finite(
            center[:, 1].std(unbiased=False)
        ),
        "decoded_radius_mean": _finite(radius.mean()),
        "decoded_radius_std": _finite(radius.std(unbiased=False)),
        "decoded_radius_p50": radius_p50,
        "decoded_radius_p90": radius_p90,
        "decoded_near_count_le_20m": float((radius <= 20.0).sum().item()),
        "decoded_mid_count_20_40m": float(
            ((radius > 20.0) & (radius <= 40.0)).sum().item()
        ),
        "decoded_far_count_gt_40m": float((radius > 40.0).sum().item()),
        "decoded_quadrant_pp_count": float(
            ((center[:, 0] >= 0.0) & (center[:, 1] >= 0.0)).sum().item()
        ),
        "decoded_quadrant_pn_count": float(
            ((center[:, 0] >= 0.0) & (center[:, 1] < 0.0)).sum().item()
        ),
        "decoded_quadrant_np_count": float(
            ((center[:, 0] < 0.0) & (center[:, 1] >= 0.0)).sum().item()
        ),
        "decoded_quadrant_nn_count": float(
            ((center[:, 0] < 0.0) & (center[:, 1] < 0.0)).sum().item()
        ),
        "decoded_aabb_area_mean": _finite(area.mean()),
        "decoded_aabb_area_std": _finite(area.std(unbiased=False)),
        "decoded_aabb_area_p50": area_p50,
        "decoded_aabb_area_p90": area_p90,
        "decoded_grid_10m_occupancy": float(
            torch.unique(
                torch.floor(center / 10.0).to(torch.int64),
                dim=0,
            ).shape[0]
        ),
    })
    return result


def _aabb_iou(
    current_min: torch.Tensor,
    current_max: torch.Tensor,
    reference_min: torch.Tensor,
    reference_max: torch.Tensor,
) -> torch.Tensor:
    if current_min.shape[0] == 0 or reference_min.shape[0] == 0:
        return torch.zeros(
            (current_min.shape[0], reference_min.shape[0]),
            dtype=torch.float32,
        )
    inter_min = torch.maximum(
        current_min[:, None, :],
        reference_min[None, :, :],
    )
    inter_max = torch.minimum(
        current_max[:, None, :],
        reference_max[None, :, :],
    )
    inter_extent = (inter_max - inter_min).clamp(min=0.0)
    intersection = inter_extent[..., 0] * inter_extent[..., 1]
    current_area = (
        (current_max - current_min).clamp(min=0.0).prod(dim=1)
    )
    reference_area = (
        (reference_max - reference_min).clamp(min=0.0).prod(dim=1)
    )
    union = (
        current_area[:, None]
        + reference_area[None, :]
        - intersection
    )
    return torch.where(
        union > 1e-12,
        intersection / union,
        torch.zeros_like(union),
    )


def paired_decoded_box_features(
    current_boxes: Optional[torch.Tensor],
    current_scores: Optional[torch.Tensor],
    ego_boxes: Optional[torch.Tensor],
    ego_scores: Optional[torch.Tensor],
    iou_threshold: float = 0.3,
) -> Dict[str, float]:
    current_boxes, current_scores = _normalise_inputs(
        current_boxes,
        current_scores,
    )
    ego_boxes, ego_scores = _normalise_inputs(ego_boxes, ego_scores)
    result = _zeros(PAIRED_DECODED_MATCH_FEATURES)
    current_center, current_min, current_max, _ = _box_geometry(
        current_boxes
    )
    ego_center, ego_min, ego_max, _ = _box_geometry(ego_boxes)
    ious = _aabb_iou(current_min, current_max, ego_min, ego_max)

    candidates = []
    for current_index in range(int(current_boxes.shape[0])):
        for ego_index in range(int(ego_boxes.shape[0])):
            iou = float(ious[current_index, ego_index].item())
            if iou >= float(iou_threshold):
                candidates.append((-iou, current_index, ego_index))
    candidates.sort()

    used_current = set()
    used_ego = set()
    matches = []
    for negative_iou, current_index, ego_index in candidates:
        if current_index in used_current or ego_index in used_ego:
            continue
        used_current.add(current_index)
        used_ego.add(ego_index)
        matches.append((current_index, ego_index, -negative_iou))

    added = [
        index for index in range(int(current_boxes.shape[0]))
        if index not in used_current
    ]
    removed = [
        index for index in range(int(ego_boxes.shape[0]))
        if index not in used_ego
    ]
    match_count = len(matches)
    result.update({
        "paired_match_count": float(match_count),
        "paired_added_count": float(len(added)),
        "paired_removed_count": float(len(removed)),
        "paired_match_ratio_current": (
            float(match_count) / float(max(int(current_boxes.shape[0]), 1))
        ),
        "paired_match_ratio_ego": (
            float(match_count) / float(max(int(ego_boxes.shape[0]), 1))
        ),
    })

    if matches:
        current_indices = torch.as_tensor(
            [item[0] for item in matches],
            dtype=torch.long,
        )
        ego_indices = torch.as_tensor(
            [item[1] for item in matches],
            dtype=torch.long,
        )
        match_ious = torch.as_tensor(
            [item[2] for item in matches],
            dtype=torch.float32,
        )
        score_delta = (
            current_scores[current_indices] - ego_scores[ego_indices]
        )
        center_shift = torch.linalg.vector_norm(
            current_center[current_indices] - ego_center[ego_indices],
            dim=1,
        )
        iou_p50 = _quantiles(match_ious, (0.50,))[0]
        shift_p90 = _quantiles(center_shift, (0.90,))[0]
        delta_p50, delta_p90 = _quantiles(
            score_delta,
            (0.50, 0.90),
        )
        result.update({
            "paired_match_iou_mean": _finite(match_ious.mean()),
            "paired_match_iou_p50": iou_p50,
            "paired_match_iou_min": _finite(match_ious.min()),
            "paired_center_shift_mean": _finite(center_shift.mean()),
            "paired_center_shift_p90": shift_p90,
            "paired_matched_score_delta_mean": _finite(
                score_delta.mean()
            ),
            "paired_matched_score_delta_std": _finite(
                score_delta.std(unbiased=False)
            ),
            "paired_matched_score_delta_p50": delta_p50,
            "paired_matched_score_delta_p90": delta_p90,
            "paired_matched_score_positive_ratio": _finite(
                (score_delta > 0.0).float().mean()
            ),
        })

    if added:
        added_scores = current_scores[
            torch.as_tensor(added, dtype=torch.long)
        ]
        result.update({
            "paired_added_score_mean": _finite(added_scores.mean()),
            "paired_added_score_max": _finite(added_scores.max()),
            "paired_added_high_conf_count": float(
                (added_scores >= 0.5).sum().item()
            ),
        })
    if removed:
        removed_scores = ego_scores[
            torch.as_tensor(removed, dtype=torch.long)
        ]
        result.update({
            "paired_removed_score_mean": _finite(
                removed_scores.mean()
            ),
            "paired_removed_score_max": _finite(removed_scores.max()),
            "paired_removed_high_conf_count": float(
                (removed_scores >= 0.5).sum().item()
            ),
        })
    return result
