from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from opencood.methods.arce.policies.discounted_linucb import DiscountedLinUCB


@dataclass(frozen=True)
class C2MABPolicyConfig:
    context_dim: int
    lambda_reg: float = 1.0
    discount: float = 0.97
    beta: float = 1.0
    feedback_weight_mode: str = "channel_quality"
    feedback_weight_alpha: float = 1.0
    feedback_weight_floor: float = 0.05
    statistical_weight_alpha: float = 1.0

    @classmethod
    def from_mapping(
        cls,
        cfg: Dict[str, Any],
        *,
        context_dim: int,
    ) -> "C2MABPolicyConfig":
        cfg = cfg or {}
        feedback_cfg = cfg.get("feedback_weight", None)
        if not isinstance(feedback_cfg, dict):
            feedback_cfg = cfg.get("corrupted_feedback", None)
        if not isinstance(feedback_cfg, dict):
            feedback_cfg = cfg.get("cw_c2ucb", None)
        if not isinstance(feedback_cfg, dict):
            feedback_cfg = {}

        def _get(name: str, default: Any) -> Any:
            return cfg.get(name, feedback_cfg.get(name, default))

        return cls(
            context_dim=int(context_dim),
            lambda_reg=float(cfg.get("lambda_reg", 1.0)),
            discount=float(cfg.get("discount", 0.97)),
            beta=float(cfg.get("beta", 1.0)),
            feedback_weight_mode=str(
                _get("feedback_weight_mode", "channel_quality")
            ),
            feedback_weight_alpha=float(_get("feedback_weight_alpha", 1.0)),
            feedback_weight_floor=float(_get("feedback_weight_floor", 0.05)),
            statistical_weight_alpha=float(_get("statistical_weight_alpha", 1.0)),
        )


class C2MABPolicyBank:
    """Lazy owner of per ego-sender DiscountedLinUCB policies."""

    def __init__(self, action_ids: Iterable[str], config: C2MABPolicyConfig):
        self.action_ids = list(action_ids)
        self.config = config
        self._policies: Dict[Tuple[str, str], DiscountedLinUCB] = {}

    @staticmethod
    def key(ego_id: Any, sender_id: Any) -> Tuple[str, str]:
        return (str(ego_id), str(sender_id))

    @property
    def policies(self) -> Dict[Tuple[str, str], DiscountedLinUCB]:
        return self._policies

    def get(self, ego_id: Any, sender_id: Any) -> DiscountedLinUCB:
        key = self.key(ego_id, sender_id)
        if key not in self._policies:
            cfg = self.config
            self._policies[key] = DiscountedLinUCB(
                action_ids=self.action_ids,
                context_dim=cfg.context_dim,
                lambda_reg=cfg.lambda_reg,
                discount=cfg.discount,
                beta=cfg.beta,
                feedback_weight_mode=cfg.feedback_weight_mode,
                feedback_weight_alpha=cfg.feedback_weight_alpha,
                feedback_weight_floor=cfg.feedback_weight_floor,
                statistical_weight_alpha=cfg.statistical_weight_alpha,
            )
        return self._policies[key]

    def clear(self) -> None:
        self._policies.clear()

    def __len__(self) -> int:
        return len(self._policies)
