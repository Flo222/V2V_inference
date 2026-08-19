"""CW-D-LinUCB: Discounted LinUCB with optional feedback weighting.

This implementation maintains two matrices per arm:

A/V   : single-discount Gram matrix, used for theta = A^{-1} b
Vt/V~ : squared-discount Gram matrix, used in UCB sandwich bonus

UCB(a) = theta_a^T c + beta * sqrt(c^T A_a^{-1} Vt_a A_a^{-1} c)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from opencood.methods.arce.reward.feedback_corruption import channel_corruption_weight


@dataclass
class LinUCBScore:
    action_id: str
    ucb: float
    mean: float
    bonus: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "action_id": self.action_id,
            "ucb": float(self.ucb),
            "mean": float(self.mean),
            "bonus": float(self.bonus),
        }


def channel_quality_weight(
    loss_rate: Optional[float] = None,
    rsrp_dbm: Optional[float] = None,
    alpha: float = 1.0,
    floor: float = 0.05,
) -> float:
    """Return feedback weight w from explicit corruption C.

    C_{t,i} is derived from channel indicators, mainly packet loss rate:
        C_{t,i} = clip(loss_rate, 0, 1)
        w_{t,i} = exp(-alpha * C_{t,i})
    """
    w, _info = channel_corruption_weight(
        loss_rate=loss_rate,
        rsrp_dbm=rsrp_dbm,
        alpha=alpha,
        floor=floor,
    )
    return float(w)


class DiscountedLinUCB:
    def __init__(
        self,
        action_ids: Sequence[str],
        context_dim: int = 5,
        lambda_reg: float = 1.0,
        discount: float = 0.97,
        beta: float = 1.0,
        feedback_weight_mode: str = "channel_quality",
        feedback_weight_alpha: float = 1.0,
        feedback_weight_floor: float = 0.05,
        statistical_weight_alpha: float = 1.0,
    ):
        self.action_ids = list(action_ids)
        if not self.action_ids:
            raise ValueError("DiscountedLinUCB requires at least one action id.")

        self.d = int(context_dim)
        self.lambda_reg = float(lambda_reg)
        self.discount = float(discount)
        self.beta = float(beta)

        if not (0.0 < self.discount <= 1.0):
            raise ValueError("discount should be in (0, 1].")

        mode = str(feedback_weight_mode).strip().lower()
        if mode not in ("none", "channel_quality", "statistical"):
            raise ValueError(
                "Unsupported feedback_weight_mode={!r}; expected none/channel_quality/statistical".format(mode)
            )

        self.feedback_weight_mode = mode
        self.feedback_weight_alpha = float(feedback_weight_alpha)
        self.feedback_weight_floor = float(feedback_weight_floor)
        self.statistical_weight_alpha = float(statistical_weight_alpha)

        self.A: Dict[str, np.ndarray] = {
            a: self.lambda_reg * np.eye(self.d, dtype=np.float64)
            for a in self.action_ids
        }

        # Vt / V_tilde: required by Discounted LinUCB.
        # A uses gamma; Vt uses gamma^2.
        self.Vt: Dict[str, np.ndarray] = {
            a: self.lambda_reg * np.eye(self.d, dtype=np.float64)
            for a in self.action_ids
        }

        self.b: Dict[str, np.ndarray] = {
            a: np.zeros((self.d,), dtype=np.float64)
            for a in self.action_ids
        }

        self.t = 0
        self.last_feedback_weight = {
            a: 1.0 for a in self.action_ids
        }
        self.last_feedback_corruption_C = {
            a: 0.0 for a in self.action_ids
        }
        self.last_feedback_corruption_info = {
            a: {"feedback_corruption_C": 0.0, "corruption_source": "init"}
            for a in self.action_ids
        }

    def _context(self, context: Any) -> np.ndarray:
        c = np.asarray(context, dtype=np.float64).reshape(-1)
        if c.shape[0] != self.d:
            raise ValueError(
                "Context dimension mismatch: expected {}, got {}.".format(self.d, c.shape[0])
            )
        return c

    def score(self, action_id: str, context: Any) -> LinUCBScore:
        if action_id not in self.A:
            raise KeyError("Unknown action_id: {}".format(action_id))

        c = self._context(context)
        A_inv = np.linalg.inv(self.A[action_id])
        theta = A_inv @ self.b[action_id]
        mean = float(theta @ c)

        # Correct D-LinUCB sandwich confidence term:
        # c^T A^{-1} Vt A^{-1} c
        sandwich_vec = A_inv @ c
        var = float(sandwich_vec @ self.Vt[action_id] @ sandwich_vec)
        bonus = float(self.beta * np.sqrt(max(var, 0.0)))

        return LinUCBScore(
            action_id=action_id,
            ucb=float(mean + bonus),
            mean=float(mean),
            bonus=float(bonus),
        )

    def select(self, feasible_action_ids: Iterable[str], context: Any) -> LinUCBScore:
        best: Optional[LinUCBScore] = None
        for a in feasible_action_ids:
            s = self.score(a, context)
            if best is None or s.ucb > best.ucb:
                best = s
        if best is None:
            raise ValueError("No feasible action is available for LinUCB selection.")
        return best

    def _apply_discount(self) -> None:
        eye = self.lambda_reg * np.eye(self.d, dtype=np.float64)
        discount_sq = self.discount * self.discount

        for a in self.action_ids:
            self.A[a] = self.discount * self.A[a] + (1.0 - self.discount) * eye
            self.Vt[a] = discount_sq * self.Vt[a] + (1.0 - discount_sq) * eye
            self.b[a] = self.discount * self.b[a]

    def _compute_feedback_weight(
        self,
        action_id: str,
        c: np.ndarray,
        channel_profile: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute feedback weight w_{t,i} and store explicit corruption C."""
        if self.feedback_weight_mode == "none":
            self.last_feedback_corruption_C[action_id] = 0.0
            self.last_feedback_corruption_info[action_id] = {
                "feedback_corruption_C": 0.0,
                "corruption_source": "none",
                "feedback_weight": 1.0,
                "feedback_weight_formula": "none",
            }
            return 1.0

        if self.feedback_weight_mode == "channel_quality":
            channel_profile = channel_profile or {}
            w, info = channel_corruption_weight(
                loss_rate=channel_profile.get("loss_rate"),
                rsrp_dbm=channel_profile.get("rsrp_dbm"),
                alpha=self.feedback_weight_alpha,
                floor=self.feedback_weight_floor,
            )
            self.last_feedback_corruption_C[action_id] = float(
                info.get("feedback_corruption_C", 0.0)
            )
            self.last_feedback_corruption_info[action_id] = dict(info)
            return float(w)

        # mode == "statistical": faithful CW-C2UCB, w = min(1, alpha/||c||_{A^-1})
        A_inv = np.linalg.inv(self.A[action_id])
        norm = float(np.sqrt(max(float(c @ A_inv @ c), 1e-12)))
        w = self.statistical_weight_alpha / max(norm, 1e-12)
        w = float(max(self.feedback_weight_floor, min(1.0, w)))
        self.last_feedback_corruption_C[action_id] = 0.0
        self.last_feedback_corruption_info[action_id] = {
            "feedback_corruption_C": 0.0,
            "corruption_source": "statistical_weight",
            "feedback_weight": float(w),
            "feedback_weight_formula": "min(1, alpha / ||context||_{A^-1})",
        }
        return float(w)

    def update(
        self,
        action_id: str,
        context: Any,
        reward: float,
        channel_profile: Optional[Dict[str, Any]] = None,
    ) -> float:
        if action_id not in self.A:
            raise KeyError("Unknown action_id: {}".format(action_id))

        c = self._context(context)
        w = self._compute_feedback_weight(
            action_id,
            c,
            channel_profile=channel_profile,
        )
        self.last_feedback_weight[action_id] = float(w)

        self._apply_discount()

        outer_c = np.outer(c, c)
        self.A[action_id] += float(w) * outer_c
        self.Vt[action_id] += float(w) * float(w) * outer_c
        self.b[action_id] += float(w) * float(reward) * c

        self.t += 1
        return float(w)

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "action_ids": list(self.action_ids),
            "context_dim": self.d,
            "lambda_reg": self.lambda_reg,
            "discount": self.discount,
            "beta": self.beta,
            "feedback_weight_mode": self.feedback_weight_mode,
            "feedback_weight_alpha": self.feedback_weight_alpha,
            "feedback_weight_floor": self.feedback_weight_floor,
            "statistical_weight_alpha": self.statistical_weight_alpha,
            "t": self.t,
            "A": {k: v.tolist() for k, v in self.A.items()},
            "Vt": {k: v.tolist() for k, v in self.Vt.items()},
            "b": {k: v.tolist() for k, v in self.b.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.t = int(state.get("t", 0))

        for k, v in state.get("A", {}).items():
            if k in self.A:
                self.A[k] = np.asarray(v, dtype=np.float64)

        for k, v in state.get("Vt", {}).items():
            if k in self.Vt:
                self.Vt[k] = np.asarray(v, dtype=np.float64)

        for k, v in state.get("b", {}).items():
            if k in self.b:
                self.b[k] = np.asarray(v, dtype=np.float64)


__all__ = ["DiscountedLinUCB", "LinUCBScore", "channel_quality_weight"]
